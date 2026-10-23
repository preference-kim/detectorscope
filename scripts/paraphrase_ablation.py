# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""The adversarial paraphrase experiment: one file, eight subcommands.

Does exposing a detector's interpretable features make it easier to evade? A
Gemini paraphraser rewrites AI-written dev documents under several conditions
and each detector is re-evaluated at a threshold frozen before the attack.
Nine settings (SETTINGS: Base-SAE, FT-SAE and FT-Model+SAE on COLING,
PRDB-Llama and PRDB-GPT-4o), all SGD+L1 logistic-regression heads at tau 0.85.

  python scripts/paraphrase_ablation.py <step> --setting SLUG [--root OUT]
      [--settings-yml demo/config/settings.yml] [step options]

  step        level    in -> out, under <root>/<dataset>/
  sample      dataset  dev split -> shared/sample/docs.jsonl
  exemplars   dataset  train split -> shared/sample/exemplars.jsonl
  prompts     dataset  docs (+ exemplars) -> shared/prompts/para_*.jsonl
              setting  features + spans + t -> <slug>/p1/prompts/feat.jsonl
  paraphrase  pass     prompts -> paraphrases/<arm>.jsonl (Gemini)
  thresholds  setting  eval JSON + mmap cache -> <slug>/sample/thresholds.json
  features    setting  mmap cache -> <slug>/features/doc_features.jsonl,
                       <slug>/scores/sae_orig.humans.jsonl
  score       pass     texts -> <slug>/[p<k>/]scores/sae_<arm>[.<i>-of-<n>].jsonl
                       (extractor + SAEs + head; --spans for the feat arm)
  pass        pass     p<k-1> paraphrases + scores -> p<k>/prompts/feat.jsonl
                       plus the carried (already flipped) rows

Order: sample, exemplars, prompts --arms <unguided>, paraphrase --arm para_*
(once per dataset); then per setting thresholds, features, score --arm orig
--ai-only --spans, score --arm orig --agreement 200, prompts --arms feat, and
[paraphrase --arm feat, score --arm ...] for passes 1..3 with pass --pass k
before passes 2 and 3. Dataset-level steps take any slug of the
dataset (or --dataset). Every step rewrites the same files at the same
relative paths, so a rerun resumes and the stored paper run
(outputs/adversarial_paraphrase/full) stays readable. Prompt templates are
scripts/prompts/adversarial/*.txt. The per-arm metrics, figure and tables are
computed from the score files in scripts/visualization/ (stats.py and
analysis.ipynb). The conditions are described at the prompts step and each
file's schema at the step that writes it.
"""

import argparse
import collections
import functools
import glob
import json
import os
import re
import time

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_DIR = os.path.join(REPO, "scripts", "prompts", "adversarial")
DEFAULT_OUT_ROOT = os.path.join(
    REPO, "outputs", "adversarial_paraphrase", "full"
)
DEFAULT_SETTINGS_YML = os.path.join(REPO, "demo", "config", "settings.yml")

# "COLING_dev" is the HF `dev` split kept in the dir named "test", with SAE
# caches COLING_test_* (the established naming inversion). PRDB's dev split is
# data/data/PRDB/test (171,837 rows; text column `review`, no label column:
# label = model != "human").
DATASETS = {
    "coling": {
        "path": os.path.join(REPO, "data/data/COLING_tokenized/test"),
        "train_path": os.path.join(REPO, "data/data/COLING_tokenized/train"),
        "text_col": "text",
        "model_col": "model",
        "meta_cols": ["source", "sub_source"],
        "ai_filter": None,  # all generators
        "eval_split": "COLING_dev",
        "eval_block": "all",
        "eval_rows": None,
        "cache_prefix": "COLING_test_",
        "target": "coling",
        "label": "COLING",
        "tex_label": "COLING (dev)",
        # few-shot exemplars: train humans of the same sub_source, at most
        # 400 words
        "exemplar_group_col": "sub_source",
        "exemplar_exclude_col": None,
        "exemplar_max_words": 400,
    },
    "prdb": {
        "path": os.path.join(REPO, "data/data/PRDB/test"),
        "train_path": os.path.join(REPO, "data/data/PRDB/train"),
        "text_col": "review",
        "model_col": "model",
        "meta_cols": ["conference", "paper_id", "review_id"],
        "ai_filter": "llama",  # in-domain generator only
        "eval_split": "PRDB_llama",
        "eval_block": "llama",
        "eval_rows": ("human", "llama"),
        "cache_prefix": "PRDB_test_",
        "target": "llama",
        "label": "PRDB-Llama",
        "tex_label": "PRDB (Llama)",
        # few-shot exemplars: train human reviews of the same conference whose
        # paper is not in the test set
        "exemplar_group_col": "conference",
        "exemplar_exclude_col": "paper_id",
        "exemplar_max_words": 450,
    },
}
# PRDB with the GPT-4o generator: same split, caches and descriptions;
# GPT-4o-trained heads and eval block.
DATASETS["prdb_gpt4o"] = {
    **DATASETS["prdb"],
    "ai_filter": "gpt4o",
    "eval_split": "PRDB_gpt4o",
    "eval_block": "gpt4o",
    "eval_rows": ("human", "gpt4o"),
    "target": "gpt4o",
    "label": "PRDB-GPT-4o",
    "tex_label": "PRDB (GPT-4o)",
}
FAMILY_LABEL = {
    "base_sae": "Base-SAE",
    "ft_sae": "FT-SAE",
    "ft_model_sae": "FT-Model+SAE",
}
# Per setting: dataset key, detector family, the demo settings.yml entry
# (extractor + SAEs + description dir), the mmap cache suffix, the head and
# eval-result directories. Everything else is derived in load_setting().
SETTINGS = {
    "coling_base_sae": {
        "dataset": "coling",
        "family": "base_sae",
        "yaml": "COLING · Gemma Scope (gemma-2-2b)",
        "cache": "gemmascope",
        "head_dir": "COLING_train_gemmascope",
        "eval_dir": "logreg_gemmascope",
    },
    "coling_ft_sae": {
        "dataset": "coling",
        "family": "ft_sae",
        "yaml": "COLING · Gemma-2 base",
        "cache": "base_gemma2",
        "head_dir": "COLING_train_base_gemma2",
        "eval_dir": "logreg_base_gemma2",
    },
    "coling_ft_model_sae": {
        "dataset": "coling",
        "family": "ft_model_sae",
        "yaml": "COLING · Gemma-2 fine-tuned",
        "cache": "gemma2",
        "head_dir": "COLING_train_gemma2",
        "eval_dir": "logreg_gemma2",
    },
    "prdb_base_sae": {
        "dataset": "prdb",
        "family": "base_sae",
        "yaml": "PRDB · Gemma Scope (gemma-2-2b)",
        "cache": "gemmascope",
        "head_dir": "PRDB_train_gemmascope",
        "eval_dir": "logreg_gemmascope",
    },
    "prdb_ft_sae": {
        "dataset": "prdb",
        "family": "ft_sae",
        "yaml": "PRDB · Gemma-2 base",
        "cache": "base_gemma2",
        "head_dir": "PRDB_train_base_gemma2",
        "eval_dir": "logreg_base_gemma2",
    },
    "prdb_ft_model_sae": {
        "dataset": "prdb",
        "family": "ft_model_sae",
        "yaml": "PRDB · Gemma-2 fine-tuned (llama)",
        "cache": "gemma2_llama",
        "head_dir": "PRDB_train_gemma2_llama",
        "eval_dir": "logreg_gemma2",
    },
    "prdb4o_base_sae": {
        "dataset": "prdb_gpt4o",
        "family": "base_sae",
        "yaml": "PRDB · Gemma Scope (gemma-2-2b)",
        "cache": "gemmascope",
        "head_dir": "PRDB_train_gemmascope",
        "eval_dir": "logreg_gemmascope",
    },
    "prdb4o_ft_sae": {
        "dataset": "prdb_gpt4o",
        "family": "ft_sae",
        "yaml": "PRDB · Gemma-2 base",
        "cache": "base_gemma2",
        "head_dir": "PRDB_train_base_gemma2",
        "eval_dir": "logreg_base_gemma2",
    },
    "prdb4o_ft_model_sae": {
        "dataset": "prdb_gpt4o",
        "family": "ft_model_sae",
        "yaml": "PRDB · Gemma-2 fine-tuned (gpt4o)",
        "cache": "gemma2_gpt4o",
        "head_dir": "PRDB_train_gemma2_gpt4o",
        "eval_dir": "logreg_gemma2",
    },
}
SETTING_ORDER = list(SETTINGS)
TAU = "0.85"
HEAD_STEM = "sgd_lr1e-02_l1-1e-01_threshold{tau}_{target}_logreg_ep3.pkl"
HUMAN_LABEL = "human"
DESC_MODEL = "gemini-3.1-flash-lite-preview"
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"  # the paraphraser
SAE_DIM = 16384
LAYERS = [7, 13, 17, 22]
EXTRACTOR_MAX_TOKENS = 1024  # detectorscope/model/extractor.py truncation

# Arms: "orig" = the sampled originals, "para_*" = unguided paraphrases made
# once per dataset, "feat" = the feature-guided attack (per setting and pass).
UNGUIDED_ARMS = [
    "para_free",
    "para_min",
    "para_fewshot",
    "para_fewshot_min",
    "para_antitell",
    "para_antitell_min",
]
FEWSHOT_ARMS = ["para_fewshot", "para_fewshot_min"]
PARA_ARMS = UNGUIDED_ARMS + ["feat"]
ARMS = ["orig"] + PARA_ARMS
ARMS_OF_PASS = {1: PARA_ARMS, 2: ["feat"], 3: ["feat"]}
N_EXEMPLARS = 3
MIN_WORDS, MAX_GEMMA_TOKENS = 30, 900  # sample length filter
# N_PER_DOC: features per direction shown by the feat prompt. N_GLOBAL /
# N_GLOBAL_HUMAN / MIN_DOCS_GLOBAL: the training-mean lists written to
# features/global_<slug>.json for the record (v5.1 prompts do not show them).
N_PER_DOC, N_GLOBAL, N_GLOBAL_HUMAN, MIN_DOCS_GLOBAL = 10, 15, 25, 10
TARGET_MARGIN = 3.0  # logits beyond the boundary the attacker is asked to clear
PROMPT_VERSION = "v5.1"  # feat-arm template: per-doc lists, spans, <plan> block
N_SPANS = 3  # firing passages shown per feature
N_PASSES = 3
SEED = 42


def _yaml_setting(settings_yml: str, name: str) -> dict:
  """The demo settings.yml entry called `name`, merged over its defaults.

  `sae_paths` is filled in for settings that give `sae_dir` instead.
  """
  raw = yaml.safe_load(open(settings_yml))
  for cfg in raw["settings"]:
    if cfg["name"] == name:
      cfg = {**raw.get("defaults", {}), **cfg}
      if "sae_paths" not in cfg:  # local .pth SAEs, one per hook point
        cfg["sae_paths"] = [
            f"{cfg['sae_dir']}/{h}_input_finetuned_sae.pth"
            for h in cfg["hook_points"]
        ]
      return cfg
  raise KeyError(name)


def load_setting(slug: str, root: str, settings_yml: str) -> dict:
  """Every per-setting path, derived from the registries.

  `head` is the SGD+L1 logreg pickle at tau 0.85, `eval_json`/`eval_block`
  the dev eval results the threshold comes from, `mmap_cache` the dev-split
  max-activation cache, `desc_dir` the auto-interp descriptions, `cfg` the demo
  settings entry (extractor + SAEs) and `root` the output tree.
  """
  if slug not in SETTINGS:
    raise SystemExit(f"--setting must be one of {SETTING_ORDER} (got {slug!r})")
  s = SETTINGS[slug]
  d = DATASETS[s["dataset"]]
  cfg = _yaml_setting(settings_yml, s["yaml"])
  return {
      "slug": slug,
      "dataset": s["dataset"],
      "family": s["family"],
      "label": FAMILY_LABEL[s["family"]],
      "root": os.path.abspath(root),
      "head": os.path.join(
          REPO,
          "outputs/logreg_sgd_l1_ablation/models",
          s["head_dir"],
          HEAD_STEM.format(tau=TAU, target=d["target"]),
      ),
      "eval_json": os.path.join(
          REPO,
          "eval_results",
          d["eval_split"],
          s["eval_dir"],
          f"threshold{TAU}_{d['target']}_logreg_results.json",
      ),
      "eval_block": d["eval_block"],
      "feature_stats": os.path.join(
          REPO,
          "outputs/logreg_sgd_l1_ablation/feature_stats",
          f"{s['head_dir']}_{d['target']}_feature_stats.npz",
      ),
      "mmap_cache": os.path.join(
          REPO, "outputs/mmap_caches", d["cache_prefix"] + s["cache"]
      ),
      "desc_dir": cfg["data_dir"],
      "model_path": cfg["model_path"],
      "cfg": cfg,
  }


def setting_from_args(args) -> dict:
  """load_setting() for --setting, or for --dataset's first setting."""
  slug = args.setting
  if slug is None:
    slug = next(
        (k for k in SETTING_ORDER if SETTINGS[k]["dataset"] == args.dataset),
        None,
    )
  if slug is None:
    raise SystemExit("give --setting SLUG or --dataset KEY")
  return load_setting(slug, args.root, args.settings_yml)


# <root>/<dataset>/shared/   sample + unguided prompts and paraphrases, shared
#                            by the dataset's settings
# <root>/<dataset>/<slug>/   thresholds, feature lists, originals' scores
# <root>/<dataset>/<slug>/p<k>/  pass k of the feature-guided attack (p1 also
#                            holds this setting's scores of the unguided arms)
def _p(*parts: str) -> str:
  """Join, create the parent directory, return the path."""
  path = os.path.join(*parts)
  os.makedirs(os.path.dirname(path), exist_ok=True)
  return path


def shared_path(s: dict, *parts: str) -> str:
  """A file under <root>/<dataset>/shared/."""
  return _p(s["root"], s["dataset"], "shared", *parts)


def setting_path(s: dict, *parts: str) -> str:
  """A file under <root>/<dataset>/<slug>/."""
  return _p(s["root"], s["dataset"], s["slug"], *parts)


def pass_path(s: dict, k: int, *parts: str) -> str:
  """A file under <root>/<dataset>/<slug>/p<k>/."""
  return setting_path(s, f"p{k}", *parts)


def prompts_path(s: dict, arm: str, k: int = 1) -> str:
  """Prompt records of an arm (shared if unguided, per pass for feat)."""
  if arm in UNGUIDED_ARMS:
    return shared_path(s, "prompts", f"{arm}.jsonl")
  return pass_path(s, k, "prompts", f"{arm}.jsonl")


def paraphrases_path(s: dict, arm: str, k: int = 1) -> str:
  """Paraphrase records of an arm; placed like prompts_path()."""
  if arm in UNGUIDED_ARMS:
    return shared_path(s, "paraphrases", f"{arm}.jsonl")
  return pass_path(s, k, "paraphrases", f"{arm}.jsonl")


def scores_dir(s: dict, arm: str, k: int = 1) -> str:
  """orig: the setting; unguided arms: pass 1 (scored once); feat: pass k."""
  if arm == "orig":
    return setting_path(s, "scores", "x")[:-2]
  return pass_path(s, 1 if arm in UNGUIDED_ARMS else k, "scores", "x")[:-2]


def score_file(
    s: dict, arm: str, k: int = 1, shard: int = 0, num_shards: int = 1
) -> str:
  """Score file of an arm, or one shard of num_shards."""
  name = (
      f"sae_{arm}.jsonl"
      if num_shards == 1
      else f"sae_{arm}.{shard:02d}-of-{num_shards:02d}.jsonl"
  )
  return os.path.join(scores_dir(s, arm, k), name)


def score_files(s: dict, arm: str, k: int = 1) -> list[str]:
  """Every score file of an arm.

  Unsharded, shards, plus carried rows (feat) or the human rows (orig).
  """
  d = scores_dir(s, arm, k)
  files = [
      p for p in [os.path.join(d, f"sae_{arm}.jsonl")] if os.path.exists(p)
  ]
  files += sorted(glob.glob(os.path.join(d, f"sae_{arm}.*-of-*.jsonl")))
  extra = {"feat": "sae_feat.carried.jsonl", "orig": "sae_orig.humans.jsonl"}
  if arm in extra and os.path.exists(os.path.join(d, extra[arm])):
    files.append(os.path.join(d, extra[arm]))
  return files


def read_jsonl(path: str) -> list:
  """Read a jsonl file, skipping blank lines."""
  rows = []
  with open(path) as f:
    for line in f:
      if line.strip():
        rows.append(json.loads(line))
  return rows


def write_jsonl(path: str, rows: list) -> None:
  """Write rows atomically (tmp file + os.replace), one object per line."""
  tmp = path + ".tmp"
  with open(tmp, "w") as f:
    for r in rows:
      f.write(json.dumps(r, ensure_ascii=False) + "\n")
  os.replace(tmp, path)


def append_jsonl(path: str, rows: list) -> None:
  """Append rows and flush, so a killed job keeps what it wrote (--resume)."""
  with open(path, "a") as f:
    for r in rows:
      f.write(json.dumps(r, ensure_ascii=False) + "\n")
    f.flush()


def write_json(path: str, obj, indent: int = 2) -> None:
  """json.dump with an indent."""
  with open(path, "w") as f:
    json.dump(obj, f, indent=indent)


def dataset_labels(ds, model_col: str) -> np.ndarray:
  """Labels (1 = AI) from the model column; COLING's label column must agree."""
  y = np.array(
      [int(str(m) != HUMAN_LABEL) for m in ds[model_col]], dtype=np.int64
  )
  if "label" in ds.column_names:
    assert np.array_equal(
        y, np.asarray(ds["label"], dtype=np.int64)
    ), "label column disagrees with model column"
  return y


def eval_row_mask(ds, d: dict) -> np.ndarray:
  """Rows that the setting's eval block scores.

  PRDB: the humans plus the in-domain generator; COLING: every row.
  """
  if d["eval_rows"] is None:
    return np.ones(len(ds), dtype=bool)
  gens = np.asarray([str(m) for m in ds[d["model_col"]]])
  return np.isin(gens, list(d["eval_rows"]))


WORD = re.compile(r"\s+|[^\s]+")


def words(text: str) -> list[str]:
  """Whitespace-delimited tokens; punctuation stays attached to its word."""
  return [t for t in WORD.findall(text or "") if not t.isspace()]


def load_head(path: str) -> dict:
  """Weights of a logreg head as numpy: w (mask applied) and b."""
  from detectorscope.model.detectors.heads.logreg import LogRegHead

  head = LogRegHead(path)
  w = head.detector.weight.detach().squeeze(0) * head.interp_mask
  return {
      "w": w.double().cpu().numpy(),
      "b": float(head.detector.bias.detach().cpu()[0]),
  }


def load_thresholds(s: dict) -> dict:
  """This setting's entry of thresholds.json ({t, t_fpr5, t_fpr1, ...})."""
  path = setting_path(s, "sample", "thresholds.json")
  return json.load(open(path))["models"][s["slug"]]


def load_detector(s: dict):
  """The setting's extractor, SAEs and head as an AuditableDetector.

  Gemma Scope SAEs come from the HF cache (HF_HOME), so compute nodes without
  network access run with HF_HUB_OFFLINE=1.
  """
  from detectorscope.model.detectors.auditable import AuditableDetector

  cfg = s["cfg"]
  return AuditableDetector.from_paths(
      cfg["model_path"],
      "logreg",
      s["head"],
      hook_points=cfg["hook_points"],
      sae_model_paths=cfg["sae_paths"],
  )


def load_descriptions(desc_dir: str) -> dict[int, dict]:
  """feature_id -> {terse, keywords, long, corr}; verbatim, last record wins.

  The per-layer auto-interp files of `desc_dir` are found with
  detectorscope.data.cache.get_desc_files; the records are read here rather
  than through load_feature_table because that table cannot tell a missing
  correlation (None here) from a stored NaN one (kept as nan). Duplicate rows
  exist (COLING_train_base_gemma2 layer 7): the last record with a terse
  description wins. Ids are layer_rank * SAE_DIM + feature_index, matching
  the caches and the heads.
  """
  from detectorscope.data.cache import get_desc_files

  files = get_desc_files(desc_dir, DESC_MODEL)
  layers = [layer for layer, _ in files]
  assert layers == LAYERS, (desc_dir, layers, LAYERS)
  out = {}
  for rank, (_, path) in enumerate(files):
    with open(path) as f:
      for line in f:
        if not line.strip():
          continue
        r = json.loads(line)
        d = r.get("description") or {}
        if not d.get("terse"):
          continue
        corr = r.get("correlation")
        out[rank * SAE_DIM + int(r["feature_index"])] = {
            "terse": d["terse"],
            "keywords": d.get("keywords", ""),
            "long": d.get("long", ""),
            "corr": None if not corr else corr[0],
        }
  return out


def split_keywords(kw) -> list[str]:
  """`"a", "b"` or `'a', 'b'` -> [a, b]."""
  if not isinstance(kw, str) or not kw.strip():
    return []
  s = kw.strip()
  q = s[0] if s[0] in "\"'" else None
  if q and s.endswith(q) and len(s) >= 2:
    toks = s[1:-1].split(f"{q}, {q}")
  else:
    toks = [k.strip() for k in s.split(",")]
  return [t for t in toks if t != ""]


def feature_name(fid: int) -> str:
  """Display name "<layer>_<local index>" of a flat feature id."""
  return f"{LAYERS[fid // SAE_DIM]}_{fid % SAE_DIM}"


def top_contributions(fids: np.ndarray, cf: np.ndarray, k: int = N_PER_DOC):
  """(ai, human): the k largest positive and k most negative contributions."""
  order = np.argsort(-cf)
  ai = [(int(fids[i]), float(cf[i])) for i in order if cf[i] > 0][:k]
  hu = [(int(fids[i]), float(cf[i])) for i in order[::-1] if cf[i] < 0][:k]
  return ai, hu


def sae_lists(
    fids: np.ndarray, cf: np.ndarray, desc: dict, k: int = N_PER_DOC
) -> dict:
  """Per-doc AI / human lists from one feature vector's contributions.

  This is the per-model record of doc_features.jsonl; a listed feature
  without an auto-interp description is a hard error.
  """
  ai, hu = top_contributions(fids, cf, k)
  out = {
      "ai": [],
      "human": [],
      "n_active": int(len(fids)),
      "ai_mass": float(cf[cf > 0].sum()),
      "human_mass": float(cf[cf < 0].sum()),
      "n_ai_features": int((cf > 0).sum()),
      "n_human_features": int((cf < 0).sum()),
  }
  for side, lst in (("ai", ai), ("human", hu)):
    for fid, v in lst:
      d = desc.get(fid)
      if not d:
        raise KeyError(
            f"feature {fid} ({feature_name(fid)}) has no auto-interp description"
        )
      out[side].append(
          {
              "fid": fid,
              "name": feature_name(fid),
              "effect": v,
              "terse": d["terse"],
              "keywords": d["keywords"],
              "corr": d["corr"],
              "long": d.get("long", ""),
          }
      )
  out["covered_share"] = float(
      sum(abs(e["effect"]) for s in ("ai", "human") for e in out[s])
      / max(np.abs(cf).sum(), 1e-12)
  )
  return out


def attach_spans(lists: dict, spans: dict, doc_id: int) -> None:
  """Attach the first N_SPANS firing runs to every listed feature, in place."""
  for side in ("ai", "human"):
    for e in lists[side]:
      sp = spans.get(str(e["fid"]))
      if sp is None:
        raise SystemExit(
            f"doc {doc_id} feature {e['fid']} has a contribution but no span record"
        )
      e["spans"] = {
          "n_runs": sp["n_runs"],
          "n_firing": sp["n_firing"],
          "n_tokens": sp["n_tokens"],
          "runs": sp["runs"][:N_SPANS],
      }


def features_shown(s: dict, df: dict, t: float) -> dict:
  """The features_shown record of a feat prompt row."""
  return {
      "model": s["slug"],
      "doc_ai": [e["fid"] for e in df["ai"]],
      "doc_human": [e["fid"] for e in df["human"]],
      "ai_mass": df["ai_mass"],
      "human_mass": df["human_mass"],
      "logit_in": df["logit"],
      "threshold": t,
  }


# Templates live in scripts/prompts/adversarial/<name>.txt, one per arm
# (para_free, para_min, para_fewshot, para_fewshot_min, para_antitell,
# para_antitell_min, feat); they are str.format templates. Every prompt ends
# with the <rewrite></rewrite> convention of
# scripts/counterfactual_ablation.py; feat adds <plan></plan>.
@functools.lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
  """The template `name`; the file's final newline is not part of it."""
  with open(os.path.join(PROMPT_DIR, f"{name}.txt")) as f:
    text = f.read()
  return text[:-1] if text.endswith("\n") else text


def render_exemplars(exemplars: list[dict]) -> str:
  """Number the exemplars as "### Example i" blocks for the few-shot arms."""
  return "\n\n".join(
      f"### Example {i + 1}\n{e['text'].strip()}"
      for i, e in enumerate(exemplars)
  )


def clean_keywords(kws: list[str]) -> list[str]:
  """Strip the auto-interp span markers (<< >>) from keywords.

  Bare marker tokens and duplicates are dropped; order is kept.
  """
  out = []
  for k in kws:
    k2 = k.replace("<<", "").replace(">>", "").strip()
    if k2 and k2 not in out:
      out.append(k2)
  return out


def _mark(run: dict) -> str:
  # One firing run as "...before[[core]]after...", newlines flattened.
  t = (run["before"] + "[[" + run["core"] + "]]" + run["after"]).replace(
      "\n", " / "
  )
  return "..." + t.strip() + "..."


def render_feature(entry: dict) -> str:
  """One bullet for a feature entry {terse, keywords, effect, spans?}."""
  line = entry["terse"]
  kws = clean_keywords(split_keywords(entry.get("keywords")))
  if kws:
    line += " (e.g., " + ", ".join(f'"{k}"' for k in kws[:6]) + ")"
  out = f"- {line} [effect on the AI score: {entry['effect']:+.2f}]"
  sp = entry.get("spans")
  if sp:
    extra = sp["n_runs"] - len(sp["runs"])
    for r in sp["runs"]:
      out += f"\n    fires at: {_mark(r)}"
    if extra > 0:
      out += (
          f"\n    (+{extra} more passage{'s' if extra > 1 else ''} like these)"
      )
  return out


def render_list(entries: list[dict]) -> str:
  """Bulleted feature list, or "- (none)" when empty."""
  if not entries:
    return "- (none)"
  return "\n".join(render_feature(e) for e in entries)


def gap_to_boundary(doc_feats: dict, threshold: float | None) -> float:
  """How far the score sits above the decision boundary (0 if already human).

  With a threshold the gap is logit - t (the head's constant and the detector's
  Youden cut both enter); without one it is the AI-minus-human mass, which
  equals logit - t only when bias == t.
  """
  if threshold is not None and doc_feats.get("logit") is not None:
    return max(float(doc_feats["logit"]) - float(threshold), 0.0)
  return max(doc_feats["ai_mass"] + doc_feats["human_mass"], 0.0)


def build_prompt(
    arm: str,
    text: str,
    doc_feats: dict | None = None,
    threshold: float | None = None,
    exemplars: list[dict] | None = None,
) -> str:
  """Render the prompt of one arm for one document.

  Args:
    arm: an entry of PARA_ARMS.
    text: the document (an original, or the previous pass's paraphrase).
    doc_feats: feat arm only; the sae_lists record with spans attached and
      the logit.
    threshold: the setting's decision cut t.
    exemplars: human exemplar rows for the few-shot arms.
  """
  if arm in FEWSHOT_ARMS:
    assert exemplars, f"{arm} needs exemplars"
    return load_prompt(arm).format(
        n=len(exemplars), exemplars=render_exemplars(exemplars), text=text
    )
  if arm in UNGUIDED_ARMS:
    return load_prompt(arm).format(text=text)
  assert arm == "feat", arm
  gap = gap_to_boundary(doc_feats, threshold)
  return load_prompt("feat").format(
      ai_mass=doc_feats["ai_mass"],
      human_mass=doc_feats["human_mass"],
      gap=gap,
      target=gap + TARGET_MARGIN,
      doc_ai=render_list(doc_feats["ai"]),
      doc_human=render_list(doc_feats["human"]),
      text=text,
  )


def parse_response(response: str | None) -> tuple[str, bool]:
  """(paraphrase, had_tags); mirrors CounterfactualAblator.parse_response.

  Without rewrite tags the whole response is kept, minus any <plan> block.
  """
  if not response:
    return "", False
  start_tag, end_tag = "<rewrite>", "</rewrite>"
  if start_tag in response and end_tag in response:
    start = response.index(start_tag) + len(start_tag)
    end = response.index(end_tag, start)
    return response[start:end].strip(), True
  body = response
  if "</plan>" in body:  # plan present, rewrite tags missing: keep the rest
    body = body[body.index("</plan>") + len("</plan>") :]
  return body.strip(), False


def parse_plan(response: str | None) -> str:
  """The <plan> block of a feat response, or "" when absent."""
  if not response or "<plan>" not in response:
    return ""
  start = response.index("<plan>") + len("<plan>")
  end = response.find("</plan>", start)
  return response[start : end if end >= 0 else None].strip()


def run_sample(args) -> None:
  """Sample N AI + N human documents of the dataset's dev split (seed 42).

  Candidates are drawn uniformly per class, filtered to MIN_WORDS+ words and
  at most MAX_GEMMA_TOKENS Gemma tokens (the extractor truncates at 1024, so
  a lengthened paraphrase is still fully scored); AI candidates are restricted
  to DATASETS[..]["ai_filter"] when set. Writes shared/sample/docs.jsonl,
  {doc_id, row, id, label (1 = AI), model, meta{...}, text, n_words,
  n_gemma_tokens} with the AI docs first (doc_id 0..n-1, humans n..2n-1), and
  sample_summary.json.
  """
  import datasets
  import transformers

  s = setting_from_args(args)
  d = DATASETS[s["dataset"]]
  text_col, model_col = d["text_col"], d["model_col"]
  t0 = time.time()
  ds = datasets.load_from_disk(d["path"])
  drop = [
      c
      for c in ("input_ids", "attention_mask", "labels", "lang")
      if c in ds.column_names
  ]
  if drop:
    ds = ds.remove_columns(drop)
  y = dataset_labels(ds, model_col)
  gens = np.asarray([str(m) for m in ds[model_col]])
  print(
      f"{s['dataset']}: {len(ds)} rows, {int((y == 1).sum())} AI / {int((y == 0).sum())} human ({time.time() - t0:.0f}s)"
  )
  tok = transformers.AutoTokenizer.from_pretrained(s["model_path"])

  rng = np.random.default_rng(args.seed)
  counts = collections.Counter()
  docs = []

  def consider(r: int, label: int):
    """Doc record of dataset row r, or None when it fails a length filter."""
    row = ds[int(r)]
    text = row[text_col] or ""
    nw = len(words(text))
    if nw < MIN_WORDS:
      counts["excluded_short"] += 1
      return None
    ntok = len(tok(text, truncation=False)["input_ids"])
    if ntok > MAX_GEMMA_TOKENS:
      counts["excluded_long"] += 1
      return None
    meta = {c: row.get(c) for c in d["meta_cols"] if c in row}
    return {
        "row": int(r),
        "id": row.get("id"),
        "label": int(label),
        "model": str(row[model_col]),
        "meta": meta,
        "text": text,
        "n_words": nw,
        "n_gemma_tokens": ntok,
    }

  for label in (1, 0):
    pool = np.flatnonzero(y == label)
    if label == 1 and d["ai_filter"]:
      pool = pool[gens[pool] == d["ai_filter"]]
    rows = rng.permutation(pool)
    taken = 0
    for r in rows:
      doc = consider(int(r), label)
      if doc is None:
        continue
      doc["doc_id"] = len(docs)
      docs.append(doc)
      taken += 1
      if taken % 500 == 0:
        print(
            f"  label {label}: {taken}/{args.n} ({time.time() - t0:.0f}s)",
            flush=True,
        )
      if taken >= args.n:
        break
    if taken < args.n:
      raise SystemExit(f"only {taken} docs of label {label} passed the filters")

  write_jsonl(shared_path(s, "sample", "docs.jsonl"), docs)
  gen_hist = collections.Counter(x["model"] for x in docs if x["label"] == 1)
  summary = {
      "dataset": s["dataset"],
      "data_path": d["path"],
      "n_per_class": args.n,
      "seed": args.seed,
      "ai_filter": d["ai_filter"],
      "tokenizer": s["model_path"],
      "counts": dict(counts),
      "generators": dict(gen_hist.most_common()),
      "mean_words_ai": float(
          np.mean([x["n_words"] for x in docs if x["label"] == 1])
      ),
      "mean_words_human": float(
          np.mean([x["n_words"] for x in docs if x["label"] == 0])
      ),
  }
  write_json(shared_path(s, "sample", "sample_summary.json"), summary)
  print(json.dumps(summary, indent=1))
  print(f"wrote {len(docs)} docs to {shared_path(s, 'sample', 'docs.jsonl')}")


def run_exemplars(args) -> None:
  """Few-shot exemplars: k TRAIN-split humans per sampled AI document.

  Seed SEED + doc_id, no repeats within a prompt, from the same group
  (exemplar_group_col: COLING sub_source, PRDB conference) when it has enough
  candidates, else from all humans. Candidates are limited to
  exemplar_max_words words and, when at least k remain, to 0.5-1.5x the
  target's word count. PRDB excludes train reviews of papers that also have
  human reviews in the test set (exemplar_exclude_col). Writes
  shared/sample/exemplars.jsonl, {doc_id, group, group_matched, exemplars:
  [{id, row, group, n_words, text}]}, and exemplar_summary.json.
  """
  import datasets

  s = setting_from_args(args)
  d = DATASETS[s["dataset"]]
  text_col, model_col = d["text_col"], d["model_col"]
  gcol, xcol, max_words = (
      d["exemplar_group_col"],
      d["exemplar_exclude_col"],
      d["exemplar_max_words"],
  )
  t0 = time.time()
  docs = [
      x
      for x in read_jsonl(shared_path(s, "sample", "docs.jsonl"))
      if x["label"]
  ]
  train = datasets.load_from_disk(d["train_path"])
  cols = [
      c for c in {model_col, gcol, xcol, "id"} if c and c in train.column_names
  ]
  meta = train.select_columns(cols).to_pandas()
  human_rows = np.flatnonzero(
      meta[model_col].astype(str).to_numpy() == HUMAN_LABEL
  )
  excluded = 0
  if xcol:
    test = (
        datasets.load_from_disk(d["path"])
        .select_columns([model_col, xcol])
        .to_pandas()
    )
    test_keys = set(
        test.loc[test[model_col].astype(str) == HUMAN_LABEL, xcol].astype(str)
    )
    keep = ~meta.loc[human_rows, xcol].astype(str).isin(test_keys).to_numpy()
    excluded = int((~keep).sum())
    human_rows = human_rows[keep]
  print(
      f"{s['dataset']}: {len(human_rows)} train humans usable ({excluded} excluded via {xcol}); "
      f"{len(docs)} AI docs ({time.time() - t0:.0f}s)",
      flush=True,
  )

  # Word counts of the human pool (only the text column, only the pool rows).
  texts = train.select_columns([text_col])
  n_words = np.zeros(len(human_rows), dtype=np.int32)
  for i in range(0, len(human_rows), 20000):
    idx = human_rows[i : i + 20000]
    n_words[i : i + len(idx)] = [
        len(words(t)) for t in texts.select(idx.tolist())[text_col]
    ]
  groups = (
      meta.loc[human_rows, gcol].astype(str).to_numpy()
      if gcol in meta
      else np.array([""] * len(human_rows))
  )
  ok_len = n_words <= max_words
  by_group = collections.defaultdict(list)
  for j in np.flatnonzero(ok_len):
    by_group[groups[j]].append(j)
  all_ok = np.flatnonzero(ok_len)
  print(
      f"  pool after the {max_words}-word cap: {len(all_ok)} docs in {len(by_group)} groups",
      flush=True,
  )

  rows, stats = [], collections.Counter()
  for x in docs:
    rng = np.random.default_rng(SEED + x["doc_id"])
    group = str(x.get("meta", {}).get(gcol, ""))
    cand = np.array(by_group.get(group, []), dtype=np.int64)
    matched = len(cand) >= args.k
    if not matched:
      cand = all_ok
    lo, hi = 0.5 * x["n_words"], 1.5 * x["n_words"]
    near = cand[(n_words[cand] >= lo) & (n_words[cand] <= hi)]
    if len(near) >= args.k:
      cand = near
      stats["length_matched"] += 1
    pick = rng.choice(cand, size=args.k, replace=False)
    stats["group_matched"] += matched
    ex = []
    for j in pick:
      r = int(human_rows[j])
      ex.append(
          {
              "id": str(meta.iloc[r]["id"]) if "id" in meta else str(r),
              "row": r,
              "group": groups[j],
              "n_words": int(n_words[j]),
              "text": texts[r][text_col],
          }
      )
    rows.append(
        {
            "doc_id": x["doc_id"],
            "group": group,
            "group_matched": matched,
            "exemplars": ex,
        }
    )
  out = shared_path(s, "sample", "exemplars.jsonl")
  write_jsonl(out, rows)
  summary = {
      "dataset": s["dataset"],
      "train_path": d["train_path"],
      "k": args.k,
      "group_col": gcol,
      "exclude_col": xcol,
      "n_excluded": excluded,
      "max_words": max_words,
      "n_docs": len(rows),
      "group_match_rate": stats["group_matched"] / max(len(rows), 1),
      "length_match_rate": stats["length_matched"] / max(len(rows), 1),
      "mean_exemplar_words": float(
          np.mean([e["n_words"] for r in rows for e in r["exemplars"]])
      ),
  }
  write_json(shared_path(s, "sample", "exemplar_summary.json"), summary)
  print(json.dumps(summary, indent=1))
  print(f"wrote {out} ({time.time() - t0:.0f}s)")


def global_sae(s: dict, desc: dict) -> dict:
  """Rank the head's features by training-mean effect w * act_mean.

  Features need a non-zero weight, MIN_DOCS_GLOBAL active training docs and a
  description; the N_GLOBAL strongest AI-direction and N_GLOBAL_HUMAN strongest
  human-direction entries form the features/global_<slug>.json record.
  """
  z = np.load(s["feature_stats"])
  act_mean = np.asarray(z["act_mean"], dtype=np.float64)
  n_active = np.asarray(z["n_active"])
  w = load_head(s["head"])["w"]
  used = np.flatnonzero((w != 0) & (n_active >= MIN_DOCS_GLOBAL))
  n_undescribed = int(sum(int(f) not in desc for f in used))
  used = np.array([f for f in used if int(f) in desc], dtype=np.int64)
  mean_cf = w[used] * act_mean[used]
  order = np.argsort(-mean_cf)
  ai = [(used[i], mean_cf[i]) for i in order if mean_cf[i] > 0][:N_GLOBAL]
  hu = [(used[i], mean_cf[i]) for i in order[::-1] if mean_cf[i] < 0][
      :N_GLOBAL_HUMAN
  ]
  out = {
      "model": s["slug"],
      "source": s["feature_stats"],
      "min_docs": MIN_DOCS_GLOBAL,
      "n_nonzero": int((w != 0).sum()),
      "n_ranked": int(len(used)),
      "n_skipped_undescribed": n_undescribed,
      "ai": [],
      "human": [],
  }
  for side, lst in (("ai", ai), ("human", hu)):
    for fid, v in lst:
      d = desc[int(fid)]
      out[side].append(
          {
              "fid": int(fid),
              "name": feature_name(int(fid)),
              "effect": float(v),
              "weight": float(w[fid]),
              "act_mean": float(act_mean[fid]),
              "n_docs_active": int(n_active[fid]),
              "terse": d["terse"],
              "keywords": d["keywords"],
              "corr": d["corr"],
          }
      )
  return out


def run_prompts(args) -> None:
  """Pass-1 prompts of the unguided arms (per dataset) and the feat arm.

  Conditions, one str.format template scripts/prompts/adversarial/<arm>.txt
  each ({text} is the document):
    para_free           rewrite so it reads as human-written, no edit budget
    para_min            same goal, change as few words as possible
    para_fewshot[_min]  the free / min-edit task plus {n} human passages
                        ({exemplars}) from the TRAIN split of the same source
                        (exemplars step): "match the register, sentence rhythm,
                        vocabulary and level of polish of these authors"
    para_antitell[_min] the free / min-edit task plus six generic rules against
                        machine-writing habits (no transition-word openers, no
                        summarising close, varied sentence length, no stock
                        vocabulary, no example lists, no added structure),
                        written from general knowledge, not from any output
    feat                the v5.1 template: the detector's N_PER_DOC strongest
                        AI- and human-direction features of THIS document with
                        their firing passages ({doc_ai}, {doc_human}; [[...]]
                        marks the firing words), the max-pooling counting rule,
                        the evidence masses ({ai_mass}, {human_mass}), a
                        numeric target ({gap}, {target} = gap + TARGET_MARGIN)
                        and a <plan> block before the rewrite
  Every template ends with the <rewrite></rewrite> convention of
  scripts/counterfactual_ablation.py. The target is
  threshold-aware: gap = logit - t, so the head's bias and the Youden cut both
  enter and the number is right for heads whose bias != t. Only AI documents
  are prompted. The feat arm needs <slug>/features/doc_features.jsonl
  (features), the spans of the originals (<slug>/scores/sae_orig.* from score
  --arm orig --ai-only --spans) and <slug>/sample/thresholds.json; it also
  writes the training-mean lists to features/global_<slug>.json for the
  record. Writes prompts/<arm>.jsonl: {doc_id, arm, prompt, features_shown
  ({model, doc_ai, doc_human, ai_mass, human_mass, logit_in, threshold}, null
  for the unguided arms), version, pass} (+ exemplar_ids for the few-shot
  arms).
  """
  arms = [a for a in args.arms.split(",") if a]
  assert all(a in PARA_ARMS for a in arms), arms
  if "feat" in arms and not args.setting:
    raise SystemExit("--arms feat is per setting: give --setting SLUG")
  s = setting_from_args(args)
  docs = {
      x["doc_id"]: x
      for x in read_jsonl(shared_path(s, "sample", "docs.jsonl"))
      if x["label"] == 1
  }
  exemplars = {}
  if any(a in FEWSHOT_ARMS for a in arms):
    exemplars = {
        r["doc_id"]: r["exemplars"]
        for r in read_jsonl(shared_path(s, "sample", "exemplars.jsonl"))
    }
    assert set(docs) <= set(
        exemplars
    ), "exemplars.jsonl lacks sampled AI docs (run the exemplars step)"

  feats, spans, t = {}, {}, None
  if "feat" in arms:
    desc = load_descriptions(s["desc_dir"])
    gf = global_sae(s, desc)
    with open(
        setting_path(s, "features", f"global_{s['slug']}.json"), "w"
    ) as f:
      json.dump(gf, f, indent=1, ensure_ascii=False)
    print(
        f"{s['slug']}: {len(gf['ai'])} AI / {len(gf['human'])} human global features (skipped undescribed: "
        f"{gf['n_skipped_undescribed']}); top AI {gf['ai'][0]['terse']!r}"
    )
    t = load_thresholds(s)["t"]
    feats = {
        f["doc_id"]: f["models"][s["slug"]]
        for f in read_jsonl(setting_path(s, "features", "doc_features.jsonl"))
    }
    assert set(docs) <= set(feats), "doc_features.jsonl lacks sampled AI docs"
    span_files = score_files(s, "orig")
    assert (
        span_files
    ), "run score --arm orig --ai-only --spans before prompts --arms feat"
    for fpath in span_files:
      for r in read_jsonl(fpath):
        if "spans" in r:
          spans[r["doc_id"]] = r["spans"]
    missing = [x for x in docs if x not in spans]
    assert (
        not missing
    ), f"{len(missing)} AI docs have no spans (e.g. {missing[:5]}); rerun the orig pass with --spans"
    for x in docs:
      df = feats[x]
      df["ai"], df["human"] = df["ai"][: args.n_doc], df["human"][: args.n_doc]
      attach_spans(df, spans[x], x)

  for arm in arms:
    rows = []
    for doc_id in sorted(docs):
      text = docs[doc_id]["text"]
      if arm == "feat":
        df = feats[doc_id]
        prompt = build_prompt(arm, text, df, threshold=t)
        shown, version = features_shown(s, df, t), PROMPT_VERSION
      else:
        ex = exemplars.get(doc_id) if arm in FEWSHOT_ARMS else None
        prompt, shown, version = (
            build_prompt(arm, text, exemplars=ex),
            None,
            "v1",
        )
      row = {
          "doc_id": doc_id,
          "arm": arm,
          "prompt": prompt,
          "features_shown": shown,
          "version": version,
          "pass": 1,
      }
      if arm in FEWSHOT_ARMS:
        row["exemplar_ids"] = [e["id"] for e in exemplars[doc_id]]
      rows.append(row)
    write_jsonl(prompts_path(s, arm), rows)
    print(
        f"{arm}: {len(rows)} prompts, mean {np.mean([len(r['prompt']) for r in rows]):.0f} chars -> {prompts_path(s, arm)}"
    )

  if args.show is not None and "feat" in arms:
    r = next(
        x
        for x in read_jsonl(prompts_path(s, "feat"))
        if x["doc_id"] == args.show
    )
    print(f"\n{'=' * 30} feat / doc {args.show} {'=' * 30}\n{r['prompt']}")


def run_paraphrase(args) -> None:
  """Gemini paraphrases for one arm of one pass.

  Reads prompts/<arm>.jsonl, sends prompts in chunks through
  GeminiGenerator.async_generate (--batch requests in flight) and appends
  {doc_id, arm, ok, has_tags, paraphrase, plan, response_raw, model,
  temperature, version} to paraphrases/<arm>.jsonl after every chunk, so an
  interrupted run loses at most one chunk. --resume skips doc_ids already in
  the output; without it an existing output is an error. A prompt that stays
  empty after the generator's retries is written with ok=false; analysis
  excludes those documents. Passes >= 2 pre-seed the output with the carried
  (already flipped) documents, which --resume leaves alone.
  """
  s = setting_from_args(args)
  k = args.pass_no
  if not os.environ.get("GEMINI_API_KEY"):
    raise SystemExit("GEMINI_API_KEY is not set")
  from detectorscope.model.gemini import GeminiGenerator

  prompts = read_jsonl(args.prompts or prompts_path(s, args.arm, k))
  if args.n:
    prompts = prompts[: args.n]
  out = args.out or paraphrases_path(s, args.arm, k)
  done = set()
  if args.resume and os.path.exists(out):
    done = {r["doc_id"] for r in read_jsonl(out)}
    print(f"resume: {len(done)} of {len(prompts)} already done")
  elif os.path.exists(out):
    raise SystemExit(f"{out} exists; pass --resume to continue or remove it")
  todo = [p for p in prompts if p["doc_id"] not in done]
  print(
      f"{args.arm}: {len(todo)} prompts to send with {args.model}, batch {args.batch}, chunk {args.chunk}"
  )
  if not todo:
    print(f"{args.arm}: nothing to do ({len(done)} records already in {out})")
    return

  gen = GeminiGenerator(model_name=args.model)
  gen.async_batch_size = args.batch
  config = None
  if args.temperature is not None:
    from google.genai import types

    config = types.GenerateContentConfig(
        temperature=args.temperature, top_p=0.95, top_k=64
    )
  n_ok = n_fail = n_notag = 0
  t0 = time.time()
  for start in range(0, len(todo), args.chunk):
    chunk = todo[start : start + args.chunk]
    t1 = time.time()
    responses = gen.async_generate([p["prompt"] for p in chunk], config)
    rows = []
    for p, resp in zip(chunk, responses):
      text, has_tags = parse_response(resp)
      ok = bool(text)
      n_ok += ok
      n_fail += not ok
      n_notag += ok and not has_tags
      rows.append(
          {
              "doc_id": p["doc_id"],
              "arm": args.arm,
              "ok": ok,
              "has_tags": has_tags,
              "paraphrase": text,
              "plan": parse_plan(resp),
              "response_raw": resp or "",
              "model": args.model,
              "temperature": args.temperature,
              "version": p.get("version"),
          }
      )
    append_jsonl(out, rows)
    print(
        f"  {start + len(chunk)}/{len(todo)}  chunk {time.time() - t1:.0f}s  total {time.time() - t0:.0f}s  "
        f"ok {n_ok} fail {n_fail} no-tags {n_notag}",
        flush=True,
    )

  total = len(read_jsonl(out))
  print(
      f"{args.arm}: {total} records in {out}; this run ok={n_ok} fail={n_fail} no_tags={n_notag}"
  )
  if n_fail / len(todo) > 0.02:
    print(
        f"WARNING: {n_fail / len(todo):.1%} of prompts failed; investigate before analysis"
    )


def prf(y: np.ndarray, s: np.ndarray, t: float) -> dict:
  """Confusion counts and P/R/F1/accuracy/FPR of the rule s >= t."""
  pred = s >= t
  tp = int((pred & (y == 1)).sum())
  fp = int((pred & (y == 0)).sum())
  fn = int((~pred & (y == 1)).sum())
  tn = int((~pred & (y == 0)).sum())
  p = tp / max(tp + fp, 1)
  r = tp / max(tp + fn, 1)
  return {
      "precision": p,
      "recall": r,
      "f1": 2 * p * r / max(p + r, 1e-12),
      "accuracy": (tp + tn) / len(y),
      "fpr": fp / max(fp + tn, 1),
      "tp": tp,
      "fp": fp,
      "fn": fn,
      "tn": tn,
  }


def run_thresholds(args) -> None:
  """Frozen decision threshold and full-split scores of one setting.

  The threshold is the Youden cut of the setting's dev evaluation
  (`optimal_threshold` in the eval JSON, logit scale; block "all" for COLING,
  the in-domain generator for PRDB). Re-scores every row of the split from
  the mmap cache in row blocks, exits unless F1 and recall at that cut
  reproduce the JSON on the block's rows, and records secondary cuts (5% / 1%
  FPR on the block's humans, balanced-F1). Writes <slug>/sample/
  thresholds.json, {data_path, n_rows, setting, models: {<slug>: {t, t_fpr5,
  t_fpr1, json_f1, f1_full_dev, auroc_full_dev, bias, ...}}, created}, and
  cache_scores.npz (label, generator, eval_mask, logit_<slug>).
  """
  import datasets
  from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

  from detectorscope.data.cache import LayerStack

  s = setting_from_args(args)
  d = DATASETS[s["dataset"]]
  slug = s["slug"]
  t0 = time.time()
  ds = datasets.load_from_disk(d["path"])
  y_all = dataset_labels(ds, d["model_col"])
  gens = np.asarray([str(m) for m in ds[d["model_col"]]])
  mask = eval_row_mask(ds, d)
  n = len(y_all)
  print(
      f"{d['path']}: {n} rows; eval block {s['eval_block']} covers {int(mask.sum())} rows "
      f"({int((y_all[mask] == 1).sum())} AI / {int((y_all[mask] == 0).sum())} human) ({time.time() - t0:.0f}s)"
  )

  head = load_head(s["head"])
  cache = LayerStack.open(s["mmap_cache"])
  assert cache.n_rows == n, (cache.n_rows, n)
  t1 = time.time()
  s_all = (
      cache.score(
          head["w"],
          chunk_rows=args.chunk_rows,
          log=lambda m: print(m, flush=True),
      )
      + head["b"]
  )
  print(f"cache scoring: {time.time() - t1:.0f}s")
  y, sc = y_all[mask], s_all[mask]
  # PRDB thresholds come from the in-domain block (llama / gpt4o): the "all"
  # block pools three out-of-domain generators and gives a very different cut.
  ev = json.load(open(s["eval_json"]))[s["eval_block"]]
  t = float(ev["optimal_threshold"])
  base = prf(y, sc, t)
  fpr, tpr, thr = roc_curve(y, sc)
  youden = float(thr[np.argmax(tpr - fpr)])
  sw = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
  p, r, th = precision_recall_curve(y, sc, sample_weight=sw)
  f = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
  hum = sc[y == 0]
  rec = {
      "t": t,
      "source_json": s["eval_json"],
      "eval_block": s["eval_block"],
      "json_f1": float(ev["f1"]),
      "json_recall": float(ev["recall"]),
      "json_precision": float(ev["precision"]),
      "json_auroc": float(ev.get("auroc", np.nan)),
      "f1_full_dev": base["f1"],
      "recall_full_dev": base["recall"],
      "precision_full_dev": base["precision"],
      "fpr_full_dev": base["fpr"],
      "auroc_full_dev": float(roc_auc_score(y, sc)),
      "youden_recomputed": youden,
      "bias": head["b"],
      "n_nonzero_weights": int((head["w"] != 0).sum()),
      "t_fpr5": float(np.quantile(hum, 0.95)),
      "t_fpr1": float(np.quantile(hum, 0.99)),
      "t_f1_balanced": float(th[np.argmax(f)]),
      "n_rows_block": int(mask.sum()),
      "n_human_block": int((y == 0).sum()),
      "n_ai_block": int((y == 1).sum()),
  }
  print(
      f"{slug}: t={t:+.4f} (recomputed Youden {youden:+.4f})  F1 {base['f1']:.4f} vs json {ev['f1']:.4f}; "
      f"recall {base['recall']:.4f} vs {ev['recall']:.4f}; FPR {base['fpr']:.4f}; AUROC {rec['auroc_full_dev']:.4f}"
  )
  if (
      abs(base["f1"] - ev["f1"]) > 1e-3
      or abs(base["recall"] - ev["recall"]) > 1e-3
  ):
    raise SystemExit(
        f"{slug}: cache scoring does not reproduce {s['eval_json']}"
    )

  np.savez(
      setting_path(s, "sample", "cache_scores.npz"),
      label=y_all,
      generator=gens,
      eval_mask=mask,
      **{f"logit_{slug}": s_all.astype(np.float32)},
  )
  out = setting_path(s, "sample", "thresholds.json")
  write_json(
      out,
      {
          "data_path": d["path"],
          "n_rows": n,
          "setting": slug,
          "models": {slug: rec},
          "created": time.strftime("%Y-%m-%d %H:%M:%S"),
      },
  )
  print(f"wrote {out} and {setting_path(s, 'sample', 'cache_scores.npz')}")


def run_features(args) -> None:
  """Feature lists of the sampled docs from this setting's cache.

  For every sampled document the contributions w_j x_j of the head are read
  from the mmap cache. AI docs get their top-k AI-direction and human-direction
  features (description, keywords, long description; an undescribed feature is
  a hard error). Human docs get a score record in scores/sae_orig.humans.jsonl
  (the cache logit; verified against the raw-text path by score --agreement).
  Writes <slug>/features/doc_features.jsonl, {doc_id, models: {<slug>: {ai:
  [{fid, name, effect, terse, keywords, corr, long}], human: [...], n_active,
  ai_mass, human_mass, n_ai_features, n_human_features, covered_share,
  logit}}}; features/cache_logits.json, {doc_id: cache logit} for every
  sampled doc; and scores/sae_orig.humans.jsonl, score records (see score)
  with source "cache" and no spans.
  """
  from detectorscope.data.cache import LayerStack

  s = setting_from_args(args)
  slug = s["slug"]
  t0 = time.time()
  docs = read_jsonl(shared_path(s, "sample", "docs.jsonl"))
  desc = load_descriptions(s["desc_dir"])
  head = load_head(s["head"])
  cache = LayerStack.open(s["mmap_cache"])
  w, b = head["w"], head["b"]
  feats, humans, logits, covered = [], [], {}, []
  for x in docs:
    fids, vals = cache.row(x["row"])
    cf = w[fids] * vals
    nz = cf != 0
    fids, cf = fids[nz], cf[nz].astype(np.float64)
    logit = float(cf.sum() + b)
    logits[x["doc_id"]] = logit
    if x["label"] == 1:
      lst = sae_lists(fids, cf, desc, args.n_per_doc)
      lst["logit"] = logit
      covered.append(lst["covered_share"])
      feats.append({"doc_id": x["doc_id"], "models": {slug: lst}})
    else:
      humans.append(
          {
              "doc_id": x["doc_id"],
              "arm": "orig",
              "label": 0,
              "source": "cache",
              "n_tokens": min(x["n_gemma_tokens"], EXTRACTOR_MAX_TOKENS),
              "truncated": x["n_gemma_tokens"] > EXTRACTOR_MAX_TOKENS,
              slug: {
                  "logit": logit,
                  "bias": b,
                  "cf": {int(f): round(float(v), 6) for f, v in zip(fids, cf)},
              },
          }
      )
  out_f = setting_path(s, "features", "doc_features.jsonl")
  out_h = setting_path(s, "scores", "sae_orig.humans.jsonl")
  out_l = setting_path(s, "features", "cache_logits.json")
  write_jsonl(out_f, feats)
  write_jsonl(out_h, humans)
  with open(out_l, "w") as f:
    json.dump({str(k): v for k, v in logits.items()}, f)
  n_ai = np.mean([len(x["models"][slug]["ai"]) for x in feats])
  n_hu = np.mean([len(x["models"][slug]["human"]) for x in feats])
  print(
      f"{slug}: {len(feats)} AI docs (mean {n_ai:.1f} AI / {n_hu:.1f} human features listed, covered share "
      f"{np.mean(covered):.3f}), {len(humans)} human score records from the cache ({time.time() - t0:.0f}s)"
  )
  print(f"wrote {out_f}, {out_h}, {out_l}")


def load_texts(s: dict, arm: str, k: int, ai_only: bool) -> list[dict]:
  """{doc_id, label, text} items of an arm.

  orig: the sample (AI documents only with ai_only); other arms: their ok,
  non-carried paraphrases.
  """
  docs = {
      x["doc_id"]: x for x in read_jsonl(shared_path(s, "sample", "docs.jsonl"))
  }
  if arm == "orig":
    return [
        {"doc_id": x["doc_id"], "label": x["label"], "text": x["text"]}
        for x in docs.values()
        if not (ai_only and x["label"] == 0)
    ]
  return [
      {
          "doc_id": r["doc_id"],
          "label": docs[r["doc_id"]]["label"],
          "text": r["paraphrase"],
      }
      for r in read_jsonl(paraphrases_path(s, arm, k))
      if r["ok"] and not r.get("carried")
  ]


def spans_for(
    tokens: list[str], token_feats, fids: list[int], context=6, top=3
):
  """Top-`top` contiguous runs where each feature fires, with +-context tokens.

  Returns {fid: {n_runs, n_firing, n_tokens, runs: [{start, end, max_act,
  before, core, after}]}}; runs are ordered by max activation and the token
  strings are concatenated as-is. `token_feats` is the sparse [seq, d]
  tensor of an Extraction.
  """
  import torch

  out = {}
  dense = token_feats.index_select(1, torch.tensor(fids)).to_dense()
  for k, fid in enumerate(fids):
    act = dense[:, k].numpy()
    pos = np.flatnonzero(act > 0)
    if len(pos) == 0:
      continue
    runs, start = [], pos[0]
    for a, b in zip(pos[:-1], pos[1:]):
      if b != a + 1:
        runs.append((start, a))
        start = b
    runs.append((start, pos[-1]))
    runs.sort(key=lambda r: -float(act[r[0] : r[1] + 1].max()))
    out[fid] = {
        "n_runs": len(runs),
        "n_firing": int(len(pos)),
        "n_tokens": int(len(tokens)),
        "runs": [
            {
                "start": int(a),
                "end": int(b),
                "max_act": float(act[a : b + 1].max()),
                "before": "".join(tokens[max(0, a - context) : a]),
                "core": "".join(tokens[a : b + 1]),
                "after": "".join(tokens[b + 1 : b + 1 + context]),
            }
            for a, b in runs[:top]
        ],
    }
  return out


def run_score(args) -> None:
  """Score texts through the setting's extractor, SAEs and head (raw text).

  Every record carries the logit, every nonzero contribution w_j x_j, the
  Gemma token count and, with --spans, the top-3 firing passages of every
  active feature (one document per forward pass; needed by prompts --arms
  feat for the originals and by pass for the feat arm of the previous pass).
  --ai-only restricts `orig` to the AI documents (the humans' scores come from
  the cache, see features); --agreement N instead scores N random human
  documents raw and checks them against the cache (writes
  scores/human_agreement.json, exits 1 on disagreement). Carried paraphrase
  rows are skipped: their records are copied by pass. Halves the batch on
  CUDA OOM; asserts sum(cf) + bias matches the logit to 1e-2. Score record:
  {doc_id, arm, label, pass, n_tokens, truncated, <slug>: {logit, bias, cf:
  {fid: w_j x_j}} [, spans: {fid: {n_runs, n_firing, n_tokens, runs: [{start,
  end, max_act, before, core, after}]}}]}; human_agreement.json: {n,
  max_abs_diff, mean_abs_diff, n_verdict_changes, threshold, tol, ok, docs}.
  """
  import torch

  s = setting_from_args(args)
  slug, k = s["slug"], args.pass_no
  if args.agreement:
    assert args.arm == "orig"
    out = setting_path(s, "scores", "human_agreement.json")
  else:
    out = args.out or score_file(s, args.arm, k, args.shard, args.num_shards)
  if args.exit_if_output_path_exists and os.path.exists(out):
    print(f"{out} exists, exiting")
    return

  if args.agreement:
    humans = [it for it in load_texts(s, "orig", k, False) if it["label"] == 0]
    rng = np.random.default_rng(SEED)
    pick = rng.choice(
        len(humans), size=min(args.agreement, len(humans)), replace=False
    )
    items = [humans[i] for i in sorted(pick)]
  else:
    items = load_texts(s, args.arm, k, args.ai_only)
    if args.ids:
      want = {int(x) for x in args.ids.split(",")}
      items = [it for it in items if it["doc_id"] in want]
    items = items[args.shard :: args.num_shards]
  print(
      f"{slug} {args.arm} pass {k}: {len(items)} texts (shard {args.shard}/{args.num_shards}) -> {out}"
  )

  t0 = time.time()
  det = load_detector(s)
  head = det.head
  device = det.extractor.device
  print(
      f"extractor + SAEs + head loaded on {device} ({time.time() - t0:.0f}s)",
      flush=True,
  )
  if device == "cpu":
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
  bias = float(head.predict(torch.zeros(1, head.interp_mask.numel()))[0])

  tok = det.extractor.tokenizer
  results = []
  t0 = time.time()
  batch = args.batch
  i = 0
  while i < len(items):
    chunk = items[i : i + batch]
    texts = [c["text"] for c in chunk]
    try:
      with torch.no_grad():
        if args.spans:  # one doc per forward pass; keep per-token activations
          exs = [det.extract(t) for t in texts]
          x = torch.cat([e.x for e in exs], 0)
        else:
          x = det.get_feature_vectors(texts)
    except torch.cuda.OutOfMemoryError:
      torch.cuda.empty_cache()
      if batch == 1:
        raise
      batch = max(1, batch // 2)
      print(f"  OOM, batch -> {batch}", flush=True)
      continue
    recs = []
    for j, it in enumerate(chunk):
      n_full = len(tok(it["text"], truncation=False)["input_ids"])
      logit = float(head.predict(x[j : j + 1])[0])
      contrib = head.get_importance_scores(x[j : j + 1])[0].numpy()
      nz = np.flatnonzero(contrib)
      rec = {
          "doc_id": it["doc_id"],
          "arm": args.arm,
          "label": it["label"],
          "pass": k,
          "n_tokens": min(n_full, EXTRACTOR_MAX_TOKENS),
          "truncated": n_full > EXTRACTOR_MAX_TOKENS,
          slug: {
              "logit": logit,
              "bias": bias,
              "cf": {int(f): round(float(contrib[f]), 6) for f in nz},
          },
      }
      assert abs(sum(rec[slug]["cf"].values()) + bias - logit) < 1e-2, (
          it["doc_id"],
          logit,
      )
      recs.append(rec)
    if args.spans:
      for j, ex in enumerate(exs):
        recs[j]["spans"] = spans_for(
            ex.tokens,
            ex.token_feats,
            sorted(int(f) for f in recs[j][slug]["cf"]),
        )
    results.extend(recs)
    i += len(chunk)
    if (i // batch) % 25 == 0 or i == len(items):
      rate = (time.time() - t0) / max(i, 1)
      print(
          f"  {i}/{len(items)}  {rate:.2f} s/doc  eta {rate * (len(items) - i) / 60:.0f} min",
          flush=True,
      )

  if args.agreement:
    cache = {
        r["doc_id"]: r[slug]["logit"]
        for r in read_jsonl(setting_path(s, "scores", "sae_orig.humans.jsonl"))
    }
    t = load_thresholds(s)["t"]
    diffs = np.array([r[slug]["logit"] - cache[r["doc_id"]] for r in results])
    flips = int(
        sum(
            (r[slug]["logit"] >= t) != (cache[r["doc_id"]] >= t)
            for r in results
        )
    )
    rep = {
        "n": len(results),
        "max_abs_diff": float(np.abs(diffs).max()),
        "mean_abs_diff": float(np.abs(diffs).mean()),
        "n_verdict_changes": flips,
        "threshold": t,
        "tol": args.tol,
        "ok": bool(np.abs(diffs).max() <= args.tol and flips == 0),
        "docs": {
            str(r["doc_id"]): {
                "raw": r[slug]["logit"],
                "cache": cache[r["doc_id"]],
            }
            for r in results
        },
    }
    write_json(out, rep, indent=1)
    print(
        f"human agreement: n={rep['n']} max|diff|={rep['max_abs_diff']:.4f} mean={rep['mean_abs_diff']:.4f} "
        f"verdict changes={flips} -> {'OK' if rep['ok'] else 'FAIL'} ({out})"
    )
    if not rep["ok"]:
      raise SystemExit(1)
    return

  write_jsonl(out, results)
  print(f"wrote {len(results)} records to {out} in {time.time() - t0:.0f}s")
  cache_logits = setting_path(s, "features", "cache_logits.json")
  if args.arm == "orig" and os.path.exists(cache_logits):
    cache = json.load(open(cache_logits))
    diffs = [
        abs(r[slug]["logit"] - cache[str(r["doc_id"])])
        for r in results
        if str(r["doc_id"]) in cache
    ]
    if diffs:
      print(
          f"  raw-text vs cache logit |diff| max {max(diffs):.4f} mean {np.mean(diffs):.4f} over {len(diffs)} docs"
      )
      if max(diffs) > args.tol:
        print(f"  WARNING: raw-vs-cache disagreement above {args.tol}")
  n_trunc = sum(r["truncated"] for r in results)
  if n_trunc:
    print(
        f"  NOTE: {n_trunc} texts exceeded {EXTRACTOR_MAX_TOKENS} tokens and were truncated by the extractor"
    )


def lists_from_record(s: dict, rec: dict, desc: dict, k: int) -> dict:
  """Feature lists of a scored paraphrase, with its firing spans attached.

  Args:
    s: the setting.
    rec: pass k-1 score record (score --spans), keyed by the slug.
    desc: auto-interp descriptions from load_descriptions().
    k: features per direction.

  Returns:
    sae_lists() output plus "logit"; each entry's "spans" keeps the first
    N_SPANS runs.
  """
  cf_d = {int(f): v for f, v in rec[s["slug"]]["cf"].items()}
  fids = np.array(sorted(cf_d), dtype=np.int64)
  cf = np.array([cf_d[f] for f in fids], dtype=np.float64)
  out = sae_lists(fids, cf, desc, k)
  out["logit"] = rec[s["slug"]]["logit"]
  assert (
      "spans" in rec
  ), f"doc {rec['doc_id']}: previous pass was scored without --spans"
  attach_spans(out, rec["spans"], rec["doc_id"])
  return out


def run_pass(args) -> None:
  """Pass k >= 2 of the iterative feature-guided attack.

  Reads pass k-1's feat paraphrases and their scores (scored with --spans),
  re-derives every document's feature lists from that text and writes pass
  k's prompts. Stop rule (default): a document whose pass k-1 paraphrase
  already scores below the frozen cut is not prompted again; its paraphrase
  row is copied into p<k>/paraphrases/feat.jsonl with carried=True and
  flipped_pass=<first pass below the cut>, and its score record into
  p<k>/scores/sae_feat.carried.jsonl, so paraphrase --resume and score leave
  it alone and analyze sees the kept text (a carried row is the paraphrase or
  score record plus {carried: true, flipped_pass, pass}). Also writes
  features/pass_features.jsonl, {doc_id, arm, pass, lists}, and
  sample/pass_info.json, {pass, prev_root, prompt_version, stop_flipped,
  threshold, counts}. Documents whose pass k-1 paraphrase failed or was never
  scored are dropped; more than 2% of them is an error.
  """
  s = setting_from_args(args)
  slug, k = s["slug"], args.pass_no
  assert k >= 2
  t = load_thresholds(s)["t"]
  docs = {
      x["doc_id"]: x
      for x in read_jsonl(shared_path(s, "sample", "docs.jsonl"))
      if x["label"] == 1
  }
  desc = load_descriptions(s["desc_dir"])

  prev_rows = read_jsonl(paraphrases_path(s, "feat", k - 1))
  para = {r["doc_id"]: r for r in prev_rows if r["ok"]}
  recs = {}
  for f in score_files(s, "feat", k - 1):
    for r in read_jsonl(f):
      recs[r["doc_id"]] = r
  counts = {
      "prompted": 0,
      "carried": 0,
      "missing_input": 0,
      "prev_failed": sum(1 for r in prev_rows if not r["ok"]),
      "flipped_pass": {},
  }
  rows, carried, carried_recs, feature_rows = [], [], [], []
  for doc_id in sorted(docs):
    if doc_id not in para or doc_id not in recs:
      counts["missing_input"] += 1
      continue
    row, rec = para[doc_id], recs[doc_id]
    if args.stop_flipped and rec[slug]["logit"] < t:
      fp = row.get("flipped_pass", k - 1)
      carried.append({**row, "carried": True, "flipped_pass": fp, "pass": k})
      carried_recs.append(
          {**rec, "pass": k, "carried": True, "flipped_pass": fp}
      )
      counts["carried"] += 1
      counts["flipped_pass"][str(fp)] = (
          counts["flipped_pass"].get(str(fp), 0) + 1
      )
      continue
    df = lists_from_record(s, rec, desc, N_PER_DOC)
    rows.append(
        {
            "doc_id": doc_id,
            "arm": "feat",
            "prompt": build_prompt("feat", row["paraphrase"], df, threshold=t),
            "version": PROMPT_VERSION,
            "pass": k,
            "features_shown": features_shown(s, df, t),
        }
    )
    feature_rows.append(
        {"doc_id": doc_id, "arm": "feat", "pass": k, "lists": df}
    )
    counts["prompted"] += 1

  out = prompts_path(s, "feat", k)
  write_jsonl(out, rows)
  para_out = paraphrases_path(s, "feat", k)
  if os.path.exists(para_out):  # a re-run may overwrite a pure pre-seed only
    assert all(
        r.get("carried") for r in read_jsonl(para_out)
    ), f"{para_out} holds generated rows; refusing to overwrite"
  if carried:
    write_jsonl(para_out, carried)
    write_jsonl(
        pass_path(s, k, "scores", "sae_feat.carried.jsonl"), carried_recs
    )
  write_jsonl(pass_path(s, k, "features", "pass_features.jsonl"), feature_rows)
  write_json(
      pass_path(s, k, "sample", "pass_info.json"),
      {
          "pass": k,
          "prev_root": pass_path(s, k - 1, "x")[:-2],
          "prompt_version": PROMPT_VERSION,
          "stop_flipped": args.stop_flipped,
          "threshold": t,
          "counts": counts,
      },
      indent=1,
  )
  print(
      f"pass {k}: {counts['prompted']} prompts (mean {np.mean([len(r['prompt']) for r in rows]) if rows else 0:.0f} chars), "
      f"{counts['carried']} carried as already flipped, {counts['missing_input']} missing -> {out}"
  )
  if counts["missing_input"]:  # failed (ok=false) or unscored; they stay failed
    print(
        f"  NOTE: {counts['missing_input']} documents have no usable pass-{k - 1} paraphrase/score and are dropped "
        f"(analysis counts them in n_failed)"
    )
    assert counts["missing_input"] <= 0.02 * len(
        docs
    ), "more than 2% of documents lack an input; investigate"


if __name__ == "__main__":
  # Eight subcommands sharing --setting / --root / --settings-yml; each step's
  # function is its set_defaults(func=...).
  common_parser = argparse.ArgumentParser(add_help=False)
  common_parser.add_argument(
      "--setting", default=None, help="slug: " + ", ".join(SETTING_ORDER)
  )
  common_parser.add_argument(
      "--root",
      default=DEFAULT_OUT_ROOT,
      help="output tree (default: %(default)s)",
  )
  common_parser.add_argument(
      "--settings-yml",
      default=DEFAULT_SETTINGS_YML,
      help="demo settings file (default: %(default)s)",
  )
  parser = argparse.ArgumentParser(
      description="Adversarial paraphrase experiment: one step per subcommand."
  )
  subparsers = parser.add_subparsers(dest="step", required=True)

  def dataset_level(p: argparse.ArgumentParser) -> None:
    """--dataset as an alternative to --setting for dataset-level steps."""
    p.add_argument(
        "--dataset",
        default=None,
        choices=list(DATASETS),
        help="instead of --setting: the dataset (its first setting is used)",
    )

  sample_parser = subparsers.add_parser(
      "sample",
      parents=[common_parser],
      help="sample N AI + N human dev documents (per dataset)",
  )
  dataset_level(sample_parser)
  sample_parser.add_argument(
      "-n", type=int, default=2500, help="documents per class"
  )
  sample_parser.add_argument("--seed", type=int, default=SEED)
  sample_parser.set_defaults(func=run_sample)

  exemplars_parser = subparsers.add_parser(
      "exemplars",
      parents=[common_parser],
      help="few-shot human exemplars from the train split (per dataset)",
  )
  dataset_level(exemplars_parser)
  exemplars_parser.add_argument("-k", type=int, default=N_EXEMPLARS)
  exemplars_parser.set_defaults(func=run_exemplars)

  prompts_parser = subparsers.add_parser(
      "prompts",
      parents=[common_parser],
      help="pass-1 prompts: unguided arms (per dataset), feat (per setting)",
  )
  dataset_level(prompts_parser)
  prompts_parser.add_argument(
      "--arms",
      default=",".join(PARA_ARMS),
      help="comma-separated subset of " + ",".join(PARA_ARMS),
  )
  prompts_parser.add_argument(
      "--n-doc",
      type=int,
      default=N_PER_DOC,
      help="feat: per-doc features per direction",
  )
  prompts_parser.add_argument(
      "--show", type=int, default=None, help="print the feat prompt of one doc"
  )
  prompts_parser.set_defaults(func=run_prompts)

  paraphrase_parser = subparsers.add_parser(
      "paraphrase",
      parents=[common_parser],
      help="Gemini paraphrases of one arm of one pass (GEMINI_API_KEY)",
  )
  paraphrase_parser.add_argument("--arm", required=True, choices=PARA_ARMS)
  paraphrase_parser.add_argument("--pass", dest="pass_no", type=int, default=1)
  paraphrase_parser.add_argument("--resume", action="store_true")
  paraphrase_parser.add_argument("--model", default=GEMINI_MODEL)
  paraphrase_parser.add_argument(
      "--batch", type=int, default=20, help="requests in flight"
  )
  paraphrase_parser.add_argument(
      "--chunk", type=int, default=100, help="prompts per async_generate call"
  )
  paraphrase_parser.add_argument(
      "-n",
      type=int,
      default=None,
      help="only the first N prompts (smoke tests)",
  )
  paraphrase_parser.add_argument(
      "--temperature",
      type=float,
      default=None,
      help="override the generator's default (1.0)",
  )
  paraphrase_parser.add_argument(
      "--out", default=None, help="override the output path (experiments)"
  )
  paraphrase_parser.add_argument(
      "--prompts", default=None, help="override the prompts file (experiments)"
  )
  paraphrase_parser.set_defaults(func=run_paraphrase)

  thresholds_parser = subparsers.add_parser(
      "thresholds",
      parents=[common_parser],
      help="frozen Youden cut + full-split cache scores (per setting)",
  )
  thresholds_parser.add_argument("--chunk-rows", type=int, default=20_000)
  thresholds_parser.set_defaults(func=run_thresholds)

  features_parser = subparsers.add_parser(
      "features",
      parents=[common_parser],
      help="per-doc feature lists and human cache scores (per setting)",
  )
  features_parser.add_argument("--n-per-doc", type=int, default=N_PER_DOC)
  features_parser.set_defaults(func=run_features)

  score_parser = subparsers.add_parser(
      "score",
      parents=[common_parser],
      help="score an arm's texts through extractor + SAEs + head",
  )
  score_parser.add_argument("--arm", required=True, choices=ARMS)
  score_parser.add_argument("--pass", dest="pass_no", type=int, default=1)
  score_parser.add_argument("--shard", type=int, default=0)
  score_parser.add_argument("--num_shards", type=int, default=1)
  score_parser.add_argument("--batch", type=int, default=8)
  score_parser.add_argument(
      "--spans", action="store_true", help="keep the top-3 firing passages"
  )
  score_parser.add_argument(
      "--ai-only", action="store_true", help="orig: score the AI documents only"
  )
  score_parser.add_argument(
      "--agreement",
      type=int,
      default=0,
      help="orig: raw-score N random humans and compare with the cache",
  )
  score_parser.add_argument("--tol", type=float, default=1e-2)
  score_parser.add_argument(
      "--ids", default=None, help="comma-separated doc_ids to restrict to"
  )
  score_parser.add_argument(
      "--out", default=None, help="override the output path"
  )
  score_parser.add_argument("--exit_if_output_path_exists", action="store_true")
  score_parser.set_defaults(func=run_score)

  pass_parser = subparsers.add_parser(
      "pass",
      parents=[common_parser],
      help="prompts of pass k >= 2 from pass k-1's paraphrases and scores",
  )
  pass_parser.add_argument("--pass", dest="pass_no", type=int, required=True)
  pass_parser.add_argument(
      "--no-stop-flipped",
      dest="stop_flipped",
      action="store_false",
      help="re-paraphrase every document, even those already below the cut",
  )
  pass_parser.set_defaults(func=run_pass)

  args = parser.parse_args()
  args.func(args)
