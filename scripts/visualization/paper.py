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

"""Registry, labels, paths, plot style and shared formatting of the paper.

The nine detector settings (SETTINGS) are the three SAE conditions on COLING,
PRDB-Llama and PRDB-GPT-4o; every table and figure under
scripts/visualization/ names its data through them. The paper heads are the
SGD+L1 sweep pickles `head_filename(**PAPER_HEAD, tau=..., target=...)`.
`tex` is the one LaTeX escaper, `iter_results` the one eval-JSON walker and
`style` the seaborn / Google Sans setup every figure used.
"""

import dataclasses
import json
from pathlib import Path
import re
import warnings

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parents[2]
FIGURES = REPO / "figures"
CACHE_ROOT = REPO / "data" / "sae_activations"
TABLES = FIGURES / "tables"
EVAL_ROOT = REPO / "eval_results"
SWEEP_ROOT = REPO / "outputs/logreg_sgd_l1_ablation"
GRAM2VEC_ROOT = REPO / "outputs/gram2vec_sgd_l1_ablation"
XGB_DENSITY = REPO / "outputs/head_comparison/density"
ACTIVE_PER_DOC = REPO / "outputs/active_per_doc"
CF_ROOT = REPO / "outputs/counterfactual_results"
ADV_ROOT = REPO / "outputs/adversarial_paraphrase/full"
ADV_FIGURES = FIGURES / "adversarial_paraphrase/full"
WORKED_EXAMPLES = FIGURES / "worked_examples"
FONT_DIR = REPO / "scripts/scratch/Google_Sans/static"

DESC_MODEL = "gemini-3.1-flash-lite-preview"
SAE_DIM = 16384
LAYERS = [7, 13, 17, 22]
PAPER_HEAD = dict(lr="1e-02", l1="1e-01", epoch=3)
MASK_TAU = 0.7  # the main results tables' "w/ Mask" rows
TOP_TAU = "0.85"  # top-feature, adversarial and worked-example heads

SAE_ORDER = ["base_sae", "ft_sae", "ft_model_sae"]
SAE_LABEL = {
    "base_sae": "Base-SAE",
    "ft_sae": "FT-SAE",
    "ft_model_sae": "FT-Model+SAE",
}
SAE_TEX = {k: rf"\textsc{{{v}}}" for k, v in SAE_LABEL.items()}
TARGET_LABEL = {"coling": None, "gpt4o": "GPT-4o", "llama": "Llama"}
GENERATOR_LABEL = {
    "gpt4o": "GPT-4o",
    "llama": "Llama 3.1 70B",
    "qwen": "Qwen 2.5 72B",
    "claude": "Claude 3.5 Sonnet",
    "gemini": "Gemini 1.5 Pro",
}
DATASET_LABEL = {
    "coling": "COLING",
    "prdb": "PRDB-Llama",
    "prdb_gpt4o": "PRDB-GPT-4o",
}
DATASET_TEX = {
    "coling": "COLING (dev)",
    "prdb": "PRDB (Llama)",
    "prdb_gpt4o": "PRDB (GPT-4o)",
}
DATASET_NAME = {"coling": "COLING", "prdb": "PRDB", "prdb_gpt4o": "PRDB"}
DATASET_CITE = {
    "coling": r"\cite{wang-etal-2025-genai}",
    "prdb": r"\cite{yu2026paperreviewedllmbenchmarking}",
    "prdb_gpt4o": r"\cite{yu2026paperreviewedllmbenchmarking}",
}
DOMAIN_TEXT = {
    "coling": "General Domain Text",
    "prdb": "AI-written Paper Reviews",
    "prdb_gpt4o": "AI-written Paper Reviews",
}
COLING_SPLIT_LABEL = {
    "dev": "COLING (dev)",
    "devtest": "COLING (devtest)",
    "test": "COLING (test)",
    "raid": "RAID (test)",
}
TAU_LABEL = {0.0: "w/o Mask", 0.7: "w/ Mask (> 0.7)"}
EVAL_SPLIT = {
    "coling": "COLING_dev",
    "prdb": "PRDB_llama",
    "prdb_gpt4o": "PRDB_gpt4o",
}
EVAL_BLOCK = {"coling": "all", "prdb": "llama", "prdb_gpt4o": "gpt4o"}
# Adversarial paraphrase: arms of the experiment -> the paper's condition names;
# "feat3" is the feat arm after the last of N_PASSES passes.
N_PASSES = 3
COND_OF_ARM = {
    "orig": "base",
    "para_free": "free",
    "para_min": "min",
    "para_fewshot": "fewshot",
    "para_fewshot_min": "fewshot_min",
    "para_antitell": "antitell",
    "para_antitell_min": "antitell_min",
    "feat": "feat",
}
COND_ORDER = [
    "base",
    "free",
    "min",
    "fewshot",
    "fewshot_min",
    "antitell",
    "antitell_min",
    "feat",
    "feat3",
]
TABLE_CONDS = ["base", "min", "fewshot_min", "antitell_min", "feat", "feat3"]
FIG_CONDS = TABLE_CONDS[1:]
FIG_COND_LABELS = {
    "min": "Baseline",
    "fewshot_min": "Few Shot",
    "antitell_min": "Guided",
    "feat": "SAE",
    "feat3": "SAE \u00d73",
}
COND_COLORS = {
    "base": "#c8c8c8",
    "min": "#bdbdbd",
    "fewshot_min": "#2EA84A",
    "antitell_min": "#FFD400",
    "feat": "#FF2D2D",
    "feat3": "#1F6FFF",
}
# results.csv columns carried into summary.csv, and the subset of the figure and
# appendix tables (<stem>_data.csv).
SUMMARY_METRICS = [
    "f1",
    "f1_lo",
    "f1_hi",
    "precision",
    "recall",
    "recall_lo",
    "recall_hi",
    "recall_fpr5",
    "recall_fpr1",
    "auroc",
    "flip",
    "flip_lo",
    "flip_hi",
    "n_flipped",
    "n_detected_base",
    "wlev",
    "wlev_lo",
    "wlev_hi",
    "word_lev_norm_max_median",
    "frac_words_changed_mean",
    "len_ratio_mean",
    "cosine_sim_mean",
    "logit_mean",
    "n_ai",
    "n_failed",
    "n_carried",
    "n_human",
    "fp",
    "threshold",
]
FIG_METRICS = [
    "f1",
    "f1_lo",
    "f1_hi",
    "flip",
    "flip_lo",
    "flip_hi",
    "wlev",
    "n_ai",
    "n_human",
]
COND_LABELS = {
    "base": "Base",
    "free": "Paraphrase (free)",
    "min": "Paraphrase (min-edit)",
    "fewshot": "Few-shot (free)",
    "fewshot_min": "Few-shot (min-edit)",
    "antitell": "Anti-tell (free)",
    "antitell_min": "Anti-tell (min-edit)",
    "feat": "+features",
    "feat3": "+features ×3",
}
SAVE_KW = dict(format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.01)


@dataclasses.dataclass(frozen=True)
class Setting:
  """One detector setting: an SAE condition trained on one dataset / target.

  `train_dir` is the training activation cache (heads, feature stats and
  descriptions are keyed by it), `cache_dir` the evaluation-split cache,
  `eval_dir` the eval_results/<split>/ directory of the logreg head and `key`
  the file key of the top-feature tables (`ftsae_gpt4o`).
  """

  slug: str
  dataset: str
  sae: str
  target: str
  train_dir: str
  eval_dir: str
  cache_dir: str
  label: str
  tex_label: str
  key: str

  @property
  def stats_stem(self) -> str:
    return f"{self.train_dir}_{self.target}"


_CACHE = {
    "base_sae": "gemmascope",
    "ft_sae": "base_gemma2",
    "ft_model_sae": "gemma2",
}
_SHORT = {"coling": "coling", "prdb": "prdb", "prdb_gpt4o": "prdb4o"}
_TARGET = {"coling": "coling", "prdb": "llama", "prdb_gpt4o": "gpt4o"}


def _setting(dataset: str, sae: str) -> Setting:
  target, split = _TARGET[dataset], DATASET_NAME[dataset]
  cache = _CACHE[sae] + (
      f"_{target}" if sae == "ft_model_sae" and target != "coling" else ""
  )
  suffix = f" ({TARGET_LABEL[target]})" if TARGET_LABEL[target] else ""
  return Setting(
      slug=f"{_SHORT[dataset]}_{sae}",
      dataset=dataset,
      sae=sae,
      target=target,
      train_dir=f"{split}_train_{cache}",
      eval_dir=f"logreg_{_CACHE[sae]}",
      cache_dir=f"{split}_test_{cache}",
      label=SAE_LABEL[sae] + suffix,
      tex_label=SAE_TEX[sae] + suffix,
      key=sae.replace("_", "") + (f"_{target}" if suffix else ""),
  )


SETTINGS = {
    s.slug: s for s in (_setting(d, sae) for d in _SHORT for sae in SAE_ORDER)
}
SETTING_ORDER = list(SETTINGS)
MAIN_SETTINGS = [s for s in SETTINGS if not s.startswith("prdb4o")]
GPT4O_SETTINGS = [s for s in SETTINGS if not s.startswith("prdb_")]
TOP_FEATURE_ORDER = [
    "coling_ft_sae",
    "coling_ft_model_sae",
    "coling_base_sae",
    "prdb4o_base_sae",
    "prdb_base_sae",
    "prdb4o_ft_sae",
    "prdb_ft_sae",
    "prdb4o_ft_model_sae",
    "prdb_ft_model_sae",
]


def head_filename(lr: str, l1: str, tau, target: str, epoch: int) -> str:
  """The sweep's head pickle name, `sgd_lr1e-02_l1-1e-01_threshold0.85_gpt4o_logreg_ep3.pkl`."""
  return f"sgd_lr{lr}_l1-{l1}_threshold{tau}_{target}_logreg_ep{epoch}.pkl"


_HEAD = re.compile(
    r"sgd_lr(?P<lr>[^_]+)_l1-(?P<l1>[^_]+)_threshold(?P<tau>[\d.]+)_(?P<target>[a-z0-9]+)"
    r"(?:_logreg)?(?:_ep(?P<epoch>\d+))?"
)


def parse_head(name: str) -> dict:
  """lr, l1, tau (strings) and target, epoch (int or None) of a head or density stem."""
  match = _HEAD.search(name)
  if match is None:
    raise ValueError(f"not a sweep head name: {name}")
  found = match.groupdict()
  found["epoch"] = int(found["epoch"]) if found["epoch"] else None
  return found


_ESC = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_ESC_RE = re.compile("|".join(re.escape(k) for k in _ESC))
NON_LATIN = re.compile(
    r"[\u2E80-\u9FFF\uAC00-\uD7AF\uF900-\uFAFF\uFF00-\uFFEF\U00020000-\U0002FFFF]"
)
CODE_CHARS = re.compile(r"[#*{}\[\]<>\\|`\"=]")


def tex(s) -> str:
  """LaTeX-escape `s`, also defusing the `<<`, `>>` and `--` ligatures."""
  s = str(s).replace("\n", "\\n").replace("\t", "\\t")
  s = _ESC_RE.sub(lambda m: _ESC[m.group(0)], s)
  return s.replace("<<", "<{}<").replace(">>", ">{}>").replace("--", "-{}-")


def split_keywords(kw) -> list[str]:
  """Verbatim tokens of an auto-interp keyword string (`"a", "b"` or `a, b`).

  The wrapping quotes and the `<<`/`>>` span markers are format, not keyword;
  CJK-script tokens are dropped because pdflatex cannot typeset them.
  """
  if not isinstance(kw, str) or not kw.strip():
    return []
  s = kw.strip()
  quote = s[0] if s[0] in "\"'" else None
  if quote and s.endswith(quote) and len(s) >= 2:
    tokens = s[1:-1].split(f"{quote}, {quote}")
  else:
    tokens = [k.strip() for k in s.split(",")]
  tokens = [t.replace("<<", "").replace(">>", "") or t for t in tokens]
  return [t for t in tokens if t and not NON_LATIN.search(t)]


def fmt_keyword(k, escape=tex) -> str:
  """Quote a keyword with ``...''; `\\texttt` for code-like or symbol tokens."""
  k = str(k)
  if not re.search(r"[A-Za-z]", k) or CODE_CHARS.search(k):
    return r"\texttt{" + escape(k) + "}"
  t = escape(k)
  if t.endswith("'"):
    t += "{}"
  return "``" + t + "''"


def description_cell(row, keywords=True, log=None, escape=tex) -> str:
  """Verbatim terse description plus (e.g., keywords) when they add anything.

  `row` is a feature record (mapping or attribute access) with `terse`,
  `keywords` and `feat` / `name`. Keywords are omitted when `keywords` is
  false, there are none or every one already occurs in the description; the
  two latter cases append a line to `log` when one is given.
  """
  field = (
      row.get if hasattr(row, "get") else lambda k, d=None: getattr(row, k, d)
  )
  name = field("feat") or field("name")
  terse = field("terse") or f"feature {name} (no description)"
  cell = escape(terse)
  kws = split_keywords(field("keywords")) if keywords else []
  if not kws:
    if log is not None and keywords:
      log.append(f"  {name}: no usable keywords ({field('keywords')!r})")
    return cell
  if all(k.strip().lower() in terse.lower() for k in kws):
    if log is not None:
      log.append(
          f"  {name}: keywords omitted, all already in description ({kws})"
      )
    return cell
  return (
      cell
      + r" (\emph{e.g.,} "
      + ", ".join(fmt_keyword(k, escape) for k in kws)
      + ")"
  )


def style(font_dir=None) -> None:
  """Seaborn ticks theme and Google Sans when the font directory exists."""
  sns.set_theme(style="ticks")
  font_dir = Path(font_dir or FONT_DIR)
  if font_dir.is_dir():
    for ttf in font_dir.glob("GoogleSans*.ttf"):
      fm.fontManager.addfont(str(ttf))
  plt.rcParams["font.sans-serif"] = ["Google Sans", "Arial", "DejaVu Sans"]
  plt.rcParams["font.family"] = "sans-serif"


# Every eval_results/<split>/<detector> directory: `base` model, `type` (the
# figure hue, shared by the two heads) and `head`. Baselines carry head
# "baseline" so they are drawn in every figure; the two linguistic baselines
# are head-specific.
ATTRIBUTES = {
    "gram2vec": {"base": "gram2vec", "type": "linguistic", "head": "xgboost"},
    "logreg_gram2vec": {
        "base": "gram2vec",
        "type": "linguistic",
        "head": "logreg",
    },
    "hf_gemma2": {"base": "gemma2", "type": "neural", "head": "baseline"},
    "hf_gemma3": {"base": "gemma3", "type": "neural", "head": "baseline"},
    "binoculars": {
        "base": "binoculars",
        "type": "baseline",
        "head": "baseline",
    },
    "fastdetectgpt": {
        "base": "fastdetectgpt",
        "type": "baseline",
        "head": "baseline",
    },
    "remodetect": {
        "base": "remodetect",
        "type": "baseline",
        "head": "baseline",
    },
    "specdetect": {
        "base": "specdetect",
        "type": "baseline",
        "head": "baseline",
    },
    "gemmascope": {"base": "gemma2", "type": "base_sae", "head": "xgboost"},
    "gemmascope2": {"base": "gemma3", "type": "base_sae", "head": "xgboost"},
    "interp_base_gemma2": {
        "base": "gemma2",
        "type": "ft_sae",
        "head": "xgboost",
    },
    "interp_base_gemma3": {
        "base": "gemma3",
        "type": "ft_sae",
        "head": "xgboost",
    },
    "interp_gemma2": {
        "base": "gemma2",
        "type": "ft_model_sae",
        "head": "xgboost",
    },
    "interp_gemma3": {
        "base": "gemma3",
        "type": "ft_model_sae",
        "head": "xgboost",
    },
    "logreg_gemmascope": {
        "base": "gemma2",
        "type": "base_sae",
        "head": "logreg",
    },
    "logreg_base_gemma2": {
        "base": "gemma2",
        "type": "ft_sae",
        "head": "logreg",
    },
    "logreg_gemma2": {
        "base": "gemma2",
        "type": "ft_model_sae",
        "head": "logreg",
    },
}
METRICS = ["auroc", "f1", "recall_at_5_fpr", "recall_at_1_fpr"]


def eval_dir(sae: str, head: str) -> str:
  """The gemma2 eval_results directory of an SAE condition and head type."""
  (name,) = [
      k
      for k, a in ATTRIBUTES.items()
      if a["type"] == sae and a["head"] == head and a["base"] == "gemma2"
  ]
  return name


def parse_threshold(name: str) -> float | None:
  """Mask threshold of a result file name, None for the untuned baselines."""
  match = re.match(r"^threshold(0\.\d+|1\.0+)_", name)
  return float(match.group(1)) if match else None


def iter_results(eval_root=EVAL_ROOT) -> pd.DataFrame:
  """One row per (split, detector, threshold, target) under eval_root.

  Every detector directory must be listed in ATTRIBUTES; a (split, detector,
  threshold) with two JSON files keeps the last in path order and warns.
  """
  paths = sorted(Path(eval_root).rglob("*.json"))
  if not paths:
    raise FileNotFoundError(f"no result JSON under {eval_root}")
  if unknown := sorted({p.parent.name for p in paths} - ATTRIBUTES.keys()):
    raise KeyError(f"add these result dirs to ATTRIBUTES: {unknown}")
  keep = {}
  for path in paths:
    key = (path.parents[1].name, path.parent.name, parse_threshold(path.name))
    if key in keep:
      warnings.warn(
          f"duplicate result files for {key}: keeping {path.name}, dropping "
          f"{keep[key].name}"
      )
    keep[key] = path
  frames = []
  for (split_dir, model, tau), path in keep.items():
    dataset, split = split_dir.split("_", 1)
    base = {
        "dataset": dataset,
        "split": split,
        "model": model,
        "threshold": tau,
    }
    frames.append(
        pd.DataFrame(
            [
                base | ATTRIBUTES[model] | {"target": target, **metrics}
                for target, metrics in json.loads(path.read_text()).items()
            ]
        )
    )
  results = pd.concat(frames, ignore_index=True)
  for metric in METRICS:
    results[metric] = results[metric].astype(float)
  return results
