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

"""The statistics behind the paper's tables and figures, from the stored runs.

Three experiments reach analysis.ipynb and render.py through this module: the
SGD+L1 sweep (sweep_metrics, select_operating_point), the counterfactual
experiment (the parquet run cache, per-feature agreement and its permutation
nulls, auto-interp bins, most-frequent features) and the adversarial paraphrase
attack (per-document edit metrics, per-arm detection metrics with bootstrap
CIs). Every number in the paper came from these functions, so the float32
casts of the counterfactual cache and the row orders the permutation nulls
depend on are part of the contract. Nothing here draws or typesets.
"""

import collections
import difflib
import json
import os
from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from detectorscope.data.cache import get_desc_files, get_lor_files
import paper
from paper import ADV_ROOT, CF_ROOT, COND_OF_ARM, LAYERS, SAE_DIM, SAE_ORDER, SETTINGS

# scripts/hparam_sweep.py writes one flat JSON per (head, eval split, checkpoint) under
# <root>/metrics/, with exactly these keys (notes/briefs/gates/viz_refactor/
# sweep/SCHEMA.md): the "all" block of the eval JSON, the masked global support
# of the checkpoint and its mean density on the held-out split.
SWEEP_METRIC_KEYS = [
    "head",
    "target",
    "split",
    "lr",
    "l1",
    "tau",
    "epoch",
    "auroc",
    "f1",
    "recall_at_1_fpr",
    "recall_at_5_fpr",
    "nonzero_coefs",
    "active_per_doc",
    "head_path",
    "eval_json",
]
# target -> (selection split, block within its eval JSON): the development
# AUROC that chooses an operating point.
SELECTION_BLOCK = {
    "coling": ("COLING_dev", "all"),
    "gpt4o": ("PRDB_gpt4o", "gpt4o"),
    "llama": ("PRDB_llama", "llama"),
}
SELECTION_METRICS = ["auroc", "f1", "recall_at_5_fpr", "recall_at_1_fpr"]


def sweep_metrics(root: Path) -> pd.DataFrame:
  """Every <root>/metrics/*.json as one frame with the SWEEP_METRIC_KEYS columns."""
  paths = sorted((Path(root) / "metrics").glob("*.json"))
  return pd.DataFrame([json.loads(p.read_text()) for p in paths])[
      SWEEP_METRIC_KEYS
  ]


def select_operating_point(frame: pd.DataFrame, head: str) -> pd.DataFrame:
  """The chosen grid point per target of one head, one row each.

  Best held-out AUROC on the target's development block (SELECTION_BLOCK,
  read from each candidate's eval JSON since the frame carries the "all"
  block); ties break toward fewer non-zero coefficients, then the earlier
  epoch. The SELECTION_METRICS columns of the result are the block's.
  """
  chosen = []
  for target, (split, block) in SELECTION_BLOCK.items():
    rows = frame[
        (frame["head"] == head)
        & (frame["target"] == target)
        & (frame["split"] == split)
    ]
    if rows.empty:
      continue
    blocks = [json.loads(Path(p).read_text())[block] for p in rows["eval_json"]]
    rows = rows.assign(
        block=block,
        **{m: [b.get(m) for b in blocks] for m in SELECTION_METRICS},
    )
    chosen.append(
        rows.sort_values(
            ["auroc", "nonzero_coefs", "epoch"], ascending=[False, True, True]
        ).iloc[0]
    )
  return pd.DataFrame(chosen)


CF_CACHE = CF_ROOT / "_cache"
# Run directory -> arm. A KeyError here is deliberate: a new sweep directory
# should fail loudly rather than be silently skipped. The directory names
# predate paper.SETTINGS and include a PRDB-Llama arm, hence a local registry.
CF_DIRS = {
    "prdb_gemma2_gpt4o": dict(
        dataset="PRDB", target="gpt4o", sae_type="ft_model_sae"
    ),
    "prdb_gemma2_llama": dict(
        dataset="PRDB", target="llama", sae_type="ft_model_sae"
    ),
    "coling_gemma2": dict(
        dataset="COLING", target="all", sae_type="ft_model_sae"
    ),
    "prdb_base_gemma2_gpt4o": dict(
        dataset="PRDB", target="gpt4o", sae_type="ft_sae"
    ),
    "prdb_base_gemma2_llama": dict(
        dataset="PRDB", target="llama", sae_type="ft_sae"
    ),
    "coling_base_gemma2": dict(
        dataset="COLING", target="all", sae_type="ft_sae"
    ),
    "prdb_gemmascope_gpt4o": dict(
        dataset="PRDB", target="gpt4o", sae_type="base_sae"
    ),
    "prdb_gemmascope_llama": dict(
        dataset="PRDB", target="llama", sae_type="base_sae"
    ),
    "coling_gemmascope": dict(
        dataset="COLING", target="all", sae_type="base_sae"
    ),
}
# Path components that mark a run as not part of the sweep: pilots, the cache
# itself, the abandoned Adam logreg heads, smoke tests and the first 25-doc
# trial run.
SKIP_PARTS = frozenset(
    {"_pilot", "_cache", "_adam_logreg", "_smoke", "initial_run_n25_d10"}
)
# tau is captured as a string but cast to float: the xgboost runs wrote
# "threshold0.7" and the logreg sweep "threshold0.70".
_RUN_FILE = re.compile(
    r"^threshold(?P<tau>\d\.\d+)_(?P<head>xgbmodel|logreg)_counterfactual\.jsonl$"
)
_HEAD_NAME = {"xgbmodel": "xgboost", "logreg": "logreg"}
# Columns kept from the run JSONL; the text fields are ~96% of the file.
# Dropped deliberately: text/paraphrase/long_description/keywords, description
# (hoisted into a per-feature frame), para_score (recoverable from
# para_logit), feature_weight (a duplicate of the cf delta).
_KEEP = [
    "doc_id",
    "label",
    "feature_id",
    "autointerp_score",
    "orig_score",
    "orig_logit",
    "para_logit",
    "counterfactual_logit_delta",
    "paraphrase_logit_delta",
    "orig_activation",
    "para_activation",
    "mean_activation",
    "target_feature_deviation",
    "avg_non_target_deviation",
    "num_active_features",
]
REMOVAL_BINS = [-np.inf, 0.0, 0.5, 0.999, np.inf]
REMOVAL_LABELS = ["not removed", "partly", "mostly", "fully"]

# The twelve arms of the paper's agreement figure and table: SGD+L1 logreg
# heads, COLING (all generators) and PRDB (GPT-4o), tau 0.0 / 0.7, three SAE
# settings. A PRDB-Llama arm would share PRDB's rows, so it is not in the
# layout.
TAU_ORDER = [0.0, 0.7]
LAYOUT_TARGET = {"COLING": "all", "PRDB": "gpt4o"}
LAYOUT_HEAD = "logreg"
# The arm typeset in tab:cf_autointerp_bins.
BINS_STEM = "PRDB-gpt4o-ft_model_sae-tau0.70-logreg"
N_BINS = 4
# Permutation nulls: 50 draws for the table (the paper csv), 200 for the bins.
N_PERM_TABLE, N_PERM_BINS, SEED = 50, 200, 0
# The paper-table filters: features with |mean cf| below MIN_CF or sampled in
# fewer than MIN_DOCS documents are dropped; TOP_FREQUENT rows per SAE block.
MIN_DOCS, MIN_CF, TOP_FREQUENT = 10, 0.01, 6

ARM_COLS = ["dataset", "target", "sae_type", "threshold", "head", "config_stem"]
STATS_META = ["dataset", "sae_type", "threshold", "config_stem"]
POINT_COLS = [
    "doc_id",
    "feature_id",
    "cf_delta",
    "para_delta",
    "autointerp_score",
] + ARM_COLS
TABLE_COLS = POINT_COLS + [
    "specificity",
    "is_noop",
    "agree",
    "removal_stratum",
    "target_dev",
    "nontarget_dev",
]
OUT_COLS = [
    "dataset",
    "sae_type",
    "threshold",
    "config_stem",
    "feature_id",
    "n_docs",
    "mean_cf",
    "mean_para",
    "autointerp",
    "description",
    "kept",
]
MOST_FREQUENT = {
    "prdb": ("PRDB", "gpt4o"),
    "coling": ("COLING", "all"),
}


def parse_run_path(path: Path) -> dict | None:
  """Arm of a run JSONL from its directory and file name.

  Returns:
    dataset, target, sae_type, threshold (float), head and config_stem, or
    None when the file is not a recognised run.
  """
  match = _RUN_FILE.match(path.name)
  if match is None or path.parent.name not in CF_DIRS:
    return None
  cfg = dict(CF_DIRS[path.parent.name])
  cfg["threshold"] = float(match.group("tau"))
  cfg["head"] = _HEAD_NAME[match.group("head")]
  cfg["config_stem"] = (
      f"{cfg['dataset']}-{cfg['target']}-{cfg['sae_type']}"
      f"-tau{cfg['threshold']:.2f}-{cfg['head']}"
  )
  return cfg


def stream_run(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
  """One JSONL -> (numeric long frame, per-feature description frame).

  Streams line by line and projects to _KEEP inside the loop; pandas'
  read_json would materialize the text columns first, roughly tripling peak
  memory and wall time on a 200 MB file.
  """
  cols = {k: [] for k in _KEEP}
  descs: dict[int, str] = {}
  with open(path) as f:
    for line in f:
      if not line.strip():
        continue
      r = json.loads(line)
      for k in _KEEP:
        cols[k].append(r.get(k))
      fid = r["feature_id"]
      if fid not in descs:
        descs[fid] = r.get("description")
  df = pd.DataFrame(cols)
  feats = pd.DataFrame(
      {"feature_id": list(descs), "description": list(descs.values())}
  )
  return df, feats


def _derive(df: pd.DataFrame, human_label: str = "human") -> pd.DataFrame:
  """Rename and add the analysis columns; cast to float32 / int32.

  Every paper number was computed from the float32 parquet this produces, so
  the casts are part of the contract.
  """
  df = df.rename(
      columns={
          "counterfactual_logit_delta": "cf_delta",
          "paraphrase_logit_delta": "para_delta",
          "target_feature_deviation": "target_dev",
          "avg_non_target_deviation": "nontarget_dev",
      }
  )
  df["label_binary"] = np.where(df["label"] == human_label, "human", "ai")
  df["specificity"] = df["target_dev"] - df["nontarget_dev"]
  # How much of the targeted feature the rewrite actually removed. This is
  # the compliance axis of the dose-response analysis; 1.0 means the feature
  # was driven to exactly zero.
  with np.errstate(invalid="ignore", divide="ignore"):
    df["rel_drop"] = (df["orig_activation"] - df["para_activation"]) / df[
        "orig_activation"
    ]
  df["removal_stratum"] = pd.cut(
      df["rel_drop"], REMOVAL_BINS, labels=REMOVAL_LABELS
  )
  df["agree"] = np.sign(df["cf_delta"]) == np.sign(df["para_delta"])
  # An exactly-zero paraphrase delta means the rewrite left the score
  # untouched, so its sign carries no information.
  df["is_noop"] = df["para_delta"] == 0.0
  df["abs_cf"] = df["cf_delta"].abs()
  df["abs_para"] = df["para_delta"].abs()
  for c in df.select_dtypes("float64").columns:
    df[c] = df[c].astype("float32")
  for c in ("doc_id", "feature_id", "num_active_features"):
    df[c] = df[c].astype("int32")
  return df


def find_runs(cf_root: Path) -> tuple[list, list]:
  """Run JSONL files under cf_root, split into (recognised, unmatched).

  Returns:
    Two lists of (path, config) pairs; config is None in the unmatched list.
    Files under a SKIP_PARTS directory are in neither.
  """
  found, unmatched = [], []
  for path in sorted(cf_root.rglob("threshold*_counterfactual.jsonl")):
    if SKIP_PARTS & set(path.parts):
      continue
    cfg = parse_run_path(path)
    (found if cfg else unmatched).append((path, cfg))
  return found, unmatched


def build_cache(
    cf_root: Path = CF_ROOT, cache: Path = CF_CACHE, refresh: bool = False
) -> list[str]:
  """Writes <cache>/runs/<stem>.parquet and .features.parquet for every run.

  A per-run mtime:size stamp in <cache>/stamps.json skips JSONL files that
  have not changed unless `refresh`. Unrecognised files are listed, never
  skipped silently.

  Returns:
    The config_stem of every recognised run (cached before or now).
  """
  (cache / "runs").mkdir(parents=True, exist_ok=True)
  stamp_path = cache / "stamps.json"
  stamps = json.loads(stamp_path.read_text()) if stamp_path.exists() else {}
  found, unmatched = find_runs(cf_root)
  if unmatched:
    print("WARNING: files that did not match the expected naming:")
    for path, _ in unmatched:
      print("   ", path.relative_to(cf_root))
  for path, cfg in found:
    st = path.stat()
    key = cfg["config_stem"]
    stamp = f"{st.st_mtime_ns}:{st.st_size}"
    out = cache / "runs" / f"{key}.parquet"
    if not refresh and out.exists() and stamps.get(key) == stamp:
      continue
    print(f"parsing {key} ({st.st_size / 1e6:.0f} MB) ...", flush=True)
    df, feats = stream_run(path)
    df = _derive(df)
    for k, v in cfg.items():
      df[k] = v
      feats[k] = v
    df.to_parquet(out, index=False)
    feats.to_parquet(cache / "runs" / f"{key}.features.parquet", index=False)
    stamps[key] = stamp
  stamp_path.write_text(json.dumps(stamps, indent=2))
  return [cfg["config_stem"] for _, cfg in found]


def load_all(cache: Path = CF_CACHE, columns=None) -> pd.DataFrame:
  """Every cached run concatenated; `columns` restricts the parquet read."""
  parts = sorted((cache / "runs").glob("*.parquet"))
  parts = [p for p in parts if not p.name.endswith(".features.parquet")]
  if not parts:
    raise FileNotFoundError(f"no cached runs in {cache}; build_cache first")
  return pd.concat(
      [pd.read_parquet(p, columns=columns) for p in parts], ignore_index=True
  )


def load_features(cache: Path = CF_CACHE) -> pd.DataFrame:
  """Every run's feature_id -> description frame, concatenated."""
  parts = sorted((cache / "runs").glob("*.features.parquet"))
  return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def perm_null(df, by="doc_id", n_perm=200, seed=0):
  """Within-document permutation null for rho and sign agreement.

  The pooled correlation is NOT tested against zero: shuffling the paraphrase
  deltas within each document still leaves a substantial correlation, because
  documents differ in how edit-sensitive they are overall. This returns the
  honest chance level.

  Returns:
    rho_mean / rho_std of the pooled Spearman(cf_delta, shuffled para_delta)
    and acc_mean / acc_std of the sign agreement over the non-zero shuffled
    deltas, across n_perm shuffles.
  """
  rng = np.random.default_rng(seed)
  cf = df["cf_delta"].to_numpy(float)
  para = df["para_delta"].to_numpy(float)
  groups = list(df.groupby(by, sort=False).indices.values())
  rhos, accs = np.empty(n_perm), np.empty(n_perm)
  for i in range(n_perm):
    shuffled = para.copy()
    for idx in groups:
      shuffled[idx] = rng.permutation(shuffled[idx])
    rhos[i] = stats.spearmanr(cf, shuffled).correlation
    keep = shuffled != 0.0
    accs[i] = (np.sign(cf[keep]) == np.sign(shuffled[keep])).mean()
  return {
      "rho_mean": float(rhos.mean()),
      "rho_std": float(rhos.std()),
      "acc_mean": float(accs.mean()),
      "acc_std": float(accs.std()),
  }


def in_layout(meta: dict) -> bool:
  """Whether an arm is one of the twelve panels of the paper figure."""
  return (
      meta["head"] == LAYOUT_HEAD
      and LAYOUT_TARGET.get(meta["dataset"]) == meta["target"]
      and meta["sae_type"] in SAE_ORDER
      and meta["threshold"] in TAU_ORDER
  )


def load_arms(
    cache: Path, columns: list[str]
) -> tuple[list[tuple[dict, pd.DataFrame]], pd.DataFrame]:
  """The cached arms of the paper layout, in config_stem order.

  Returns:
    A list of (arm metadata, pairs) with the pairs in cache row order (the
    permutation null depends on it), and the feature description frame.
  """
  everything = load_all(cache, columns=columns)
  arms, skipped = [], []
  for _, arm in everything.groupby("config_stem", sort=True):
    meta = arm.iloc[0][ARM_COLS].to_dict()
    (arms if in_layout(meta) else skipped).append((meta, arm))
  if skipped:
    print(
        f"skipping {len(skipped)} cached arm(s) outside the paper layout:",
        ", ".join(meta["config_stem"] for meta, _ in skipped),
    )
  if len(arms) != len(SAE_ORDER) * len(TAU_ORDER) * len(LAYOUT_TARGET):
    print(f"note: {len(arms)} layout arms in {cache} (the paper figure has 12)")
  return arms, load_features(cache)


def feature_means(pairs: pd.DataFrame, min_docs: int, min_cf: float):
  """Per-feature doc count and mean deltas, with the `kept` filter flag."""
  means = (
      pairs.groupby("feature_id")
      .agg(
          n_docs=("doc_id", "nunique"),
          mean_cf=("cf_delta", "mean"),
          mean_para=("para_delta", "mean"),
          autointerp=("autointerp_score", "mean"),
      )
      .reset_index()
  )
  means["kept"] = (means["mean_cf"].abs() >= min_cf) & (
      means["n_docs"] >= min_docs
  )
  return means


def agreement_points(
    cache: Path = CF_CACHE, min_docs: int = MIN_DOCS, min_cf: float = MIN_CF
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Per-feature means for every layout arm, plus stats on the kept features.

  Returns:
    A `(points, arm_stats)` pair. `points` has one row per (arm, feature) with
    n_docs, mean_cf, mean_para, the mean auto-interp score, the description
    and the `kept` filter flag. `arm_stats` has one row per arm with n_total,
    n_kept, Spearman rho, Pearson r and the least-squares slope and intercept,
    all computed over the kept features only.
  """
  arms, feats = load_arms(cache, POINT_COLS)
  points, rows = [], []
  for meta, pairs in arms:
    means = feature_means(pairs, min_docs, min_cf)
    desc = feats.loc[
        feats["config_stem"] == meta["config_stem"],
        ["feature_id", "description"],
    ]
    means = means.merge(desc, on="feature_id", how="left")
    for k in STATS_META:
      means[k] = meta[k]
    points.append(means)
    kept = means[means["kept"]]
    slope, intercept = np.polyfit(kept["mean_cf"], kept["mean_para"], 1)
    rows.append(
        {c: meta[c] for c in STATS_META}
        | {
            "n_total": len(means),
            "n_kept": len(kept),
            "rho": stats.spearmanr(
                kept["mean_cf"], kept["mean_para"]
            ).correlation,
            "pearson": stats.pearsonr(kept["mean_cf"], kept["mean_para"])[0],
            "slope": slope,
            "intercept": intercept,
        }
    )
  return pd.concat(points, ignore_index=True)[OUT_COLS], pd.DataFrame(rows)


def arm_table_row(
    meta: dict, pairs: pd.DataFrame, min_docs: int, min_cf: float
) -> dict:
  """One tab:counterfactual row: counts, agreement and nulls over kept pairs."""
  means = feature_means(pairs, min_docs, min_cf).set_index("feature_id")
  kept = means[means["kept"]]
  kept_pairs = pairs[pairs["feature_id"].isin(kept.index)]
  signed = kept_pairs[~kept_pairs["is_noop"]]
  null = perm_null(kept_pairs, n_perm=N_PERM_TABLE, seed=SEED)
  return dict(
      config=meta["config_stem"],
      dataset=meta["dataset"],
      sae=meta["sae_type"],
      tau=float(meta["threshold"]),
      head=meta["head"],
      n_docs=pairs["doc_id"].nunique(),
      feat_all=len(means),
      feat_cf=int((means["mean_cf"].abs() >= min_cf).sum()),
      feat_kept=len(kept),
      pairs_all=len(pairs),
      pairs_kept=len(kept_pairs),
      feat_rho=stats.spearmanr(kept["mean_cf"], kept["mean_para"]).correlation,
      # The feature means are float32; the paper csv carries the float64
      # correlation (scipy >= 1.15 would otherwise keep float32).
      feat_r=stats.pearsonr(
          kept["mean_cf"].astype(float), kept["mean_para"].astype(float)
      )[0],
      feat_sign_agree=(
          np.sign(kept["mean_cf"]) == np.sign(kept["mean_para"])
      ).mean(),
      pooled_rho=stats.spearmanr(
          kept_pairs["cf_delta"], kept_pairs["para_delta"]
      ).correlation,
      pooled_null=null["rho_mean"],
      pair_sign_acc=signed["agree"].mean(),
      pair_null=null["acc_mean"],
      noop=kept_pairs["is_noop"].mean(),
      fully=(kept_pairs["removal_stratum"] == "fully").mean(),
      mean_spec=kept_pairs["specificity"].mean(),
      mean_tdev=kept_pairs["target_dev"].mean(),
      mean_ntdev=kept_pairs["nontarget_dev"].mean(),
      feat_rho_unfilt=stats.spearmanr(
          means["mean_cf"], means["mean_para"]
      ).correlation,
  )


def agreement_table(
    cache: Path = CF_CACHE, min_docs: int = MIN_DOCS, min_cf: float = MIN_CF
) -> pd.DataFrame:
  """tab:counterfactual, one row per layout arm (sorted dataset/sae/head/tau)."""
  arms, _ = load_arms(cache, TABLE_COLS)
  rows = [arm_table_row(meta, pairs, min_docs, min_cf) for meta, pairs in arms]
  return pd.DataFrame(rows).sort_values(["dataset", "sae", "head", "tau"])


def autointerp_bins(cache: Path = CF_CACHE) -> pd.DataFrame:
  """tab:cf_autointerp_bins: rho by auto-interp score quantile, BINS_STEM.

  The typeset block: the unfiltered arm, record-level quantiles (pd.qcut over
  the pairs' scores; each feature is assigned its modal bin), N_BINS bins.
  Per bin: the Spearman rho over per-feature mean deltas (the paper's
  column), the pooled rho over the pairs and its within-document permutation
  null, plus a whole-arm row. Row order of the pairs is preserved: the null
  draws depend on it.
  """
  pairs = pd.read_parquet(cache / "runs" / f"{BINS_STEM}.parquet")
  features = pairs.groupby("feature_id").agg(
      n=("doc_id", "nunique"),
      mean_cf=("cf_delta", "mean"),
      mean_para=("para_delta", "mean"),
      score=("autointerp_score", "first"),
  )
  codes, edges = pd.qcut(
      pairs.autointerp_score,
      N_BINS,
      labels=False,
      retbins=True,
      duplicates="drop",
  )
  pairs = pairs.assign(bin=codes)
  features = features.assign(
      bin=pairs.groupby("feature_id").bin.agg(lambda s: s.mode().iloc[0])
  )
  rows = []
  for b, group in pairs.groupby("bin"):
    in_bin = features[features.bin == b]
    rows.append(
        dict(
            bin=int(b),
            score_lo=edges[b],
            score_hi=edges[b + 1],
            n_feat=len(in_bin),
            n_pairs=len(group),
            rho_feat_means=stats.spearmanr(
                in_bin.mean_cf, in_bin.mean_para
            ).correlation,
            rho_pooled=stats.spearmanr(
                group.cf_delta, group.para_delta
            ).correlation,
            pooled_null=perm_null(group, n_perm=N_PERM_BINS, seed=SEED)[
                "rho_mean"
            ],
        )
    )
  rows.append(
      dict(
          bin="all",
          score_lo=features.score.min(),
          score_hi=features.score.max(),
          n_feat=len(features),
          n_pairs=len(pairs),
          rho_feat_means=stats.spearmanr(
              features.mean_cf, features.mean_para
          ).correlation,
          rho_pooled=stats.spearmanr(
              pairs.cf_delta, pairs.para_delta
          ).correlation,
          pooled_null=perm_null(pairs, n_perm=N_PERM_BINS, seed=SEED)[
              "rho_mean"
          ],
      )
  )
  return pd.DataFrame(rows)


def most_frequent(
    cache: Path,
    dataset: str,
    target: str,
    top: int = TOP_FREQUENT,
    min_docs: int = MIN_DOCS,
    min_cf: float = MIN_CF,
) -> pd.DataFrame:
  """tab:cf_frequent_<dataset>: the top features by document count per SAE.

  One block per SAE setting of the tau 0.70 logreg arm. Per feature (means
  over the (document, feature) pairs it was sampled in): n_docs, mean_cf
  (zero-ablation logit delta), mean_para (paraphrase logit delta), targ_dev
  (normalised change of the targeted feature; 1 = removed), nontarg_dev (the
  same over the other active features) and the mean auto-interp score, with
  the description verbatim from the run. The paper-table filters apply; the
  block counts needed by the captions ride along as columns.
  """
  blocks = []
  for sae in SAE_ORDER:
    stem = f"{dataset}-{target}-{sae}-tau0.70-logreg"
    pairs = pd.read_parquet(cache / "runs" / f"{stem}.parquet")
    desc = pd.read_parquet(
        cache / "runs" / f"{stem}.features.parquet",
        columns=["feature_id", "description"],
    )
    means = (
        pairs.groupby("feature_id")
        .agg(
            n_docs=("doc_id", "nunique"),
            mean_cf=("cf_delta", "mean"),
            mean_para=("para_delta", "mean"),
            targ_dev=("target_dev", "mean"),
            nontarg_dev=("nontarget_dev", "mean"),
            autointerp=("autointerp_score", "mean"),
            n_noop=("is_noop", "sum"),
        )
        .reset_index()
        .merge(desc, on="feature_id", how="left")
    )
    kept = means[
        (means["mean_cf"].abs() >= min_cf) & (means["n_docs"] >= min_docs)
    ]
    out = kept.sort_values(
        ["n_docs", "feature_id"], ascending=[False, True]
    ).head(top)
    fid = out["feature_id"].to_numpy()
    blocks.append(
        out.assign(
            feature=[f"{LAYERS[f // SAE_DIM]}_{f % SAE_DIM}" for f in fid],
            dataset=dataset,
            target=target,
            sae=sae,
            tau=0.70,
            n_features=len(means),
            n_kept=len(kept),
            docs_total=pairs["doc_id"].nunique(),
            pairs_total=len(pairs),
            noop_total=int(pairs["is_noop"].sum()),
            min_docs=min_docs,
            min_cf=min_cf,
        )
    )
  cols = [
      "dataset", "target", "sae", "tau", "feature", "feature_id", "n_docs",
      "mean_cf", "mean_para", "targ_dev", "nontarg_dev", "autointerp", "n_noop",
      "description", "n_features", "n_kept", "docs_total", "pairs_total",
      "noop_total", "min_docs", "min_cf",
  ]  # fmt: skip
  return pd.concat(blocks, ignore_index=True)[cols]


def counterfactual_analysis(
    cache: Path = CF_CACHE,
    min_docs: int = MIN_DOCS,
    min_cf: float = MIN_CF,
    top: int = TOP_FREQUENT,
) -> dict[str, pd.DataFrame]:
  """Every counterfactual table, keyed by its csv stem under <root>/analysis."""
  points, arm_stats = agreement_points(cache, min_docs, min_cf)
  tables = {
      "feature_agreement_points": points,
      "feature_agreement_stats": arm_stats,
      "feature_agreement_table": agreement_table(cache, min_docs, min_cf),
      "autointerp_bins": autointerp_bins(cache),
  }
  for key, (dataset, target) in MOST_FREQUENT.items():
    tables[f"most_frequent_{key}"] = most_frequent(
        cache, dataset, target, top, min_docs, min_cf
    )
  return tables


# Every paper extractor is a Gemma-2-2b; the fine-tuned ones tokenize
# identically to the base model, so one tokenizer serves the token metrics.
TOKENIZER = "google/gemma-2-2b"
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
ARMS_OF_PASS = {1: UNGUIDED_ARMS + ["feat"], 2: ["feat"], 3: ["feat"]}
# Per-document edit metrics summarised as <col>_mean / _median per arm in
# results; wlev is word_lev_norm_max (word Levenshtein / longer text).
EDIT_COLS = [
    "word_lev_norm_max",
    "word_lev_norm_orig",
    "frac_words_changed",
    "frac_gemma_tokens_changed",
    "len_ratio",
    "cosine_sim",
    "tok_lev_norm_max",
    "tok_lev_norm_orig",
    "word_jaccard_set",
    "word_jaccard_bag",
    "tok_jaccard_set",
    "tok_jaccard_bag",
]
N_BOOT = 1000
MIN_GENERATOR_DOCS = 30
WORD = re.compile(r"\s+|[^\s]+")


def words(text: str) -> list[str]:
  """Whitespace-delimited tokens; punctuation stays attached to its word."""
  return [t for t in WORD.findall(text or "") if not t.isspace()]


def edit_stats(a: str, b: str) -> dict:
  """Word-level edit statistics of b relative to a (difflib)."""
  ta, tb = words(a), words(b)
  matcher = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
  changed = n_del = n_ins = 0
  for op, i1, i2, j1, j2 in matcher.get_opcodes():
    if op != "equal":
      changed += max(i2 - i1, j2 - j1)
      n_del += i2 - i1
      n_ins += j2 - j1
  n = max(len(ta), 1)
  return {
      "n_words_orig": len(ta),
      "n_words_para": len(tb),
      "frac_words_changed": changed / n,
      "n_deleted": n_del,
      "n_inserted": n_ins,
      "len_ratio": len(tb) / n,
  }


def levenshtein(a: list, b: list) -> int:
  """Minimum number of single-element insertions, deletions and substitutions."""
  if not a:
    return len(b)
  if not b:
    return len(a)
  codes = {}
  ai = np.array([codes.setdefault(x, len(codes)) for x in a])
  bi = np.array([codes.setdefault(x, len(codes)) for x in b])
  prev = np.arange(len(bi) + 1)
  cols = np.arange(len(bi) + 1)
  for i, x in enumerate(ai, 1):
    row = np.empty_like(prev)
    row[0] = i
    row[1:] = np.minimum(prev[1:] + 1, prev[:-1] + (bi != x))
    prev = np.minimum.accumulate(row - cols) + cols
  return int(prev[-1])


def jaccard(a: list, b: list, multiset: bool = False) -> float:
  """Jaccard similarity as sets or (multiset=True) bags; 1.0 if both empty."""
  if multiset:
    ca, cb = collections.Counter(a), collections.Counter(b)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
  else:
    sa, sb = set(a), set(b)
    inter, union = len(sa & sb), len(sa | sb)
  return inter / union if union else 1.0


def similarity_stats(a: list, b: list, prefix: str) -> dict:
  """Levenshtein (normalised two ways) and Jaccard for two token sequences."""
  distance = levenshtein(a, b)
  return {
      f"{prefix}_lev": distance,
      f"{prefix}_lev_norm_max": distance / max(len(a), len(b), 1),
      f"{prefix}_lev_norm_orig": distance / max(len(a), 1),
      f"{prefix}_jaccard_set": jaccard(a, b),
      f"{prefix}_jaccard_bag": jaccard(a, b, multiset=True),
  }


def frac_ids_changed(ids_a: list[int], ids_b: list[int]) -> float:
  """frac_words_changed of edit_stats, over token id sequences."""
  matcher = difflib.SequenceMatcher(None, ids_a, ids_b, autojunk=False)
  changed = sum(
      max(i2 - i1, j2 - j1)
      for op, i1, i2, j1, j2 in matcher.get_opcodes()
      if op != "equal"
  )
  return changed / max(len(ids_a), 1)


def edit_metrics(original: str, paraphrase: str, tokenizer) -> dict:
  """Every per-document edit metric of a paraphrase: words, then Gemma tokens."""
  encode = lambda text: tokenizer(
      text, truncation=False, add_special_tokens=False
  )["input_ids"]
  ids_orig, ids_para = encode(original), encode(paraphrase)
  return {
      **edit_stats(original, paraphrase),
      "frac_gemma_tokens_changed": frac_ids_changed(ids_orig, ids_para),
      **similarity_stats(
          [w.lower() for w in words(original)],
          [w.lower() for w in words(paraphrase)],
          "word",
      ),
      **similarity_stats(ids_orig, ids_para, "tok"),
  }


def read_jsonl(path: Path) -> list[dict]:
  """One record per non-blank line."""
  with open(path) as f:
    return [json.loads(line) for line in f if line.strip()]


# <root>/<dataset>/shared/      the sample and the unguided paraphrases, shared
#                               by the dataset's settings
# <root>/<dataset>/<slug>/      thresholds and the originals' scores
# <root>/<dataset>/<slug>/p<k>/ pass k of the feature-guided attack (p1 also
#                               holds this setting's scores of the unguided arms)
def setting_dir(root: Path, setting: str) -> Path:
  """<root>/<dataset>/<slug> of a paper setting."""
  return Path(root) / SETTINGS[setting].dataset / setting


def shared_dir(root: Path, setting: str) -> Path:
  """<root>/<dataset>/shared of a paper setting."""
  return Path(root) / SETTINGS[setting].dataset / "shared"


def paraphrases_path(root: Path, setting: str, arm: str, pass_k: int) -> Path:
  """Paraphrase records of an arm (shared if unguided, per pass for feat)."""
  if arm in UNGUIDED_ARMS:
    return shared_dir(root, setting) / "paraphrases" / f"{arm}.jsonl"
  return (
      setting_dir(root, setting) / f"p{pass_k}" / "paraphrases" / f"{arm}.jsonl"
  )


def score_files(root: Path, setting: str, arm: str, pass_k: int) -> list[Path]:
  """Every score file of an arm: unsharded, shards, carried (feat) or humans (orig).

  orig is scored once per setting, the unguided arms once in pass 1, feat in
  every pass.
  """
  folder = setting_dir(root, setting)
  if arm != "orig":
    folder = folder / f"p{1 if arm in UNGUIDED_ARMS else pass_k}"
  folder = folder / "scores"
  files = [p for p in [folder / f"sae_{arm}.jsonl"] if p.exists()]
  files += sorted(folder.glob(f"sae_{arm}.*-of-*.jsonl"))
  extra = {"feat": "sae_feat.carried.jsonl", "orig": "sae_orig.humans.jsonl"}
  if arm in extra and (folder / extra[arm]).exists():
    files.append(folder / extra[arm])
  return files


def load_scores(
    root: Path, setting: str, arms: list[str], pass_k: int
) -> dict[str, pd.DataFrame]:
  """Each arm's score records as a frame indexed by doc_id.

  The logit is this setting's entry; the last record wins on duplicate
  doc_ids. Arms without score files are left out.
  """
  out = {}
  for arm in arms:
    files = score_files(root, setting, arm, pass_k)
    if not files:
      continue
    rows = {}
    for path in files:
      for r in read_jsonl(path):
        rows[r["doc_id"]] = {
            "doc_id": r["doc_id"],
            "arm": arm,
            "label": r["label"],
            "logit": r[setting]["logit"],
            "n_tokens": r.get("n_tokens"),
            "truncated": r.get("truncated"),
            "carried": bool(r.get("carried", False)),
            "flipped_pass_rec": r.get("flipped_pass"),
        }
    out[arm] = pd.DataFrame(list(rows.values())).set_index("doc_id")
  return out


def prf_arm(h: np.ndarray, a: np.ndarray, t: float) -> dict:
  """Confusion counts and P/R/F1 of AI logits a against human logits h at t."""
  tp = int((a >= t).sum())
  fp = int((h >= t).sum())
  fn = len(a) - tp
  tn = len(h) - fp
  p = tp / max(tp + fp, 1)
  r = tp / max(len(a), 1)
  return {
      "f1": 2 * p * r / max(p + r, 1e-12),
      "precision": p,
      "recall": r,
      "accuracy": (tp + tn) / max(len(a) + len(h), 1),
      "fpr": fp / max(len(h), 1),
      "tp": tp,
      "fp": fp,
      "fn": fn,
      "tn": tn,
  }


def bootstrap(h, a_orig, a_arm, t, edit, n_boot, seed=0) -> dict:
  """95% CIs over resampled AI docs (humans fixed).

  Returns f1_ci, recall_ci and flip_ci as (lo, hi) tuples, plus wlev_ci for
  the mean edit metric when `edit` has one value per AI doc and is not all
  NaN. Flip is measured against `a_orig` on the same resample.
  """
  rng = np.random.default_rng(seed)
  fp = int((h >= t).sum())
  n = len(a_arm)
  det_o, det_a = a_orig >= t, a_arm >= t
  idx = rng.integers(0, n, size=(n_boot, n))
  tp = det_a[idx].sum(1)
  p = tp / np.maximum(tp + fp, 1)
  r = tp / n
  f1 = 2 * p * r / np.maximum(p + r, 1e-12)
  do, da = det_o[idx], det_a[idx]
  flip = ((do & ~da).sum(1)) / np.maximum(do.sum(1), 1)
  q = lambda x: (float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975)))
  out = {"f1_ci": q(f1), "recall_ci": q(r), "flip_ci": q(flip)}
  if edit is not None and len(edit) == n and not np.isnan(edit).all():
    out["wlev_ci"] = q(np.nanmean(edit[idx], axis=1))
  return out


def per_document(
    scores: dict, paras: dict, docs: dict, t: float, pass_k: int, cosine: bool
) -> pd.DataFrame:
  """One row per scored (document, arm): logit, detection and edit metrics.

  Carried documents keep the text and score of the pass in which they first
  flipped, so the edit metrics are measured on the kept text; `flipped_pass`
  is the pass that first pushed the document below t. With `cosine`, MiniLM
  cosine similarity of every (original, paraphrase) pair is appended.
  """
  import transformers  # pylint: disable=g-import-not-at-top

  tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER)
  rows, pairs = [], []
  for arm, frame in scores.items():
    for doc_id, r in frame.iterrows():
      doc = docs[doc_id]
      rec = {
          "doc_id": doc_id,
          "arm": arm,
          "cond": COND_OF_ARM[arm],
          "pass": pass_k,
          "label": doc["label"],
          "generator": doc["model"],
          "n_words_orig": doc["n_words"],
          "n_tokens": r.get("n_tokens"),
          "truncated": r.get("truncated"),
          "logit": float(r["logit"]),
          "det": bool(r["logit"] >= t),
          "carried": bool(r.get("carried", False)),
      }
      if arm != "orig":
        prow = paras[arm][doc_id]
        text = prow["paraphrase"]
        rec.update(edit_metrics(doc["text"], text, tokenizer))
        rec["has_tags"] = prow.get("has_tags")
        rec["flipped_pass"] = (
            prow.get("flipped_pass")
            if prow.get("carried")
            else (pass_k if rec["logit"] < t else None)
        )
        pairs.append((len(rows), doc["text"], text))
      rows.append(rec)
  per_doc = pd.DataFrame(rows)
  if cosine and pairs:
    from detectorscope.autointerpret import compute_embedding_matrix  # pylint: disable=g-import-not-at-top

    a = compute_embedding_matrix([p[1] for p in pairs])
    b = compute_embedding_matrix([p[2] for p in pairs])
    per_doc["cosine_sim"] = np.nan
    per_doc.loc[[p[0] for p in pairs], "cosine_sim"] = (a * b).sum(1)
  return per_doc


def per_arm(
    scores: dict,
    per_doc: pd.DataFrame,
    thresholds: dict,
    ai_ids: list,
    hu_ids: list,
    setting: str,
    pass_k: int,
    n_boot: int,
) -> pd.DataFrame:
  """One row per arm: detection metrics with the humans fixed.

  F1 / precision / recall / accuracy at t with the human scores fixed and the
  AI scores replaced by the arm's paraphrase scores (AI docs restricted to
  those scored in both the arm and orig; n_failed counts the rest); flip =
  share of AI docs detected at base that no longer are; recall at the 5% / 1%
  FPR cuts; AUROC; 95% bootstrap CIs on F1, recall, flip and mean wlev;
  <edit>_mean / _median over EDIT_COLS; n_flipped_by_pass<j> counts.
  """
  t = thresholds["t"]
  orig = scores["orig"]
  h = orig.loc[[d for d in hu_ids if d in orig.index], "logit"].to_numpy()
  rows = []
  for arm, frame in scores.items():
    avail = [d for d in ai_ids if d in frame.index and d in orig.index]
    a_arm = frame.loc[avail, "logit"].to_numpy()
    a_orig = orig.loc[avail, "logit"].to_numpy()
    edits = per_doc[(per_doc.arm == arm) & (per_doc.label == 1)].set_index(
        "doc_id"
    )
    wl = (
        edits.loc[avail, "word_lev_norm_max"].to_numpy(dtype=float)
        if arm != "orig"
        else None
    )
    det_o = a_orig >= t
    ci = bootstrap(h, a_orig, a_arm, t, wl, n_boot)
    y = np.r_[np.zeros(len(h)), np.ones(len(a_arm))]
    row = {
        "setting": setting,
        "arm": arm,
        "cond": COND_OF_ARM[arm],
        "pass": pass_k,
        "threshold": t,
        "n_ai": len(avail),
        "n_failed": len(ai_ids) - len(avail),
        "n_human": len(h),
        "n_carried": int(frame.loc[avail, "carried"].sum()),
        **prf_arm(h, a_arm, t),
        "flip": float((det_o & (a_arm < t)).sum() / max(det_o.sum(), 1)),
        "n_detected_base": int(det_o.sum()),
        "n_flipped": int((det_o & (a_arm < t)).sum()),
        "auroc": float(roc_auc_score(y, np.r_[h, a_arm]))
        if len(a_arm)
        else np.nan,
        "f1_lo": ci["f1_ci"][0],
        "f1_hi": ci["f1_ci"][1],
        "recall_lo": ci["recall_ci"][0],
        "recall_hi": ci["recall_ci"][1],
        "flip_lo": ci["flip_ci"][0],
        "flip_hi": ci["flip_ci"][1],
        "recall_fpr5": float((a_arm >= thresholds["t_fpr5"]).mean())
        if len(a_arm)
        else np.nan,
        "recall_fpr1": float((a_arm >= thresholds["t_fpr1"]).mean())
        if len(a_arm)
        else np.nan,
        "logit_mean": float(a_arm.mean()) if len(a_arm) else np.nan,
        "n_truncated": int(frame.loc[avail, "truncated"].sum()),
    }
    if arm != "orig":
      row["wlev"] = float(np.nanmean(wl))
      row["wlev_lo"], row["wlev_hi"] = ci.get("wlev_ci", (np.nan, np.nan))
      for c in EDIT_COLS:
        if c in edits:
          row[f"{c}_mean"] = float(edits[c].mean())
          row[f"{c}_median"] = float(edits[c].median())
      flipped = edits.loc[avail, "flipped_pass"]
      for j in range(1, pass_k + 1):
        row[f"n_flipped_by_pass{j}"] = int((flipped <= j).sum())
    rows.append(row)
  return pd.DataFrame(rows)


def per_generator(per_doc: pd.DataFrame) -> pd.DataFrame:
  """Recall per (arm, generator) over generators with MIN_GENERATOR_DOCS docs."""
  rows = []
  for arm in per_doc.arm.unique():
    sub = per_doc[(per_doc.arm == arm) & (per_doc.label == 1)]
    for generator, group in sub.groupby("generator"):
      if len(group) >= MIN_GENERATOR_DOCS:
        rows.append(
            {
                "arm": arm,
                "generator": generator,
                "n": len(group),
                "recall": float(group["det"].mean()),
            }
        )
  return pd.DataFrame(rows)


def adversarial_results(
    root: Path,
    setting: str,
    pass_k: int,
    n_boot: int = N_BOOT,
    cosine: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Metrics of one (setting, pass): the per_doc, results and per_generator frames.

  The threshold t is the setting's frozen Youden cut (sample/thresholds.json,
  written by the thresholds stage of paraphrase_ablation.py). Pass k >= 2
  evaluates the feat arm only. The frames are what the pipeline used to write
  as p<k>/analysis/{per_doc,results,per_generator}.csv.
  """
  docs = {
      x["doc_id"]: x
      for x in read_jsonl(shared_dir(root, setting) / "sample" / "docs.jsonl")
  }
  thresholds = json.loads(
      (setting_dir(root, setting) / "sample" / "thresholds.json").read_text()
  )["models"][setting]
  scores = load_scores(root, setting, ["orig"] + ARMS_OF_PASS[pass_k], pass_k)
  assert "orig" in scores, "orig scores are required"
  paras = {
      arm: {r["doc_id"]: r for r in read_jsonl(path)}
      for arm in ARMS_OF_PASS[pass_k]
      if (path := paraphrases_path(root, setting, arm, pass_k)).exists()
  }
  ai_ids = sorted(d for d, x in docs.items() if x["label"] == 1)
  hu_ids = sorted(d for d, x in docs.items() if x["label"] == 0)
  per_doc = per_document(scores, paras, docs, thresholds["t"], pass_k, cosine)
  results = per_arm(
      scores, per_doc, thresholds, ai_ids, hu_ids, setting, pass_k, n_boot
  )
  return per_doc, results, per_generator(per_doc)


# Loaders of the notebook's remaining inputs: the activation caches (auto-interp
# descriptions, log-odds ratios), the paper heads and their feature statistics,
# the eval and density trees of the head comparison, the corpora's metadata, the
# adversarial results and the worked-example analyses.
CACHE_ROOT = paper.CACHE_ROOT
DATA_ROOT = paper.REPO / "data/data"
DEMO_SETTINGS = paper.REPO / "demo/config/settings.yml"


def _numpy(x) -> np.ndarray:
  """A pickled head field (torch tensor or array) as a numpy array."""
  return np.asarray(x.detach().cpu() if hasattr(x, "detach") else x)


def read_descriptions(train_dir: str, cache_root=CACHE_ROOT) -> pd.DataFrame:
  """Every auto-interp jsonl line of a train cache, one row each.

  Columns feature_id (layer rank * SAE_DIM + feature_index), terse, long,
  keywords and corr (rho of the stored (rho, p) pair). A feature described
  twice keeps both rows: the active-feature counts are jsonl line counts.
  """
  files = get_desc_files(str(Path(cache_root) / train_dir), paper.DESC_MODEL)
  assert [layer for layer, _ in files] == LAYERS, (train_dir, files)
  frames = []
  for rank, (_, path) in enumerate(files):
    with open(path) as f:
      records = [json.loads(line) for line in f if line.strip()]
    frames.append(
        pd.DataFrame(
            {
                "feature_id": [
                    rank * SAE_DIM + int(r["feature_index"]) for r in records
                ],
                "terse": [r["description"]["terse"] for r in records],
                "long": [r["description"].get("long") for r in records],
                "keywords": [
                    r["description"].get("keywords", "") for r in records
                ],
                "corr": [r["correlation"][0] for r in records],
            }
        )
    )
  return pd.concat(frames, ignore_index=True)


def feature_stats(stats_stem: str) -> dict:
  """The feature_stats npz of a setting (paper.Setting.stats_stem), hist dropped."""
  z = np.load(
      paper.SWEEP_ROOT / "feature_stats" / f"{stats_stem}_feature_stats.npz"
  )
  return {k: z[k] for k in z.files if k != "hist"}


def paper_head(train_dir: str, target: str, tau) -> dict:
  """The unpickled PAPER_HEAD sweep pickle of a train cache at threshold tau."""
  name = paper.head_filename(tau=tau, target=target, **paper.PAPER_HEAD)
  with open(paper.SWEEP_ROOT / "models" / train_dir / name, "rb") as f:
    return pickle.load(f)


def head_weight(train_dir: str, target: str, tau) -> np.ndarray:
  """Flat weight vector of the paper head trained at threshold tau."""
  state = paper_head(train_dir, target, tau)["state_dict"]
  (key,) = [k for k in state if k.endswith("weight")]
  return _numpy(state[key]).ravel()


def feature_block(setting, cache_root=CACHE_ROOT) -> pd.DataFrame:
  """Nonzero masked features of a setting's TOP_TAU head, with stats and text.

  One row per feature with nonzero weight inside the interp mask: layer,
  local index, weight, activation stats, mean_cf = weight * act_mean (the
  mean zero-ablation logit change over the training documents the feature
  fires in; positive pushes toward AI), density_ai / density_human, the
  verbatim terse / long / keywords and simulation corr of the auto-interp
  jsonl, feat = <layer>_<local>, train_ds and target.
  """
  st = feature_stats(setting.stats_stem)
  layers = [int(l) for l in np.asarray(st["layers"])]
  assert int(st["sae_dim"]) == SAE_DIM and layers == LAYERS, setting
  head = paper_head(setting.train_dir, setting.target, paper.TOP_TAU)
  w = _numpy(head["state_dict"]["weight"]).ravel().astype(np.float64)
  mask = _numpy(head["interp_mask"]).ravel().astype(bool)
  used = np.flatnonzero((w != 0) & mask)
  df = pd.DataFrame(
      {
          "feature_id": used,
          "layer": np.asarray(layers)[used // SAE_DIM],
          "local": used % SAE_DIM,
          "weight": w[used],
          "act_mean": st["act_mean"][used],
          "n_docs_active": st["n_active"][used],
          "n_active_ai": st["n_active_ai"][used],
          "n_active_human": st["n_active_human"][used],
      }
  )
  df["mean_cf"] = df["weight"] * df["act_mean"]
  df["density_ai"] = df["n_active_ai"] / int(st["n_ai_docs"])
  df["density_human"] = df["n_active_human"] / int(st["n_human_docs"])
  desc = read_descriptions(setting.train_dir, cache_root)
  df = df.join(
      desc.drop_duplicates("feature_id").set_index("feature_id"),
      on="feature_id",
  )
  df["feat"] = df["layer"].astype(str) + "_" + df["local"].astype(str)
  df["train_ds"], df["target"] = setting.train_dir, setting.target
  return df


def lor_join(df: pd.DataFrame, setting, cache_root=CACHE_ROOT) -> pd.DataFrame:
  """Adds lor, human_density and target_density from the setting's LOR csv files."""
  folder = str(Path(cache_root) / setting.train_dir)
  frames = []
  for rank, (_, path) in enumerate(get_lor_files(folder, setting.target)):
    lor = pd.read_csv(path)
    lor["feature_id"] = rank * SAE_DIM + lor["feature_id"].astype(int)
    frames.append(lor[["feature_id", "lor", "human_density", "target_density"]])
  return df.join(pd.concat(frames).set_index("feature_id"), on="feature_id")


# (head-filename tau, csv column label, strict threshold; None = every described
# feature, the unmasked head's set) of tab:active_feature_counts.
COUNT_THRESHOLDS = [
    ("0.0", "Total", None),
    ("0.70", "t>0.7", 0.7),
    ("0.80", "t>0.8", 0.8),
    ("0.85", "t>0.85", 0.85),
    ("0.90", "t>0.9", 0.9),
]


def active_feature_counts(cache_root=CACHE_ROOT) -> pd.DataFrame:
  """Global counts of tab:active_feature_counts, one row per COLING / PRDB-GPT-4o setting.

  Per COUNT_THRESHOLDS label: described features with score > tau (jsonl
  line counts), how many of those carry a positive / negative weight in the
  paper head trained at that tau, and the mean number that fire on a
  training document (feature_stats n_active / n_train_docs).
  """
  rows = []
  for slug in [s for s in paper.SETTING_ORDER if not s.startswith("prdb_")]:
    s = SETTINGS[slug]
    desc = read_descriptions(s.train_dir, cache_root)
    st = feature_stats(s.stats_stem)
    n_active, n_docs = st["n_active"], int(st["n_train_docs"])
    row = dict(
        group=paper.DATASET_NAME[s.dataset].lower(),
        sae_type=s.sae,
        n_train_docs=n_docs,
    )
    for tau, label, cut in COUNT_THRESHOLDS:
      kept = desc if cut is None else desc[desc["corr"] > cut]
      w = head_weight(s.train_dir, s.target, tau)
      passing = np.zeros(len(w), bool)
      passing[kept["feature_id"].unique()] = True
      positive = passing & (w > 0)
      row[label] = len(kept)
      row[f"{label} w>0"] = int(positive.sum())
      row[f"{label} w<0"] = int((passing & (w < 0)).sum())
      row[f"{label} act/doc"] = n_active[passing].sum() / n_docs
      row[f"{label} w>0 act/doc"] = n_active[positive].sum() / n_docs
    rows.append(row)
  return pd.DataFrame(rows)


# LogReg vs XGBoost cells: eval dirs of the linguistic baseline per head, and
# the density trees (logreg: the sweep's; gram2vec: by selection.json's stem).
LINGUISTIC_DIR = {"logreg": "logreg_gram2vec", "xgboost": "gram2vec"}
LOGREG_DENSITY = paper.SWEEP_ROOT / "density"


def eval_file(split: str, head_dir: str, tau, target: str) -> Path:
  """The one eval JSON of (split, head dir, tau, target); raises otherwise.

  XGBoost files carry the target token on PRDB_* and COLING_dev but not on
  the other splits, and thresholds are written 0.7 (xgb) or 0.70 (logreg),
  so matching is by parsed float plus an optional target token.
  """
  folder = paper.EVAL_ROOT / split / head_dir
  found = [
      p for p in folder.glob("*.json") if paper.parse_threshold(p.name) == tau
  ]
  tagged = [
      p
      for p in found
      if f"_{target}_" in p.name or p.name.startswith(f"{target}_")
  ]
  found = tagged or found
  if len(found) != 1:
    raise FileNotFoundError(f"{folder} tau={tau} target={target}: {found}")
  return found[0]


def density(head: str, setting: str, tau: str, target: str) -> dict:
  """nonzero_coefs and active_per_doc of one (head, setting key, tau, target)."""
  split = "COLING_dev" if target == "coling" else f"PRDB_{target}"
  if head == "xgboost":
    path = (
        paper.XGB_DENSITY
        / f"xgb_{setting}_{split}_threshold{tau}_{target}.json"
    )
  elif setting == "gram2vec":
    stem = json.loads((paper.GRAM2VEC_ROOT / "selection.json").read_text())[
        target
    ]["stem"]
    path = (
        paper.GRAM2VEC_ROOT / "density" / f"logreg_gram2vec_{split}_{stem}.json"
    )
  else:
    tau_lr = {"0.0": "0.0", "0.7": "0.70"}[tau]
    stem = f"sgd_lr{paper.PAPER_HEAD['lr']}_l1-{paper.PAPER_HEAD['l1']}_threshold{tau_lr}_{target}_ep{paper.PAPER_HEAD['epoch']}"
    path = LOGREG_DENSITY / f"logreg_{setting}_{split}_{stem}.json"
  d = json.loads(path.read_text())
  if d.get("head_type", head) != head:
    raise ValueError(f"{path} is a {d['head_type']} measurement, not {head}")
  return {
      "nonzero_coefs": int(d["nonzero_coefs"]),
      "active_per_doc": float(d["active_per_doc_mean"]),
  }


def head_comparison_cells(
    target: str, eval_cols: list, row_specs: list
) -> pd.DataFrame:
  """Tidy frame of every cell of one training target of tab:head_comparison_*.

  `eval_cols` are (split, block) pairs, one per column group; `row_specs`
  the (block, label, head, tau) rows (render.hc_row_specs). The setting key
  of a row comes from its label; the eval dir from paper.eval_dir.
  """
  sae_of = {tex: sae for sae, tex in paper.SAE_TEX.items()}
  records = []
  for block_name, label, head, tau in row_specs:
    if label == "gram2vec":
      setting, head_dir = "gram2vec", LINGUISTIC_DIR[head]
      tau_eval = None if head == "xgboost" else float(tau)
    else:
      sae = sae_of[label]
      setting = SETTINGS[f"coling_{sae}"].eval_dir.removeprefix("logreg_")
      head_dir, tau_eval = paper.eval_dir(sae, head), float(tau)
    dens = density(head, setting, tau, target)
    for split, block in eval_cols:
      path = eval_file(split, head_dir, tau_eval, target)
      metrics = json.loads(path.read_text())[block]
      records.append(
          {
              "train_target": target,
              "block": block_name,
              "label": label,
              "setting": setting,
              "head": head,
              "tau": tau,
              "split": split,
              "generator": block,
              "eval_file": os.path.relpath(path, paper.REPO),
              **{
                  m: float(metrics[m])
                  for m in ("auroc", "f1", "recall_at_5_fpr")
              },
              **dens,
          }
      )
  return pd.DataFrame(records)


# tab:datasets. Generator normalisation: raw name -> (canonical name, display
# family); canonical merges only spelling variants of the same model, the
# family is what the table prints. Every raw name must be listed.
HUMAN = "human"
MODEL_FAMILY = {
    "gpt-3.5-turbo": ("gpt-3.5-turbo", "GPT-3.5"),
    "gpt-35": ("gpt-3.5-turbo", "GPT-3.5"),
    "GPT3.5": ("gpt-3.5-turbo", "GPT-3.5"),
    "ChatGPT": ("gpt-3.5-turbo", "GPT-3.5"),
    "ChatGPT-turbo": ("gpt-3.5-turbo", "GPT-3.5"),
    "chatgpt": ("gpt-3.5-turbo", "GPT-3.5"),
    "gpt4": ("gpt-4", "GPT-4/4o"),
    "GPT4": ("gpt-4", "GPT-4/4o"),
    "GPT-4": ("gpt-4", "GPT-4/4o"),
    "gpt4o": ("gpt-4o", "GPT-4/4o"),
    "gpt-4o": ("gpt-4o", "GPT-4/4o"),
    "gpt-4o-2024-08-06": ("gpt-4o-2024-08-06", "GPT-4/4o"),
    "gpt-4o-mini-2024-07-18": ("gpt-4o-mini", "GPT-4/4o"),
    "davinci": ("davinci", "GPT-3 (davinci)"),
    "text-davinci-002": ("text-davinci-002", "GPT-3 (davinci)"),
    "text-davinci-003": ("text-davinci-003", "GPT-3 (davinci)"),
    "gpt3": ("text-davinci-002", "GPT-3 (davinci)"),
    "7B": ("llama-7b", "Llama"),
    "13B": ("llama-13b", "Llama"),
    "30B": ("llama-30b", "Llama"),
    "65B": ("llama-65b", "Llama"),
    "llama3-8b": ("llama3-8b", "Llama"),
    "llama3-70b": ("llama3-70b", "Llama"),
    "Llama2": ("llama2", "Llama"),
    "Llama3": ("llama3", "Llama"),
    "llama-3.1-70b-versatile": ("llama-3.1-70b", "Llama"),
    "llama-3.1-70B-versatile": ("llama-3.1-70b", "Llama"),
    "llama": ("llama", "Llama"),
    "llama-chat": ("llama-chat", "Llama"),
    **{
        m: (m, "OPT")
        for m in [
            "opt_125m", "opt_350m", "opt_1.3b", "opt_2.7b", "opt_6.7b",
            "opt_13b", "opt_30b", "opt_iml_max_1.3b", "opt_iml_30b",
        ]
    },  # fmt: skip
    **{
        m: (m, "FLAN-T5")
        for m in [
            "flan_t5_small",
            "flan_t5_base",
            "flan_t5_large",
            "flan_t5_xl",
            "flan_t5_xxl",
        ]
    },
    "gemma-7b-it": ("gemma-7b-it", "Gemma"),
    "gemma2-9b-it": ("gemma2-9b-it", "Gemma"),
    "gemini1.5": ("gemini-1.5", "Gemini"),
    "gemini": ("gemini", "Gemini"),
    "t0_3b": ("t0_3b", "T0/BLOOM"),
    "t0_11b": ("t0_11b", "T0/BLOOM"),
    "bloom_7b": ("bloom_7b", "T0/BLOOM"),
    "bloomz": ("bloomz", "T0/BLOOM"),
    "gpt_j": ("gpt-j", "GPT-J/NeoX"),
    "gpt_neox": ("gpt-neox", "GPT-J/NeoX"),
    "gpt2": ("gpt2", "GPT-2"),
    "GLM130B": ("glm-130b", "GLM"),
    "ChatGLM": ("chatglm", "GLM"),
    "mixtral-8x7b": ("mixtral-8x7b", "Mistral"),
    "mistral-7b": ("mistral-7b", "Mistral"),
    "mistral": ("mistral", "Mistral"),
    "mistral-chat": ("mistral-chat", "Mistral"),
    "cohere": ("cohere", "Cohere"),
    "cohere-chat": ("cohere-chat", "Cohere"),
    "dolly": ("dolly", "Dolly"),
    "dolly-v2-12b": ("dolly-v2-12b", "Dolly"),
    "Dolly": ("dolly", "Dolly"),
    "mpt": ("mpt", "MPT"),
    "mpt-chat": ("mpt-chat", "MPT"),
    "Qwen": ("qwen", "Qwen"),
    "qwen": ("qwen", "Qwen"),
    "Baichuan": ("baichuan", "Baichuan"),
    "StableLM": ("stablelm", "StableLM"),
    "claude": ("claude", "Claude"),
    # LLM-DetectAIve "humanized": generated by X, rewritten by Y -> family of Y
    "gpt-3.5-turbo_to_gemini1.5": (
        "gemini-1.5 (humanized)",
        "Gemini (humanized)",
    ),
    "gpt-3.5-turbo_to_mistral-7b": (
        "mistral-7b (humanized)",
        "Mistral (humanized)",
    ),
    # MixSet rows with model=None: GPT-4 / Llama-2 edits of human text
    "None": ("mixset-mixed", "GPT-4/Llama-2 (mixed)"),
}
# Domain grouping: the coarse label the table prints (authored as LaTeX).
COLING_DOMAIN_GROUP = {
    "reddit_eli5": "Q\\&A", "eli5": "Q\\&A", "squad": "Q\\&A", "finance": "Q\\&A",
    "medicine": "Q\\&A", "open_qa": "Q\\&A", "wiki_csai": "Q\\&A",
    "wikihow": "WikiHow", "wikipedia": "Wikipedia", "arxiv": "arXiv abstracts",
    "sci_gen": "scientific tables", "reddit": "Reddit", "tldr": "Reddit",
    "cmv": "opinion (CMV)", "yelp": "Yelp reviews", "xsum": "news", "wp": "stories",
    "roct": "stories", "hswag": "commonsense", "outfox": "student essays",
    "peerread": "peer reviews", "cnn": "misc.", "imdb": "misc.", "pubmed": "misc.",
    "dialogsum": "misc.",
}  # fmt: skip
COLING_TEST_GROUP = {
    "CUDRT": "editing tasks (expand, polish, QA, refine, rewrite, summarise)",
    "ieltsduck": "IELTS essays",
    "LLM-DetectAIve-IELTS": "IELTS essays",
    "NLPeer-F1000": "peer reviews",
    "NLPeer-ARR22": "peer reviews",
    "NLPeer-COLING20": "peer reviews",
    "peersum": "peer-review summaries",
    "Mixset": "mixed human--AI text (news, blog, email, speech, reviews, abstracts, QA)",
}
# (dataset, split label, role, split directory); COLING's "test" directory
# holds the shared task's dev split, "realtest" the test split.
CORPUS_SPLITS = [
    ("COLING", "train", "train", "COLING_tokenized/train"),
    ("COLING", "dev", "dev", "COLING_tokenized/test"),
    ("COLING", "devtest", "dev", "COLING_tokenized/devtest"),
    ("COLING", "test", "test", "COLING_tokenized/realtest"),
    ("PRDB", "train", "train", "PRDB/train"),
    ("PRDB", "test", "test", "PRDB/test"),
    ("RAID", "test", "test", "RAID/test"),
]


def split_metadata(path: Path, cols: list[str]) -> pd.DataFrame:
  """Those of `cols` an on-disk split has, as a frame (never text)."""
  import datasets  # pylint: disable=g-import-not-at-top

  ds = datasets.load_from_disk(str(path))
  return ds.select_columns(
      [c for c in cols if c in ds.column_names]
  ).to_pandas()


def corpus_split(dataset: str, path: Path) -> pd.DataFrame:
  """One split as (source, domain, domain_group, model, label) rows.

  COLING's domain is sub_source (train / dev) or the domain column with
  NaNs filled from source (test); devtest has no per-document domain and
  gets an empty group (the table prints no breakdown for it). MixSet rows
  have model None, mapped to the string "None" for MODEL_FAMILY.
  """
  if dataset == "COLING":
    df = split_metadata(
        path, ["source", "sub_source", "model", "label", "domain"]
    )
    df["model"] = df["model"].fillna("None")
    if "sub_source" in df:
      df["domain"] = df["sub_source"]
      df["domain_group"] = df["sub_source"].map(COLING_DOMAIN_GROUP)
    elif "domain" in df:
      df["domain"] = df["domain"].fillna(df["source"])
      df["domain_group"] = df["source"].map(COLING_TEST_GROUP)
    else:
      df["domain"], df["domain_group"] = df["source"], ""
  elif dataset == "PRDB":
    raw = split_metadata(path, ["model", "conference"])
    years = raw["conference"].str.extract(r"(\d{4})")[0].astype(int)
    df = pd.DataFrame(
        {
            "source": "PRDB",
            "domain": raw["conference"],
            "domain_group": f"ICLR/NeurIPS peer reviews ({years.min()}--{years.max()})",
            "model": raw["model"],
            "label": (raw["model"] != HUMAN).astype(int),
        }
    )
  else:
    raw = split_metadata(path, ["model", "domain", "attack"])
    assert (raw["attack"] == "none").all(), raw["attack"].value_counts()
    df = pd.DataFrame(
        {
            "source": "RAID",
            "domain": raw["domain"],
            "domain_group": raw["domain"],
            "model": raw["model"],
            "label": (raw["model"] != HUMAN).astype(int),
        }
    )
  return df[["source", "domain", "domain_group", "model", "label"]]


def generator_families(group: pd.DataFrame) -> str:
  """Families in descending AI-doc order; '(k)' counts a family's generators."""
  fam = (
      group.groupby("family")
      .agg(n=("n", "sum"), k=("canonical", "nunique"))
      .sort_values("n", ascending=False)
  )
  return ", ".join(
      f"{paper.tex(f)} ({r.k})" if r.k > 1 else paper.tex(f)
      for f, r in fam.iterrows()
  )


def domain_groups(counts: pd.DataFrame) -> str:
  """Domain groups in descending doc order (authored as LaTeX already)."""
  grp = counts.groupby("domain_group")["n"].sum().sort_values(ascending=False)
  return ", ".join(g for g in grp.index if g)


def dataset_composition(
    data_root=DATA_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """tab:datasets' inputs from the corpora's metadata columns.

  Returns:
    `rows`, the long-format counts (one row per dataset, split, source,
    domain, generator, label) and `summary`, one row per split with the
    table's cells and the counts behind them. `label` is authoritative:
    COLING test's MixSet "HWT" rows carry a generator name but count as
    human. Raises on a generator name missing from MODEL_FAMILY or a domain
    without a group.
  """
  rows, summary = [], []
  for dataset, split, role, folder in CORPUS_SPLITS:
    df = corpus_split(dataset, Path(data_root) / folder)
    unknown = sorted(set(df["model"]) - set(MODEL_FAMILY) - {HUMAN})
    assert not unknown, f"{dataset}/{split}: unmapped generator names {unknown}"
    assert (
        df["domain_group"].notna().all()
    ), f"{dataset}/{split}: unmapped domains"
    counts = (
        df.groupby(
            ["source", "domain", "domain_group", "model", "label"],
            observed=True,
        )
        .size()
        .reset_index(name="n")
    )
    counts.insert(0, "split", split)
    counts.insert(0, "dataset", dataset)
    counts["canonical"] = counts["model"].map(
        lambda m: HUMAN if m == HUMAN else MODEL_FAMILY[m][0]
    )
    counts["family"] = counts["model"].map(
        lambda m: HUMAN if m == HUMAN else MODEL_FAMILY[m][1]
    )
    rows.append(counts)
    ai = counts[counts["label"] == 1]
    n, n_ai = int(counts["n"].sum()), int(ai["n"].sum())
    summary.append(
        dict(
            dataset=dataset,
            split=split,
            role=role,
            n=n,
            n_ai=n_ai,
            pct_ai=100 * n_ai / n,
            n_gen_raw=ai["model"].nunique(),
            n_gen=ai["canonical"].nunique(),
            n_domains=counts["domain"].nunique(),
            generators=generator_families(ai),
            domains=domain_groups(counts),
        )
    )
  long = pd.concat(rows, ignore_index=True).rename(
      columns={
          "model": "generator_raw",
          "canonical": "generator_canonical",
          "family": "generator_family",
      }
  )
  cols = [
      "dataset", "split", "source", "domain", "domain_group", "generator_raw",
      "generator_canonical", "generator_family", "label", "n",
  ]  # fmt: skip
  return long[cols], pd.DataFrame(summary)


# fig:corr / fig:lor_hist: the nine settings in plot order (the hue order of
# the histograms is the order of appearance), the LOR splits per dataset as
# (column prefix, eval-dir split, csv target) and the old dataset keys.
HIST_ORDER = [
    f"{d}_{sae}"
    for d in ("coling", "prdb4o", "prdb")
    for sae in ("ft_sae", "ft_model_sae", "base_sae")
]
HIST_DATASET = {
    "coling": "coling",
    "prdb_gpt4o": "prdb_gpt4o",
    "prdb": "prdb_llama",
}
LOR_SPLITS = {
    "coling": [
        ("dev", "test", None),
        ("devtest", "devtest", None),
        ("test", "realtest", None),
    ],
    "prdb": [
        (g, "test", g) for g in ["gpt4o", "llama", "gemini", "claude", "qwen"]
    ],
}


def add_lor(
    df: pd.DataFrame, lor_dir: Path, prefix: str, target
) -> pd.DataFrame:
  """Inner-joins the dir's layer<N>_[<target>_]log_odds_ratios.csv files as <prefix>_* columns."""
  lor = pd.concat(
      [
          pd.read_csv(path).assign(layer=layer)
          for layer, path in get_lor_files(str(lor_dir), target)
      ]
  ).add_prefix(f"{prefix}_")
  df = df.merge(
      lor,
      left_on=["old_feature_index", "layer"],
      right_on=[f"{prefix}_feature_id", f"{prefix}_layer"],
  )
  return df.drop(columns=[f"{prefix}_feature_id", f"{prefix}_layer"])


def described_features(cache_root=CACHE_ROOT) -> pd.DataFrame:
  """Every described feature of the nine settings with its per-split log-odds ratios.

  One row per auto-interp jsonl line with sae_type, dataset (coling /
  prdb_gpt4o / prdb_llama), target, layer, old_feature_index, feature_index
  (layer-major global id), correlation, p_value and one <split>_lor column
  per LOR split of the dataset; a feature is kept only if it has a LOR in
  every split (inner joins).
  """
  frames = []
  for slug in HIST_ORDER:
    s = SETTINGS[slug]
    parts = []
    for layer, path in get_desc_files(
        str(Path(cache_root) / s.train_dir), paper.DESC_MODEL
    ):
      with open(path) as f:
        parts.append(
            pd.json_normalize([json.loads(line) for line in f]).assign(
                layer=layer
            )
        )
    df = pd.concat(parts)
    df["p_value"] = df["correlation"].str[1]
    df["correlation"] = df["correlation"].str[0]
    rank = {layer: i for i, layer in enumerate(sorted(df["layer"].unique()))}
    df["old_feature_index"] = df["feature_index"]
    df["feature_index"] = df["layer"].map(rank) * SAE_DIM + df["feature_index"]
    for prefix, split, target in LOR_SPLITS[
        paper.DATASET_NAME[s.dataset].lower()
    ]:
      df = add_lor(
          df,
          Path(cache_root) / s.train_dir.replace("train", split),
          prefix,
          target,
      )
    df["sae_type"], df["dataset"], df["target"] = (
        s.sae,
        HIST_DATASET[s.dataset],
        s.target,
    )
    frames.append(df)
  return pd.concat(frames, ignore_index=True)


def in_dist_key(target: str) -> str:
  """The LOR split the head of `target` was trained on."""
  return "dev" if target == "coling" else target


# fig:l1_grid_search: the data csv columns (summary.csv called l1 `lambda`).
GRID_COLS = ["head", "tau", "lr", "lambda", "epoch", "auroc", "f1", "recall_at_1_fpr", "nonzero_coefs", "active_per_doc"]  # fmt: skip


def l1_grid(frame: pd.DataFrame) -> pd.DataFrame:
  """The 120 figure rows of a sweep_metrics frame: COLING dev, epoch 3, tau 0 / 0.7.

  lr and lambda become the grid's tags (1e-02, 3e-01, 1e+00) and the rows
  are sorted by head, tau, lr, lambda.
  """
  keep = (
      (frame["split"] == "COLING_dev")
      & (frame["epoch"] == 3)
      & frame["tau"].isin([0.0, 0.7])
  )
  rows = frame[keep].assign(
      lr=frame["lr"].map("{:.0e}".format),
      **{"lambda": frame["l1"].map("{:.0e}".format)},
  )
  return (
      rows[GRID_COLS]
      .sort_values(["head", "tau", "lr", "lambda"])
      .reset_index(drop=True)
  )


def adversarial_analysis(root: Path, setting: str, pass_k: int) -> Path:
  """<root>/<dataset>/<slug>/p<k>/analysis/, computed when its csv files are missing.

  The stored run has every analysis; adversarial_results (tokenizer, edit
  distances, MiniLM cosines) takes 10+ minutes per setting and pass.
  """
  folder = setting_dir(root, setting) / f"p{pass_k}" / "analysis"
  names = ["per_doc", "results", "per_generator"]
  if not all((folder / f"{name}.csv").exists() for name in names):
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(names, adversarial_results(root, setting, pass_k)):
      frame.to_csv(folder / f"{name}.csv", index=False)
  return folder


def adversarial_rows(settings, conds, metrics, root=ADV_ROOT) -> pd.DataFrame:
  """One row per (setting, condition) from the per-pass results.csv files.

  The feat3 row is the feat arm of pass N_PASSES; "pass" records where each
  row came from. The `metrics` columns are floats, NaN when results.csv
  lacks them. Settings without a pass-1 analysis are skipped.
  """
  rows = []
  for slug in settings:
    s = SETTINGS[slug]
    first = pd.read_csv(
        adversarial_analysis(root, slug, 1) / "results.csv"
    ).set_index("cond")
    last = pd.read_csv(
        adversarial_analysis(root, slug, paper.N_PASSES) / "results.csv"
    ).set_index("cond")
    for cond in conds:
      if cond == "feat3":
        src = last.loc["feat"] if "feat" in last.index else None
      else:
        src = first.loc[cond] if cond in first.index else None
      if src is None:
        continue
      row = {
          "setting": slug,
          "dataset": s.dataset,
          "family": s.sae,
          "cond": cond,
          "pass": int(src["pass"]) if "pass" in src else 1,
      }
      for m in metrics:
        row[m] = float(src[m]) if m in src and pd.notna(src[m]) else np.nan
      rows.append(row)
  return pd.DataFrame(rows)


def adversarial_figure_rows(settings, root=ADV_ROOT) -> pd.DataFrame:
  """The TABLE_CONDS x FIG_METRICS rows of the figure and appendix tables.

  Adds the family label as "group"; base has no flip or wlev and the
  document counts are integers.
  """
  df = adversarial_rows(settings, paper.TABLE_CONDS, paper.FIG_METRICS, root)
  df["group"] = df["family"].map(paper.SAE_LABEL)
  df.loc[df["cond"] == "base", ["flip", "wlev"]] = np.nan
  df[["n_ai", "n_human"]] = df[["n_ai", "n_human"]].astype(int)
  return df[
      ["setting", "dataset", "family", "cond", "group"] + paper.FIG_METRICS
  ]


# Worked examples: the analyze record of a recipe's documents, cached as
# analysis.jsonl (the meta line, then one pair per line) so that a routine run
# never loads a model.
SENTENCE_END = re.compile(r"[.!?][\"'\)\]]*(?=\s|$)")


def worked_documents(source, doc_ids: list[str]) -> list[dict]:
  """{id, label, text} of the recipe's documents, in doc_ids order.

  `source` is a jsonl of such records, or None for the PRDB test split
  (label = the model column).
  """
  if source is not None:
    rows = {r["id"]: r for r in read_jsonl(Path(source))}
    return [{k: rows[d][k] for k in ("id", "label", "text")} for d in doc_ids]
  df = split_metadata(
      DATA_ROOT / "PRDB/test", ["id", "model", "review"]
  ).set_index("id")
  return [
      {"id": d, "label": df.loc[d, "model"], "text": df.loc[d, "review"]}
      for d in doc_ids
  ]


def detector_config(train_dir: str) -> dict:
  """The demo settings.yml entry whose data_dir is `train_dir`, defaults merged, sae_paths filled."""
  import yaml  # pylint: disable=g-import-not-at-top

  raw = yaml.safe_load(DEMO_SETTINGS.read_text())
  (cfg,) = [
      c for c in raw["settings"] if os.path.basename(c["data_dir"]) == train_dir
  ]
  cfg = raw.get("defaults", {}) | cfg
  cfg.setdefault(
      "sae_paths",
      [
          f"{cfg['sae_dir']}/{h}_input_finetuned_sae.pth"
          for h in cfg["hook_points"]
      ],
  )
  return cfg


def truncate(token_strings, text: str, max_tokens: int) -> str:
  """First `max_tokens` tokens, cut back to the last sentence end past the halfway point."""
  tokens = token_strings(text)
  if len(tokens) <= max_tokens:
    return text.rstrip()
  prefix = "".join(tokens[:max_tokens])
  if not text.startswith(prefix):
    prefix = " ".join(text.split()[: int(0.75 * max_tokens)])
  ends = [m.end() for m in SENTENCE_END.finditer(prefix)]
  return (
      prefix[: ends[-1]] if ends and ends[-1] >= 0.5 * len(prefix) else prefix
  ).rstrip()


def analyze_document(
    detector, head, features: pd.DataFrame, text: str, max_tokens: int
) -> dict:
  """Truncate, extract, score and explain one document.

  Returns tokens, logit, bias, P(AI) and one entry per active feature with
  `cf` = w_i x_i and the tokens (and activations) it fires on; asserts the
  logreg identity cf.sum() + bias == logit.
  """
  from detectorscope.data.cache import evidence  # pylint: disable=g-import-not-at-top

  text = truncate(detector.extractor.get_token_strings, text, max_tokens)
  detection = detector.explain(text, head)
  assert (
      abs(detection.residual) < 1e-3
  ), f"cf sum + bias != logit ({detection.residual})"
  highlights = {f.id: f.highlights for f in detection.features}
  feats = []
  for fid, row in evidence(features, detection).iterrows():
    idx, act = zip(*highlights[fid])
    layer = int(re.search(r"\d+", str(row["layer"])).group())
    clean = lambda v: None if isinstance(v, float) and np.isnan(v) else v
    feats.append(
        {
            "feature_id": int(fid),
            "name": f"{layer}_{int(row['local_index'])}",
            "cf": float(row["cf_delta"]),
            "terse": clean(row.get("description.terse")),
            "keywords": clean(row.get("description.keywords")),
            "correlation": clean(row.get("correlation")),
            "token_idx": list(idx),
            "token_act": [round(v, 4) for v in act],
        }
    )
  return {
      "text": text,
      "tokens": detection.tokens,
      "logit": detection.logit,
      "bias": detection.bias,
      "prob": detection.p_ai,
      "n_tokens": len(detection.tokens),
      "n_active": len(feats),
      "features": feats,
  }


def analyze_documents(setting, tau, docs: list[dict], max_tokens: int) -> dict:
  """Runs the setting's detector and TOP_TAU-style paper head at `tau` on `docs`.

  Returns the analyze record {meta, pairs}: meta names the setting, its
  data_dir, the head and head_spec (target, tau) and max_tokens; pairs holds
  one single-document pair per document.
  """
  from detectorscope.data.cache import load_feature_table  # pylint: disable=g-import-not-at-top
  from detectorscope.model.detectors.auditable import AuditableDetector  # pylint: disable=g-import-not-at-top
  from detectorscope.model.detectors.heads.head import DetectorHead  # pylint: disable=g-import-not-at-top

  cfg = detector_config(setting.train_dir)
  head_path = (
      paper.SWEEP_ROOT
      / "models"
      / setting.train_dir
      / paper.head_filename(tau=tau, target=setting.target, **paper.PAPER_HEAD)
  )
  head = DetectorHead.create_detector_head(
      "logreg", detector_path=str(head_path)
  )
  features = load_feature_table(
      cfg["data_dir"],
      cfg["desc_model"],
      SAE_DIM,
      setting.target,
      len(cfg["hook_points"]),
  )
  detector = AuditableDetector.from_paths(
      cfg["model_path"],
      "logreg",
      None,
      hook_points=cfg["hook_points"],
      sae_model_paths=cfg["sae_paths"],
  )
  pairs = [
      {
          "pair_id": d["id"],
          "docs": [
              d
              | analyze_document(
                  detector, head, features, d["text"], max_tokens
              )
          ],
      }
      for d in docs
  ]
  meta = {
      "setting": cfg["name"],
      "data_dir": cfg["data_dir"],
      "head": str(head_path),
      "head_spec": {
          "head_type": "logreg",
          "target": setting.target,
          "tau": tau,
      },
      "max_tokens": max_tokens,
  }
  return {"meta": meta, "pairs": pairs}


def worked_example_analysis(
    cache: Path, stored: Path, setting, tau, max_tokens, doc_ids, source
) -> dict:
  """The analyze record of a recipe's documents, from the cache when present.

  Order: `cache` (analysis.jsonl), else the pairs of `stored` (a full
  analyze JSON) that hold the documents, else analyze_documents on the
  recipe's documents; either of the latter is written to `cache`.
  """
  if cache.exists():
    lines = cache.read_text().splitlines()
    return {
        "meta": json.loads(lines[0]),
        "pairs": [json.loads(l) for l in lines[1:]],
    }
  if stored.exists():
    data = json.loads(stored.read_text())
    pairs = [
        p for p in data["pairs"] if any(d["id"] in doc_ids for d in p["docs"])
    ]
    data = {"meta": data["meta"], "pairs": pairs}
  else:
    data = analyze_documents(
        setting, tau, worked_documents(source, doc_ids), max_tokens
    )
  cache.parent.mkdir(parents=True, exist_ok=True)
  with open(cache, "w") as f:
    f.write(
        "\n".join(json.dumps(x) for x in [data["meta"], *data["pairs"]]) + "\n"
    )
  return data
