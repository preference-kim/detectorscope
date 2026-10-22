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

"""One grid point of the paper's SGD + L1 sweep: train, evaluate, density.

The grid is GRID: 4 learning rates x 5 L1 lambdas x 2 interp thresholds for
each of the nine (SAE head, target) pairs, then 4 x 5 at tau 0.0 for the three
gram2vec baselines, 420 points. A point trains one head for 10 epochs with
checkpoints at 3 and 10 (SGD, momentum 0.9, batch 1024; SAE features divided
by their RMS, gram2vec features z-scored), scores each checkpoint on the
target's held-out splits, measures how many supported features fire per
document on the last of them, and writes one metrics JSON per (split,
checkpoint) under <root>/metrics/ (schema:
notes/briefs/gates/viz_refactor/sweep/SCHEMA.md). Checkpoints, eval JSONs and
metrics that already exist are reused, so a point can be resubmitted.

Usage (slurm arrays pass $SLURM_ARRAY_TASK_ID as --index):
  python scripts/hparam_sweep.py --index 0
  python scripts/hparam_sweep.py --index 419 --root outputs/gram2vec_sgd_l1_ablation
"""

import argparse
import dataclasses
import json
import os
import pickle

import datasets
import numpy as np

from detectorscope.data.cache import LayerStack
from detectorscope.evaluate import compute_metrics
from detectorscope.evaluate import evaluate
from detectorscope.evaluate import load_dataset
from detectorscope.model.detector import Detector
from detectorscope.model.detectors.heads.training import train_logreg

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_ROOT = "data/sae_activations"
DESC_MODEL = "gemini-3.1-flash-lite-preview"
LRS = (1e-3, 3e-3, 1e-2, 3e-2)
L1S = (1e-2, 3e-2, 1e-1, 3e-1, 1e0)
TAUS = (0.0, 0.70)
EPOCHS = (3, 10)
METRICS = ("auroc", "f1", "recall_at_1_fpr", "recall_at_5_fpr")
EVAL_BATCH = 1024
DENSITY_DOCS = 20000
BLOCK_SIZE = 500


@dataclasses.dataclass(frozen=True)
class Setting:
  """One (head, target) pair and the cache it trains on.

  `train` names the cache under --mmap_root (SAE) or --gram2vec_root; the
  eval caches follow from it (eval_cache).
  """

  head: str
  target: str
  train: str
  sae: bool = True


SETTINGS = [
    Setting("logreg_gemma2", "gpt4o", "PRDB_train_gemma2_gpt4o"),
    Setting("logreg_gemma2", "llama", "PRDB_train_gemma2_llama"),
    Setting("logreg_gemma2", "coling", "COLING_train_gemma2"),
    Setting("logreg_base_gemma2", "gpt4o", "PRDB_train_base_gemma2"),
    Setting("logreg_base_gemma2", "llama", "PRDB_train_base_gemma2"),
    Setting("logreg_base_gemma2", "coling", "COLING_train_base_gemma2"),
    Setting("logreg_gemmascope", "gpt4o", "PRDB_train_gemmascope"),
    Setting("logreg_gemmascope", "llama", "PRDB_train_gemmascope"),
    Setting("logreg_gemmascope", "coling", "COLING_train_gemmascope"),
    Setting("logreg_gram2vec", "gpt4o", "PRDB_train", sae=False),
    Setting("logreg_gram2vec", "llama", "PRDB_train", sae=False),
    Setting("logreg_gram2vec", "coling", "COLING_train", sae=False),
]
GRID = [
    (setting, lr, l1, tau)
    for setting in SETTINGS
    for lr in LRS
    for l1 in L1S
    for tau in (TAUS if setting.sae else TAUS[:1])
]

# Eval splits per target, ascending cache size; density is measured on the
# last. COLING_dev is the COLING_tokenized/test split and COLING_test the
# realtest one (the established naming inversion).
SPLITS = {
    "coling": ("COLING_devtest", "RAID_test", "COLING_test", "COLING_dev"),
    "gpt4o": ("PRDB_gpt4o",),
    "llama": ("PRDB_llama",),
}
SAE_CACHE = {
    "COLING_devtest": "COLING_devtest",
    "RAID_test": "RAID_test",
    "COLING_test": "COLING_realtest",
    "COLING_dev": "COLING_test",
    "PRDB_gpt4o": "PRDB_test",
    "PRDB_llama": "PRDB_test",
}
GRAM2VEC_DATASETS = {
    "COLING_train": "data/data/COLING_tokenized/train",
    "COLING_dev": "data/data/COLING_tokenized/test",
    "COLING_devtest": "data/data/COLING_tokenized/devtest",
    "COLING_test": "data/data/COLING_tokenized/realtest",
    "RAID_test": "data/data/RAID/test",
    "PRDB_train": "data/data/PRDB/train",
    "PRDB_test": "data/data/PRDB/test",
}


def stem(lr: float, l1: float, tau: float, target: str) -> str:
  """The recipe's checkpoint / eval file stem, without the _ep<N> suffix."""
  tau_tag = "0.0" if tau == 0 else f"{tau:.2f}"
  return f"sgd_lr{lr:.0e}_l1-{l1:.0e}_threshold{tau_tag}_{target}"


def models_dir(setting: Setting) -> str:
  if setting.sae:
    return setting.train
  return "coling" if setting.target == "coling" else "prdb"


def eval_cache(setting: Setting, split: str) -> str:
  """Cache name of an eval split, from the training cache's SAE suffix."""
  if setting.sae:
    return SAE_CACHE[split] + setting.train.split("_train", 1)[1]
  return "PRDB_test" if split.startswith("PRDB") else split


def source(
    setting: Setting, cache: str, args: argparse.Namespace
) -> tuple[str, str]:
  """(dataset with the labels, feature cache directory) of a cache name."""
  if setting.sae:
    return f"{args.cache_root}/{cache}", f"{args.mmap_root}/{cache}"
  return f"{REPO}/{GRAM2VEC_DATASETS[cache]}", f"{args.gram2vec_root}/{cache}"


def prefault(directory: str) -> None:
  """Reads the cache arrays front to back once so random gathers hit RAM."""
  for name in sorted(os.listdir(directory)):
    if name.endswith(".npy"):
      with open(os.path.join(directory, name), "rb") as f:
        while f.read(1 << 26):
          pass


def train(
    setting: Setting,
    lr: float,
    l1: float,
    tau: float,
    model_path: str,
    args: argparse.Namespace,
) -> None:
  """Trains the point's head with the recipe's flags, checkpointing EPOCHS."""
  data_path, cache = source(setting, setting.train, args)
  prefault(cache)
  train_logreg.run(
      argparse.Namespace(
          data_path=data_path,
          use_mmap_cache=cache,
          num_examples=None,
          label_col="model",
          interp_threshold=tau,
          desc_gen_model=DESC_MODEL,
          target_model=None if setting.target == "coling" else setting.target,
          feature_scale=f"{cache}/feature_scale.npy" if setting.sae else None,
          standardize="none" if setting.sae else "zscore",
          optimizer="sgd",
          momentum=0.9,
          l1_lambda=l1,
          l2_lambda=0.0,
          epochs=max(EPOCHS),
          batch_size=1024,
          lr=lr,
          checkpoint_epochs=",".join(str(epoch) for epoch in EPOCHS),
          threads=None,
          cpu_only=False,
          exit_if_trained_model_exists=False,
          output_path=model_path,
      )
  )


def evaluate_split(
    setting: Setting,
    split: str,
    head_path: str,
    eval_json: str,
    args: argparse.Namespace,
) -> None:
  """Scores a checkpoint on one split from its cache, as evaluate.py does."""
  data_path, features_dir = source(setting, eval_cache(setting, split), args)
  prefault(features_dir)
  features = LayerStack.open(features_dir)
  if setting.sae:
    detector = Detector.create_detector(
        "auditable",
        "google/gemma-2-2b",
        head_type="logreg",
        head_path=head_path,
        head_only=True,
    )
  else:
    detector = Detector.create_detector(
        "gram2vec", head_path, head_type="logreg"
    )
  ds = evaluate(
      load_dataset(data_path, None),
      detector,
      None,
      EVAL_BATCH,
      features,
      np.arange(features.n_rows),
      "model",
  )
  os.makedirs(os.path.dirname(eval_json), exist_ok=True)
  with open(eval_json, "w") as f:
    json.dump(compute_metrics(ds, "human", "model", "predictions"), f, indent=2)


def contributing(head_path: str) -> np.ndarray:
  """Features the head can respond to: non-zero weight and masked in."""
  with open(head_path, "rb") as f:
    artifact = pickle.load(f)
  weight = np.asarray(artifact["state_dict"]["weight"]).ravel()
  mask = np.asarray(artifact["interp_mask"]).ravel().astype(bool)
  return (weight != 0) & mask


def sample_blocks(n_rows: int) -> list[tuple[int, int]]:
  """DENSITY_DOCS rows as evenly spaced blocks of BLOCK_SIZE consecutive rows.

  A block's column indices are one contiguous span of the mmap, so sampling
  blocks rather than rows costs one sequential read each.
  """
  if DENSITY_DOCS >= n_rows:
    return [(0, n_rows)]
  n_blocks = max(1, DENSITY_DOCS // BLOCK_SIZE)
  size = min(BLOCK_SIZE, DENSITY_DOCS)
  starts = np.linspace(0, n_rows - size, n_blocks).astype(np.int64)
  return [(int(start), int(start) + size) for start in starts]


def active_per_doc(stack: LayerStack, contrib: np.ndarray) -> float:
  """Mean number of contributing features that fire in a sampled document."""
  blocks = sample_blocks(stack.n_rows)
  counts = np.zeros(sum(end - start for start, end in blocks), dtype=np.int64)
  for rank, layer in enumerate(stack.layers):
    offset = rank * stack.d_sae
    written = 0
    for start, end in blocks:
      lo, hi = int(layer.crow[start]), int(layer.crow[end])
      hits = contrib[offset + np.asarray(layer.col[lo:hi])]
      row_ids = np.repeat(
          np.arange(end - start), np.diff(layer.crow[start : end + 1])
      )
      counts[written : written + end - start] += np.bincount(
          row_ids[hits], minlength=end - start
      )
      written += end - start
  return float(counts.mean())


def run(args: argparse.Namespace) -> None:
  """Trains, evaluates and measures GRID[args.index], skipping finished work."""
  setting, lr, l1, tau = GRID[args.index]
  point = stem(lr, l1, tau, setting.target)
  root = os.path.abspath(args.root)
  splits = SPLITS[setting.target]
  checkpoints = {
      epoch: f"{root}/models/{models_dir(setting)}/{point}_logreg_ep{epoch}.pkl"
      for epoch in EPOCHS
  }

  def metrics_path(epoch: int, split: str) -> str:
    return f"{root}/metrics/{setting.head}_{split}_{point}_ep{epoch}.json"

  def eval_path(epoch: int, split: str) -> str:
    name = f"{point}_ep{epoch}_logreg_results.json"
    return f"{root}/eval_results/{split}/{setting.head}/{name}"

  print(f"point {args.index}: {setting.head} {point} -> {root}")
  if args.dry_run:
    for epoch in EPOCHS:
      print(
          checkpoints[epoch], *(eval_path(epoch, s) for s in splits), sep="\n"
      )
    return
  todo = [
      epoch
      for epoch in EPOCHS
      if not all(os.path.exists(metrics_path(epoch, s)) for s in splits)
  ]
  if not todo:
    print("metrics exist, nothing to do")
    return
  if not all(os.path.exists(path) for path in checkpoints.values()):
    train(
        setting,
        lr,
        l1,
        tau,
        checkpoints[max(EPOCHS)].replace(f"_ep{max(EPOCHS)}", ""),
        args,
    )

  density_stack = LayerStack.open(
      source(setting, eval_cache(setting, splits[-1]), args)[1]
  )
  for epoch in todo:
    for split in splits:
      if not os.path.exists(eval_path(epoch, split)):
        evaluate_split(
            setting, split, checkpoints[epoch], eval_path(epoch, split), args
        )
    contrib = contributing(checkpoints[epoch])
    density = active_per_doc(density_stack, contrib)
    print(
        f"epoch {epoch}: nonzero_coefs={int(contrib.sum())} "
        f"active_per_doc={density:.2f}"
    )
    os.makedirs(f"{root}/metrics", exist_ok=True)
    for split in splits:
      with open(eval_path(epoch, split)) as f:
        scores = json.load(f)["all"]
      record = {
          "head": setting.head,
          "target": setting.target,
          "split": split,
          "lr": lr,
          "l1": l1,
          "tau": tau,
          "epoch": epoch,
          **{metric: scores[metric] for metric in METRICS},
          "nonzero_coefs": int(contrib.sum()),
          "active_per_doc": density,
          "head_path": checkpoints[epoch],
          "eval_json": eval_path(epoch, split),
      }
      with open(metrics_path(epoch, split), "w") as f:
        json.dump(record, f, indent=2)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description=__doc__.split("\n\n", maxsplit=1)[0]
  )
  parser.add_argument(
      "--index",
      type=int,
      required=True,
      help=(
          f"Grid point, 0..{len(GRID) - 1} (points {len(GRID) - 60}+ are "
          "gram2vec)"
      ),
  )
  parser.add_argument(
      "--root",
      type=str,
      default=None,
      help=(
          "Sweep tree holding models/, eval_results/ and metrics/ (default "
          "outputs/logreg_sgd_l1_ablation, outputs/gram2vec_sgd_l1_ablation "
          "for the gram2vec points)"
      ),
  )
  parser.add_argument(
      "--cache_root",
      type=str,
      default=CACHE_ROOT,
      help="Directory of the SAE split datasets (labels and description files)",
  )
  parser.add_argument(
      "--mmap_root",
      type=str,
      default="outputs/mmap_caches",
      help="Directory of the converted SAE caches, one sub-directory per split",
  )
  parser.add_argument(
      "--gram2vec_root",
      type=str,
      default="outputs/gram2vec_features",
      help="Directory of the gram2vec caches, one sub-directory per split",
  )
  parser.add_argument(
      "--dry_run",
      action="store_true",
      help="Print the point's checkpoint and eval paths and exit",
  )
  args = parser.parse_args()
  if args.root is None:
    args.root = (
        "outputs/logreg_sgd_l1_ablation"
        if GRID[args.index][0].sae
        else "outputs/gram2vec_sgd_l1_ablation"
    )
  datasets.disable_caching()
  run(args)
