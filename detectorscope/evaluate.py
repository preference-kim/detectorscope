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

"""Evaluates a detector on a huggingface dataset (pipeline stage 7).

Takes in a detector model of type Detector and runs it on the entire dataset,
then writes one JSON (-o) with a block of metrics per generator and for
"all", including the Youden-optimal threshold that later experiments freeze;
the paper's eval_results/<split>/<detector dir>/ files are its output. Can
specify whether or not the activations are already cached: the interpretable
detector (-dt auditable --head_only, no extractor) then scores the
max-activation caches (memory-mapped, --use_mmap_cache DIR, or the .pth
layers beside the dataset, --use_max_activation_caches), and -dt gram2vec
scores the split's gram2vec cache the same way (--use_mmap_cache DIR). If
not cached then we need the full HFExtractor + SAE model.

Usage:
  python -m detectorscope.evaluate -d data/data/PRDB/test -tc review -c model \
      -dt auditable -ht logreg --head_only -e models/heads/<head>.pkl \
      --use_mmap_cache outputs/mmap_caches/PRDB_test_gemma2_gpt4o \
      -o eval_results/PRDB_gpt4o/logreg_gemma2/<head>_results.json
"""

import argparse
import os
import sys

import datasets
import torch
import numpy as np
import json
from sklearn.metrics import roc_auc_score, roc_curve

from detectorscope.model.detector import Detector
from detectorscope.data.cache import LayerStack
from detectorscope.data.cache import SAE_COLUMN_INFIX


def activation_columns(ds: datasets.Dataset) -> list[str]:
  """The per-token CSR activation columns of an activation dataset.

  Matching the infix rather than listing hook names keeps this correct across
  models, layer counts and layer indices. evaluate() passes these columns to
  ds.map(remove_columns=...): map writes every surviving input column to a new
  Arrow file, and the activation columns are 99.6% of the bytes (about 209 GiB
  for the 171,837-row PRDB test split), while only the metadata and the
  predictions column are ever read back. datasets.disable_caching() does not
  avoid the write, it only moves it to a temporary directory.
  """
  return [c for c in ds.column_names if SAE_COLUMN_INFIX in c]


def load_dataset(
    data_path: str,
    num_examples: int,
) -> datasets.Dataset:
  """Loads the split in torch format, truncated to num_examples when set."""
  ds = datasets.load_from_disk(data_path)
  ds = ds.with_format("torch")

  if num_examples:
    ds = ds.select(range(num_examples))

  return ds


def compute_metrics(
    ds: datasets.Dataset, human_label: str, model_col: str, preds_col: str
):
  """Computes detection metrics per generator and for all AI text pooled.

  Each generator in model_col (and "all") is scored against the same human
  rows: AUROC, recall at 5% and 1% FPR, and accuracy, precision, recall and
  F1 at the Youden-optimal threshold (argmax of TPR - FPR), recorded as
  optimal_threshold and frozen by the later experiments. Scores >= threshold
  count as AI.

  Returns:
    {generator: {metric: value}} with values rounded to 4 decimals plus the
    class counts n_human and n_ai; when the ROC cannot be built AUROC and the
    FPR recalls are NaN and the threshold falls back to 0.5.
  """
  df = ds.select_columns([model_col, preds_col]).to_pandas()

  human_preds = df[df[model_col] == human_label][preds_col].values
  n_human = len(human_preds)
  y_true_human = np.zeros(n_human)

  results = {}
  for m in df[model_col].unique().tolist() + ["all"]:
    if m == human_label:
      continue

    mask = (
        (df[model_col] != human_label) if m == "all" else (df[model_col] == m)
    )

    ai_preds = df[mask][preds_col].values
    n_ai = len(ai_preds)
    y_true_ai = np.ones(n_ai)

    y_true = np.concatenate([y_true_human, y_true_ai])
    y_scores = np.concatenate([human_preds, ai_preds])

    try:
      auroc = float(roc_auc_score(y_true, y_scores))
      fpr, tpr, thresholds = roc_curve(y_true, y_scores)

      recall_at_5_fpr = float(np.interp(0.05, fpr, tpr))
      recall_at_1_fpr = float(np.interp(0.01, fpr, tpr))

      optimal_idx = np.argmax(tpr - fpr)
      opt_thresh = thresholds[optimal_idx]

    except ValueError:
      auroc = float("nan")
      recall_at_5_fpr = float("nan")
      recall_at_1_fpr = float("nan")
      opt_thresh = 0.5  # Fallback threshold if ROC fails

    tp = float(np.sum(ai_preds >= opt_thresh))
    fn = float(np.sum(ai_preds < opt_thresh))
    fp = float(np.sum(human_preds >= opt_thresh))
    tn = float(np.sum(human_preds < opt_thresh))

    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn)

    results[m] = {
        "accuracy": round(acc, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "auroc": round(auroc, 4),
        "recall_at_5_fpr": round(recall_at_5_fpr, 4),
        "recall_at_1_fpr": round(recall_at_1_fpr, 4),
        "optimal_threshold": round(opt_thresh, 4),
        "n_human": n_human,
        "n_ai": n_ai,
    }

  return results


def evaluate(
    ds: datasets.Dataset,
    detector: Detector,
    text_col: str,
    batch_size: int,
    features: LayerStack | None = None,
    row_map: np.ndarray | None = None,
    model_col: str | None = None,
    cpu_only: bool = False,
) -> datasets.Dataset:
  """Scores every row of ds and returns it with a "predictions" column.

  With a cache the detector's `predict_cached` sees the dense rows
  `features.fetch(row_map[indices])` of each batch and only `model_col` is
  decoded from ds; without one the text goes through `detector.predict`. The
  per-token activation columns are dropped from the output either way (see
  activation_columns), so the result holds the metadata and predictions only.

  Args:
    ds: the split, in torch format.
    detector: the detector to score with.
    text_col: text column, read only when scoring from text.
    batch_size: rows per map batch.
    features: the split's cache (SAE max activations or gram2vec) aligned
      with the dataset rows, or None to score the text.
    row_map: cache row of each dataset row; required with features.
    model_col: the one column handed to map when scoring from features.
    cpu_only: keep the head on CPU even when CUDA is available.
  """
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if cpu_only:
    device = torch.device("cpu")
  activation_cols = activation_columns(ds)

  if features is None:

    def apply_to_text(batch: dict[str, list]):
      predictions = detector.predict(batch[text_col])
      return {"predictions": predictions.cpu().tolist()}

    return ds.map(
        apply_to_text,
        batched=True,
        batch_size=batch_size,
        remove_columns=activation_cols,
    )

  def apply_to_features(_col: list, indices: list[int]):
    inputs = torch.from_numpy(features.fetch(row_map[indices])).to(device)
    predictions = detector.predict_cached(inputs)
    return {"predictions": predictions.cpu().tolist()}

  return ds.map(
      apply_to_features,
      batched=True,
      batch_size=batch_size,
      with_indices=True,
      input_columns=[model_col],
      remove_columns=activation_cols,
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-d",
      "--data_path",
      type=str,
      help="Path to the dataset",
  )
  parser.add_argument(
      "-m",
      "--extractor_model_path",
      type=str,
      help="Path to the extractor model",
  )
  parser.add_argument(
      "-tp",
      "--extractor_tokenizer_path",
      type=str,
      default=None,
      help="(Optional) path to the extractor model's tokenizer",
  )
  parser.add_argument(
      "-dt",
      "--detector_type",
      type=str,
      choices=[
          "huggingface",
          "auditable",
          "interpretable",
          "gram2vec",
          "binoculars",
          "fastdetectgpt",
          "remodetect",
          "specdetect",
      ],
      help="Type of detector to evaluate",
  )
  parser.add_argument(
      "-ht",
      "--head_type",
      type=str,
      required=False,
      choices=["logreg", "xgboost"],
      help="Type of detector head to load",
  )
  parser.add_argument(
      "-e",
      "--detector_head_path",
      type=str,
      required=False,
      help="Path to the detector head to evaluate",
  )
  parser.add_argument(
      "-b",
      "--batch_size",
      type=int,
      default=4,
      help="Batch size for evaluation",
  )
  parser.add_argument(
      "-sm",
      "--sae_model_paths",
      type=str,
      nargs="+",
      required=False,
      help="One SAE per hook point: a state_dict .pth, or a Gemma Scope id",
  )
  parser.add_argument(
      "-hp",
      "--hook_points",
      type=str,
      nargs="+",
      required=False,
      help="hook points for SAEs to connect to",
  )
  parser.add_argument(
      "-c",
      "--model_col",
      type=str,
      default="model",
      help="Column in the dataset that contains the class",
  )
  parser.add_argument(
      "-tc",
      "--text_col",
      type=str,
      default="review",
      help="Column in the dataset that contains the text",
  )
  parser.add_argument(
      "-hl",
      "--human_label",
      type=str,
      default="human",
      help="Label for the human class in the dataset",
  )
  parser.add_argument(
      "-n",
      "--num_examples",
      type=int,
      help="Num examples to eval on (Optional, defaults to full dataset)",
  )
  parser.add_argument(
      "--use_max_activation_caches",
      action="store_true",
      help="If true, use pre-computed max activation caches from disk",
  )
  parser.add_argument(
      "--use_mmap_cache",
      type=str,
      default=None,
      help=(
          "Directory of converted max activation caches, gathered per batch "
          "instead of loaded into RAM (see process.py from-pth)"
      ),
  )
  parser.add_argument(
      "--exit_if_output_path_exists",
      action="store_true",
      help="If true, exit script if the output path file already exists",
  )
  parser.add_argument(
      "--head_only",
      action="store_true",
      help="If true, load only the detector head",
  )
  parser.add_argument(
      "--gemma_scope_repo",
      type=str,
      default="google/gemma-scope-2b-pt-res",
      help="Hub repo of the Gemma Scope ids among --sae_model_paths",
  )
  parser.add_argument(
      "--cpu_only",
      action="store_true",
      help="If true, load on cpu",
  )
  parser.add_argument(
      "-o",
      "--output_path",
      type=str,
      default="results.json",
      help="Path for the output results to be printed to",
  )
  args = parser.parse_args()

  # Disable caching to ensure that huggingface doesn't save the intermediate
  # steps of computation. This takes up twice as much disk space and can
  # cause OOM errors on large datasets.
  # NOTE: This is not necessary for machines with lots of extra disk space.
  datasets.disable_caching()

  if args.use_mmap_cache and args.use_max_activation_caches:
    parser.error(
        "--use_mmap_cache and --use_max_activation_caches are mutually "
        "exclusive; pick one feature source"
    )

  if args.exit_if_output_path_exists and os.path.exists(args.output_path):
    print(f"Found results file at {args.output_path}, exiting")
    sys.exit(0)

  det_args = {}
  if args.detector_type in ("auditable", "interpretable"):
    det_args |= {
        "head_type": args.head_type,
        "head_path": args.detector_head_path,
        "head_only": args.head_only,
    }
    if not args.head_only:
      det_args |= {
          "hook_points": args.hook_points,
          "sae_model_paths": args.sae_model_paths,
          "gemma_scope_repo": args.gemma_scope_repo,
      }

  if args.detector_type == "gram2vec" and args.head_type:
    det_args |= {"head_type": args.head_type}

  if args.detector_type == "huggingface" and args.extractor_tokenizer_path:
    det_args |= {"tokenizer_path": args.extractor_tokenizer_path}

  # Load the dataset
  ds = load_dataset(args.data_path, args.num_examples)

  # Load the requested feature source: memory-mapped <block>.*.npy arrays
  # (SAE layers or the gram2vec block), or the resident .pth layers that sit
  # beside the dataset.
  features = None
  row_map = None
  if args.use_mmap_cache or args.use_max_activation_caches:
    features = LayerStack.open(args.use_mmap_cache or args.data_path)
    n_full = len(datasets.load_from_disk(args.data_path))
    n_cached = features.n_rows
    # A cache built with -n covers a prefix of the dataset; accept it for an
    # eval over no more than that prefix (smoke tests).
    partial_ok = bool(args.num_examples) and n_cached >= args.num_examples
    if n_cached != n_full and not partial_ok:
      raise ValueError(
          f"the cache has {n_cached} rows but the dataset has {n_full}; "
          f"was it built from {args.data_path}?"
      )
    row_map = np.arange(n_cached)
    if args.num_examples:
      row_map = row_map[: args.num_examples]

  # Load the detector
  detector = Detector.create_detector(
      args.detector_type, model_path=args.extractor_model_path, **det_args
  )

  # Run evaluation
  ds = evaluate(
      ds,
      detector,
      args.text_col,
      args.batch_size,
      features=features,
      row_map=row_map,
      model_col=args.model_col,
      cpu_only=args.cpu_only,
  )

  # Compute accuracy, precision, recall, f1 score on the elements
  results = compute_metrics(ds, args.human_label, args.model_col, "predictions")

  print(results)

  with open(args.output_path, "w") as f:
    json.dump(results, f, indent=2)
