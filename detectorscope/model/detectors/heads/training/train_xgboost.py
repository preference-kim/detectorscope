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

"""Trains an XGBoost head to detect AI-generated text (the appendix arm).

A fixed-hyperparameter booster (150 trees, depth 6, lr 0.1, seed 42) over
the same feature sources as train_logreg.py: the interp-masked SAE max
activations of a split (every layer is concatenated in RAM), or the dense
gram2vec feature cache of the linguistic baseline. A stratified 10% split
(seed 42) is held out for the printed accuracy.

The output pickles ``model``, ``interp_mask`` and ``means`` for XGBoostHead;
heads trained on a cache that names its columns (gram2vec) add
``feature_names``, ``feature_cache`` and ``target_model``. Usage:

  python -m detectorscope.model.detectors.heads.training.train_xgboost \
      -d DATA_PATH -t 0.7 -tm gpt4o -o OUT.pkl
  python -m detectorscope.model.detectors.heads.training.train_xgboost \
      -d data/data/PRDB/train --use_mmap_cache outputs/gram2vec_features/PRDB_train \
      -tm gpt4o -o OUT.pkl
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import torch
from tqdm.auto import tqdm
import xgboost as xgb

from detectorscope.data import loading


class TqdmCallback(xgb.callback.TrainingCallback):
  """Callback to display a tqdm progress bar during XGBoost training.

  Not picklable: the trainer drops it from the classifier before saving.
  """

  def __init__(self, n_estimators: int):
    self.pbar = tqdm(total=n_estimators, desc="Training XGBoost")

  def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
    """Advances the bar by one tree; False keeps training."""
    self.pbar.update(1)
    return False

  def after_training(self, model):
    """Closes the bar and returns the model unchanged."""
    self.pbar.close()
    return model


def train_xgboost(
    y: np.ndarray,
    interp_mask: np.ndarray,
    x: sp.csr_matrix | np.ndarray,
    cpu_only: bool = False,
    feature_names: list[str] | None = None,
) -> tuple[xgb.XGBClassifier, np.ndarray]:
  """Trains an XGBoost classifier over a feature matrix.

  Args:
    y: int 0/1 label per row of x (1 = AI).
    interp_mask: Mask of shape [input_dim] for interpretability.
    x: [n_docs, input_dim] feature matrix: sparse CSR for the SAE max
      activations, dense float32 for a gram2vec cache.
    cpu_only: If true, train only on cpu regardless of device availability.
    feature_names: Column names recorded in the booster (dense features
      only), so it validates text-path inputs by name.

  Returns:
    The trained XGBoost classifier and an array of shape [1, input_dim] of
    feature-level mean activations, averaged over the documents where each
    feature fires (0 where it never fires). The means are saved in the
    artifact for importance scoring.
  """
  # Apply the interp mask
  if sp.issparse(x):
    # multiply returns COO in recent scipy; CSR is what the row split and the
    # booster index (sklearn's train_test_split made the same conversion).
    x_masked = x.multiply(interp_mask).tocsr()
    x_masked.eliminate_zeros()
  else:
    x_masked = x * interp_mask.astype(x.dtype)

  # Compute feature-level means over the nonzero entries only (used in
  # importance scoring later).
  col_sums = np.asarray(x_masked.sum(axis=0))
  nonzero_counts = np.asarray((x_masked != 0).sum(axis=0))
  with np.errstate(invalid="ignore", divide="ignore"):
    means = np.where(nonzero_counts > 0, col_sums / nonzero_counts, 0.0)

  # Split on row indices: the same partition and order as splitting the
  # matrix itself.
  train_idx, test_idx, y_train, y_test = train_test_split(
      np.arange(x_masked.shape[0]),
      y,
      test_size=0.1,
      random_state=42,
      stratify=y,
  )
  x_train, x_test = x_masked[train_idx], x_masked[test_idx]
  if feature_names is not None:
    x_train = pd.DataFrame(x_train, columns=feature_names)
    x_test = pd.DataFrame(x_test, columns=feature_names)

  # Train the model
  device = "cuda" if torch.cuda.is_available() else "cpu"
  device = "cpu" if cpu_only else device
  n_estimators = 150
  clf = xgb.XGBClassifier(
      n_estimators=n_estimators,
      max_depth=6,
      learning_rate=0.1,
      tree_method="hist",
      device=device,
      n_jobs=-1,
      random_state=42,
      callbacks=[TqdmCallback(n_estimators=n_estimators)],
  ).fit(x_train, y_train)

  # Evaluate
  preds = clf.predict(x_test)
  print(
      f"Accuracy: {accuracy_score(y_test, preds):.4f}\n"
      f"{classification_report(y_test, preds)}"
  )

  return clf, means


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Train an XGBoost head (the appendix head comparison)."
  )
  parser.add_argument(
      "-d",
      "--data_path",
      type=str,
      default=None,
      help=(
          "Split directory: the Hugging Face dataset with the label column, "
          "its layer{N}_max_activations.pth caches and the description files"
      ),
  )
  parser.add_argument(
      "--use_mmap_cache",
      type=str,
      default=None,
      help=(
          "Directory of layer{N}.*.npy caches (process.py). Batches are "
          "gathered from the memory-mapped arrays instead of the resident "
          ".pth layers in --data_path; same training data"
      ),
  )
  parser.add_argument(
      "-n",
      "--num_examples",
      type=int,
      default=None,
      help="Keep only the first N rows (after the target filter)",
  )
  parser.add_argument(
      "-l",
      "--label_col",
      type=str,
      default="model",
      help=(
          'Dataset column holding the generator name; "human" is the '
          "negative class and every other value is AI"
      ),
  )
  parser.add_argument(
      "-t",
      "--interp_threshold",
      type=float,
      default=None,
      help=(
          "tau: train only on features whose description simulation score is "
          ">= tau (SAE caches only). Unset or 0.0 keeps every feature"
      ),
  )
  parser.add_argument(
      "-m",
      "--desc_gen_model",
      type=str,
      default="gemini-3.1-flash-lite-preview",
      help="Tag of the layer{N}_{tag}_feature_results.jsonl description files",
  )
  parser.add_argument(
      "-tm",
      "--target_model",
      type=str,
      default=None,
      help=(
          'Keep only "human" rows and this generator\'s rows (the PRDB '
          "per-generator heads); default trains on every row"
      ),
  )
  parser.add_argument(
      "--exit_if_trained_model_exists",
      action="store_true",
      help="If true, check if a model exists at output_path and if so, exit",
  )
  parser.add_argument(
      "--cpu_only",
      action="store_true",
      help="If true, run on cpu regardless of device",
  )
  parser.add_argument(
      "-o",
      "--output_path",
      type=str,
      default="xgbmodel.pkl",
      help=("Output pickle: model + interp_mask + means (XGBoostHead)"),
  )
  args = parser.parse_args()

  if args.exit_if_trained_model_exists and os.path.exists(args.output_path):
    print(f"Found model at {args.output_path}, exiting")
    raise SystemExit(0)

  source = loading.load_source(args)
  y = (source.labels != "human").astype(int)
  print(
      f"{len(y):,} training rows ({len(y) - int(y.sum()):,} human, "
      f"{int(y.sum()):,} AI) x {source.dim} features"
  )

  # The booster needs the whole matrix in memory. SAE layers go in as one
  # CSR, where an absent entry is "missing"; a named block (gram2vec) is a
  # dense feature table, so its rows are gathered dense to keep zeros zeros.
  feature_names = source.extra_artifact.get("feature_names")
  if source.features.layer_ids:
    x = source.features.to_scipy()[source.row_map]
  else:
    x = source.features.fetch(source.row_map)

  clf, means = train_xgboost(
      y, source.interp_mask, x, args.cpu_only, feature_names
  )

  # Remove the un-pickle-able tqdm callback before saving.
  clf.callbacks = None

  os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
  print(f"Saving model to {args.output_path}")
  with open(args.output_path, "wb") as f:
    pickle.dump(
        {
            "model": clf,
            "interp_mask": source.interp_mask,
            "means": means,
            **source.extra_artifact,
        },
        f,
    )
