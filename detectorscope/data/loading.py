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

"""The training data behind every head: feature source, labels and rows.

`train_logreg.py` and `train_xgboost.py` define the same feature-source flags
and hand them to `load_source`. A head is trained on the caches of a split:
`-d` names the dataset with the labels, and the features come from the
resident `.pth` layers beside it or, with `--use_mmap_cache`, from a
memory-mapped cache directory, either the SAE layers of the split or the
`gram2vec` block extract.py writes. The interp mask comes from the split's
description files (`-t`); every feature of a cache without them is kept. A
cache that names its columns (feature_names.json) has them recorded in the
head. The trainer gets a feature source with `fetch(rows)`, the string label
of every kept row, the cache row each kept row maps to, and the extra keys
the saved head records.

`training_rows` reproduces, for the analysis scripts under scripts/, the rows
a head was trained on: the target filter, then the 90% side of
`train_test_split(np.arange(n), y, test_size=0.1, random_state=42,
stratify=y)`. Any change to the trainers' filtering or split must be mirrored
there.
"""

import argparse
import dataclasses
import json
import os

import datasets
import numpy as np
from sklearn.model_selection import train_test_split

from detectorscope.data.cache import get_interp_mask
from detectorscope.data.cache import LayerStack

# Features whose std falls below this are treated as constant (see
# standardisation).
SCALE_EPS = 1e-12


@dataclasses.dataclass
class TrainingSource:
  """What a trainer needs from the feature-source flags.

  Attributes:
    features: the cache's LayerStack; `fetch(rows)` gives dense float32 rows.
    labels: str label of every kept row, in training order.
    row_map: cache row of every kept row, so `features.fetch(row_map[i])` is
      the feature vector of labels[i].
    interp_mask: [features.dim] mask, float64 all ones when every feature is
      kept and int64 {0, 1} from the description files otherwise.
    extra_artifact: keys merged into every saved head so the artifact is
      self-describing (feature names and provenance for gram2vec).
  """

  features: LayerStack
  labels: np.ndarray
  row_map: np.ndarray
  interp_mask: np.ndarray
  extra_artifact: dict

  @property
  def dim(self) -> int:
    return self.features.dim


def kept_rows(labels: np.ndarray, target: str | None) -> np.ndarray:
  """Cache rows a head trains on: "human" plus target, or every row."""
  labels = np.asarray(labels)
  if target:
    return np.flatnonzero((labels == "human") | (labels == target))
  return np.arange(len(labels))


def load_source(args: argparse.Namespace, mmap: bool = True) -> TrainingSource:
  """Opens the feature source the trainer flags describe and filters rows.

  The target filter and then `[:num_examples]` are applied to the row map,
  never to the cache itself, so a filtered run keeps contiguous cache rows.

  Args:
    args: the trainer's parsed flags (-d, --use_mmap_cache, -l, -t, -m, -tm, -n).
    mmap: memory-map the cache arrays (False loads them into memory).

  Raises:
    ValueError: on contradictory flags, a cache that does not match the
      dataset row for row, or a target that leaves no rows.
  """
  if not args.data_path:
    raise ValueError("-d (the split's dataset) is required")
  # Disable caching so Hugging Face does not write the intermediate steps
  # of computation to disk; on large datasets that doubles the disk use.
  datasets.disable_caching()
  ds = datasets.load_from_disk(args.data_path)
  labels = np.asarray(ds[args.label_col]).astype(str)
  cache_dir = args.use_mmap_cache or args.data_path
  features = LayerStack.open(cache_dir, mmap)
  if features.n_rows != len(labels):
    raise ValueError(
        f"the caches have {features.n_rows} rows but the dataset has "
        f"{len(labels)}; were they built from {args.data_path}?"
    )
  # The falsy rule: tau 0.0 and "no tau" both keep every feature with a
  # float64 all-ones mask; any tau > 0 reads the description files and
  # yields an int64 {0, 1} mask. Saved heads carry whichever they got.
  interp_mask = np.ones(features.dim)
  if args.interp_threshold:
    interp_mask = get_interp_mask(
        args.data_path,
        args.desc_gen_model,
        args.interp_threshold,
        features.d_sae,
    ).numpy()
  # A cache that names its columns (gram2vec) has them recorded in the head,
  # with its path and the target, so the artifact is self-describing.
  extra = {}
  names_path = os.path.join(cache_dir, "feature_names.json")
  if os.path.exists(names_path):
    with open(names_path) as f:
      names = json.load(f)
    if len(names) != features.dim:
      raise ValueError(
          f"{names_path}: {len(names)} names for {features.dim} features"
      )
    extra = {
        "feature_names": names,
        "feature_cache": os.path.abspath(cache_dir),
        "target_model": args.target_model,
    }

  keep = kept_rows(labels, args.target_model)
  if args.num_examples:
    keep = keep[: args.num_examples]
  if keep.size == 0:
    raise ValueError(f"no rows left for target {args.target_model!r}")
  return TrainingSource(features, labels[keep], keep, interp_mask, extra)


def standardisation(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Per-feature mean and std over the given rows.

  Constant features (std below SCALE_EPS) get sigma = 1 and mu = 0 so they
  pass through as zeros instead of dividing by ~0, mirroring the "never
  fires -> scale 1" rule of LayerStack.feature_rms.

  Returns:
    (mu, sigma) as float32 arrays of shape [n_features].
  """
  mu = x.mean(axis=0, dtype=np.float64)
  sigma = x.std(axis=0, dtype=np.float64)
  constant = sigma < SCALE_EPS
  sigma[constant] = 1.0
  mu[constant] = 0.0
  return mu.astype(np.float32), sigma.astype(np.float32)


def training_rows(
    labels: np.ndarray, target: str | None, all_rows: bool
) -> tuple[np.ndarray, np.ndarray]:
  """Cache rows a checkpoint trained on, and their binary labels.

  Mirrors the trainers: `keep` is the target filter applied at load time
  (its row_map), and the split is taken on np.arange(len(keep)) with the same
  seed and stratification, so the returned rows are the ones the training
  loop actually gathered.

  Args:
    labels: the dataset's label column, one entry per cache row.
    target: generator to keep alongside "human"; None keeps every row.
    all_rows: skip the 90/10 split and return the whole population.

  Returns:
    (rows, is_ai): cache row indices and a matching float32 0/1 label array,
    1 for every non-human label.
  """
  labels = np.asarray(labels)
  keep = kept_rows(labels, target)
  y = (labels[keep] != "human").astype(np.float32)
  if all_rows:
    train_idx = np.arange(len(keep))
  else:
    train_idx, _, _, _ = train_test_split(
        np.arange(len(keep)), y, test_size=0.1, random_state=42, stratify=y
    )
  return keep[train_idx], y[train_idx]
