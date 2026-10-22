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

"""Builds the max-activation caches that the detector heads train on.

Six sub-commands, the first four in pipeline order (README steps 3 and 4):

  from-dataset   per-token SAE activations (extract.py sae output) ->
                 one [n_docs, sae_dim] sparse CSR tensor of per-document max
                 activations, saved as <data_path>/<layer>_max_activations.pth
                 and loaded by model.cache.MaxActivationCache.
  from-pth       <data_path>/layer*_max_activations.pth -> memory-mappable
                 layer<N>.{crow,col,val,shape}.npy arrays, one layer per
                 invocation if wanted, loaded by LayerStack.open.
  feature-scale  per-feature RMS over a converted (mmap) cache, written as
                 feature_scale.npy for the SGD + L1 trainer
                 (model/detectors/heads/training/train_logreg.py
                 --feature_scale).
  lor            per-feature log odds ratio of firing in human vs. target
                 documents, from a .pth cache and the dataset's model column,
                 written as <data_path>/<layer>_[<target>_]log_odds_ratios.csv
                 (autointerpret.py's feature selection and the demo's Log
                 Odds column, via the demo).
  feature-stats  per-feature activation statistics over the rows a logreg
                 head trains on (documents per class, sum, sum of squares and
                 a log-spaced histogram, hence mean, std and quantiles), one
                 npz per training population; for a linear head every
                 per-feature statistic of the contribution w_j * x_ij is
                 one of these times w_j (the paper's feature tables).
  active-counts  per-document counts, over an eval split's mmap cache, of
                 the features that pass each interp threshold and also carry
                 a non-zero / positive weight in the paper-final SGD+L1 head
                 trained at that threshold; npz plus a json summary
                 (tab:active_feature_counts).

Row i of every cache is row i of the Hugging Face dataset at --data_path;
lor and evaluate.py rely on that alignment.

Usage: python -m detectorscope.data.process <sub-command> -d <dir> [options]

What lor computes: each feature's document density in the human and target
classes, both rescaled to the smaller class so the classes carry equal weight,
then log((h + 0.5) / (n - h + 0.5)) - log((t + 0.5) / (n - t + 0.5)) with
add-0.5 smoothing. Positive values are human-leaning. Features firing in fewer
than --frequency_threshold rescaled documents, both classes combined, get NaN
(an empty csv field). The target is one generator (--target_model) or, by
default, every non-human row.

Why feature-scale divides rather than centres: SAE activations are
unnormalised and vary in magnitude by orders of magnitude across features,
which makes plain SGD diverge. Dividing each feature by its RMS over the
corpus, sqrt(mean(x_j^2)), keeps 0 -> 0 so the caches stay sparse; centring
would turn every stored zero into a non-zero. Because the head is linear,
training on x/s and then dividing the learned weights by s yields a model that
operates on raw features, so nothing downstream of training needs to know.
The L1 penalty then acts in the standardised space, which is the usual lasso
convention. A feature that never fires has scale 0 and is left at 1.0.
"""

import argparse
import json
import math
import os
import pickle
import time
from typing import Any, Iterator

import datasets
import numpy as np
import pandas as pd
import torch
import tqdm

from detectorscope.data.cache import CSR_KEYS
from detectorscope.data.cache import get_cache_files
from detectorscope.data.cache import get_desc_files
from detectorscope.data.cache import LayerStack
from detectorscope.data.cache import MaxActivationCache
from detectorscope.data.cache import NNZ_CHUNK
from detectorscope.data.loading import training_rows

QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
# active-counts: (head file-name tag, strict > threshold; None keeps every
# described feature, the unmasked head's set) and the weight conditions.
THRESHOLDS = (
    ("0.0", None),
    ("0.70", 0.7),
    ("0.80", 0.8),
    ("0.85", 0.85),
    ("0.90", 0.9),
)
VARIANTS = ("mask", "wnz", "wpos")
DESC_MODEL = "gemini-3.1-flash-lite-preview"


def get_max_activation(t: torch.Tensor) -> torch.Tensor:
  """Computes the maximum activation of each feature over one document's tokens.

  Args:
    t: Batched sparse CSR tensor of shape [1, n_tokens, sae_dim], one row per
      token.

  Returns:
    Sparse COO tensor of shape [1, sae_dim] with each feature's maximum over
    all tokens; features that never fire are absent.
  """
  unique_ids, inverse_indices = torch.unique(
      t.col_indices(), return_inverse=True
  )

  max_values = torch.zeros(unique_ids.shape)[None, :]

  max_values.scatter_reduce_(
      1, inverse_indices, t.values(), reduce="amax", include_self=False
  )

  return torch.sparse_coo_tensor(
      indices=torch.stack((torch.zeros(unique_ids.shape), unique_ids)),
      values=max_values[0, :],
      size=(1, t.size()[2]),
  )


def iterate_slice(ds_slice: datasets.Dataset) -> Iterator[dict[str, Any]]:
  """Yields one document's CSR components (CSR_KEYS) per slice row."""
  for i in range(len(ds_slice[CSR_KEYS[0]])):
    yield {col: ds_slice[col][i] for col in CSR_KEYS}


def ds_to_sparse_tensor(
    ds: datasets.Dataset,
    batch_size: int = 100,
    separate_columns: bool = True,
    values_col: str = "hook_sae_acts_post",
) -> torch.Tensor:
  """Converts a per-token activation dataset to a [n_docs, sae_dim] CSR tensor.

  Args:
    ds: Dataset with one row per document.
    batch_size: Documents whose max-activation rows are concatenated at once.
    separate_columns: If True the CSR components are separate columns named by
      CSR_KEYS; otherwise each row of values_col holds them as a dict.
    values_col: Column read when separate_columns is False.

  Returns:
    Sparse CSR tensor of per-document max activations, row i = dataset row i.
  """
  iterator = iterate_slice if separate_columns else (lambda x: x[values_col])

  ds = ds.with_transform(
      lambda x: {
          "sae_activation": [
              get_max_activation(torch.sparse_csr_tensor(**row))
              for row in iterator(x)
          ]
      }
  )

  batches = [
      torch.cat(ds[i : i + batch_size]["sae_activation"], dim=0)
      for i in tqdm.tqdm(range(0, len(ds), batch_size))
  ]

  return torch.cat(batches, dim=0).to_sparse_csr()


def run_from_dataset(args: argparse.Namespace) -> None:
  """Writes <data_path>/<output_file> from a per-token activation dataset.

  With --csr_separate_columns the CSR components are read from the bare
  CSR_KEYS columns or, with -l <layer_name>, from
  <layer_name>_sae_acts_post_{crow_indices,col_indices,values,size}.
  """
  dataset = datasets.load_from_disk(args.data_path)

  if args.csr_separate_columns:
    if args.layer_name:
      dataset = dataset.rename_columns(
          {f"{args.layer_name}_sae_acts_post_{key}": key for key in CSR_KEYS}
      )

    dataset = dataset.select_columns(CSR_KEYS)

  sparse_tensor = ds_to_sparse_tensor(
      dataset, args.batch_size, args.csr_separate_columns
  )

  torch.save(sparse_tensor, os.path.join(args.data_path, args.output_file))


def convert_layer(pth_path: str, output_path: str, layer: int) -> None:
  """Writes one cache as layer<N>.{crow,col,val,shape}.npy in output_path."""
  MaxActivationCache.from_pth(pth_path).save_npy(output_path, f"layer{layer}")


def run_from_pth(args: argparse.Namespace) -> None:
  """Converts every (or one) layer*_max_activations.pth under --data_path."""
  os.makedirs(args.output_path, exist_ok=True)

  cache_files = get_cache_files(args.data_path)
  if args.layer is not None:
    cache_files = [(l, p) for l, p in cache_files if l == args.layer]
    if not cache_files:
      raise ValueError(f"No cache found for layer {args.layer}")

  for layer, path in cache_files:
    prefix = os.path.join(args.output_path, f"layer{layer}")
    if args.exit_if_converted and os.path.exists(f"{prefix}.val.npy"):
      print(f"Layer {layer} already converted, skipping")
      continue
    print(f"Converting layer {layer}: {path}")
    convert_layer(path, args.output_path, layer)
    print(f"Wrote {prefix}.{{crow,col,val,shape}}.npy")


def run_feature_scale(args: argparse.Namespace) -> None:
  """Writes feature_scale.npy for the mmap cache directory --mmap_cache.

  Defaults to writing inside the cache directory, next to the arrays it
  describes; pass the same path to train_logreg.py --feature_scale.
  """
  out_path = args.output_path or os.path.join(
      args.mmap_cache, "feature_scale.npy"
  )
  if args.exit_if_output_path_exists and os.path.exists(out_path):
    print(f"Found scale at {out_path}, exiting")
    return

  stack = LayerStack.open(args.mmap_cache)
  print(
      f"{args.mmap_cache}: {stack.n_rows:,} rows x "
      f"{stack.d_sae:,} features x {len(stack)} layers"
  )
  scale = stack.feature_rms()
  np.save(out_path, scale)
  print(
      f"wrote {out_path}  shape={scale.shape}  "
      f"min={scale.min():.4g} max={scale.max():.4g}"
  )


def class_density(cache: MaxActivationCache, rows: np.ndarray) -> list[int]:
  """Number of documents among `rows` (a boolean mask) each feature fires in."""
  return cache.to_scipy()[rows].getnnz(axis=0).tolist()


def compute_log_odds_ratio(
    human_density: list[int],
    target_density: list[int],
    n_h: int,
    n_t: int,
    pooled_freq_threshold: int,
) -> list[float]:
  """Computes the normalized log odds ratio per feature with add-0.5 smoothing.

  Both feature densities are rescaled to n_min = min(n_h, n_t) so the classes
  carry equal weight whatever their sizes. Positive values are human-leaning.

  Args:
    human_density: Documents each feature fires in among the n_h human ones.
    target_density: Documents each feature fires in among the n_t target ones.
    n_h: Number of human documents.
    n_t: Number of target documents.
    pooled_freq_threshold: Features firing in fewer rescaled documents than
      this, human and target combined, get NaN.

  Returns:
    One LOR per feature, in feature order.
  """
  n_min = min(n_h, n_t)
  d_h = [d * n_min / n_h for d in human_density]
  d_t = [d * n_min / n_t for d in target_density]
  return [
      math.log((h + 0.5) / (n_min - h + 0.5))
      - math.log((t + 0.5) / (n_min - t + 0.5))
      if h + t >= pooled_freq_threshold
      else float("nan")
      for h, t in zip(d_h, d_t)
  ]


def run_lor(args: argparse.Namespace) -> None:
  """Writes <data_path>/<layer>_<output_file> for one layer's .pth cache.

  The csv has the columns layer, feature_id, lor, human_density and
  target_density (document counts rescaled to the smaller class) and
  density_fraction (fraction of all documents the feature fires in), sorted
  by lor descending with NaN sorted as 0.
  """
  prefix = f"{args.layer_name}_" if args.layer_name else ""
  cache_path = os.path.join(args.data_path, f"{prefix}max_activations.pth")
  if not os.path.exists(cache_path):
    raise FileNotFoundError(f"{cache_path}: run process from-dataset first")

  models = np.array(datasets.load_from_disk(args.data_path)[args.model_col])
  cache = MaxActivationCache.from_pth(cache_path)
  is_human = models == "human"
  is_target = models == args.target_model if args.target_model else ~is_human
  n_h, n_t = int(is_human.sum()), int(is_target.sum())
  n_min = min(n_h, n_t)

  human_density = class_density(cache, is_human)
  target_density = class_density(cache, is_target)
  lor = compute_log_odds_ratio(
      human_density, target_density, n_h, n_t, args.frequency_threshold
  )

  rows = sorted(
      zip(
          range(cache.shape[1]),
          lor,
          human_density,
          target_density,
          cache.get_feature_density(),
      ),
      key=lambda r: 0 if math.isnan(r[1]) else r[1],
      reverse=True,
  )
  lor_df = pd.DataFrame(
      [
          {
              "layer": args.layer_name,
              "feature_id": f_id,
              "lor": f_lor,
              "human_density": round(h * n_min / n_h),
              "target_density": round(t * n_min / n_t),
              "density_fraction": float(d) / float(len(models)),
          }
          for f_id, f_lor, h, t, d in rows
      ]
  )
  out_path = os.path.join(args.data_path, f"{prefix}{args.output_file}")
  lor_df.to_csv(out_path, index=False)
  print(
      f"wrote {out_path}  {n_h:,} human vs. {n_t:,} target documents, "
      f"{lor_df['lor'].notna().sum():,} / {len(lor_df):,} features above "
      f"the frequency threshold"
  )


def chunk_rows(crow: np.ndarray, start: int, end: int) -> np.ndarray:
  """Row id of every nnz entry in [start, end), from the CSR row pointer."""
  first = int(np.searchsorted(crow, start, side="right")) - 1
  last = int(np.searchsorted(crow, end, side="left"))
  starts = np.maximum(crow[first:last], start)
  ends = np.minimum(crow[first + 1 : last + 1], end)
  return np.repeat(np.arange(first, last, dtype=np.int64), ends - starts)


def layer_stats(
    cache: MaxActivationCache,
    keep_row: np.ndarray,
    ai_row: np.ndarray,
    n_bins: int,
    log_min: float,
    log_max: float,
) -> dict[str, np.ndarray | int]:
  """Streams one layer and returns its per-feature accumulators.

  Only entries whose row is in `keep_row` count; `ai_row` splits them by
  class. `hist` comes back as [sae_dim, n_bins]; values outside
  [log_min, log_max) land in the edge bins and are tallied in `n_below` /
  `n_above`.
  """
  sae_dim = cache.shape[1]
  crow = np.asarray(cache.crow, dtype=np.int64)
  scale = n_bins / np.log10(log_max / log_min)
  count = np.zeros(sae_dim, dtype=np.int64)
  count_ai = np.zeros(sae_dim, dtype=np.int64)
  act_sum = np.zeros(sae_dim, dtype=np.float64)
  act_sumsq = np.zeros(sae_dim, dtype=np.float64)
  hist = np.zeros(sae_dim * n_bins, dtype=np.int64)
  n_below = n_above = 0
  started = time.time()
  for start in range(0, cache.nnz, NNZ_CHUNK):
    end = min(start + NNZ_CHUNK, cache.nnz)
    rows = chunk_rows(crow, start, end)
    kept = keep_row[rows]
    cols = np.asarray(cache.col[start:end])[kept]
    vals = np.asarray(cache.val[start:end])[kept].astype(np.float64)
    is_ai = ai_row[rows[kept]]
    del rows, kept
    count += np.bincount(cols, minlength=sae_dim)
    count_ai += np.bincount(cols[is_ai], minlength=sae_dim)
    act_sum += np.bincount(cols, weights=vals, minlength=sae_dim)
    act_sumsq += np.bincount(cols, weights=vals * vals, minlength=sae_dim)
    n_below += int((vals < log_min).sum())
    n_above += int((vals >= log_max).sum())
    bins = np.floor(np.log10(np.maximum(vals, log_min) / log_min) * scale)
    bins = np.clip(bins.astype(np.int64), 0, n_bins - 1)
    hist += np.bincount(cols * n_bins + bins, minlength=sae_dim * n_bins)
    print(
        f"    {end:,}/{cache.nnz:,} ({end / cache.nnz:.0%})  "
        f"{time.time() - started:.0f}s",
        flush=True,
    )
  return {
      "n_active": count,
      "n_active_ai": count_ai,
      "act_sum": act_sum,
      "act_sumsq": act_sumsq,
      "hist": hist.reshape(sae_dim, n_bins),
      "n_below": n_below,
      "n_above": n_above,
  }


def histogram_quantiles(
    hist: np.ndarray,
    count: np.ndarray,
    n_bins: int,
    log_min: float,
    log_max: float,
) -> dict[str, np.ndarray]:
  """Quantiles, min and max per feature from the log-spaced histogram.

  A quantile is the geometric centre of the first bin whose cumulative count
  reaches q * n_active; min/max are the outer edges of the first/last
  non-empty bin. Features that never fire get NaN.

  Returns:
    `act_qNN` (one float32 array per QUANTILES entry), `act_min`, `act_max`.
  """
  scale = n_bins / np.log10(log_max / log_min)
  edges = log_min * 10 ** (np.arange(n_bins + 1) / scale)
  centres = np.sqrt(edges[:-1] * edges[1:])
  cdf = np.cumsum(hist, axis=1)
  fired = count > 0
  quantiles = {}
  for q in QUANTILES:
    target = np.ceil(q * count)[:, None]
    bin_index = np.minimum((cdf < target).sum(axis=1), n_bins - 1)
    values = centres[bin_index].astype(np.float32)
    values[~fired] = np.nan
    quantiles[f"act_q{int(round(q * 100)):02d}"] = values
  nonempty = hist > 0
  first = np.argmax(nonempty, axis=1)
  last = n_bins - 1 - np.argmax(nonempty[:, ::-1], axis=1)
  act_min = edges[first].astype(np.float32)
  act_max = edges[last + 1].astype(np.float32)
  act_min[~fired] = np.nan
  act_max[~fired] = np.nan
  return quantiles | {"act_min": act_min, "act_max": act_max}


def run_feature_stats(args: argparse.Namespace) -> None:
  """Writes the feature_stats npz of one training population.

  Training rows replicate train_logreg.py (loading.training_rows): the cache
  filtered to label in {human, --target_model} (every row without a target),
  then the 90% side of the stratified split, or the whole filtered population
  with --all_rows. The cache is streamed in NNZ_CHUNK entries, never densified.
  """
  if args.exit_if_output_path_exists and os.path.exists(args.output_path):
    print(f"Found stats at {args.output_path}, exiting")
    return
  caches = LayerStack.open(args.mmap_cache)
  n_docs, sae_dim = caches.n_rows, caches.d_sae
  print(
      f"{args.mmap_cache}: {n_docs:,} rows x {sae_dim:,} features x "
      f"{len(caches)} layers {caches.layer_ids}"
  )
  labels = np.asarray(datasets.load_from_disk(args.data_path)[args.label_col])
  if len(labels) != n_docs:
    raise ValueError(f"dataset has {len(labels)} rows but cache has {n_docs}")
  rows, is_ai = training_rows(labels, args.target_model, args.all_rows)
  keep_row = np.zeros(n_docs, dtype=bool)
  keep_row[rows] = True
  ai_row = np.zeros(n_docs, dtype=bool)
  ai_row[rows[is_ai == 1]] = True
  n_ai = int(is_ai.sum())
  print(
      f"training rows: {len(rows):,} of {n_docs:,} "
      f"(ai={n_ai:,} human={len(rows) - n_ai:,}) "
      f"target={args.target_model} all_rows={args.all_rows}"
  )

  per_layer = []
  for name, cache in zip(caches.names, caches):
    print(f"{name}: nnz={cache.nnz:,}")
    per_layer.append(
        layer_stats(
            cache, keep_row, ai_row, args.n_bins, args.log_min, args.log_max
        )
    )
  n_active = np.concatenate([s["n_active"] for s in per_layer])
  n_active_ai = np.concatenate([s["n_active_ai"] for s in per_layer])
  act_sum = np.concatenate([s["act_sum"] for s in per_layer])
  act_sumsq = np.concatenate([s["act_sumsq"] for s in per_layer])
  hist = np.concatenate([s["hist"] for s in per_layer])
  n_below = sum(s["n_below"] for s in per_layer)
  n_above = sum(s["n_above"] for s in per_layer)
  with np.errstate(divide="ignore", invalid="ignore"):
    act_mean = act_sum / n_active
    act_std = np.sqrt(np.maximum(act_sumsq / n_active - act_mean**2, 0.0))

  os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
  np.savez_compressed(
      args.output_path,
      n_active=n_active,
      n_active_ai=n_active_ai,
      n_active_human=n_active - n_active_ai,
      act_sum=act_sum,
      act_sumsq=act_sumsq,
      act_mean=act_mean.astype(np.float32),
      act_std=act_std.astype(np.float32),
      **histogram_quantiles(
          hist, n_active, args.n_bins, args.log_min, args.log_max
      ),
      hist=hist.astype(np.int32),
      train_rows=rows,
      train_is_ai=is_ai.astype(np.int8),
      n_train_docs=len(rows),
      n_ai_docs=n_ai,
      n_human_docs=len(rows) - n_ai,
      n_cache_docs=n_docs,
      sae_dim=sae_dim,
      n_layers=len(caches),
      layers=np.asarray(caches.layer_ids),
      n_bins=args.n_bins,
      log_min=args.log_min,
      log_max=args.log_max,
      n_below_log_min=n_below,
      n_above_log_max=n_above,
      mmap_cache=args.mmap_cache,
      data_path=args.data_path,
      target_model=args.target_model or "",
      all_rows=args.all_rows,
  )
  print(
      f"wrote {args.output_path}: {int((n_active > 0).sum()):,}/"
      f"{n_active.size:,} features fire on the training rows; active-per-doc "
      f"mean={n_active.sum() / len(rows):.1f}; histogram clamps: {n_below:,} "
      f"below {args.log_min:g}, {n_above:,} above {args.log_max:g}"
  )


def load_desc(data_path: str, desc_model: str, sae_dim: int) -> pd.DataFrame:
  """Described features as (fid, corr) rows, one per jsonl line.

  fid is the layer-major global id (rank * sae_dim + feature_index) and corr
  the auto-interp score the interp mask thresholds on.
  """
  rows = []
  for rank, (_, path) in enumerate(get_desc_files(data_path, desc_model)):
    with open(path) as f:
      for line in f:
        record = json.loads(line)
        rows.append(
            (
                rank * sae_dim + int(record["feature_index"]),
                record["correlation"][0],
            )
        )
  return pd.DataFrame(rows, columns=["fid", "corr"])


def select_desc(desc: pd.DataFrame, threshold: float | None) -> pd.DataFrame:
  """Rows with score > threshold (strict); None keeps every described one."""
  return desc if threshold is None else desc[desc["corr"] > threshold]


def sweep_head(sweep_root: str, train_dir: str, target: str, tau: str) -> str:
  """Path of the paper-final SGD+L1 head (lr 1e-2, lambda 1e-1, 3 epochs)."""
  name = f"sgd_lr1e-02_l1-1e-01_threshold{tau}_{target}_logreg_ep3.pkl"
  return f"{sweep_root}/models/{train_dir}/{name}"


def load_weight(head_path: str) -> np.ndarray:
  """Flat weight vector of a pickled logreg head."""
  with open(head_path, "rb") as f:
    artifact = pickle.load(f)
  return np.asarray(artifact["state_dict"]["weight"]).ravel()


def build_indicators(
    desc: pd.DataFrame,
    sweep_root: str,
    train_dir: str,
    target: str,
    n_features: int,
) -> tuple[np.ndarray, list[str]]:
  """[n_features, n_conditions] uint8 indicators and their condition names.

  Conditions are THRESHOLDS x VARIANTS, named "<tau:.2f>_<variant>": the
  features passing the threshold, those also with w != 0 and those with
  w > 0 in the head trained at that threshold.
  """
  columns, names = [], []
  for tau, threshold in THRESHOLDS:
    passing = np.zeros(n_features, bool)
    passing[np.unique(select_desc(desc, threshold).fid.to_numpy())] = True
    weight = load_weight(sweep_head(sweep_root, train_dir, target, tau))
    conditions = (passing, passing & (weight != 0), passing & (weight > 0))
    for variant, condition in zip(VARIANTS, conditions):
      columns.append(condition.astype(np.uint8))
      names.append(f"{float(tau):.2f}_{variant}")
      print(
          f"  tau={tau} {variant}: {int(condition.sum()):,} features",
          flush=True,
      )
  return np.stack(columns, axis=1), names


def select_rows(
    data_path: str, target: str, split: str
) -> tuple[np.ndarray, np.ndarray]:
  """Row indices into the eval cache and their is_ai flags for one split.

  "train" replicates the head's exact 90% training rows; "test" keeps every
  row with label human or the target (every row for coling). Rows with a
  None model (COLING_realtest's unlabelled AI) count as AI.
  """
  raw = datasets.load_from_disk(data_path)["model"]
  labels = np.asarray([x if x is not None else "unknown_ai" for x in raw])
  rows, is_ai = training_rows(
      labels, None if target == "coling" else target, all_rows=split != "train"
  )
  return np.asarray(rows), np.asarray(is_ai)


def count_layer(
    cache: MaxActivationCache,
    indicators: np.ndarray,
    rows: np.ndarray,
    chunk_nnz: int = 50_000_000,
) -> np.ndarray:
  """[len(rows), n_conditions] counts of indicated features present per row.

  The conditions are packed into one uint16 bitmask per feature so each nnz
  entry costs a single 2-byte gather; per-condition counts come from a cumsum
  over the extracted bit, differenced at the CSR row boundaries.
  """
  n_cond = indicators.shape[1]
  bits = np.zeros(indicators.shape[0], dtype=np.uint16)
  for k in range(n_cond):
    bits |= indicators[:, k].astype(np.uint16) << k
  crow, col = cache.crow, cache.col
  n_rows_total = len(crow) - 1
  counts = np.zeros((n_rows_total, n_cond), dtype=np.int32)
  start_row = 0
  while start_row < n_rows_total:
    end_row = (
        int(np.searchsorted(crow, crow[start_row] + chunk_nnz, side="right"))
        - 1
    )
    end_row = min(max(end_row, start_row + 1), n_rows_total)
    first, last = crow[start_row], crow[end_row]
    hit = bits[np.asarray(col[first:last])]
    bounds = crow[start_row : end_row + 1] - first
    cumulative = np.empty(last - first + 1, dtype=np.int64)
    cumulative[0] = 0
    for k in range(n_cond):
      np.cumsum((hit >> k) & 1, out=cumulative[1:])
      counts[start_row:end_row, k] = (
          cumulative[bounds[1:]] - cumulative[bounds[:-1]]
      )
    start_row = end_row
  return counts[rows]


def run_active_counts(args: argparse.Namespace) -> None:
  """Counts active features per document for one (train dir, eval dir, split).

  Writes --output_path (npz: counts [n_docs, n_conditions], rows, is_ai,
  condition_names, provenance) and a sibling .json with each condition's mean
  / median / std and per-class means.
  """
  stack = LayerStack.open(f"{args.mmap_root}/{args.eval_dir}")
  print(f"indicators from {args.train_dir} / {args.target}", flush=True)
  desc = load_desc(
      f"{args.cache_root}/{args.train_dir}", args.desc_gen_model, stack.d_sae
  )
  indicators, names = build_indicators(
      desc, args.sweep_root, args.train_dir, args.target, stack.dim
  )
  rows, is_ai = select_rows(
      f"{args.cache_root}/{args.eval_dir}", args.target, args.split
  )
  print(f"  {len(rows):,} documents ({int(is_ai.sum()):,} AI)", flush=True)
  total = np.zeros((len(rows), indicators.shape[1]), dtype=np.int64)
  for rank, (name, layer) in enumerate(zip(stack.names, stack)):
    started = time.time()
    total += count_layer(
        layer, indicators[rank * stack.d_sae : (rank + 1) * stack.d_sae], rows
    )
    print(f"  {name}: {time.time() - started:.0f}s", flush=True)

  summary = {}
  for j, name in enumerate(names):
    counts = total[:, j]
    summary[name] = dict(
        mean=float(counts.mean()),
        median=float(np.median(counts)),
        std=float(counts.std()),
        mean_ai=float(counts[is_ai == 1].mean()),
        mean_human=float(counts[is_ai == 0].mean()),
        n_features=int(indicators[:, j].sum()),
    )
    print(
        f"  {name:>10}: mean {counts.mean():10.1f}  median "
        f"{np.median(counts):10.1f}  ai {summary[name]['mean_ai']:10.1f}  "
        f"human {summary[name]['mean_human']:10.1f}",
        flush=True,
    )
  os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
  provenance = dict(
      train_dir=args.train_dir,
      eval_dir=args.eval_dir,
      target=args.target,
      split=args.split,
  )
  np.savez(
      args.output_path,
      counts=total,
      rows=rows,
      is_ai=is_ai,
      condition_names=np.array(names),
      **provenance,
  )
  with open(
      args.output_path.replace(".npz", ".json"), "w", encoding="utf-8"
  ) as f:
    json.dump(
        provenance
        | dict(n_docs=int(len(rows)), n_ai=int(is_ai.sum()), summary=summary),
        f,
        indent=1,
    )
  print("wrote", args.output_path)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description=__doc__.split("\n\n", maxsplit=1)[0]
  )
  subparsers = parser.add_subparsers(dest="command", required=True)

  from_dataset_parser = subparsers.add_parser(
      "from-dataset",
      help="per-token activation dataset -> <layer>_max_activations.pth",
  )
  from_dataset_parser.add_argument("-d", "--data_path", type=str, required=True)
  from_dataset_parser.add_argument(
      "-b",
      "--batch_size",
      type=int,
      default=100,
      help="Batch size for sparse vector concatenation",
  )
  from_dataset_parser.add_argument(
      "-l",
      "--layer_name",
      type=str,
      default=None,
      help="(Optional) Name for the layer. Defaults to empty str",
  )
  from_dataset_parser.add_argument(
      "--csr_separate_columns",
      action="store_true",
      help="If true, csr columns are separated in the source dataset",
  )
  from_dataset_parser.add_argument(
      "-o",
      "--output_file",
      type=str,
      default="max_activations.pth",
      help="File name to write inside --data_path",
  )
  from_dataset_parser.set_defaults(func=run_from_dataset)

  from_pth_parser = subparsers.add_parser(
      "from-pth", help="layer*_max_activations.pth -> memory-mappable .npy"
  )
  from_pth_parser.add_argument(
      "-d",
      "--data_path",
      type=str,
      required=True,
      help="Directory containing the layer*_max_activations.pth caches",
  )
  from_pth_parser.add_argument(
      "-o",
      "--output_path",
      type=str,
      required=True,
      help="Directory to write the .npy arrays to",
  )
  from_pth_parser.add_argument(
      "-l",
      "--layer",
      type=int,
      default=None,
      help="(Optional) convert only this layer number instead of all layers",
  )
  from_pth_parser.add_argument(
      "--exit_if_converted",
      action="store_true",
      help="Skip a layer whose .npy arrays already exist",
  )
  from_pth_parser.set_defaults(func=run_from_pth)

  feature_scale_parser = subparsers.add_parser(
      "feature-scale", help="per-feature RMS of an mmap cache -> .npy"
  )
  feature_scale_parser.add_argument(
      "-d",
      "--mmap_cache",
      type=str,
      required=True,
      help="Directory of converted mmap caches",
  )
  feature_scale_parser.add_argument(
      "-o",
      "--output_path",
      type=str,
      default=None,
      help="Where to write feature_scale.npy (default: inside the cache dir)",
  )
  feature_scale_parser.add_argument(
      "--exit_if_output_path_exists",
      action="store_true",
      help="If true, exit if the scale file already exists",
  )
  feature_scale_parser.set_defaults(func=run_feature_scale)

  lor_parser = subparsers.add_parser(
      "lor", help="per-feature log odds ratio, human vs. target -> csv"
  )
  lor_parser.add_argument(
      "-d",
      "--data_path",
      type=str,
      required=True,
      help="Dataset directory holding <layer>_max_activations.pth",
  )
  lor_parser.add_argument(
      "-l",
      "--layer_name",
      type=str,
      default=None,
      help="(Optional) Name for the layer. Defaults to empty str",
  )
  lor_parser.add_argument(
      "-m",
      "--model_col",
      type=str,
      default="model",
      help='Dataset column naming each document\'s generator ("human" for humans)',
  )
  lor_parser.add_argument(
      "-tm",
      "--target_model",
      type=str,
      default="",
      help="Generator compared against the humans (default: every non-human row)",
  )
  lor_parser.add_argument(
      "-f",
      "--frequency_threshold",
      type=int,
      default=25,
      help="Features firing in fewer rescaled documents than this get NaN",
  )
  lor_parser.add_argument(
      "-o",
      "--output_file",
      type=str,
      default="log_odds_ratios.csv",
      help=(
          "File name to write inside --data_path, after the <layer>_ prefix;"
          " <target>_log_odds_ratios.csv keeps a per-target copy"
      ),
  )
  lor_parser.set_defaults(func=run_lor)

  feature_stats_parser = subparsers.add_parser(
      "feature-stats",
      help="per-feature activation statistics over a head's training rows",
  )
  feature_stats_parser.add_argument(
      "-d",
      "--data_path",
      type=str,
      required=True,
      help="HF dataset directory (only its label column is read)",
  )
  feature_stats_parser.add_argument(
      "--mmap_cache",
      type=str,
      required=True,
      help="Directory of converted mmap caches for the same split",
  )
  feature_stats_parser.add_argument(
      "-tm",
      "--target_model",
      type=str,
      default=None,
      help="Keep rows labelled human or this model (omit to keep every row)",
  )
  feature_stats_parser.add_argument(
      "-l", "--label_col", type=str, default="model"
  )
  feature_stats_parser.add_argument(
      "-o", "--output_path", type=str, required=True
  )
  feature_stats_parser.add_argument("--n_bins", type=int, default=1024)
  feature_stats_parser.add_argument("--log_min", type=float, default=1.0)
  feature_stats_parser.add_argument("--log_max", type=float, default=1e4)
  feature_stats_parser.add_argument(
      "--all_rows",
      action="store_true",
      help="Use the whole filtered population rather than the 90%% train split",
  )
  feature_stats_parser.add_argument(
      "--exit_if_output_path_exists",
      action="store_true",
      help="If true, exit if the stats file already exists",
  )
  feature_stats_parser.set_defaults(func=run_feature_stats)

  active_counts_parser = subparsers.add_parser(
      "active-counts",
      help="active interpretable features per document -> npz + json",
  )
  active_counts_parser.add_argument(
      "--cache_root",
      type=str,
      required=True,
      help=(
          "Directory holding the <train dir>/ and <eval dir>/ datasets and "
          "the description files"
      ),
  )
  active_counts_parser.add_argument(
      "--mmap_root",
      type=str,
      required=True,
      help="Directory holding the converted <eval dir>/ mmap caches",
  )
  active_counts_parser.add_argument(
      "--sweep_root",
      type=str,
      default="outputs/logreg_sgd_l1_ablation",
      help="SGD+L1 sweep tree whose models/<train dir>/ holds the heads",
  )
  active_counts_parser.add_argument(
      "--train_dir",
      type=str,
      required=True,
      help="Dir holding descriptions and heads, e.g. PRDB_train_gemmascope",
  )
  active_counts_parser.add_argument(
      "--eval_dir",
      type=str,
      required=True,
      help="Dir whose mmap cache and labels to stream, e.g. PRDB_test_gemma2",
  )
  active_counts_parser.add_argument(
      "--target", type=str, required=True, help="gpt4o | llama | coling"
  )
  active_counts_parser.add_argument(
      "--split",
      choices=["train", "test"],
      required=True,
      help=(
          "train: the head's 90%% training rows; test: every human / target "
          "row"
      ),
  )
  active_counts_parser.add_argument(
      "--desc_gen_model",
      type=str,
      default=DESC_MODEL,
      help="Tag of the layer{N}_{tag}_feature_results.jsonl description files",
  )
  active_counts_parser.add_argument(
      "-o", "--output_path", type=str, required=True, help=".npz path"
  )
  active_counts_parser.set_defaults(func=run_active_counts)

  args = parser.parse_args()
  args.func(args)
