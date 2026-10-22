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

"""Document-level max SAE activation caches and dense feature caches.

A cache holds, for one SAE layer, the max activation of every feature over
every document of a split as one CSR matrix [n_docs, d_sae]; row i is row i
of the Hugging Face dataset the split was built from. process.py writes it
in one of two layouts that hold the same three arrays:

  layer{N}_max_activations.pth   one torch.sparse_csr_tensor (resident)
  layer{N}.{crow,col,val,shape}.npy   flat arrays, memory-mappable

`MaxActivationCache` opens either layout and answers per-feature queries and
row gathers. `LayerStack` opens every block of a cache directory, the SAE
layers of a split in ascending order (feature id = rank * d_sae + local
index) or one named block such as the `gram2vec` features extract.py writes,
and presents one concatenated feature space, so the trainers and evaluate.py
do not care which kind of feature they score.

The module also finds a split's per-layer files (`get_layer_files` and the
three wrappers for caches, descriptions and log-odds ratios), builds the
interp mask from the description files (`get_interp_mask`) and joins the
descriptions with the log-odds ratios into one feature table
(`load_feature_table`), which `evidence` ranks for one document.
"""

import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

# Entries per pass of the streaming reductions. The val/col arrays run to
# billions of entries, so they are read in slices of this many.
NNZ_CHUNK = 200_000_000

# Column suffixes of one sparse CSR activation as extract.py sae writes it.
CSR_KEYS = ["crow_indices", "col_indices", "values", "size"]
SAE_COLUMN_INFIX = "_sae_acts_post_"

# Blocks of a cache directory: numbered SAE layers, layer{N}.*.npy, or one
# named block such as gram2vec.*.npy.
_NPY_BLOCK = re.compile(r"(.+)\.crow\.npy")
_LAYER_BLOCK = re.compile(r"layer(\d+)")
_PTH_LAYER = re.compile(r"layer(\d+)_max_activations\.pth")


def sae_hook(layer: int) -> str:
  """Hook point of GemmaScope layer N: the SAE reads the input of block N+1."""
  return f"model.layers.{layer + 1}_input"


def sae_columns(hook: str) -> dict[str, str]:
  """CSR_KEYS -> the dataset columns holding one hook's per-token features."""
  return {key: f"{hook}{SAE_COLUMN_INFIX}{key}" for key in CSR_KEYS}


class MaxActivationCache:
  """One layer's document-level max activations as a CSR matrix.

  `crow` ([n_docs + 1] int64) is always resident; `col` ([nnz] int64) and
  `val` ([nnz] float32) are memory-mapped when opened with `from_npy(...,
  mmap=True)` and resident otherwise. The three arrays and `shape` are the
  public surface the streaming scripts slice directly.
  """

  def __init__(
      self,
      crow: np.ndarray,
      col: np.ndarray,
      val: np.ndarray,
      shape: tuple[int, int],
  ):
    self.crow = crow
    self.col = col
    self.val = val
    self.shape = (int(shape[0]), int(shape[1]))

  @classmethod
  def from_pth(cls, path: str) -> "MaxActivationCache":
    """Opens a `torch.save`d sparse CSR tensor (any file name)."""
    t = torch.load(path)
    assert isinstance(t, torch.Tensor) and t.layout == torch.sparse_csr
    return cls(
        t.crow_indices().numpy(),
        t.col_indices().numpy(),
        t.values().numpy(),
        tuple(t.shape),
    )

  @classmethod
  def from_npy(
      cls, directory: str, name: str, mmap: bool = True
  ) -> "MaxActivationCache":
    """Opens <name>.{crow,col,val,shape}.npy under directory."""
    prefix = os.path.join(directory, name)
    mode = "r" if mmap else None
    return cls(
        np.load(f"{prefix}.crow.npy"),
        np.load(f"{prefix}.col.npy", mmap_mode=mode),
        np.load(f"{prefix}.val.npy", mmap_mode=mode),
        tuple(np.load(f"{prefix}.shape.npy").tolist()),
    )

  def save_npy(self, directory: str, name: str) -> None:
    """Writes the four <name>.*.npy arrays `from_npy` reads."""
    prefix = os.path.join(directory, name)
    np.save(f"{prefix}.crow.npy", np.asarray(self.crow))
    np.save(f"{prefix}.col.npy", np.asarray(self.col))
    np.save(f"{prefix}.val.npy", np.asarray(self.val))
    np.save(f"{prefix}.shape.npy", np.array(self.shape))

  @property
  def nnz(self) -> int:
    return int(self.col.shape[0])

  def to_torch(self) -> torch.Tensor:
    """The cache as a resident torch.sparse_csr_tensor."""
    return torch.sparse_csr_tensor(
        torch.from_numpy(np.asarray(self.crow)),
        torch.from_numpy(np.asarray(self.col)),
        torch.from_numpy(np.asarray(self.val)),
        size=self.shape,
    )

  def to_scipy(self) -> sp.csr_matrix:
    """The cache as a scipy CSR matrix wrapping the same arrays."""
    return sp.csr_matrix(
        (np.asarray(self.val), np.asarray(self.col), np.asarray(self.crow)),
        shape=self.shape,
    )

  def fetch_dense_rows(
      self, row_indices: np.ndarray, out: np.ndarray, col_offset: int = 0
  ) -> None:
    """Scatters the requested rows into a dense output buffer.

    Only the non-zero entries are written; the caller zeroes `out` first.

    Args:
      row_indices: rows to gather, in the order they should appear in out.
      out: dense buffer of shape [len(row_indices), total_dim] to write into.
      col_offset: column at which this layer's features start in out, so
        several layers can be written into one buffer to form the
        concatenated feature vector.
    """
    for i, row in enumerate(row_indices):
      start, end = self.crow[row], self.crow[row + 1]
      out[i, col_offset + self.col[start:end]] = self.val[start:end]

  def _resident(self) -> None:
    """Loads col and val into memory once; the per-feature queries scan them."""
    if isinstance(self.col, np.memmap):
      self.col = np.array(self.col)
    if isinstance(self.val, np.memmap):
      self.val = np.array(self.val)

  def get_feature_density(self) -> list[int]:
    """Number of documents each feature is active on (a count, not a rate)."""
    self._resident()
    return np.bincount(self.col, minlength=self.shape[1]).tolist()

  def get_active_docs(self, feature_index: int) -> list[int]:
    """Documents where the feature is active, in row order."""
    self._resident()
    positions = np.flatnonzero(self.col == feature_index)
    return (np.searchsorted(self.crow, positions, side="right") - 1).tolist()

  def get_max_activations(self, feature_index: int) -> list[tuple[int, float]]:
    """(doc_id, max_activation) of the feature on each active document.

    Inactive documents are omitted, which is how the auto-interp validator
    finds its negatives.
    """
    self._resident()
    active = self.col == feature_index
    docs = np.searchsorted(self.crow, np.flatnonzero(active), side="right") - 1
    return list(zip(docs.tolist(), self.val[active].tolist()))


class LayerStack:
  """The blocks of one cache directory as one concatenated feature space.

  A directory holds either the SAE layers of a split, `layer{N}` blocks
  stacked in ascending N so that feature id = rank * d_sae + local index
  (the layout `get_interp_mask` and the heads use), or exactly one named
  block such as `gram2vec`, whose features are its columns as they are.
  `open` reads `<block>.*.npy` arrays (memory-mapped) or, when a directory
  has none, its `layer{N}_max_activations.pth` files (resident); the pth
  match is exact so the `layer7_gpt4o_max_activations.pth` split caches of
  process.py lor are never mistaken for a layer.
  """

  def __init__(self, layers: list[MaxActivationCache], names: list[str]):
    if not layers:
      raise ValueError("a LayerStack needs at least one block")
    shapes = {c.shape for c in layers}
    if len(shapes) != 1:
      raise ValueError(f"blocks disagree on shape: {sorted(shapes)}")
    self.layers = layers
    self.names = list(names)
    self.n_rows, self.d_sae = layers[0].shape
    self.dim = self.d_sae * len(layers)

  @property
  def layer_ids(self) -> list[int]:
    """The N of every layer{N} block, in stack order; empty for a named block."""
    return [
        int(m.group(1)) for n in self.names if (m := _LAYER_BLOCK.fullmatch(n))
    ]

  @classmethod
  def open(cls, directory: str, mmap: bool = True) -> "LayerStack":
    """Opens the npy blocks of directory, or its pth layers when it has none."""
    entries = os.listdir(directory)
    blocks = sorted(
        m.group(1) for n in entries if (m := _NPY_BLOCK.fullmatch(n))
    )
    if blocks:
      layers = [b for b in blocks if _LAYER_BLOCK.fullmatch(b)]
      if layers and len(layers) != len(blocks):
        raise ValueError(
            f"{directory} mixes layer blocks with named ones: {blocks}"
        )
      if layers:
        blocks = sorted(
            layers, key=lambda b: int(_LAYER_BLOCK.fullmatch(b).group(1))
        )
      elif len(blocks) != 1:
        raise ValueError(f"{directory} holds several named blocks: {blocks}")
      return cls(
          [MaxActivationCache.from_npy(directory, b, mmap) for b in blocks],
          blocks,
      )
    pth = sorted(
        (int(m.group(1)), n) for n in entries if (m := _PTH_LAYER.fullmatch(n))
    )
    if not pth:
      raise ValueError(
          f"no <block>.crow.npy or layer*_max_activations.pth in {directory}"
      )
    return cls(
        [
            MaxActivationCache.from_pth(os.path.join(directory, n))
            for _, n in pth
        ],
        [f"layer{l}" for l, _ in pth],
    )

  def __len__(self) -> int:
    return len(self.layers)

  def __iter__(self):
    return iter(self.layers)

  def fetch(
      self, rows: np.ndarray, out: np.ndarray | None = None
  ) -> np.ndarray:
    """Dense float32 [len(rows), dim] of the requested cache rows."""
    if out is None:
      out = np.zeros((len(rows), self.dim), dtype=np.float32)
    for i, layer in enumerate(self.layers):
      layer.fetch_dense_rows(rows, out, col_offset=i * self.d_sae)
    return out

  def to_scipy(self) -> sp.csr_matrix:
    """All layers side by side as one [n_rows, dim] scipy CSR matrix."""
    return sp.hstack([l.to_scipy() for l in self.layers], format="csr")

  def row(self, r: int) -> tuple[np.ndarray, np.ndarray]:
    """(feature ids, max activations) of the active features of row r."""
    ids, vals = [], []
    for i, layer in enumerate(self.layers):
      a, b = int(layer.crow[r]), int(layer.crow[r + 1])
      ids.append(i * self.d_sae + np.asarray(layer.col[a:b], dtype=np.int64))
      vals.append(np.asarray(layer.val[a:b], dtype=np.float32))
    return np.concatenate(ids), np.concatenate(vals)

  def score(
      self, w: np.ndarray, chunk_rows: int = 20_000, log=None
  ) -> np.ndarray:
    """Logits (without bias) of every row under a linear head w [dim].

    Scored per layer in blocks of chunk_rows, so memory stays at one block's
    nnz; layers whose weights are all zero are skipped.
    """
    logits = np.zeros(self.n_rows, dtype=np.float64)
    for i, layer in enumerate(self.layers):
      wl = w[i * self.d_sae : (i + 1) * self.d_sae]
      if not np.any(wl):
        continue
      crow = layer.crow
      for a in range(0, self.n_rows, chunk_rows):
        b = min(a + chunk_rows, self.n_rows)
        s, e = int(crow[a]), int(crow[b])
        if e == s:
          continue
        x = sp.csr_matrix(
            (
                np.asarray(layer.val[s:e], dtype=np.float32),
                np.asarray(layer.col[s:e], dtype=np.int64),
                crow[a : b + 1] - s,
            ),
            shape=(b - a, self.d_sae),
        )
        logits[a:b] += x @ wl
      if log:
        log(f"  {self.names[i]} scored")
    return logits

  def feature_rms(self, eps: float = 1e-12) -> np.ndarray:
    """Per-feature RMS over the corpus, sqrt(mean(x_j^2)), float32 [dim].

    The SGD + L1 trainer divides every feature by this scale: SAE activations
    span orders of magnitude across features, which makes plain SGD diverge,
    and dividing (rather than centring) keeps zeros at zero so the caches stay
    sparse. A feature that never fires has scale 0 and is left at 1.0 so the
    division is a no-op rather than a NaN.
    """
    scales = []
    for layer in self.layers:
      sumsq = np.zeros(self.d_sae, dtype=np.float64)
      for start in range(0, layer.nnz, NNZ_CHUNK):
        end = min(start + NNZ_CHUNK, layer.nnz)
        cols = np.asarray(layer.col[start:end])
        vals = np.asarray(layer.val[start:end], dtype=np.float64)
        sumsq += np.bincount(cols, weights=vals * vals, minlength=self.d_sae)
      rms = np.sqrt(sumsq / self.n_rows)
      rms[rms < eps] = 1.0
      scales.append(rms.astype(np.float32))
    return np.concatenate(scales)


# A split's cache directory holds one file per layer, ``layer{N}_<suffix>``
# (max activations, feature descriptions, log-odds ratios). The helpers below
# find those files sorted by N, and that order fixes the concatenated feature
# id ``rank * sae_dim + local_index`` that the heads, the demo and the
# counterfactual scripts all share.


def get_layer_files(data_path: str, suffix: str) -> list[tuple[int, str]]:
  """Finds files named layer{N}_{suffix} in data_path, sorted by N.

  The fullmatch is load-bearing: the glob for "max_activations.pth" also
  matches e.g. layer7_gpt4o_max_activations.pth, which must be rejected.

  Returns:
    (N, path) pairs in ascending N. The position of a pair in this list is
    the layer's offset in the concatenated feature ids.
  """
  pattern = re.compile(rf"layer(\d+)_{re.escape(suffix)}")
  return sorted(
      (int(m.group(1)), str(f))
      for f in Path(data_path).glob(f"layer*_{suffix}")
      if (m := pattern.fullmatch(f.name))
  )


def get_cache_files(data_path: str) -> list[tuple[int, str]]:
  """Finds and sorts cache files by layer number."""
  return get_layer_files(data_path, "max_activations.pth")


def get_desc_files(data_path: str, desc_model: str) -> list[tuple[int, str]]:
  """Finds and sorts feature description files by layer number."""
  return get_layer_files(data_path, f"{desc_model}_feature_results.jsonl")


def get_lor_files(
    data_path: str, target_model: str | None = None
) -> list[tuple[int, str]]:
  """Finds layer{N}_[{target_model}_]log_odds_ratios.csv files, sorted by N."""
  prefix = f"{target_model}_" if target_model else ""
  return get_layer_files(data_path, f"{prefix}log_odds_ratios.csv")


LOR_COLS = ["lor", "human_density", "target_density"]


def load_layer_features(desc_path: str, lor_path: str | None) -> pd.DataFrame:
  """One layer's descriptions, joined with its log-odds-ratio csv when present.

  Splits the stored `correlation` (rho, p_value) pair into `correlation` and
  `p_value` columns and guarantees the LOR_COLS exist, NaN when no csv.
  """
  with open(desc_path) as f:
    df = pd.json_normalize([json.loads(l) for l in f if l.strip()])
  # Some description files were generated in partitions and repeat features
  # (e.g. COLING_train_base_gemma2 layer 7); keep the most recent entry.
  df = df.rename(columns={"feature_index": "local_index"}).drop_duplicates(
      "local_index", keep="last"
  )
  if lor_path:  # LOR covers every feature id; descriptions are sparse
    lor = pd.read_csv(lor_path).drop(columns="layer")
    df = lor.rename(columns={"feature_id": "local_index"}).merge(
        df, on="local_index", how="left"
    )
  df["p_value"] = df["correlation"].str[1]
  df["correlation"] = df["correlation"].str[0]
  return df.reindex(
      columns=df.columns.union(LOR_COLS)
  )  # NaN LOR columns if absent


def load_feature_table(
    data_path: str,
    desc_model: str,
    sae_dim: int,
    target_model: str | None = None,
    n_layers: int | None = None,
) -> pd.DataFrame:
  """Descriptions + LOR of every layer, indexed by concatenated feature id.

  Uses the same sorted-by-layer discovery and `rank * sae_dim + local_index`
  offset as get_interp_mask, so ids line up with the heads' input. The LOR
  csv is the target-qualified one if it exists, else the untargeted one,
  else absent. `n_layers`, when given, must match the description files.
  """
  desc_files = get_desc_files(data_path, desc_model)
  lor_files = dict(
      get_lor_files(data_path, target_model) or get_lor_files(data_path)
  )
  if n_layers is not None and len(desc_files) != n_layers:
    raise ValueError(
        f"{len(desc_files)} description files for {n_layers} layers"
    )
  frames = [
      load_layer_features(path, lor_files.get(layer)).assign(
          layer=layer,
          feature_id=lambda d, r=rank: r * sae_dim + d["local_index"],
      )
      for rank, (layer, path) in enumerate(desc_files)
  ]
  return pd.concat(frames).set_index("feature_id")


def feature_frame(table: dict[int, dict]) -> pd.DataFrame:
  """A detector's feature_table (its features.jsonl records) as evidence() reads it.

  Indexed by feature id, with the nested description flattened to
  `description.terse` / `.keywords` / `.long`, the stored (rho, p_value)
  pair split into `correlation` and `p_value`, and LOR_COLS present (NaN
  for features without log-odds statistics).
  """
  df = pd.json_normalize(list(table.values())).set_index("id")
  df["p_value"] = df["correlation"].str[1]
  df["correlation"] = df["correlation"].str[0]
  return df.reindex(columns=df.columns.union(LOR_COLS))


def evidence(features: pd.DataFrame, det) -> pd.DataFrame:
  """Feature table rows of a Detection's features, in its |importance| order.

  Adds `cf_delta`, the importance, and `token_frac`, the fraction of the
  document's tokens the feature fires on. Features missing from the table
  keep NaN descriptions rather than being dropped.
  """
  df = features.reindex([f.id for f in det.features])
  df["cf_delta"] = [f.importance for f in det.features]
  df["token_frac"] = [
      len(f.highlights or ()) / len(det.tokens) for f in det.features
  ]
  return df


def get_interp_mask(
    data_path: str, desc_model: str, threshold: float, sae_dim: int
) -> torch.Tensor:
  """Creates the interp mask: 1 where the simulation score >= threshold.

  Reads every layer{N}_{desc_model}_feature_results.jsonl in data_path in
  ascending N and indexes features as ``rank * sae_dim + feature_index``, the
  layout of the heads' concatenated input. ``correlation`` is stored as the
  (rho, p_value) pair of scipy's spearmanr; only rho is compared. Features
  without a description stay 0. The trainers skip this call when tau is 0.0
  and use an all-ones mask.

  Args:
    data_path: split cache directory holding the description files.
    desc_model: description-model tag in the file names.
    threshold: tau, compared against rho.
    sae_dim: width of one SAE.

  Returns:
    Int tensor of shape [sae_dim * n_layers] with entries in {0, 1}.
  """
  # Load all description files into one pandas dataframe
  dfs = []
  for i, (layer, path) in enumerate(get_desc_files(data_path, desc_model)):
    with open(path, "r") as file:
      json_lines = [json.loads(line) for line in file]
    df = pd.json_normalize(json_lines)
    df["correlation"] = df["correlation"].str[0]
    # index feature ID on a per-layer basis
    df["feature_id"] = (i * sae_dim) + df["feature_index"].astype(int)
    dfs.append(df)

  df = pd.concat(dfs)

  # filter to only interpretable features
  df = df[(df["correlation"] >= threshold)]
  interp_feats = df["feature_id"].unique().tolist()

  # Create binary tensor (0 = below threshold, 1 = above threshold)
  mask = torch.zeros(sae_dim * len(dfs), dtype=int)
  mask[interp_feats] = 1

  return mask
