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

"""Extracts per-document features from a text dataset.

Two feature kinds, one sub-command each:

  sae       Runs every document through the extractor LM, applies one SAE per
            hook point and saves each token's SAE features as sparse CSR
            columns, one set per hook (`cache.sae_columns`), in a copy of the
            dataset. process.py from-dataset collapses them to per-document
            caches; autointerpret.py reads the tokens to highlight them.
  gram2vec  Computes the 618 gram2vec stylometric features of every document
            with spaCy and writes them straight into the per-document cache
            layout, `<out_dir>/gram2vec.{crow,col,val,shape}.npy` plus
            `feature_names.json`, which `cache.LayerStack` opens exactly like
            an SAE cache. There is no token level, so nothing to pool.

Row i of either output is row i of the dataset at --data_path, which is where
the trainers and evaluate.py read the labels.

Usage:
  python -m detectorscope.data.extract sae -d data/data/PRDB/train -t review \
      -m models/detectors/gpt4o_detect_gemma -k model.layers.8 ... \
      -p models/saes/<setting>/model.layers.8_input_finetuned_sae.pth ... \
      -o data/sae_activations/PRDB_train_gemma2_gpt4o
  python -m detectorscope.data.extract gram2vec -d data/data/PRDB/train \
      -t review -o outputs/gram2vec_features/PRDB_train

gram2vec throughput: gram2vec.vectorizer.from_documents forks four spaCy
workers per call, which is the bottleneck. Each worker here runs the same
pipeline single-process (emoji strip -> nlp.pipe -> Gram2VecDocument ->
_apply_features_to_docs), which is bit-identical to from_documents; the
parent asserts this on the first documents before doing any real work.
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import datasets
from detectorscope.data.cache import MaxActivationCache
from detectorscope.data.cache import sae_columns
from detectorscope.model.extractor import HFExtractor
from detectorscope.model.sae import JumpReLUSAE
import numpy as np
import scipy.sparse as sp
import torch

SPACY_BATCH = 64


def extract_sae(
    extractor: HFExtractor,
    ds: datasets.Dataset,
    saes: dict[str, JumpReLUSAE],
    text_col: str,
) -> datasets.Dataset:
  """Adds one set of CSR columns per hook: the SAE features of every token.

  Args:
    extractor: a wrapper around the LLM.
    ds: the dataset to run processing on.
    saes: a dict of saes indexed by hook point; the keys are the extractor's
      `<hook>_input` names.
    text_col: the column name in the dataset containing the text.

  Returns:
    A new dataset object with additional columns, one for each SAE activation
    saved.
  """

  def sparsify(x: torch.Tensor):
    """Splits x into its CSR arrays; size is a plain list for Arrow."""
    x = x.detach().to_sparse(layout=torch.sparse_csr).cpu()
    return {
        "crow_indices": x.crow_indices(),
        "col_indices": x.col_indices(),
        "values": x.values(),
        "size": list(x.size()),
    }

  def process_row(row: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extracts one row's activations into the new columns."""
    output = {}
    acts = extractor.get_activations(row[text_col], on_device=True)
    for hook, sae in saes.items():
      _, features = sae.forward(acts[hook])
      columns = sae_columns(hook)
      for key, value in sparsify(features).items():
        output[columns[key]] = value
    return output

  return ds.map(process_row, writer_batch_size=50)


def run_sae(args: argparse.Namespace) -> None:
  """Writes the per-token activation dataset to --output_path."""
  if len(args.sae_paths) != len(args.hook_points):
    raise ValueError("pass one SAE (-p) per hook point (-k)")

  # Disable caching to ensure that huggingface doesn't save the intermediate
  # steps of computation. This takes up twice as much disk space and can
  # cause OOM errors on large datasets.
  datasets.disable_caching()
  dataset = datasets.load_from_disk(args.data_path)
  if args.num_examples:
    dataset = dataset.select(range(args.num_examples))

  extractor = HFExtractor(
      model_path=args.model_path, default_hooks=args.hook_points
  )

  saes = {}
  for sae_path, hook in zip(args.sae_paths, args.hook_points):
    if os.path.exists(sae_path):
      sae = JumpReLUSAE.from_state_dict(sae_path)
    else:
      sae = JumpReLUSAE.from_gemma_scope(sae_path, args.gemma_scope_repo)
    saes[f"{hook}_input"] = sae.to(extractor.device).eval()

  extract_sae(extractor, dataset, saes, args.text_col).save_to_disk(
      args.output_path
  )


def featurize_chunk(texts: list[str]) -> tuple[list[str], np.ndarray]:
  """Computes the gram2vec feature matrix of a list of texts, single process.

  Mirrors gram2vec.vectorizer._process_documents + _apply_features_to_docs
  exactly, minus the n_process=4 fork inside nlp.pipe.

  Returns:
    (column names, float32 array [len(texts), n_features]); NaN where a
    document had zero tokens.
  """
  from gram2vec import vectorizer as g2v
  from gram2vec._load_spacy import nlp

  cleaned = [g2v._remove_emojis(t) for t in texts]
  docs = list(nlp.pipe(cleaned, batch_size=SPACY_BATCH, n_process=1))
  g2v_docs = [
      g2v.Gram2VecDocument(raw, doc, len(doc), len(list(doc.sents)))
      for raw, doc in zip(texts, docs)
  ]
  df = g2v._apply_features_to_docs(g2v_docs, None, False)
  return df.columns.tolist(), df.to_numpy(dtype=np.float32)


def reference_features(texts: list[str]) -> tuple[list[str], np.ndarray]:
  """The library's own path, used only to assert parity at startup."""
  from gram2vec import vectorizer as g2v

  df = g2v.from_documents(texts)
  return df.columns.tolist(), df.to_numpy(dtype=np.float32)


def clean_texts(values: list) -> list[str]:
  """Replaces None and other non-string cells with "" for spaCy."""
  return [v if isinstance(v, str) else "" for v in values]


def run_gram2vec(args: argparse.Namespace) -> None:
  """Writes <output_dir>/gram2vec.*.npy and feature_names.json.

  Checks featurize_chunk against the library's from_documents on the first
  five documents, then forks --workers processes over --chunk_size chunks.
  Empty documents (NaN features) are stored as zeros.
  """
  os.makedirs(args.output_dir, exist_ok=True)
  shape_path = os.path.join(args.output_dir, "gram2vec.shape.npy")

  ds = datasets.load_from_disk(args.data_path)
  if args.num_examples:
    ds = ds.select(range(min(args.num_examples, len(ds))))
  n = len(ds)

  if args.exit_if_output_exists and os.path.exists(shape_path):
    have = int(np.load(shape_path)[0])
    if have == n:
      print(f"Found {args.output_dir} ({have} rows), exiting")
      return
    print(f"{args.output_dir} has {have} rows, expected {n}; recomputing")

  texts = clean_texts(ds[args.text_col])

  # spaCy and torch both spin up thread pools; one thread per worker process is
  # what keeps a fork pool from oversubscribing the allocation. Load spaCy once
  # in the parent so forked workers share the model pages.
  os.environ.setdefault("OMP_NUM_THREADS", "1")
  os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
  torch.set_num_threads(1)
  t0 = time.time()
  import gram2vec  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import

  print(f"gram2vec import {time.time() - t0:.1f}s", flush=True)

  # Parity guard: the manual single-process pipeline must equal the library's
  # from_documents on real documents, NaN pattern included.
  probe = texts[: min(5, n)]
  if probe:
    ref_cols, ref = reference_features(probe)
    got_cols, got = featurize_chunk(probe)
    if ref_cols != got_cols or not np.array_equal(ref, got, equal_nan=True):
      raise RuntimeError("single-process pipeline differs from from_documents")
    print(
        f"parity vs from_documents ok on {len(probe)} docs "
        f"({len(ref_cols)} features)",
        flush=True,
    )

  chunks = [
      texts[i : i + args.chunk_size] for i in range(0, n, args.chunk_size)
  ]
  ctx = mp.get_context("fork")
  cols = None
  parts = []
  done = 0
  t0 = time.time()
  with ctx.Pool(args.workers) as pool:
    for chunk_cols, feats in pool.imap(featurize_chunk, chunks, chunksize=1):
      if cols is None:
        cols = chunk_cols
      elif chunk_cols != cols:
        raise RuntimeError("feature order changed between chunks")
      parts.append(feats)
      done += feats.shape[0]
      elapsed = time.time() - t0
      print(
          f"  {done:,}/{n:,}  {elapsed:.0f}s  {done / max(elapsed, 1e-9):.1f} docs/s",
          flush=True,
      )

  feats = (
      np.concatenate(parts, axis=0)
      if parts
      else np.zeros((0, len(cols or [])), np.float32)
  )
  assert feats.shape[0] == n, (feats.shape, n)
  n_nan = int(np.isnan(feats).any(axis=1).sum())
  feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

  m = sp.csr_matrix(feats)
  MaxActivationCache(
      m.indptr.astype(np.int64),
      m.indices.astype(np.int64),
      m.data.astype(np.float32),
      m.shape,
  ).save_npy(args.output_dir, "gram2vec")
  with open(os.path.join(args.output_dir, "feature_names.json"), "w") as f:
    json.dump(cols or [], f)
  print(
      f"wrote {args.output_dir}/gram2vec.*.npy  shape={feats.shape}  "
      f"nnz={m.nnz:,}  nan_docs={n_nan}  {time.time() - t0:.0f}s"
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description=__doc__.split("\n\n", maxsplit=1)[0]
  )
  subparsers = parser.add_subparsers(dest="command", required=True)

  sae_parser = subparsers.add_parser(
      "sae", help="text -> per-token SAE activations as CSR dataset columns"
  )
  sae_parser.add_argument(
      "-d",
      "--data_path",
      type=str,
      required=True,
      help="Hugging Face dataset directory (load_from_disk)",
  )
  sae_parser.add_argument(
      "-t",
      "--text_col",
      type=str,
      required=True,
      help="Column holding the document text",
  )
  sae_parser.add_argument(
      "-n",
      "--num_examples",
      type=int,
      default=None,
      help="Only the first N rows of the dataset (smoke tests)",
  )
  sae_parser.add_argument(
      "-m",
      "--model_path",
      type=str,
      required=True,
      help="Local path (or hub id) of the extractor LM",
  )
  sae_parser.add_argument(
      "-k",
      "--hook_points",
      type=str,
      nargs="+",
      required=True,
      help="Modules whose inputs the SAEs read, e.g. model.layers.8",
  )
  sae_parser.add_argument(
      "-p",
      "--sae_paths",
      type=str,
      nargs="+",
      required=True,
      help="One SAE per hook point: a state_dict .pth, or a Gemma Scope id",
  )
  sae_parser.add_argument(
      "-b",
      "--gemma_scope_repo",
      type=str,
      default="google/gemma-scope-2b-pt-res",
      help="Hub repo of the Gemma Scope ids among --sae_paths",
  )
  sae_parser.add_argument(
      "-o",
      "--output_path",
      type=str,
      required=True,
      help="Directory the activation dataset is saved to",
  )
  sae_parser.set_defaults(func=run_sae)

  gram2vec_parser = subparsers.add_parser(
      "gram2vec", help="text -> per-document gram2vec features as a cache"
  )
  gram2vec_parser.add_argument(
      "-d",
      "--data_path",
      type=str,
      required=True,
      help="Hugging Face dataset directory (load_from_disk)",
  )
  gram2vec_parser.add_argument(
      "-t",
      "--text_col",
      type=str,
      required=True,
      help="Column holding the document text",
  )
  gram2vec_parser.add_argument(
      "-n",
      "--num_examples",
      type=int,
      default=None,
      help="Only the first N rows of the dataset (smoke tests)",
  )
  gram2vec_parser.add_argument(
      "-o",
      "--output_dir",
      type=str,
      required=True,
      help="Cache directory for this split",
  )
  gram2vec_parser.add_argument(
      "--workers",
      type=int,
      default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
      help="Forked single-process spaCy workers",
  )
  gram2vec_parser.add_argument(
      "--chunk_size", type=int, default=500, help="Documents per worker task"
  )
  gram2vec_parser.add_argument(
      "--exit_if_output_exists",
      action="store_true",
      help="Skip a cache that already has the right number of rows",
  )
  gram2vec_parser.set_defaults(func=run_gram2vec)

  args = parser.parse_args()
  args.func(args)
