# Copyright 2025 Google LLC
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

"""This script computes log odds ratios for features in a sparse autoencoder.

It processes activation caches from datasets, splits them based on conditions
(e.g., Human vs. GPT4), calculates log odds ratios, and outputs top-k documents
for features with extreme log odds ratios.

A histogram of the log odds ratios is also generated.
"""

import argparse
import math
import os

import datasets
from interpretable_ai_detectors.model.activation_cache import MaxActivationCache
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm


def convert_ds_to_activation_cache(
    dataset: datasets.Dataset, col_name: str, batch_size: int = 128
) -> torch.Tensor:
  """Convert activations in dataset at col_name to sparse CSR tensor."""
  # Convert to COO format first for easy concatenation
  batches = [
      dataset[i : i + batch_size][col_name].to_sparse_coo()
      for i in tqdm.tqdm(range(0, len(dataset), batch_size))
  ]

  # Concat the batches together then convert to CSR
  return torch.cat(batches, dim=0).to_sparse_csr()


def split_csr_tensor(
    csr_tensor: torch.Tensor, row_indices: list[int] | torch.Tensor
) -> torch.Tensor:
  """Selects specific rows from a sparse CSR tensor. (Written by Gemini).

  Args:
      csr_tensor (torch.Tensor): The input sparse CSR tensor.
      row_indices (list or torch.Tensor): The row indices to select.

  Returns:
      torch.Tensor: Sparse CSR tensor containing only the selected rows.
  """
  # Convert to tensor for more efficient indexing
  if not isinstance(row_indices, torch.Tensor):
    row_indices = torch.tensor(row_indices, dtype=torch.long)

  # 1. Get the underlying components of the input CSR tensor
  crow_indices = csr_tensor.crow_indices()
  col_indices = csr_tensor.col_indices()
  values = csr_tensor.values()

  # 2. Find the number of non-zero elements (NNZ) in each selected row
  starts = crow_indices.index_select(0, row_indices)
  ends = crow_indices.index_select(0, row_indices + 1)
  lengths = ends - starts

  # 3. Build the new crow_indices for the output tensor
  new_crow_indices = torch.cat(
      [torch.tensor([0]), torch.cumsum(lengths, dim=0)]
  )

  # 4. Create the new values and col_indices by concatenating the slices
  indices_to_gather = torch.cat(
      [torch.arange(s, e) for s, e in zip(starts, ends)]
  )

  new_values = values.index_select(0, indices_to_gather)
  new_col_indices = col_indices.index_select(0, indices_to_gather)

  # 5. Assemble the new CSR tensor
  new_shape = (len(row_indices), csr_tensor.shape[1])
  return torch.sparse_csr_tensor(
      crow_indices=new_crow_indices,
      col_indices=new_col_indices,
      values=new_values,
      size=new_shape,
  )


def get_top_k_docs(
    feature_index: int,
    k: int,
    dataset: datasets.Dataset,
    cache: MaxActivationCache,
) -> tuple[list[str], list[tuple[int, float]]]:
  """Get the top k documents in format: doc, (doc_id, max_activation)."""
  docs = cache.get_max_activations(feature_index)
  top_k_docs = sorted(docs, key=lambda x: x[1], reverse=True)[:k]
  return dataset[[x[0] for x in top_k_docs]]["response"], top_k_docs


def compute_log_odds(cache: MaxActivationCache) -> list[float]:
  return [
      math.log((d + 0.5) / ((cache.shape[0] - d) + 0.5))
      for d in cache.get_feature_density()
  ]


def split_cache(
    dataset: datasets.Dataset,
    cache: MaxActivationCache,
    col: str,
    condition: str,
) -> torch.Tensor:
  """Split existing max activation cache on a column of the dataset."""
  indices = np.where(np.array(dataset[col]) == condition)[0].tolist()
  return split_csr_tensor(cache.features, indices)


def print_top_k_to_file(
    dataset: datasets.Dataset,
    cache: MaxActivationCache,
    log_odds_ratios: list[tuple[int, float]],
    name: str,
    k: int = 10,
):
  with open(f"{name}_top_docs.txt", "w+") as f:
    for feature_id, feature_lor in log_odds_ratios:
      f.write(f"Feature {feature_id} (LOR: {feature_lor}) Top-{k} docs\n")
      for doc, (doc_id, activ) in get_top_k_docs(feature_id, k, dataset, cache):
        f.write(f"{doc_id} - {activ} -- {repr(doc)}\n")


def plot_lor_histogram(
    log_odds_ratios: list[tuple[int, float]],
    epsilon: float = 1e-7,
):
  """Plots a histogram of the provided log odds ratios.

  Args:
      log_odds_ratios: A list of tuples, where each tuple contains a feature
        index and its corresponding log odds ratio.
      epsilon: A small float value used to filter out LORs close to zero.
  """
  # Filter out entries that are close to 0
  filtered_lor = [r for r in log_odds_ratios if abs(r) > epsilon]

  # Create the histogram
  plt.hist(filtered_lor, bins=50, alpha=0.7, edgecolor="black")
  plt.axvline(0, color="k", linestyle=":", linewidth=2, label="Center (x=0)")

  x_min, x_max = plt.xlim()
  max_abs_val = max(abs(x_min), abs(x_max))
  plt.xlim(-max_abs_val, max_abs_val)

  plt.xlabel("Values")
  plt.ylabel("Frequency")
  plt.title("Log Odds Ratios of SAE Features")

  plt.savefig("log_odds_ratio.png")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-d",
      "--data_path",
      type=str,
      help="Path to the dataset",
  )
  parser.add_argument(
      "-s",
      "--sae_path",
      type=str,
      help="Path to the .pth file to load the sae from",
  )
  parser.add_argument(
      "-c",
      "--column_name",
      type=str,
      default="norm_input",
      help="Name of column to apply the SAE to in the source dataset",
  )
  parser.add_argument(
      "-n",
      "--top_k_docs_csv_name",
      type=str,
      default="top_k_docs.csv",
      help="Name of output csv containing top k docs for each feature",
  )
  parser.add_argument(
      "-o",
      "--output_path",
      type=str,
      help="Name of the location to save the new dataset to",
  )
  args = parser.parse_args()

  cache_path = os.path.join(args.data_path, "max_activations.pth")
  human_cache_path = os.path.join(args.data_path, "human_max_activations.pth")
  gpt4_cache_path = os.path.join(args.data_path, "gpt4_max_activations.pth")

  ds = datasets.load_from_disk(args.data_path)
  ds = ds.with_format("torch", dtype=torch.float32)

  # If max activation cache does not exist, create it
  if not os.path.exists(cache_path):
    full_cache = convert_ds_to_activation_cache(ds, args.column_name)
    torch.save(full_cache, cache_path)
  full_cache = MaxActivationCache(cache_path)

  if not os.path.exists(human_cache_path):
    human_cache = split_cache(ds, full_cache, "condition", "Human")
    torch.save(human_cache, human_cache_path)
  human_cache = MaxActivationCache(human_cache_path)

  if not os.path.exists(gpt4_cache_path):
    gpt4_cache = split_cache(ds, full_cache, "condition", "GPT4")
    torch.save(gpt4_cache, gpt4_cache_path)
  gpt4_cache = MaxActivationCache(gpt4_cache_path)

  # Compute Log odds ratio of features
  human_log_odds = compute_log_odds(human_cache)
  gpt4_log_odds = compute_log_odds(gpt4_cache)
  log_odds_ratio = [
      hlo - glo for hlo, glo in zip(human_log_odds, gpt4_log_odds)
  ]

  # Sort feature IDs by log odds ratio
  sorted_lor = sorted(
      list(enumerate(log_odds_ratio)), key=lambda x: x[1], reverse=False
  )

  # Print top k documents for each of the features to a text file
  feature_top_k = 100
  print_top_k_to_file(ds, full_cache, sorted_lor[:feature_top_k:-1], "human")
  print_top_k_to_file(ds, full_cache, sorted_lor[:feature_top_k], "gpt4")

  # Output top 25 documents in the dataset for all features
  top_k = 25
  records = []
  for i, lor in sorted_lor:
    for d, (d_id, d_activ) in get_top_k_docs(i, top_k, ds, full_cache):
      records.append(
          {
              "id": i,
              "lor": lor,
              "doc": d,
              "doc_id": d_id,
              "activ": d_activ,
          }
      )
  df = pd.DataFrame(records)
  df.to_csv(args.top_k_docs_csv_name, index=False)

  # Plot histogram of log odds ratios
  plot_lor_histogram(sorted_lor)
