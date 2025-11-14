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

"""Creates a max activation cache from a dataset containing sparse activations.

This script loads a dataset containing sparse SAE activations, computes the
maximum activation for each feature across different tokens, and saves the
result as a single sparse CSR tensor to later be loaded as a MaxActivationCache.
"""

import argparse
import os
from typing import Any, Iterator

import accelerate
import datasets
import torch
import tqdm

CSR_KEYS = ["crow_indices", "col_indices", "values", "size"]


def get_max_activation(t: torch.sparse_csr_tensor) -> torch.sparse_coo_tensor:
  """Computes the maximum activation for each feature in a sparse CSR tensor.

  Args:
    t: A sparse_csr_tensor where rows represent tokens and columns represent
      features.

  Returns:
    A sparse_coo_tensor containing the maximum activation for each feature
    across all tokens.
  """
  # Get the unique IDs
  unique_ids, inverse_indices = torch.unique(
      t.col_indices(), return_inverse=True
  )

  # Initialize the tensor of max values to all zeros
  max_values = torch.zeros(unique_ids.shape)[None, :]

  # Use scatter_reduce_ to compute efficient max
  max_values.scatter_reduce_(
      1, inverse_indices, t.values(), reduce="amax", include_self=False
  )

  # create the new max_activations tensor
  return torch.sparse_coo_tensor(
      indices=torch.stack((torch.zeros(unique_ids.shape), unique_ids)),
      values=max_values[0, :],
      size=(1, t.size()[2]),
  )


def iterate_slice(ds_slice: datasets.Dataset) -> Iterator[dict[str, Any]]:
  """Iterate over a slice of the dataset and only return the CSR keys."""
  for i in range(len(ds_slice[CSR_KEYS[0]])):
    yield {col: ds_slice[col][i] for col in CSR_KEYS}


def ds_to_sparse_tensor(
    ds: datasets.Dataset,
    batch_size: int = 100,
    separate_columns: bool = True,
    values_col: str = "hook_sae_acts_post",
) -> torch.sparse_csr_tensor:
  """Convert the dataset to a 3D sparse tensor."""
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
      torch.cat(ds[i : min(i+batch_size, len(ds)-1)]["sae_activation"], dim=0)
      for i in tqdm.tqdm(range(0, len(ds), batch_size))
  ]

  return torch.cat(batches, dim=0).to_sparse_csr()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-d",
      "--data_path",
      type=str,
      default="./activations/sae_activations_10k",
      help="Path to the dataset",
  )
  parser.add_argument(
      "-b",
      "--batch_size",
      type=int,
      default=100,
      help="Batch size for sparse vector concatenation",
  )
  parser.add_argument(
      "--csr_separate_columns",
      action="store_true",
      help="If true, csr columns are separated in the source dataset",
  )
  parser.add_argument(
      "--do_shard",
      action="store_true",
      help="If true, shard dataset",
  )
  parser.add_argument(
      "-o",
      "--output_file",
      type=str,
      default="max_activations.pth",
      help="Name of the mmap file to write to",
  )
  args = parser.parse_args()

  dataset = datasets.load_from_disk(args.data_path)

  if args.csr_separate_columns:
    dataset = dataset.select_columns(CSR_KEYS)

  if args.do_shard:
    accelerator = accelerate.Accelerator()
    dataset = dataset.shard(  # pytype: disable=attribute-error
        num_shards=accelerator.num_processes, index=accelerator.process_index
    )

  sparse_tensor = ds_to_sparse_tensor(
      dataset, args.batch_size, args.csr_separate_columns
  )

  torch.save(sparse_tensor, os.path.join(args.data_path, args.output_file))
