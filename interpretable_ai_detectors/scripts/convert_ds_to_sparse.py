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

"""Converts a dataset of SAE activations to a sparse tensor of max activations.

This script loads a dataset containing sparse SAE activations, computes the
maximum activation for each feature across different tokens, and saves the
result as a single sparse CSR tensor.
"""

import argparse
import os

import datasets
import torch
import tqdm


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


def ds_to_sparse_tensor(ds, batch_size: int = 100):
  """Convert the dataset to a 3D sparse tensor."""
  ds = ds.with_transform(
      lambda x: {
          "hook_sae_acts_post": [
              get_max_activation(torch.sparse_csr_tensor(**contents))
              for contents in x["hook_sae_acts_post"]
          ]
      }
  )

  batches = [
      torch.cat(ds[i : i + batch_size]["hook_sae_acts_post"], dim=0)
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
      "-o",
      "--output_file",
      type=str,
      default="max_activations.pth",
      help="Name of the mmap file to write to",
  )
  args = parser.parse_args()

  dataset = datasets.load_from_disk(args.data_path)
  sparse_tensor = ds_to_sparse_tensor(dataset, args.batch_size)

  torch.save(sparse_tensor, os.path.join(args.data_path, args.output_file))
