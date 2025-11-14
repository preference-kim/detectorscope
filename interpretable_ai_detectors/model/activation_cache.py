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

"""Provides a class for loading and querying a sparse max activation cache.

This module defines the `MaxActivationCache` class, which is used to load
and interact with precomputed maximum activations of SAE features across
a dataset.
"""

import torch


class MaxActivationCache:
  """Loads and provides query methods for a sparse max activation cache.

  This cache stores the maximum activation of each SAE feature for each document
  in a dataset.
  """

  def __init__(self, data_path: str):
    """Encapsulates logic for loading & querying the max activation cache.

    This cache is created by the scripts/convert_ds_to_sparse.py script and
    is a torch.Tensor of type torch.sparse_csr_tensor w/ shape
    [num_docs, num_sae_features]. It should contain the max
    activation of each SAE feature across all documents in the dataset.

    Args:
        data_path (str): path to the .pth file for the activations tensor
    """
    self.features = torch.load(data_path)
    assert isinstance(self.features, torch.Tensor)
    assert self.features.layout == torch.sparse_csr

  def get_feature_density(self) -> list[int]:
    """Get the activation density of all features on the dataset.

    This version assumes self.features is a 2D torch.sparse_csr_tensor where
    each non-zero value represents a unique document where that feature is
    active.

    Returns:
        A list where the i-th element is the number of documents in the
        dataset where feature i has a non-zero max activation.
    """
    return torch.bincount(
        self.features.col_indices(), minlength=self.features.shape[1]
    ).tolist()

  def get_active_docs(self, feature_index: int) -> list[int]:
    """Get the list of all documents where the given feature is active."""
    return (
        torch.searchsorted(
            self.features.crow_indices(),
            torch.where(self.features.col_indices() == feature_index)[0],
            right=True,
        )
        - 1
    ).tolist()

  def get_max_activations(self, feature_index: int) -> list[tuple[int, float]]:
    """Get the max activation of the feature on all docs in the dataset.

    Args:
        feature_index (int): The ID of the feature to get activations for.

    Returns:
        A list of tuples, where each tuple is (doc_id, max_activation).
    """
    # Group features by document
    unique_docs, inverse_indices = torch.unique(
        torch.tensor(self.get_active_docs(feature_index)),
        return_inverse=True,
    )

    # Get the values for the given feature_index
    values = self.features.values()[
        self.features.col_indices() == feature_index
    ]

    # Compute the max activations across the documents
    max_vals = torch.empty(unique_docs.shape[0])
    max_vals.scatter_reduce_(
        dim=0,
        index=inverse_indices,
        src=values,
        reduce="amax",
        include_self=False,
    )

    return list(zip(unique_docs.tolist(), max_vals.tolist()))
