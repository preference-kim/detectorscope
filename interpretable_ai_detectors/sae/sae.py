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

"""AutoSAE interface and factory for loading different SAE types."""

import abc

from interpretable_ai_detectors.sae.models.baseline_sae import BaselineSAE
from interpretable_ai_detectors.sae.models.jumprelu_sae import JumpReLUSAE
import sae_lens
from sae_lens.saes.jumprelu_sae import JumpReLUSAE as SAELensJumpReLUSAE
from sae_lens.saes.standard_sae import StandardSAE as SAELensStandardSAE
import torch


class AutoSAE(abc.ABC):
  """Shared interface for all SAE implementations.

  This interface follows a design pattern similar to that used in Huggingface
  where a central AutoModel class can be used to load various different
  pre-trained language models and run inference on them in a unified way.
  """

  @abc.abstractmethod
  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Takes in x and returns x_reconstructed, features."""

  @abc.abstractmethod
  def loss_fn(
      self,
      x: torch.Tensor,
      reconstructed: torch.Tensor,
      features: torch.Tensor,
      sparsity_coeff: float,
  ) -> dict[str, torch.Tensor]:
    """Takes in x & outputs of forward. Returns dict of losses."""

  @classmethod
  def from_pretrained(cls, sae: sae_lens.SAE, **kwargs):
    """Shared factory method for creating SAE instances."""
    if isinstance(sae, SAELensJumpReLUSAE):
      return JumpReLUSAE.from_pretrained(sae, **kwargs)
    elif isinstance(sae, SAELensStandardSAE):
      return BaselineSAE.from_pretrained(sae, **kwargs)
    else:
      raise ValueError(f"Invalid sae type {type(sae)}")
