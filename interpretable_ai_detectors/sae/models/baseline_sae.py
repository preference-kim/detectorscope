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

"""Baseline Sparse Auto-Encoder implementation.

This module contains the definition of the BaselineSAE class, a simple Sparse
Auto-Encoder model.
"""

import sae_lens
import torch


class BaselineSAE(torch.nn.Module):
  """Implmentation of a baseline Sparse Auto-Encoder.

  Applies a linear encoder -> ReLU activation -> linear decoder
  with d_hidden >> d_input. Sparsity is enforced by the loss_fn
  which takes in a sparsity_coeff to scale an L1 penalty.

  Args:
    d_input (int) - dimension of the input vector
    d_hidden (int) - width of the SAE

  Can also be initialized using the `.from_pretrained()`
  method, which initializes the model using pretrained weights
  from an sae_lens checkpoint.
  """

  def __init__(self, d_input: int, d_hidden: int):
    super().__init__()
    self.d_input = d_input
    self.d_hidden = d_hidden
    self.initialize()

  def initialize(self):
    self.encoder = torch.nn.Linear(self.d_input, self.d_hidden)
    self.relu = torch.nn.ReLU()
    self.decoder = torch.nn.Linear(self.d_hidden, self.d_input)

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    features = self.relu(self.encoder(x))
    reconstructed = self.decoder(features)
    return reconstructed, features

  @property
  def device(self):
    return self.encoder.device

  def loss_fn(
      self,
      x: torch.Tensor,
      reconstructed: torch.Tensor,
      features: torch.Tensor,
      sparsity_coeff: float,
  ) -> dict[str, torch.Tensor]:
    mse_loss = torch.sum((x - reconstructed) ** 2, dim=-1)
    l1_loss = sparsity_coeff * torch.mean(torch.abs(features))
    return {"mse": mse_loss, "l1": l1_loss}

  @classmethod
  def from_pretrained(cls, sae: sae_lens.SAE, **kwargs):
    del kwargs  # Unused.
    model = cls(d_input=sae.cfg.d_in, d_hidden=sae.cfg.d_sae)
    model.load_state_dict(
        {
            "encoder.weight": sae.W_enc.T,
            "encoder.bias": sae.b_enc,
            "decoder.weight": sae.W_dec.T,
            "decoder.bias": sae.b_dec,
        }
    )
    return model

  def reset_thresholds(self):
    raise ValueError("BaseSAE has no thresholds")

  def rescale_to_unit_norm(self):
    raise NotImplementedError("Rescaling not implemented")

  def undo_rescaling(self):
    raise NotImplementedError("Rescaling not implemented")
