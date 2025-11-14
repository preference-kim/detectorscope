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

"""Implementation of JumpReLU Sparse Autoencoders.

This module contains the JumpReLUSAE class, which implements the JumpReLU SAE
as described in https://arxiv.org/abs/2407.14435. It includes custom
autograd functions for the Heaviside step and JumpReLU activations,
using straight-through estimators for gradient computation.
"""

import math

import sae_lens
import torch
from torch import nn
from torch.autograd import function
import torch.nn.functional as F
import transformers


def rectangle(x: torch.Tensor) -> torch.Tensor:
  """Kernel function for straight-through estimators."""
  return ((x > -0.5) & (x < 0.5)).to(x.dtype)


class HeavisideStep(torch.autograd.Function):
  """Custom autograd function for the Heaviside step function.

  This class implements a straight-through estimator for the Heaviside step
  function, where the forward pass is a standard step function, and the
  backward pass uses a rectangular kernel for the gradient.
  """

  @classmethod
  def forward(
      cls,
      ctx: function.FunctionCtx,
      x: torch.Tensor,
      threshold: torch.Tensor,
      bandwidth: float,
      acc: float,
  ) -> torch.Tensor:
    ctx.save_for_backward(x, threshold)
    ctx.bandwidth = bandwidth
    ctx.acc = acc
    return (x > threshold).to(x.dtype)

  @classmethod
  def backward(
      cls, ctx: function.FunctionCtx, grad_output: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, threshold = ctx.saved_tensors   # pytype: disable=attribute-error
    bandwidth = ctx.bandwidth  # pytype: disable=attribute-error
    acc = ctx.acc  # pytype: disable=attribute-error
    grad_threshold = (
        -(1.0 / bandwidth)
        * rectangle((x - threshold) / bandwidth)
        * grad_output
    ) * acc
    return None, grad_threshold, None, None


class JumpReLU(torch.autograd.Function):
  """Custom autograd function for the JumpReLU activation.

  This class implements a straight-through estimator for the JumpReLU function,
  defined as `x * (x > threshold)`. The forward pass computes this value,
  and the backward pass uses a rectangular kernel for the gradient of the
  threshold.
  """

  @classmethod
  def forward(
      cls,
      ctx: function.FunctionCtx,
      x: torch.Tensor,
      threshold: torch.Tensor,
      bandwidth: float,
      acc: float,
  ) -> torch.Tensor:
    ctx.save_for_backward(x, threshold)
    ctx.bandwidth = bandwidth
    ctx.acc = acc
    return x * (x > threshold)

  @classmethod
  def backward(
      cls, ctx: function.FunctionCtx, grad_output: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, threshold = ctx.saved_tensors  # pytype: disable=attribute-error
    bandwidth = ctx.bandwidth  # pytype: disable=attribute-error
    acc = ctx.acc  # pytype: disable=attribute-error
    grad_x = (x > threshold).to(x.dtype) * grad_output
    grad_threshold = (
        -(threshold / bandwidth)
        * rectangle((x - threshold) / bandwidth)
        * grad_output
    ) * acc
    return grad_x, grad_threshold, None, None


class JumpReLUSAE(nn.Module):
  r"""Implmentation of JumpReLU SAEs (https://arxiv.org/abs/2407.14435).

  JumpReLU SAEs swap the ReLU in the standard SAE for a JumpReLU function
  defined as $H(x - \theta) * x$ where H is the heaviside step function.
  It also modifies the sparsity penalty from L1 to L0 and uses
  straight-through estimators (STEs) to compute gradients for both
  the L0 and JumpReLU threshold values.

  This implementation is a translation of the JAX pseudocode from Appendix J
  of the original paper into pytorch. We include aspects such as ReLU of
  pre-activations for maximum fidelity to the original implementation.

  Args:
     d_input (int) - dimension of the input vector
     d_hidden (int) - width of the SAE
     bandwidth (float) - bandwidth parameter for the KDE used in gradients
     acc (float) - constant to use for accelerating threshold gradients

  Can also be initialized using the `.from_pretrained()`
  method, which initializes the model using pretrained weights
  from an sae_lens checkpoint.
  """

  def __init__(
      self,
      d_input: int,
      d_hidden: int,
      bandwidth: float = 0.001,
      acc: float = 1.0,
  ):
    super().__init__()

    self.d_input = d_input
    self.d_hidden = d_hidden
    self.acc = acc
    self.bandwidth = bandwidth
    self.C = 1.0  # pylint: disable=invalid-name

    self.initialize()

    self.use_e2e_loss = False
    self.use_pre_enc_bias = False
    self.is_unit_norm_scaled = False

  @property
  def device(self):
    return self.W_dec.device

  def initialize(self):
    # Initialize W_dec weights with normalized He-uniform and W_enc weights
    # as W_dec.T following gemma scope https://arxiv.org/pdf/2408.05147
    self.W_dec = nn.Parameter(torch.randn(self.d_hidden, self.d_input))  # pylint: disable=invalid-name
    nn.init.kaiming_uniform_(self.W_dec)
    self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=1)
    self.W_enc = nn.Parameter(self.W_dec.data.T.clone())  # pylint: disable=invalid-name

    # Initialize bias vectors to zero & thresholds to 0.001
    self.b_enc = nn.Parameter(torch.zeros(self.d_hidden))
    self.b_dec = nn.Parameter(torch.zeros(self.d_input))
    self.log_threshold = nn.Parameter(torch.ones(self.d_hidden) * -3.0)

  def reset_thresholds(self):
    self.log_threshold = nn.Parameter(torch.ones(self.d_hidden) * -3.0)

  def activate_e2e_loss(self, path: str):
    self.use_e2e_loss = True
    cls_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        path
    )
    self.cls_head = nn.Sequential(
        getattr(getattr(cls_model, "model"), "norm"),
        getattr(cls_model, "score"),
    ).to(self.device)

  def rescale_to_unit_norm(self):
    """Rescale the model by self.C for training."""
    if self.is_unit_norm_scaled:
      raise ValueError("Attempted to scale model that is already scaled.")

    self.log_threshold.data -= math.log(self.C)
    self.b_dec.data /= self.C
    self.b_enc.data /= self.C

    if self.use_pre_enc_bias:
      self.b_enc.data += self.b_dec @ self.W_enc

    self.is_unit_norm_scaled = True

  def undo_rescaling(self):
    """Rescale the model to operate on raw activation inputs."""
    if not self.is_unit_norm_scaled:
      raise ValueError("Attempted to undo scaling on unscaled model.")

    if self.use_pre_enc_bias:
      self.b_enc.data -= self.b_dec @ self.W_enc

    self.log_threshold.data += math.log(self.C)
    self.b_enc.data *= self.C
    self.b_dec.data *= self.C

    self.C = 1.0
    self.is_unit_norm_scaled = False

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = x / self.C
    if self.use_pre_enc_bias:
      x -= self.b_dec
    pre_activations = F.relu(x @ self.W_enc + self.b_enc)
    features = JumpReLU.apply(
        pre_activations,
        torch.exp(self.log_threshold),
        self.bandwidth,
        self.acc,
    )
    reconstructed = self.C * (features @ self.W_dec + self.b_dec)
    return reconstructed, features

  def loss_fn(
      self,
      x: torch.Tensor,
      reconstructed: torch.Tensor,
      features: torch.Tensor,
      l0_lambda: float,
      e2e_lambda: float,
      labels: torch.Tensor | None = None,
  ) -> dict[str, torch.Tensor]:
    loss = {}

    if self.use_e2e_loss:
      predictions = self.cls_head(reconstructed)
      loss["e2e"] = (
          F.cross_entropy(predictions, labels.long(), reduction="none")
          * e2e_lambda
      )

    loss["mse"] = torch.mean((x - reconstructed) ** 2, dim=-1)
    loss["l0"] = (
        torch.sum(
            HeavisideStep.apply(
                features,
                torch.exp(self.log_threshold),
                self.bandwidth,
                self.acc,
            ),
            dim=-1,
        )
        * l0_lambda
    )

    return loss

  @classmethod
  def from_pretrained(cls, sae: sae_lens.SAE, **kwargs):
    model = cls(d_input=sae.cfg.d_in, d_hidden=sae.cfg.d_sae, **kwargs)
    model.load_state_dict(
        {
            "W_enc": sae.W_enc,
            "b_enc": sae.b_enc,
            "W_dec": sae.W_dec,
            "b_dec": sae.b_dec,
            "log_threshold": torch.log(sae.threshold),
        }
    )
    return model

  @classmethod
  def from_state_dict(cls, state_dict_path: str):
    state_dict = torch.load(state_dict_path)
    model = cls(
        d_input=state_dict["W_enc"].shape[0],
        d_hidden=state_dict["W_enc"].shape[1],
    )
    model.load_state_dict(state_dict)
    return model
