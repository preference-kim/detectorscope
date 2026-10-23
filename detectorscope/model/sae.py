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

"""Implementation of JumpReLU Sparse Autoencoders.

This module contains the JumpReLUSAE class, which implements the JumpReLU SAE
as described in https://arxiv.org/abs/2407.14435, plus custom autograd
functions for the Heaviside step and JumpReLU activations that use
straight-through estimators for the threshold gradients. These are the
paper's SAEs: train_sae.py fine-tunes them from the Gemma Scope release
(from_gemma_scope) and AuditableDetector reloads the saved state dicts with
from_state_dict.
"""

import math

import huggingface_hub
import numpy as np
import torch
from torch import nn
from torch.autograd import function
import torch.nn.functional as F

# The Gemma Scope SAE behind each `layer_N/width_W/canonical` id the paper
# fine-tunes, as the sae_lens registry resolved it.
CANONICAL = {
    "layer_7/width_16k": "average_l0_69",
    "layer_13/width_16k": "average_l0_84",
    "layer_17/width_16k": "average_l0_77",
    "layer_22/width_16k": "average_l0_72",
}


def rectangle(x: torch.Tensor) -> torch.Tensor:
  """Rectangular kernel (1 on (-0.5, 0.5), else 0) for the STEs below."""
  return ((x > -0.5) & (x < 0.5)).to(x.dtype)


class HeavisideStep(torch.autograd.Function):
  """Custom autograd function for the Heaviside step function.

  This class implements a straight-through estimator for the Heaviside step
  function, where the forward pass is a standard step function and the
  backward pass uses a rectangular kernel for the gradient. Only the
  threshold receives a gradient, not x; this is what makes the L0 penalty in
  JumpReLUSAE.loss_fn trainable.
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
    """Returns 1 where x > threshold, else 0, in x's dtype."""
    ctx.save_for_backward(x, threshold)
    ctx.bandwidth = bandwidth
    ctx.acc = acc
    return (x > threshold).to(x.dtype)

  @classmethod
  def backward(
      cls, ctx: function.FunctionCtx, grad_output: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """KDE gradient for the threshold (scaled by acc); None for the rest."""
    x, threshold = ctx.saved_tensors  # pytype: disable=attribute-error
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
  and the backward pass passes the gradient through to x where the unit is
  on and uses a rectangular kernel for the gradient of the threshold.
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
    """Returns x where x > threshold, else 0."""
    ctx.save_for_backward(x, threshold)
    ctx.bandwidth = bandwidth
    ctx.acc = acc
    return x * (x > threshold)

  @classmethod
  def backward(
      cls, ctx: function.FunctionCtx, grad_output: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masked gradient for x; KDE gradient (scaled by acc) for threshold."""
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
  r"""Implementation of JumpReLU SAEs (https://arxiv.org/abs/2407.14435).

  JumpReLU SAEs swap the ReLU in the standard SAE for a JumpReLU function
  defined as $H(x - \theta) * x$ where H is the Heaviside step function.
  They also replace the L1 sparsity penalty with L0 and use straight-through
  estimators (STEs) to compute gradients for both the L0 and the JumpReLU
  threshold values.

  This implementation is a translation of the JAX pseudocode from Appendix J
  of the original paper into PyTorch. We include aspects such as the ReLU of
  pre-activations for maximum fidelity to the original implementation.
  Thresholds are stored as `log_threshold` so they stay positive.

  Can also be initialized with `from_gemma_scope()` (a Gemma Scope SAE on
  the Hub) or `from_state_dict()` (a file saved by train_sae.py). The
  training-only switches (`use_pre_enc_bias`, `C` with `rescale_to_unit_norm`)
  default to off; train_sae.py flips them on after construction.

  Args:
    d_input: Dimension of the input vector.
    d_hidden: Width of the SAE.
    bandwidth: Bandwidth of the KDE used in the STE gradients.
    acc: Constant multiplying the threshold gradients (an accelerator).
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

    self.use_pre_enc_bias = False
    self.is_unit_norm_scaled = False

  @property
  def device(self):
    """Device of the parameters (W_dec)."""
    return self.W_dec.device

  def initialize(self):
    """(Re)creates all parameters with the GemmaScope initialisation."""
    # Initialize W_dec weights with normalized He-uniform and W_enc weights
    # as W_dec.T following gemma scope https://arxiv.org/pdf/2408.05147
    self.W_dec = nn.Parameter(torch.randn(self.d_hidden, self.d_input))  # pylint: disable=invalid-name
    nn.init.kaiming_uniform_(self.W_dec)
    self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=1)
    self.W_enc = nn.Parameter(self.W_dec.data.T.clone())  # pylint: disable=invalid-name

    # Initialize bias vectors to zero & thresholds to exp(-3) ~= 0.05
    self.b_enc = nn.Parameter(torch.zeros(self.d_hidden))
    self.b_dec = nn.Parameter(torch.zeros(self.d_input))
    self.log_threshold = nn.Parameter(torch.ones(self.d_hidden) * -3.0)

  def reset_thresholds(self):
    """Resets every threshold to its initial value, keeping other weights."""
    self.log_threshold = nn.Parameter(torch.ones(self.d_hidden) * -3.0)

  def rescale_to_unit_norm(self):
    """Rescales the model by self.C for training on unit-RMS inputs.

    train_sae.py sets C to the hook point's RMS activation norm and divides
    every batch by it while `is_unit_norm_scaled` is set; this adjusts the
    biases and thresholds to match. Call undo_rescaling before saving.
    """
    if self.is_unit_norm_scaled:
      raise ValueError("Attempted to scale model that is already scaled.")

    self.log_threshold.data -= math.log(self.C)
    self.b_dec.data /= self.C
    self.b_enc.data /= self.C

    if self.use_pre_enc_bias:
      self.b_enc.data += self.b_dec @ self.W_enc

    self.is_unit_norm_scaled = True

  def undo_rescaling(self):
    """Rescales the model back to raw activation inputs and resets C to 1."""
    if not self.is_unit_norm_scaled:
      raise ValueError("Attempted to undo scaling on unscaled model.")

    if self.use_pre_enc_bias:
      self.b_enc.data -= self.b_dec @ self.W_enc

    self.log_threshold.data += math.log(self.C)
    self.b_enc.data *= self.C
    self.b_dec.data *= self.C

    self.C = 1.0
    self.is_unit_norm_scaled = False

  def encode(self, x: torch.Tensor) -> torch.Tensor:
    """Feature activations of x, [..., d_hidden]; no STE, for inference."""
    if self.use_pre_enc_bias:
      x = x - self.b_dec
    pre_activations = F.relu(x @ self.W_enc + self.b_enc)
    return pre_activations * (pre_activations > torch.exp(self.log_threshold))

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (reconstructed, features) for x of shape [..., d_input].

    Features are [..., d_hidden]. With use_pre_enc_bias, x is modified in
    place.
    """
    if self.use_pre_enc_bias:
      x -= self.b_dec
    pre_activations = F.relu(x @ self.W_enc + self.b_enc)
    features = JumpReLU.apply(
        pre_activations,
        torch.exp(self.log_threshold),
        self.bandwidth,
        self.acc,
    )
    reconstructed = features @ self.W_dec + self.b_dec
    return reconstructed, features

  def loss_fn(
      self,
      x: torch.Tensor,
      reconstructed: torch.Tensor,
      features: torch.Tensor,
      l0_lambda: float,
  ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Per-example losses and statistics for one batch.

    Args:
      x: SAE input (already divided by C when the model is rescaled).
      reconstructed: First output of forward.
      features: Second output of forward.
      l0_lambda: Weight of the L0 penalty.

    Returns:
      (loss, stats), each mapping a name to a [batch] tensor. loss has "l0"
      and "mse"; stats has the unweighted "l0" count (via the Heaviside STE,
      so it carries a threshold gradient).
    """
    loss = {}
    stats = {}

    stats["l0"] = torch.sum(
        HeavisideStep.apply(
            features, torch.exp(self.log_threshold), self.bandwidth, self.acc
        ),
        dim=-1,
    )

    loss["l0"] = l0_lambda * stats["l0"]
    loss["mse"] = torch.sum((x - reconstructed) ** 2, dim=-1)

    return loss, stats

  @classmethod
  def from_gemma_scope(
      cls,
      sae_id: str,
      repo: str = "google/gemma-scope-2b-pt-res",
      **kwargs,
  ):
    """Downloads a Gemma Scope SAE; kwargs go to __init__.

    Args:
      sae_id: `layer_N/width_W/average_l0_K`, or `layer_N/width_W/canonical`
        for the ids in CANONICAL.
      repo: Hub repo holding one `params.npz` per SAE.
    """
    prefix, name = sae_id.rsplit("/", 1)
    if name == "canonical":
      name = CANONICAL[prefix]
    path = huggingface_hub.hf_hub_download(repo, f"{prefix}/{name}/params.npz")
    params = {k: torch.from_numpy(v) for k, v in np.load(path).items()}
    params["log_threshold"] = torch.log(params.pop("threshold"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = cls(*params["W_enc"].shape, **kwargs)
    model.load_state_dict(params)
    return model.to(device)

  @classmethod
  def from_state_dict(cls, state_dict_path: str):
    """Loads a state dict saved by train_sae.py, sizing the model from W_enc.

    Uses the default bandwidth and acc, which only matter for training.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(state_dict_path, map_location=device)
    model = cls(
        d_input=state_dict["W_enc"].shape[0],
        d_hidden=state_dict["W_enc"].shape[1],
    )
    model.load_state_dict(state_dict)
    return model.to(device)
