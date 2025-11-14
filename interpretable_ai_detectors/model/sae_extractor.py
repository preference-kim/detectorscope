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

"""Extracts activations from a pre-trained SAE model using sae_lens."""

import enum
import os

import sae_lens
import torch
import torch.nn.functional as F
from transformer_lens import hook_points


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class HookPointName(enum.StrEnum):
  """Class to keep track of the hook point names in sae-lens."""

  INPUT = "hook_sae_input"
  PRE = "hook_sae_acts_pre"
  POST = "hook_sae_acts_post"
  RECONS = "hook_sae_recons"
  OUTPUT = "hook_sae_output"


class SAEExtractor:
  """Extracts activations from a pre-trained SAE model.

  This class loads a specified SAE and a corresponding HookedSAETransformer
  model to extract activations at various hook points within the SAE.
  """

  def __init__(self):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    self.sae = sae_lens.SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_25/width_16k/canonical",
    ).to(self.device)

    self.neuronpedia_id = self.sae.cfg.metadata.neuronpedia_id
    self.model_name = self.sae.cfg.metadata.model_name
    self.prepend_bos = self.sae.cfg.metadata.prepend_bos

    self.model = sae_lens.HookedSAETransformer.from_pretrained(self.model_name)
    self.model = self.model.to(self.device).eval()

    self.model.cfg.n_ctx = 1024  # Set to avoid OOM
    self.captured_activation = {}

  def pad(self, activation: torch.Tensor):
    """Right pad a tensor up to model max ctx len."""
    pad_width = self.model.cfg.n_ctx - activation.shape[1]
    padding = (0, 0, 0, pad_width, 0, 0)
    return F.pad(input=activation, pad=padding, mode="constant", value=0)

  def _sae_hook(self, activations: torch.Tensor, hook: hook_points.HookPoint):
    """Hook function for use in run_with_hooks_with_saes."""
    self.captured_activation[hook.name.split(".")[-1]] = (
        activations.detach().cpu()
    )

  def get_activations(self, text: str) -> dict[torch.Tensor]:
    """Runs text through the LLM to extract token activations.

    Args:
        text (str): the text to extract activations for

    Returns:
        dictionary of activations indexed by hook point name with
        values being tensors of shape [1, seq_len, feature_dim].
    """
    self.captured_activation = {}

    fwd_hooks = [
        (f"{self.sae.cfg.metadata.hook_name}.{hook.value}", self._sae_hook)
        for hook in [HookPointName.INPUT, HookPointName.POST]
    ]

    self.model.run_with_hooks_with_saes(
        self.model.to_tokens(text, prepend_bos=self.prepend_bos),
        fwd_hooks=fwd_hooks,
        saes=self.sae,
        reset_saes_end=False,
    )
    return self.captured_activation
