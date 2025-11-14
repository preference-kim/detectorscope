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

"""Hugging Face model extractor for activations and gradient-based saliency."""

import contextlib
import functools

import torch
import transformers


class HFExtractor:
  """Hugging Face model extractor for activations and gradient-based saliency.

  This class provides methods to extract activations and compute gradient-based
  saliency maps from Hugging Face transformer models.
  """

  def __init__(
      self,
      model_path: str = "google/gemma-2-2b",
      default_hooks: list[str] = None,
  ):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    self.model = transformers.AutoModel.from_pretrained(model_path)
    self.model = self.model.to(self.device)
    self.captured_activation = {}
    self.default_hooks = default_hooks

    # If default hooks is None, use all possible hooks in the model
    if not self.default_hooks:
      self.default_hooks = [
          name for name, _ in self.model.named_modules() if name
      ]

  def string_to_module(self, name: str) -> torch.nn.Module:
    return functools.reduce(getattr, name.split("."), self.model)

  @contextlib.contextmanager
  def apply_hooks(self, hooks: list[str] | None, detach=True):
    """Applies forward hooks to the specified modules.

    This context manager registers forward hooks on the modules specified by
    `hooks`. The hooks capture the input and output of each module and store
    them in `self.captured_activation`. The hooks are automatically removed
    when exiting the context.

    Args:
      hooks: A list of module names (strings) to apply hooks to. If None,
        `self.default_hooks` will be used.
      detach: Whether to detach the captured tensors from the computation graph.

    Yields:
      None. The context manager yields control to the enclosed block.
    """
    if not hooks:
      hooks = self.default_hooks

    handles = []
    for hook_name in hooks:
      # NOTE: We re-define this fn within the loop to allow each hook_fn
      #       to have a different default hook_name argument
      def hook_fn(module, input_embed, output_embed, hook=hook_name):
        del module, output_embed  # Unused.
        if isinstance(input_embed, tuple):
          input_embed = (
              input_embed[0] if input_embed else torch.tensor(input_embed)
          )
        if detach:
          input_embed = input_embed.detach()
        self.captured_activation[hook + "_input"] = input_embed.cpu()

      if module := self.string_to_module(hook_name):
        handles.append(module.register_forward_hook(hook_fn))

    try:
      yield
    finally:
      for handle in handles:
        handle.remove()

  def get_activations(
      self, text: str, hooks: list[str] = None
  ) -> dict[torch.Tensor]:
    """Runs text through the LLM to extract token activations.

    Args:
      text (str): the text to extract activations for
      hooks (list[str]): the hook points to extract from

    Returns:
      dictionary of activations indexed by hook point name with
      values being tensors of shape [1, seq_len, feature_dim].
    """
    self.captured_activation = {}

    with self.apply_hooks(hooks):
      inputs = self.tokenizer.encode(
          text, max_length=1024, truncation=True, return_tensors="pt"
      )
      self.model(inputs.to(self.device))

    return self.captured_activation

  def get_gradient_based_saliency(
      self, text: str, sae: torch.nn.Module, hooks: list[str] | None = None
  ) -> tuple[list[str], torch.Tensor, dict[int, torch.tensor]]:
    """Computes gradients of token embeddings w.r.t each SAE feature.

    This borrows the method from SAE ARENA (Sec. 1.3.2) where we
    initialize a vector of scalars s and compute gradients of
    each SAE feature w.r.t. these scalars at the point s = [1, 1, ...].

    Args:
      text (str): the text to extract activations for
      sae (nn.Module): the sae model to use for decomposition
      hooks (list[str]): the hook points to extract from

    Returns:
      tuple of (tokens, features, saliency_map) where features is the SAE
      decomposition and saliency_map is a mapping from SAE feature ID
      to tensor of shape [seq_len] containing the token gradient values
    """
    if not hooks:
      hooks = self.default_hooks

    self.model.requires_grad_(False)
    self.captured_activation = {}
    saliency_map = {}

    # Tokenize the input
    inputs = self.tokenizer.encode(
        text, max_length=1024, truncation=True, return_tensors="pt"
    ).to(self.device)

    # Get token embeddings and initialize token scalars
    embed = self.model.get_input_embeddings()(inputs).to(self.device)
    s = torch.ones(embed.shape[:2], device=self.device, requires_grad=True)

    # Run inference on the scaled embeddings
    with self.apply_hooks(hooks, detach=False):
      scaled_embed = (s.unsqueeze(-1) * embed).to(self.device)
      self.model(inputs_embeds=scaled_embed)

    # Get SAE features of the input
    _, features = sae.forward(
        self.captured_activation["norm_input"][0, -1, :].to(self.device)
    )

    # Backprop once for each active SAE feature to compute salience
    for feature_id in features.nonzero().flatten():
      if s.grad is not None:
        s.grad.zero_()
      features[feature_id].backward(retain_graph=True)
      saliency_map[feature_id] = s.grad.clone().flatten()

    # NOTE: We pass in inputs.T here since inputs is a Tensor of shape
    # [1, seq_len] rather than [seq_len]
    tokens = self.tokenizer.convert_ids_to_tokens(inputs.T)

    return tokens, features.detach(), saliency_map
