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

"""Hugging Face model extractor for hidden-state activations.

Wraps a transformers model and its tokenizer so that the residual stream
entering any named module can be captured with forward hooks. The SAEs in this
project read block inputs, hence the `<module>_input` naming of hook points.
"""

import contextlib
import functools
import re

import torch
import transformers


class _ActivationsComplete(Exception):
  """Internal early return after the last requested block input is captured."""


class HFExtractor:
  """Hugging Face model extractor for hidden-state activations.

  Hooks a Hugging Face transformer and returns the activations entering the
  requested modules. Text is truncated to 1024 tokens everywhere, so
  get_token_strings and get_activations always agree on sequence length.

  Args:
    model_path: Hugging Face model id or local path.
    default_hooks: module names to hook when a call passes none; None or
      empty hooks every named module, which is only sensible for inspection.
    tokenizer_only: skip loading the model; only the tokenizer methods work.
    torch_dtype: dtype to load the model in; None keeps the checkpoint's.
  """

  def __init__(
      self,
      model_path: str = "google/gemma-2-2b",
      default_hooks: list[str] | None = None,
      tokenizer_only: bool = False,
      torch_dtype: str | None = None,
  ):
    self.tokenizer_only = tokenizer_only
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

    # Ensure the tokenizer has a pad token for batching.
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token

    # If loading in tokenizer only mode, end here. Do not load model
    if self.tokenizer_only:
      print(f"Loading extractor {model_path} in tokenizer only mode")
      return

    # Ensure that we're loading the model as the correct class
    self.config = transformers.AutoConfig.from_pretrained(model_path)
    if self.config.architectures:
      model_class = getattr(transformers, self.config.architectures[0])
    else:
      model_class = transformers.AutoModel
    self.model = model_class.from_pretrained(model_path, dtype=torch_dtype)

    self.model = self.model.to(self.device)
    self.captured_activation = {}
    self.default_hooks = default_hooks

    # If default hooks is None, use all possible hooks in the model
    if not self.default_hooks:
      self.default_hooks = [
          name for name, _ in self.model.named_modules() if name
      ]

  def string_to_module(self, name: str) -> torch.nn.Module:
    """Resolves a dotted module name such as `model.layers.8` on the model."""
    return functools.reduce(getattr, name.split("."), self.model)

  @contextlib.contextmanager
  def apply_hooks(
      self, hooks: list[str] | None, detach=True, to_cpu=True,
      stop_after_last_hook: bool = False,
  ):
    """Registers forward hooks on the named modules for the enclosed block.

    Each hook stores the module's first positional input in
    `self.captured_activation[f"{name}_input"]`; module outputs are not kept.
    The hooks are removed on exit.

    Args:
      hooks: module names to hook; None or empty uses `self.default_hooks`.
      detach: detach captured tensors from the autograd graph.
      to_cpu: move captured tensors to the cpu as they are captured.
      stop_after_last_hook: capture Gemma-2 block inputs with pre-hooks and
        stop before executing the last requested block. Inference only.

    Raises:
      ValueError: if the extractor was loaded in tokenizer-only mode.
    """
    if self.tokenizer_only:
      raise ValueError("Extractor model loaded in tokenizer only mode.")

    if not hooks:
      hooks = self.default_hooks

    stop_at = None
    if stop_after_last_hook:
      if self.model.training:
        raise ValueError("Early extraction requires model.eval().")
      if self.config.model_type != "gemma2" or not hooks or any(
          re.fullmatch(r"model\.layers\.\d+", name) is None for name in hooks
      ):
        raise ValueError("Early extraction supports Gemma-2 block inputs only.")
      stop_at = max(hooks, key=lambda name: int(name.rsplit(".", 1)[1]))

    handles = []
    try:
      for hook_name in hooks:
        def capture(input_embed, hook=hook_name):
          if detach:
            input_embed = input_embed.detach()
          if to_cpu:
            input_embed = input_embed.cpu()
          self.captured_activation[hook + "_input"] = input_embed

        module = self.string_to_module(hook_name)
        if stop_after_last_hook:
          def pre_hook(module, args, kwargs, hook=hook_name, save=capture):
            del module
            save(args[0] if args else kwargs["hidden_states"])
            if hook == stop_at:
              raise _ActivationsComplete
          handles.append(module.register_forward_pre_hook(
              pre_hook, with_kwargs=True
          ))
        else:
          def post_hook(module, args, output, save=capture):
            del module, output
            save(args[0] if args else torch.tensor(args))
          handles.append(module.register_forward_hook(post_hook))
      yield
    finally:
      for handle in handles:
        handle.remove()

  def _get_token_ids(self, text: str) -> list[int]:
    """Converts a text into token ids, truncated to 1024 tokens."""
    return self.tokenizer.encode(text, max_length=1024, truncation=True)

  def get_token_strings(self, text: str) -> list[str]:
    """Splits a text into the exact substring each token covers.

    Uses the tokenizer's offset mapping rather than decoding ids, so each
    fragment is a literal slice of the input and highlights land on the
    original spelling.
    """
    encoding = self.tokenizer(
        text, max_length=1024, truncation=True, return_offsets_mapping=True
    )
    return [text[start:end] for start, end in encoding["offset_mapping"]]

  def get_activations(
      self,
      text: str | list[str],
      hooks: list[str] | None = None,
      return_mask: bool = False,
      on_device: bool = False,
      stop_after_last_hook: bool = False,
  ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Runs text through the LLM to extract token activations.

    Args:
      text: the text(s) to extract activations for; a list is padded into
        one batch.
      hooks: module names to extract from; None uses the default hooks.
      return_mask: also return the attention mask, needed to ignore padding
        in batched extraction.
      on_device: keep extracted tensors on the model's device instead of
        moving them to the cpu.
      stop_after_last_hook: skip the unused tail and output head (Gemma-2).

    Returns:
      `{f"{hook}_input": tensor}` with tensors of shape
      [batch, seq_len, d_model], or `(activations, mask)` with the mask of
      shape [batch, seq_len] when return_mask is set.

    Raises:
      ValueError: if the extractor was loaded in tokenizer-only mode.
    """
    if self.tokenizer_only:
      raise ValueError("Extractor model loaded in tokenizer only mode")

    self.captured_activation = {}
    is_batch = isinstance(text, (list, tuple))

    with self.apply_hooks(
        hooks, to_cpu=(not on_device),
        stop_after_last_hook=stop_after_last_hook,
    ):
      inputs = self.tokenizer(
          text,
          padding=is_batch,
          max_length=1024,
          truncation=True,
          return_tensors="pt",
      ).to(self.device)

      kwargs = {"use_cache": False} if stop_after_last_hook else {}
      try:
        self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"], **kwargs
        )
      except _ActivationsComplete:
        if not stop_after_last_hook:
          raise

    if stop_after_last_hook:
      expected = {f"{hook}_input" for hook in (hooks or self.default_hooks)}
      if set(self.captured_activation) != expected:
        raise RuntimeError("Not all requested block inputs were captured.")

    if return_mask:
      mask = inputs["attention_mask"]
      return self.captured_activation, mask if on_device else mask.cpu()

    return self.captured_activation
