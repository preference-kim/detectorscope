"""Bounded disk -> CPU -> GPU weight pipeline for Gemma-2 extraction.

No full-model state dict, embedding table, or mmap of a model shard is kept.
One CPU loader thread prepares at most the next block while the current block
executes. A synchronization boundary protects the reusable CPU staging buffer
until GPU copies complete. Only the current GPU block is retained; the CPU
staging buffer is released after each forward. OS file caching remains reclaimable by the OS.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import re
import struct
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer, Gemma2RotaryEmbedding

from detectorscope.model.extractor import HFExtractor


class TensorReader:
  """Read individual safetensors ranges into owned, bounded CPU arrays."""

  def __init__(self, root, filename=None):
    self.root = Path(root)
    index = self.root / "model.safetensors.index.json"
    if filename:
      files = [filename]
      weight_map = None
    elif index.is_file():
      weight_map = json.loads(index.read_text())["weight_map"]
      files = sorted(set(weight_map.values()))
    else:
      files = ["model.safetensors"]
      weight_map = None
    self.tensors = {}
    for name in files:
      path = self.root / name
      with path.open("rb") as file:
        size = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(size))
      for key, spec in header.items():
        if key != "__metadata__" and (weight_map is None or weight_map.get(key) == name):
          self.tensors[key] = (path, 8 + size + spec["data_offsets"][0], spec)

  def shape(self, key):
    return self.tensors[key][2]["shape"]

  @staticmethod
  def _numpy_dtype(spec):
    return {"F32": np.dtype("<f4"), "F16": np.dtype("<f2"),
            "BF16": np.dtype("<u2")}[spec["dtype"]]

  @staticmethod
  def _torch(array, spec):
    tensor = torch.from_numpy(array)
    return tensor.view(torch.bfloat16) if spec["dtype"] == "BF16" else tensor

  def read(self, key):
    path, offset, spec = self.tensors[key]
    count = int(np.prod(spec["shape"]))
    with path.open("rb") as file:
      file.seek(offset)
      array = np.fromfile(file, dtype=self._numpy_dtype(spec), count=count)
    if array.size != count:
      raise IOError(f"Incomplete tensor: {key}")
    return self._torch(array.reshape(spec["shape"]), spec)

  @contextmanager
  def staging(self, enabled):
    """Own one reusable CPU read buffer until all prefetch work has joined.

    Reuse requires the caller to finish the GPU copy before submitting the
    next read. CPU execution must not reuse a buffer that it is still reading.
    """
    self._reuse_buffer = enabled
    self._buffer = None
    self.read_stats = {"bytes": 0, "read_calls": 0, "allocations": 0,
                       "reuses": 0, "max_staging_bytes": 0}
    try:
      yield
    finally:
      self._buffer = None
      self._reuse_buffer = False

  def read_many(self, keys):
    """Coalesce adjacent requested tensors, without reading gaps or other layers."""
    groups = []
    for key in sorted(keys, key=lambda k: (str(self.tensors[k][0]), self.tensors[k][1])):
      path, offset, spec = self.tensors[key]
      size = int(np.prod(spec["shape"])) * self._numpy_dtype(spec).itemsize
      if (groups and groups[-1][0] == path and groups[-1][2] == offset
          and groups[-1][3] == spec["dtype"]):
        groups[-1][2] += size
        groups[-1][4].append(key)
      else:
        groups.append([path, offset, offset + size, spec["dtype"], [key]])
    result = {}
    for path, start, end, _, names in groups:
      spec = self.tensors[names[0]][2]
      dtype = self._numpy_dtype(spec)
      size = (end - start) // dtype.itemsize
      array = None
      if len(groups) == 1 and getattr(self, "_reuse_buffer", False):
        array = self._buffer
        if array is not None and (array.size != size or array.dtype != dtype):
          self._buffer = array = None
      else:
        # Fragmented or sharded blocks must not retain an extra cached block.
        self._buffer = None
      reused = array is not None
      if array is None:
        array = np.empty(size, dtype=dtype)
      if len(groups) == 1 and getattr(self, "_reuse_buffer", False):
        self._buffer = array
      view = memoryview(array).cast("B")
      with path.open("rb", buffering=0) as file:
        file.seek(start)
        done = 0
        while done < len(view):
          count = file.readinto(view[done:])
          if not count:
            raise IOError(f"Incomplete tensor group: {names[0]}")
          done += count
          if hasattr(self, "read_stats"):
            self.read_stats["read_calls"] += 1
      if hasattr(self, "read_stats"):
        self.read_stats["bytes"] += end - start
        self.read_stats["allocations"] += int(not reused)
        self.read_stats["reuses"] += int(reused)
      flat = self._torch(array, spec)
      for key in names:
        _, offset, spec = self.tensors[key]
        count = int(np.prod(spec["shape"]))
        result[key] = flat.narrow(0, (offset - start) // dtype.itemsize, count).view(spec["shape"])
    if hasattr(self, "read_stats"):
      self.read_stats["max_staging_bytes"] = max(
          self.read_stats["max_staging_bytes"], sum(g[2] - g[1] for g in groups))
    return result

  def rows(self, key, indices):
    path, offset, spec = self.tensors[key]
    height, width = spec["shape"]
    dtype = self._numpy_dtype(spec)
    out = np.empty((len(indices), width), dtype=dtype)
    with path.open("rb") as file:
      for row, index in enumerate(indices):
        if not 0 <= index < height:
          raise ValueError("Embedding index outside checkpoint vocabulary.")
        file.seek(offset + index * width * dtype.itemsize)
        values = np.fromfile(file, dtype=dtype, count=width)
        if len(values) != width:
          raise IOError(f"Incomplete row: {key}")
        out[row] = values
    return self._torch(out, spec)

  def columns(self, key, indices):
    # A row is at most 64 KiB for the released SAE. Scanning one row at a
    # time avoids materializing its 144 MiB encoder to retain ~1 MiB of it.
    path, offset, spec = self.tensors[key]
    height, width = spec["shape"]
    dtype = self._numpy_dtype(spec)
    out = np.empty((height, len(indices)), dtype=dtype)
    with path.open("rb") as file:
      file.seek(offset)
      for row in range(height):
        values = np.fromfile(file, dtype=dtype, count=width)
        if len(values) != width:
          raise IOError(f"Incomplete row: {key}")
        out[row] = values[indices]
    return self._torch(out, spec)


class _UnpaddedMLP(torch.nn.Module):
  """Run the token-local MLP only on valid rows, restoring the batch layout."""

  def __init__(self, mlp, indices):
    super().__init__()
    self.inner = mlp
    self.register_buffer("indices", indices, persistent=False)
    self.train(mlp.training)

  def forward(self, hidden):
    flat = hidden.flatten(0, 1)
    values = self.inner(flat.index_select(0, self.indices))
    # Padding never contributes to a valid query or feature pooling. Its MLP
    # output can be zero; retain the original layout for attention and hooks.
    return torch.zeros_like(flat).index_copy_(0, self.indices, values).view_as(hidden)


class _UnpaddedAttention(torch.nn.Module):
  """Share attention weights while computing each independent valid context."""

  def __init__(self, attention, spans):
    super().__init__()
    self.inner, self.spans = attention, spans
    self.train(attention.training)

  def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
    output = torch.zeros_like(hidden_states)
    for batch, (start, end) in enumerate(self.spans):
      if start == end:
        continue
      embeddings = tuple(value[batch:batch + 1, start:end] if value.shape[0] > 1
                         else value[:, start:end] for value in position_embeddings)
      mask = attention_mask
      if mask is not None:
        mask = mask[batch:batch + 1] if mask.shape[0] > 1 else mask
        mask = mask[..., start:end, start:end]
      positions = dict(kwargs)
      if positions.get("position_ids") is not None:
        positions["position_ids"] = positions["position_ids"][..., start:end]
      if positions.get("cache_position") is not None:
        positions["cache_position"] = positions["cache_position"][start:end]
      values, _ = self.inner(hidden_states=hidden_states[batch:batch + 1, start:end],
          position_embeddings=embeddings, attention_mask=mask, **positions)
      output[batch:batch + 1, start:end] = values
    return output, None


class StreamingExtractor(HFExtractor):
  """The HF extractor's tokenizer contract with bounded Gemma block loading."""

  def __init__(self, model_path, hooks, device="cpu", torch_dtype="float32",
               attention_implementation="sdpa", compact_mlp=True, compact_attention=False):
    self.config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if self.config.model_type != "gemma2":
      raise ValueError("Streaming extraction supports Gemma-2 only.")
    if not hooks or any(re.fullmatch(r"model\.layers\.\d+", h) is None for h in hooks):
      raise ValueError("Streaming requires named Gemma-2 block inputs.")
    self.config._attn_implementation = attention_implementation
    self.compact_mlp = compact_mlp
    self.compact_attention = compact_attention
    self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token
    # This extractor has no generation/decode step. Right padding keeps all
    # padding after valid queries, so causal attention cannot expose padding to
    # valid tokens. The original mask still excludes padding from SAE pooling.
    self.tokenizer.padding_side = "right"
    self.reader = TensorReader(model_path)
    self.device = device
    if device == "mps":
      # This process owns one inference worker. Bound allocator growth as well
      # as live weights; exceeding the budget fails instead of causing swap.
      torch.mps.set_per_process_memory_fraction(
          min(1.0, 2 * 2**30 / torch.mps.recommended_max_memory())
      )
    self.dtype = getattr(torch, torch_dtype)
    self.default_hooks = list(hooks)
    self.hook_layers = {int(h.rsplit(".", 1)[1]): h for h in hooks}
    self.stop_layer = max(self.hook_layers)
    if not 0 <= self.stop_layer < self.config.num_hidden_layers:
      raise ValueError("Hook outside the checkpoint's layer range.")
    self.rotary = Gemma2RotaryEmbedding(self.config, device=device).eval()
    self.captured_activation = {}
    self.tokenizer_only = False
    self.profile = {}

  def _load_layer(self, index):
    started = time.perf_counter()
    prefix = f"model.layers.{index}."
    # Meta construction creates no random CPU/GPU parameter storage.
    with torch.device("meta"):
      layer = Gemma2DecoderLayer(self.config, index)
    keys = [key for key in self.reader.tensors if key.startswith(prefix)]
    coalesced = (self.device == "mps" and self.dtype == torch.float32
                 and all(self.reader.tensors[key][2]["dtype"] == "F32" for key in keys))
    if coalesced:
      state = {key[len(prefix):]: value for key, value in self.reader.read_many(keys).items()}
    else:
      state = {key[len(prefix):]: self.reader.read(key).to(self.dtype) for key in keys}
    layer.load_state_dict(state, strict=True, assign=True)
    layer.eval().requires_grad_(False)
    weight_bytes = sum(p.numel() * p.element_size() for p in layer.parameters())
    return layer, weight_bytes, time.perf_counter() - started

  @contextmanager
  def _weight_staging(self):
    with self.reader.staging(self.device == "mps" and self.dtype == torch.float32):
      try:
        yield
      finally:
        # Drain queued copies on exceptions too, before releasing NumPy-backed
        # storage: its completion handlers may acquire the GIL.
        if self.device == "mps" and self.dtype == torch.float32:
          self._sync()

  def _transfer_layer(self, layer):
    if self.device == "mps" and self.dtype == torch.float32:
      # Keep the original GPU parameter layout. Queue all parameter copies,
      # then let the caller's one block fence protect the CPU prefetch slot.
      layer.to(self.device, non_blocking=True)
    else:
      layer.to(self.device)

  def _sync(self):
    if self.device == "mps":
      torch.mps.synchronize()
    elif self.device == "cuda":
      torch.cuda.synchronize()

  def _release_cache(self):
    if self.device == "mps":
      torch.mps.empty_cache()
    elif self.device == "cuda":
      torch.cuda.empty_cache()

  @torch.inference_mode()
  def get_activations(self, text, hooks=None, return_mask=False, on_device=False,
                      stop_after_last_hook=True):
    forward_start = time.perf_counter()
    if hooks is not None and list(hooks) != self.default_hooks:
      raise ValueError("Streaming hooks are fixed when the runner is loaded.")
    inputs = self.tokenizer(text, padding=isinstance(text, (list, tuple)),
                            max_length=1024, truncation=True, return_tensors="pt")
    ids = inputs["input_ids"]
    unique, inverse = ids.flatten().unique(sorted=True, return_inverse=True)
    rows = self.reader.rows("model.embed_tokens.weight", unique.tolist())
    hidden = rows[inverse].reshape(*ids.shape, self.config.hidden_size)
    hidden = hidden.to(self.device, dtype=self.dtype)
    del rows
    mask = inputs["attention_mask"].to(self.device)
    real_tokens = int(inputs["attention_mask"].sum())
    mlp_indices = None
    # A one-row reduction did not shrink the model-shaped GEMM enough to pay
    # for gather/scatter. Keep the dense path for fewer than 64 padding rows.
    if self.compact_mlp and ids.numel() - real_tokens >= 64:
      # Derive indices on CPU once per forward, avoiding a GPU nonzero/sync
      # inside every block. This works with both left and right padding.
      mlp_indices = inputs["attention_mask"].flatten().nonzero().flatten().to(self.device)
    attention_spans = None
    if self.compact_attention and real_tokens < ids.numel():
      attention_spans = []
      for row in inputs["attention_mask"]:
        valid = row.nonzero().flatten()
        attention_spans.append((int(valid[0]), int(valid[-1]) + 1) if len(valid) else (0, 0))
    positions = torch.arange(ids.shape[1], device=self.device)
    position_ids = positions.unsqueeze(0)
    implicit_padding = (self.config._attn_implementation == "sdpa"
                        and self.tokenizer.padding_side == "right")
    kwargs = dict(config=self.config, input_embeds=hidden,
                  attention_mask=None if implicit_padding else mask,
                  cache_position=positions, past_key_values=None,
                  # These windows are not packed sequences. With a None mask,
                  # HF interprets supplied positions as packing metadata and
                  # disables implicit causality. RoPE still uses position_ids.
                  position_ids=None if implicit_padding else position_ids)
    masks = {"full_attention": create_causal_mask(**kwargs),
             "sliding_attention": create_sliding_window_causal_mask(**kwargs)}
    position_embeddings = self.rotary(hidden, position_ids)
    normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden.dtype)
    hidden = hidden * normalizer
    acts = {}
    profile = {"max_cpu_prefetch_blocks": 1,
               "max_gpu_blocks": int(self.device in ("mps", "cuda")),
               "max_cpu_compute_blocks": int(self.device == "cpu"),
               "batch_size": ids.shape[0], "sequence_length": ids.shape[1],
               "padding_side": self.tokenizer.padding_side,
               "real_tokens": real_tokens,
               "padded_tokens": ids.numel(),
               "mlp_tokens": real_tokens if mlp_indices is not None else ids.numel(),
               "attention_tokens": real_tokens if attention_spans is not None else ids.numel(),
               "prepare_seconds": time.perf_counter() - forward_start,
               "attention_masks": {kind: None if value is None else
                   {"shape": list(value.shape), "dtype": str(value.dtype)}
                   for kind, value in masks.items()},
               "embedding_rows_read": len(unique), "layer_weight_bytes": 0,
               "max_mps_driver_bytes": 0, "stages": []}
    with (
        self._weight_staging(),
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="weights") as pool,
    ):
      future = pool.submit(self._load_layer, 0) if self.stop_layer else None
      try:
        for index in range(self.stop_layer):
          if index in self.hook_layers:
            acts[f"{self.hook_layers[index]}_input"] = hidden
          wait_start = time.perf_counter()
          layer, weight_bytes, load_seconds = future.result()
          wait_seconds = time.perf_counter() - wait_start
          future = None  # Do not keep a Future owning the previous GPU block.
          transfer_start = time.perf_counter()
          self._transfer_layer(layer)
          if mlp_indices is not None:
            layer.mlp = _UnpaddedMLP(layer.mlp, mlp_indices)
          if attention_spans is not None:
            layer.self_attn = _UnpaddedAttention(layer.self_attn, attention_spans)
          self._sync()
          transfer_seconds = time.perf_counter() - transfer_start
          # The GPU copy is complete: the sole CPU staging buffer can now be
          # overwritten by prefetch while this block computes on the GPU.
          if index + 1 < self.stop_layer:
            future = pool.submit(self._load_layer, index + 1)
          compute_start = time.perf_counter()
          hidden = layer(hidden, position_embeddings=position_embeddings,
                         attention_mask=masks[layer.attention_type],
                         position_ids=position_ids, past_key_values=None,
                         output_attentions=False, use_cache=False,
                         cache_position=positions)[0]
          self._sync()
          compute_seconds = time.perf_counter() - compute_start
          profile["layer_weight_bytes"] = max(profile["layer_weight_bytes"], weight_bytes)
          if self.device == "mps":
            profile["max_mps_driver_bytes"] = max(profile["max_mps_driver_bytes"],
                                                  torch.mps.driver_allocated_memory())
          release_start = time.perf_counter()
          del layer
          self._release_cache()
          release_seconds = time.perf_counter() - release_start
          profile["stages"].append({"layer": index, "load_seconds": load_seconds,
                                     "wait_seconds": wait_seconds,
                                     "transfer_seconds": transfer_seconds,
                                     "compute_seconds": compute_seconds,
                                     "release_seconds": release_seconds})
      finally:
        if future is not None:
          future.cancel()
    profile["coalesced_weight_io"] = dict(self.reader.read_stats)
    acts[f"{self.hook_layers[self.stop_layer]}_input"] = hidden
    profile["forward_seconds"] = time.perf_counter() - forward_start
    self.profile = profile
    self.captured_activation = acts
    if not on_device:
      acts = {key: value.cpu() for key, value in acts.items()}
    return (acts, mask if on_device else mask.cpu()) if return_mask else acts
