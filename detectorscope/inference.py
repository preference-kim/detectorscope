"""Layer-streamed Gemma-2 inference for one fixed logistic-regression head.

Uses the original extractor, head, feature descriptions and explanation code.
Only encoder columns with nonzero effective head weights are retained. Original
feature coordinates are restored after pooling, so the head's arithmetic and
feature identities stay unchanged. No SAE decoder is needed for inference.
"""

import json
from pathlib import Path

import huggingface_hub
import torch
from torch import nn

from detectorscope.model.detectors.auditable import AuditableDetector
from detectorscope.model.detectors.auditable import Detection
from detectorscope.model.detectors.auditable import Extraction
from detectorscope.model.detectors.heads.logreg import LogRegHead
from detectorscope.model.sae import JumpReLUSAE
from detectorscope.streaming import StreamingExtractor, TensorReader


DEFAULT_MODEL = "liamdugan/detector-scope"
DEFAULT_REVISION = "9baaff9e2d1cdb79cafa90cdb27811b847e69dae"


def resolve_device(device: str = "auto") -> str:
  if device == "auto":
    if torch.cuda.is_available():
      return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"
  if device == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError("MPS is unavailable in this Python process.")
  if device not in ("cpu", "mps", "cuda"):
    raise ValueError(f"Unsupported device: {device}")
  return device


class _SelectedEncoder(nn.Module):
  """An inference-only subset of a JumpReLU encoder, preserving its dtype."""

  def __init__(self, sae: JumpReLUSAE, indices: torch.Tensor):
    super().__init__()
    indices = indices.to(sae.W_enc.device)
    self.register_buffer("weight", sae.W_enc.detach()[:, indices].clone())
    self.register_buffer("bias", sae.b_enc.detach()[indices].clone())
    self.register_buffer("log_threshold", sae.log_threshold.detach()[indices].clone())
    self.register_buffer(
        "pre_bias", sae.b_dec.detach().clone() if sae.use_pre_enc_bias else None
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x.to(self.weight.dtype)
    if self.pre_bias is not None:
      x = x - self.pre_bias
    pre = torch.relu(x @ self.weight + self.bias)
    return pre * (pre > self.log_threshold.exp())

  @classmethod
  def from_checkpoint(cls, reader, prefix, selected):
    result = cls.__new__(cls)
    nn.Module.__init__(result)
    indices = selected.tolist()
    result.register_buffer("weight", reader.columns(prefix + "W_enc", indices))
    result.register_buffer("bias", reader.read(prefix + "b_enc")[selected].clone())
    result.register_buffer("log_threshold",
                            reader.read(prefix + "log_threshold")[selected].clone())
    # The exported AuditableDetector uses the default use_pre_enc_bias=False.
    result.register_buffer("pre_bias", None)
    return result


class OptimizedDetector:
  """Inference runner for a fixed Gemma-2/logreg detector configuration.

  from_pretrained reads embedding rows and streams one transformer block at a
  time, prefetching only the next CPU block. It never loads a whole backbone or
  SAE. The constructor taking an existing detector is retained for small-model
  equivalence fixtures and shares that object's extractor. Encoder math stays
  FP32; torch_dtype controls only the language model. Calls are serialized by
  the document service because extraction state is mutable.
  """

  def __init__(self, source: AuditableDetector, device: str = "auto"):
    head = source.load_head()
    if not isinstance(head, LogRegHead):
      raise ValueError("Compact inference requires a fixed logistic head.")
    if source.extractor is None or source.extractor.config.model_type != "gemma2":
      raise ValueError("Compact inference requires a Gemma-2 extractor.")
    if not source.hooks or len(source.hooks) != len(source.saes):
      raise ValueError("Each hook must have one SAE.")

    self.device = resolve_device(device)
    self.extractor = source.extractor
    self.hooks = list(source.hooks)
    self.encoders = nn.ModuleList()
    effective_weight = (head._weight * head.interp_mask).detach().cpu()
    self.input_dim = effective_weight.numel()
    feature_ids = []
    offset = 0
    for sae in source.saes:
      width = sae.W_enc.shape[1]
      selected = effective_weight[offset:offset + width].nonzero().flatten()
      self.encoders.append(_SelectedEncoder(sae, selected))
      feature_ids.append(selected + offset)
      offset += width
    if offset != self.input_dim:
      raise ValueError("SAE dimensions do not match the classifier.")
    self.feature_ids = torch.cat(feature_ids)
    self.retained_per_layer = [len(ids) for ids in feature_ids]
    self.encoders.to(self.device).eval()
    self.extractor.model.to(self.device).eval()
    self.extractor.device = self.device

    # A head-only instance reuses the original scoring/explanation contract.
    self.explainer = AuditableDetector(None, head=source.default[0],
                                      tau=source.default[1])
    self.explainer.add_head(head, *source.default)
    self.explainer.feature_table = source.feature_table
    self.identity = {"head": source.default[0], "tau": source.default[1],
                     "device": self.device,
                     "backbone_dtype": str(next(self.extractor.model.parameters()).dtype),
                     "retained_features": len(self.feature_ids),
                     "original_features": self.input_dim}

  @classmethod
  def from_pretrained(
      cls, model_id: str = DEFAULT_MODEL, *, revision: str = DEFAULT_REVISION,
      device: str = "auto", torch_dtype: str = "float32",
      local_files_only: bool = False, **kwargs,
  ) -> "OptimizedDetector":
    # Never call AuditableDetector.from_pretrained here: it loads all weights.
    root = Path(model_id) if Path(model_id).is_dir() else Path(
        huggingface_hub.snapshot_download(model_id, revision=revision,
                                          local_files_only=local_files_only))
    config = json.loads((root / "config.json").read_text())
    kind = kwargs.pop("head", config.get("head", "logreg"))
    tau = kwargs.pop("tau", config.get("tau", 0.85))
    attention = kwargs.pop("attention_implementation", "sdpa")
    compact_mlp = kwargs.pop("compact_mlp", True)
    compact_attention = kwargs.pop("compact_attention", False)
    if kwargs:
      raise TypeError(f"Unsupported loading options: {sorted(kwargs)}")
    if kind != "logreg":
      raise ValueError("Streaming inference requires a fixed logistic head.")
    head = LogRegHead(str(root / "heads" / f"logreg_tau{tau:.2f}.safetensors"))
    result = cls.__new__(cls)
    result.device = resolve_device(device)
    result.hooks = config["hook_points"]
    result.extractor = StreamingExtractor(root / "extractor", result.hooks,
        result.device, torch_dtype, attention, compact_mlp=compact_mlp,
        compact_attention=compact_attention)
    reader = TensorReader(root)
    weight = (head._weight * head.interp_mask).detach().cpu()
    result.input_dim = weight.numel()
    result.encoders = nn.ModuleList()
    ids, offset = [], 0
    for rank in range(len(result.hooks)):
      prefix = f"saes.{rank}."
      width = reader.shape(prefix + "W_enc")[1]
      selected = weight[offset:offset + width].nonzero().flatten()
      result.encoders.append(_SelectedEncoder.from_checkpoint(reader, prefix, selected))
      ids.append(selected + offset)
      offset += width
    if offset != result.input_dim:
      raise ValueError("SAE dimensions do not match the classifier.")
    result.feature_ids = torch.cat(ids)
    result.retained_per_layer = [len(x) for x in ids]
    result.encoders.to(result.device).eval()
    result.explainer = AuditableDetector(None, head=kind, tau=tau)
    result.explainer.add_head(head, kind, tau)
    keep = set(result.feature_ids.tolist())
    with (root / "features.jsonl").open() as file:
      result.explainer.feature_table = {record["id"]: record for line in file
          if (record := json.loads(line))["id"] in keep}
    result.identity = {"model": model_id, "revision": revision, "head": kind,
                       "tau": tau, "device": result.device,
                       "backbone_dtype": torch_dtype,
                       "compact_mlp": compact_mlp,
                       "compact_attention": compact_attention,
                       "retained_features": len(result.feature_ids),
                       "original_features": result.input_dim,
                       "weight_loading": (
                           f"one {result.device.upper()} compute block + one CPU prefetch block")}
    return result

  @torch.inference_mode()
  def extract_batch(
      self, texts: list[str], highlights: bool = False,
  ) -> list[Extraction]:
    if not texts:
      return []
    acts, mask = self.extractor.get_activations(
        texts, return_mask=True, on_device=True, stop_after_last_hook=True,
    )
    pooled = []
    token_features = []
    for hook, encoder in zip(self.hooks, self.encoders):
      features = encoder(acts[f"{hook}_input"].float()) * mask[:, :, None]
      pooled.append(features.amax(dim=1))
      if highlights:
        token_features.append(features.cpu())
    compact = torch.cat(pooled, dim=1).cpu()
    x = torch.zeros((len(texts), self.input_dim), dtype=compact.dtype)
    x[:, self.feature_ids] = compact
    # Sparse tensor construction stays on CPU; MPS need not support COO ops.
    token_features = torch.cat(token_features, dim=-1) if highlights else None
    valid = mask.cpu().bool()
    out = []
    for index, text in enumerate(texts):
      ex = Extraction(x[index:index + 1])
      if highlights:
        ex.tokens = self.extractor.get_token_strings(text)
        dense = token_features[index][valid[index]]
        coords = dense.nonzero().T
        values = dense[coords[0], coords[1]]
        coords[1] = self.feature_ids[coords[1]]
        ex.token_feats = torch.sparse_coo_tensor(
            coords, values, (dense.shape[0], self.input_dim), check_invariants=True
        ).coalesce()
        if dense.shape[0] != len(ex.tokens):
          raise RuntimeError("Tokenizer and feature sequence lengths differ.")
      out.append(ex)
    self.extractor.captured_activation = {}
    return out

  def __call__(
      self, texts: str | list[str], *, return_highlights: bool = False,
      batch_size: int = 1,
  ) -> Detection | list[Detection]:
    if batch_size < 1:
      raise ValueError("batch_size must be positive.")
    single = isinstance(texts, str)
    items = [texts] if single else list(texts)
    out = []
    for start in range(0, len(items), batch_size):
      extracted = self.extract_batch(items[start:start + batch_size],
                                     return_highlights)
      out.extend(self.explainer.explain(ex) for ex in extracted)
    return out[0] if single else out
