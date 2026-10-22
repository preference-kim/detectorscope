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

"""The auditable detector: a language model, JumpReLU SAEs and a head.

AuditableDetector hooks a language model, encodes the residual stream entering
each hooked block with a JumpReLU SAE, max-pools the features over the
document and scores the pooled vector with a DetectorHead. Through
PyTorchModelHubMixin, `from_pretrained` loads a Hub repo or an exported
directory (config.json, model.safetensors with the SAEs, one head file per
(type, tau) under heads/, features.jsonl, and the language model under
extractor/); `from_paths` loads the paper's artifacts on disk. Selected with
`-dt auditable`.
"""

import dataclasses
import json
import os
import pathlib
import re
from typing import NamedTuple

import huggingface_hub
import torch
from torch import nn

from detectorscope.model.detector import Detector
from detectorscope.model.detectors.heads.head import DetectorHead
from detectorscope.model.extractor import HFExtractor
from detectorscope.model.sae import JumpReLUSAE

HEAD_EXT = {"logreg": "safetensors", "xgboost": "json"}
_HEAD_FILE = re.compile(r"(logreg|xgboost)_tau(\d\.\d\d)\.(safetensors|json)")


def head_file(head: str, tau: float) -> str:
  """Path of the (head, tau) file inside a repo."""
  return f"heads/{head}_tau{tau:.2f}.{HEAD_EXT[head]}"


def list_heads(root: str) -> list[tuple[str, float]]:
  """The (head, tau) pairs whose files a repo directory holds, sorted."""
  names = os.listdir(os.path.join(root, "heads"))
  return sorted(
      (m[1], float(m[2])) for n in names if (m := _HEAD_FILE.fullmatch(n))
  )


REPR_FEATURES = 3  # features shown by print(detection)


class Feature(NamedTuple):
  """One feature's evidence for a document."""

  id: int  # head coordinate: layer_rank * d_sae + index within the SAE
  importance: float  # zero-ablation effect on the logit; positive pushes to AI
  description: str | None  # terse auto-interp description, when known
  highlights: list[tuple[int, float]] | None  # (token index, activation)

  def __repr__(self) -> str:
    return (
        f"Feature(importance={self.importance:+.2f},"
        f" description={self.description!r})"
    )


@dataclasses.dataclass
class Extraction:
  """SAE activations of one document, before any head is applied."""

  x: torch.Tensor  # [1, d_sae * n_hooks], cpu: the head's input
  tokens: list[str] | None = None  # the exact substring of each token
  token_feats: torch.Tensor | None = None  # sparse [seq, d_sae * n_hooks], cpu


@dataclasses.dataclass(repr=False)
class Detection:
  """One document scored by one head.

  Prints as the label, P(AI) and the top REPR_FEATURES features, one per
  line; `features` holds every active one.
  """

  logit: float
  bias: float  # logit of an empty document (intercept / base margin)
  label: str  # "ai" or "human" at the head's threshold
  features: list[Feature]  # non-zero importance, |importance| descending
  tokens: list[str] | None = None

  @property
  def p_ai(self) -> float:
    """P(AI)."""
    return torch.sigmoid(torch.tensor(self.logit)).item()

  @property
  def residual(self) -> float:
    """logit - bias - importances: 0 for logreg, interactions for XGBoost."""
    return self.logit - self.bias - sum(f.importance for f in self.features)

  def __repr__(self) -> str:
    shown = [repr(f) for f in self.features[:REPR_FEATURES]]
    if len(self.features) > REPR_FEATURES:
      shown.append("...")
    body = ",\n  ".join(shown)
    return f"Detection(label={self.label!r}, p_ai={self.p_ai:.2f}, features=[\n  {body}])"


class AuditableDetector(
    nn.Module,
    Detector,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="detectorscope",
    repo_url="https://github.com/google-deepmind/detectorscope",
    license="cc-by-4.0",
    tags=["ai-text-detection", "sparse-autoencoder", "interpretability"],
):
  """Language model -> JumpReLU SAEs -> max-pool -> DetectorHead.

  `head` and `tau` name the default head among a repo's heads/ files; every
  call may override them. `base_model=None` loads no extractor: a repo's
  bundled extractor/ is loaded instead when there is one, else only
  predict_cached() works (evaluate.py --head_only).

  Args:
    base_model: Hub id or path of the language model, or None.
    hook_points: Modules whose inputs the SAEs read, e.g. model.layers.8;
      their order fixes the layer order of the feature vector.
    d_model: Width of the residual stream.
    d_sae: Width of each SAE.
    head: Default head type, "logreg" or "xgboost".
    tau: Default auto-interpretation threshold the head was trained with.
    torch_dtype: dtype to load the language model in; None keeps its own.
  """

  def __init__(
      self,
      base_model: str | None,
      hook_points: list[str] | None = None,
      d_model: int = 2304,
      d_sae: int = 16384,
      head: str = "logreg",
      tau: float = 0.85,
      torch_dtype: str | None = None,
  ):
    super().__init__()
    self.hooks = list(hook_points or [])
    self.default = (head, round(tau, 2))
    self.saes = nn.ModuleList(JumpReLUSAE(d_model, d_sae) for _ in self.hooks)
    self.extractor = (
        None
        if base_model is None
        else HFExtractor(base_model, self.hooks, torch_dtype=torch_dtype)
    )
    self.feature_table: dict[int, dict] = {}  # id -> features.jsonl record
    self.root: str | None = None  # directory holding heads/ and features.jsonl
    self._heads: dict[tuple[str, float], DetectorHead] = {}

  @classmethod
  def from_paths(
      cls,
      model_path: str,
      head_type: str,
      head_path: str | None,
      hook_points: list[str] | None = None,
      sae_model_paths: list[str] | None = None,
      head_only: bool = False,
      gemma_scope_repo: str = "google/gemma-scope-2b-pt-res",
  ) -> "AuditableDetector":
    """Builds a detector from the paper's artifacts on disk.

    Args:
      model_path: HF path of the base or fine-tuned language model.
      head_type: "logreg" or "xgboost".
      head_path: Head pickle; None attaches no head.
      hook_points: Extractor hooks, paired positionally with sae_model_paths.
      sae_model_paths: JumpReLU state dicts; a path that is not a file is a
        Gemma Scope id (JumpReLUSAE.from_gemma_scope).
      head_only: Load neither the extractor nor the SAEs.
      gemma_scope_repo: Hub repo of the Gemma Scope ids.
    """
    det = cls(
        None if head_only else model_path, None if head_only else hook_points
    )
    if not head_only:
      det.saes = nn.ModuleList(
          JumpReLUSAE.from_state_dict(p)
          if os.path.exists(p)
          else JumpReLUSAE.from_gemma_scope(p, gemma_scope_repo)
          for p in sae_model_paths
      )
    if head_path is not None:
      det.add_head(
          DetectorHead.create_detector_head(head_type, detector_path=head_path),
          *det.default,
      )
    return det

  @property
  def head(self) -> DetectorHead | None:
    """The default head, or None when none is attached."""
    return self._heads.get(self.default)

  def add_head(self, head: DetectorHead, kind: str, tau: float) -> None:
    """Attaches a head as the (kind, tau) head; save_pretrained writes it."""
    self._heads[kind, round(tau, 2)] = head

  def heads(self) -> list[tuple[str, float]]:
    """The (head, tau) pairs available: the repo's files, else the loaded."""
    return sorted(self._heads) if self.root is None else list_heads(self.root)

  def load_head(
      self, head: str | None = None, tau: float | None = None
  ) -> DetectorHead:
    """The head for (head, tau), each defaulting to the instance default.

    Raises:
      ValueError: If no such head is loaded or in the repo.
    """
    kind, tau = head or self.default[0], self.default[1] if tau is None else tau
    key = (kind, round(tau, 2))
    if key not in self._heads:
      path = self.root and os.path.join(self.root, head_file(*key))
      if not path or not os.path.exists(path):
        raise ValueError(f"no head {key}; available: {self.heads()}")
      self._heads[key] = DetectorHead.load(path)
    return self._heads[key]

  @torch.no_grad()
  def get_sae_activations(self, texts: list[str]) -> list[torch.Tensor]:
    """Per-token, attention-masked SAE activations of a batch.

    Returns:
      One tensor per hook point, in hook_points order, each of shape
      [len(texts), max_seq_len, d_sae]. Padded positions are zero.
    """
    acts, mask = self.extractor.get_activations(
        texts, return_mask=True, on_device=True
    )
    return [
        sae.encode(acts[f"{hook}_input"].float()) * mask[:, :, None]
        for hook, sae in zip(self.hooks, self.saes)
    ]

  @staticmethod
  def pool(sae_acts: list[torch.Tensor]) -> torch.Tensor:
    """Max-pools over the sequence and concatenates layers: [batch, d].

    This pooling defines the feature vectors every cache and head consume.
    """
    return torch.concat([a.max(dim=1)[0] for a in sae_acts], dim=1)

  def get_feature_vectors(self, texts: list[str]) -> torch.Tensor:
    """Max-pooled SAE feature vectors, [len(texts), d_sae * n_hooks]."""
    return self.pool(self.get_sae_activations(texts))

  def predict(self, texts: list[str]) -> torch.Tensor:
    """Logits (log-odds of AI) of the texts under the default head, [n]."""
    return self.predict_cached(self.get_feature_vectors(texts))

  def predict_cached(self, x: torch.Tensor) -> torch.Tensor:
    """Logits of feature vectors x [n, d] under the default head, [n]."""
    return self.load_head().predict(x)

  def get_global_importance(self) -> torch.Tensor:
    """The default head's global importance of every feature, [d]."""
    return self.load_head().get_global_importance()

  @torch.no_grad()
  def extract(self, text: str, highlights: bool = True) -> Extraction:
    """Runs the extractor and SAEs on one document; independent of the head.

    Args:
      text: The document.
      highlights: Also keep its tokens and per-token activations.
    """
    acts = self.get_sae_activations([text])
    ex = Extraction(self.pool(acts).cpu())
    if highlights:
      ex.tokens = self.extractor.get_token_strings(text)
      ex.token_feats = torch.cat(acts, -1)[0].to_sparse().cpu()
      assert ex.token_feats.shape[0] == len(ex.tokens), "tokenizer mismatch"
    return ex

  def explain(
      self,
      doc: str | Extraction,
      head: str | DetectorHead | None = None,
      tau: float | None = None,
  ) -> Detection:
    """Scores one document with a head and lists its feature evidence.

    Importance is the zero-ablation counterfactual (original - ablated logit)
    of every active feature, positive toward AI. Highlights are filled when
    the extraction carries token activations.

    Args:
      doc: The document, or its Extraction (one extraction, many heads).
      head: A head type resolved with `tau` among the repo's heads, a
        DetectorHead, or None for the default head.
      tau: The threshold of the head type; None for the default.
    """
    h = head if isinstance(head, DetectorHead) else self.load_head(head, tau)
    ex = self.extract(doc) if isinstance(doc, str) else doc
    logit = h.predict(ex.x)[0].item()
    bias = h.predict(torch.zeros_like(ex.x))[0].item()
    cf = h.get_importance_scores(ex.x)[0]
    ids = cf.nonzero().flatten()
    highlights = {}
    if ex.token_feats is not None:
      cols = ex.token_feats.index_select(1, ids).coalesce()
      for (t, k), a in zip(cols.indices().T.tolist(), cols.values().tolist()):
        highlights.setdefault(ids[k].item(), []).append((t, a))
    feats = [
        Feature(
            i,
            cf[i].item(),
            r["description"]["terse"]
            if (r := self.feature_table.get(i))
            else None,
            highlights.get(i),
        )
        for i in ids.tolist()
    ]
    feats.sort(key=lambda f: -abs(f.importance))
    label = "ai" if logit > h.threshold else "human"
    return Detection(logit, bias, label, feats, ex.tokens)

  def forward(
      self,
      texts: str | list[str],
      head: str | None = None,
      tau: float | None = None,
      return_highlights: bool = False,
  ) -> Detection | list[Detection]:
    """Detections of one text or of each text, under the (head, tau) head."""
    single = isinstance(texts, str)
    out = [
        self.explain(self.extract(t, return_highlights), head, tau)
        for t in ([texts] if single else texts)
    ]
    return out[0] if single else out

  def _save_pretrained(self, save_directory: pathlib.Path) -> None:
    """Writes the SAEs (the mixin), the extractor, the heads and features."""
    super()._save_pretrained(save_directory)
    if self.extractor is not None:
      self.extractor.model.save_pretrained(save_directory / "extractor")
      self.extractor.tokenizer.save_pretrained(save_directory / "extractor")
    (save_directory / "heads").mkdir(exist_ok=True)
    for (name, tau), h in self._heads.items():
      h.save(str(save_directory / head_file(name, tau)))
    with open(save_directory / "features.jsonl", "w", encoding="utf-8") as f:
      for record in self.feature_table.values():
        f.write(json.dumps(record) + "\n")

  @classmethod
  def _from_pretrained(
      cls,
      *,
      model_id: str,
      revision: str | None = None,
      cache_dir: str | None = None,
      force_download: bool = False,
      local_files_only: bool = False,
      token: str | bool | None = None,
      **kwargs,
  ) -> "AuditableDetector":
    """Loads the SAEs (the mixin), then the extractor, heads and features."""
    kwargs.setdefault("strict", True)
    model = super()._from_pretrained(
        model_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
        token=token,
        **kwargs,
    )
    model.root = (
        model_id
        if os.path.isdir(model_id)
        else huggingface_hub.snapshot_download(
            model_id,
            allow_patterns=["extractor/*", "heads/*", "features.jsonl"],
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
        )
    )
    with open(
        os.path.join(model.root, "features.jsonl"), encoding="utf-8"
    ) as f:
      model.feature_table = {r["id"]: r for r in map(json.loads, f)}
    model.load_head()  # a missing default head fails here, not at a call
    extractor = os.path.join(model.root, "extractor")
    if model.extractor is None and os.path.isdir(extractor):
      model.extractor = HFExtractor(
          extractor, model.hooks, torch_dtype=kwargs.get("torch_dtype")
      )
    device = model.extractor.device if model.extractor else "cpu"
    return model.to(device)
