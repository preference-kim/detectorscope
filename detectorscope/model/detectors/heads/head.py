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

"""Interface for loading and using a trained detector head over SAE features."""

import abc

import torch
from typing_extensions import Self


class DetectorHead(abc.ABC):
  """Shared interface for all DetectorHead models.

  This interface wraps around any model that can be used to convert SAE
  activations into a prediction in a lightweight and interpretable fashion.
  The models should support querying for feature importance scores on both an
  instance level and a global level. Inputs are max-pooled feature vectors of
  shape [n, d_sae], the concatenation over all hook points (d_sae * layers in
  AuditableDetector).

  Every head scores in the same units: predict() returns the log-odds of AI
  (P(AI) with logit=False) and get_importance_scores() returns per-feature
  contributions to that logit, positive meaning the feature pushes toward AI.

  Attributes:
    interp_mask: [d_sae] mask of the interpretable features the head was
      trained on (every concrete head sets it in its constructor).
    means: Per-feature training means when the artifact carries them.
    threshold: Decision cut on the logit, read from a head file's metadata.
    metadata: The rest of that metadata, written back by save().
  """

  interp_mask: torch.Tensor
  means: torch.Tensor | None = None
  threshold: float = 0.0
  metadata: dict[str, str] = {}

  @abc.abstractmethod
  def predict(
      self, x: torch.Tensor, logit: bool = True, **kwargs
  ) -> torch.Tensor:
    """Runs an input Tensor through the detector to get a prediction.

    Args:
      x: Input tensor of shape [n, d_sae].
      logit: If True, return the log-odds of the positive class; otherwise
        return P(AI).

    Returns:
      Tensor of shape [n] containing the detector's predictions.
    """

  @abc.abstractmethod
  def get_importance_scores(
      self, x: torch.Tensor, method: str = "counterfactual", **kwargs
  ) -> torch.Tensor:
    """Computes per-feature importance scores for a particular input.

    Args:
      x: Input feature vectors of shape [n, d_sae].
      method: "counterfactual" for the zero-ablation effect
        f(x) - f(x | x_i = 0) of each active, interpretable feature, measured
        on the logit; "shapley" for an additive SHAP decomposition of the
        logit.

    Returns:
      Tensor of shape [n, d_sae] containing the importance scores: positive
      means the feature pushes the prediction toward AI.

    Raises:
      ValueError: For a method this head does not support.
    """

  @abc.abstractmethod
  def get_global_importance(self, **kwargs) -> torch.Tensor:
    """Returns a tensor of shape [d_sae] of global importance scores."""

  @abc.abstractmethod
  def save(self, path: str) -> None:
    """Writes the head as a file load() reads, metadata and threshold inside."""

  @classmethod
  def load(cls, path: str) -> Self:
    """Loads a `.safetensors` (LogRegHead) or `.json` (XGBoostHead) file."""
    head = "logreg" if path.endswith(".safetensors") else "xgboost"
    return cls.create_detector_head(head, detector_path=path)

  @classmethod
  def create_detector_head(cls, head_name: str, **kwargs) -> Self:
    """Factory method for creating DetectorHead instances.

    Args:
      head_name: "logreg" or "xgboost".
      **kwargs: Forwarded to the head's constructor (e.g. detector_path).

    Raises:
      ValueError: For an unknown head_name.
    """
    if head_name == "logreg":
      from detectorscope.model.detectors.heads.logreg import LogRegHead

      return LogRegHead(**kwargs)
    elif head_name == "xgboost":
      from detectorscope.model.detectors.heads.xgboost import XGBoostHead

      return XGBoostHead(**kwargs)

    raise ValueError(f"Unknown detector head {head_name}")
