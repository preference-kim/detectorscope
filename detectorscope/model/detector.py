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

"""Abstract interface and factory for all detectors."""

import abc

import torch
import torch.nn.functional as F
from typing_extensions import Self


class Detector(abc.ABC):
  """Shared interface for all Detector models."""

  @abc.abstractmethod
  def predict(self, texts: list[str]) -> torch.Tensor:
    """Runs texts through the detector.

    Args:
      texts: input texts to score.

    Returns:
      Scores of shape [n]; higher means more likely AI. The scale is the
      detector's own: log-odds for the SAE heads and the gram2vec logreg,
      P(AI) for the huggingface, Fast-DetectGPT and gram2vec XGBoost paths and
      a raw criterion for the other baselines. Per-feature importances are not
      computed here; ask the head (detector.head.get_importance_scores).
    """

  def predict_cached(self, x: torch.Tensor) -> torch.Tensor:
    """Runs cached feature vectors through the detector head.

    Only the cache-backed detectors (interpretable and gram2vec) override this.

    Args:
      x: feature vectors of shape [n, d]; for the SAE detectors
        d = d_sae * n_layers with layers concatenated in ascending order (the
        activation-cache layout), for gram2vec d = n_features.

    Returns:
      Scores of shape [n], as for predict.

    Raises:
      NotImplementedError: if the detector has no feature cache.
    """
    raise NotImplementedError(
        f"{type(self).__name__} has no feature cache; use predict()"
    )

  def get_global_importance(self, **kwargs) -> torch.Tensor:
    """Returns the input-independent importance of every feature.

    Only the feature-based detectors (interpretable and gram2vec) override
    this.

    Returns:
      Tensor of shape [d], one entry per input feature of predict_cached.

    Raises:
      NotImplementedError: if the detector has no per-feature scores.
    """
    raise NotImplementedError(
        f"{type(self).__name__} doesn't support global importance"
    )

  @classmethod
  def create_detector(
      cls, detector_name: str, model_path: str, **kwargs
  ) -> Self:
    """Builds a detector by name.

    Args:
      detector_name: one of huggingface, auditable (or its old name
        interpretable), gram2vec, binoculars, fastdetectgpt, remodetect or
        specdetect.
      model_path: model path or id; ignored by binoculars, fastdetectgpt,
        remodetect and specdetect, which take their models through kwargs.
      **kwargs: passed to the detector's constructor.

    Raises:
      ValueError: if detector_name is not one of the names above.
    """
    if detector_name == "huggingface":
      from detectorscope.model.detectors.neural import HFDetector

      return HFDetector(model_path, **kwargs)
    elif detector_name in ("auditable", "interpretable"):
      from detectorscope.model.detectors.auditable import AuditableDetector

      return AuditableDetector.from_paths(model_path, **kwargs)
    elif detector_name == "gram2vec":
      from detectorscope.model.detectors.gram2vec import Gram2VecDetector

      return Gram2VecDetector(model_path, **kwargs)
    elif detector_name == "binoculars":
      from detectorscope.model.detectors.baselines.binoculars import BinocularsDetector

      return BinocularsDetector(**kwargs)
    elif detector_name == "fastdetectgpt":
      from detectorscope.model.detectors.baselines.fastdetectgpt import FastDetectGPTDetector

      return FastDetectGPTDetector(**kwargs)
    elif detector_name == "remodetect":
      from detectorscope.model.detectors.baselines.remodetect import ReMoDetectDetector

      return ReMoDetectDetector(**kwargs)
    elif detector_name == "specdetect":
      from detectorscope.model.detectors.baselines.specdetect import SpecDetectPlusPlusDetector

      return SpecDetectPlusPlusDetector(**kwargs)

    raise ValueError(f"Unknown detector name {detector_name}")
