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

"""Gram2Vec linguistic-feature detector: XGBoost or logistic-regression head.

This is the linguistic baseline, selected with `-dt gram2vec`.
"""

import numpy as np
import pickle
import torch
from gram2vec import vectorizer

from detectorscope.model.detector import Detector
from detectorscope.model.detectors.heads.head import DetectorHead


class Gram2VecDetector(Detector):
  """Scores documents from their gram2vec feature vectors through one head.

  The head is a DetectorHead pickled by one of the trainers, a LogRegHead
  (``state_dict``) or an XGBoostHead (``model``), told apart by content
  unless ``head_type`` says; both were trained on the split's gram2vec cache
  (``--use_mmap_cache``), which stores the NaN rate features of empty
  documents as 0, so inputs are NaN-filled to match. Predictions are logits
  for either head. Features may come from text (predict) or from a
  precomputed cache (predict_cached, see extract.py gram2vec).
  """

  def __init__(self, model_path: str, head_type: str | None = None):
    if head_type is None:
      with open(model_path, "rb") as f:
        head_type = "xgboost" if "model" in pickle.load(f) else "logreg"
    self.head = DetectorHead.create_detector_head(
        head_type, detector_path=model_path
    )

  def _score(self, x) -> torch.Tensor:
    """Scores a feature matrix [n, n_features] as logits of AI."""
    x = torch.nan_to_num(torch.as_tensor(np.asarray(x), dtype=torch.float32))
    return self.head.predict(x)

  def predict(self, texts: list[str]) -> torch.Tensor:
    """Vectorizes texts with gram2vec and scores them.

    Args:
      texts: Input texts to run through the detector.

    Returns:
      Logits of shape [n].
    """
    feat_df = vectorizer.from_documents(texts)
    return self._score(feat_df.to_numpy(dtype=np.float32))

  def predict_cached(self, x: torch.Tensor) -> torch.Tensor:
    """Scores precomputed gram2vec feature rows of shape [n, n_features]."""
    return self._score(x)

  def get_global_importance(self) -> torch.Tensor:
    """Global importance of shape [n_features] (|w| or XGBoost gain)."""
    return self.head.get_global_importance()
