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

"""Implementation of a Gradient Boosted Trees detector head.

Loads the pickle written by train_xgboost.py (``model``, ``interp_mask`` and
``means``), or a booster JSON written by save(), whose mask is the set of
features the trees split on and whose attributes hold the threshold and
metadata. The booster runs on the GPU when one is visible.
"""

import numpy as np
import pickle
import torch
import xgboost as xgb

from detectorscope.model.detectors.heads.head import DetectorHead

# Small constant used when converting probabilities to logits so we never take
# the log of exactly 0 or 1.
_LOGIT_EPS = 1e-7


class XGBoostHead(DetectorHead):
  """XGBoost detector head (inference and importance scoring only).

  Attributes:
    detector: The fitted xgb.XGBClassifier.
    interp_mask: Mask of shape [d_model] (n_layers * d_sae) exactly as stored
      in the artifact (int64/float64); left uncast because casting to float32
      perturbs predictions in the low bits.
    means: Per-feature means of shape [1, d_model] if the artifact has them,
      else None.
  """

  def __init__(self, detector_path: str):
    if detector_path.endswith(".json"):
      self.detector = xgb.XGBClassifier()
      self.detector.load_model(detector_path)
      booster = self.detector.get_booster()
      self.interp_mask = np.zeros(booster.num_features())
      self.interp_mask[[int(f[1:]) for f in booster.get_score()]] = 1
      self.means = None
      self.metadata = booster.attributes()
      self.threshold = float(self.metadata.pop("threshold", 0.0))
    else:
      with open(detector_path, "rb") as f:
        artifact = pickle.load(f)
      self.detector = artifact["model"]
      self.interp_mask = artifact["interp_mask"]
      self.means = artifact.get("means")
    self.detector.set_params(
        device=("cuda" if torch.cuda.is_available() else "cpu")
    )

  def save(self, path: str) -> None:
    """Writes the booster JSON; its attributes carry threshold and metadata."""
    self.detector.get_booster().set_attr(
        **self.metadata, threshold=str(self.threshold)
    )
    self.detector.save_model(path)

  def _masked(self, x: torch.Tensor) -> np.ndarray:
    """The interp-masked input as the array the booster scores.

    A plain array rather than a tensor: a booster that carries feature names
    (the gram2vec heads) rejects unnamed tensors but accepts arrays.
    """
    return (x.cpu().detach() * self.interp_mask).numpy()

  def predict(self, x: torch.Tensor, logit: bool = True) -> torch.Tensor:
    """Runs an input Tensor through the detector to get a prediction.

    Args:
      x: Input tensor of shape [n, d_model].
      logit: If True, return the log-odds of AI, with P(AI) first clamped away
        from 0 and 1 by _LOGIT_EPS; otherwise return P(AI) as the booster
        reports it.

    Returns:
      Tensor of shape [n] containing the detector's predictions.
    """
    probs = self.detector.predict_proba(self._masked(x))[:, 1]
    p = torch.from_numpy(probs)
    if not logit:
      return p
    p = torch.clamp(p, _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return torch.log(p / (1.0 - p))

  def get_importance_scores(
      self,
      x: torch.Tensor,
      method: str = "counterfactual",
      chunk_size: int = 256,
  ) -> torch.Tensor:
    """Computes per-feature importance scores for a particular input.

    Args:
      x: Input feature vectors of shape [n, d_model].
      method: "counterfactual" for the zero-ablation effect
        f(x) - f(x | x_i = 0) on the logit of every active, interpretable
        feature (see _counterfactual_importance); "shapley" for TreeSHAP
        contributions to the margin (logit), with the trailing bias column
        XGBoost appends dropped.
      chunk_size: Counterfactual only; how many (sample, feature) pairs to
        score at once. Bounds peak memory; does not affect the result.

    Returns:
      Tensor of shape [n, d_model] containing the importance scores: positive
      means the feature pushes the prediction toward AI.

    Raises:
      ValueError: For an unknown method.
    """
    if method == "counterfactual":
      return self._counterfactual_importance(x, chunk_size)
    if method == "shapley":
      return torch.from_numpy(
          self.detector.get_booster().predict(
              xgb.DMatrix(
                  self._masked(x),
                  feature_names=self.detector.get_booster().feature_names,
              ),
              pred_contribs=True,
          )
      )[:, :-1]
    raise ValueError(f"Unknown importance method {method!r}")

  def get_global_importance(self) -> torch.Tensor:
    """Returns XGBoost's feature_importances_ as a tensor of shape [d_model]."""
    return torch.from_numpy(self.detector.feature_importances_)

  def _counterfactual_importance(
      self, x: torch.Tensor, chunk_size: int
  ) -> torch.Tensor:
    """Zero-ablation importance for features active in x.

    For each sample and each active, interpretable feature i in that sample,
    computes:
        delta_i = f(x) - f(x | x_i = 0)
    on the logit, via a single batched forward pass across all samples and
    active features. A positive delta means the feature pushed the prediction
    toward AI.

    Args:
      x: Input tensor of shape [n, d_model].
      chunk_size: How many (sample, feature) pairs to score at once.

    Returns:
      Tensor of shape [n, d_model] with delta_i at active feature positions,
      0.0 elsewhere.
    """
    # Apply interpretability mask: [n, d_model]
    x_masked = x.cpu().detach() * self.interp_mask

    # Baseline logit for each sample: [n]
    baselines = self.predict(x_masked)

    # Find (sample_idx, feature_idx) pairs for every active feature.
    interp_mask_bool = torch.as_tensor(self.interp_mask, dtype=torch.bool)
    active_mask = (x_masked != 0) & interp_mask_bool  # [n, d_model]
    sample_indices, feature_indices = active_mask.nonzero(as_tuple=True)

    effects = torch.zeros_like(x_masked)

    if len(sample_indices) == 0:
      return effects

    for start in range(0, len(sample_indices), chunk_size):
      stop = min(start + chunk_size, len(sample_indices))
      rows, cols = sample_indices[start:stop], feature_indices[start:stop]

      # Advanced indexing already returns a fresh tensor, so this is safe to
      # mutate in place; no clone() needed.
      batch = x_masked[rows]  # [chunk, d_model]
      batch[torch.arange(len(cols)), cols] = 0.0

      deltas = baselines[rows] - self.predict(batch)

      # Scatter this chunk's deltas back into [n, d_model].
      effects[rows, cols] = deltas.to(effects.dtype)
    return effects
