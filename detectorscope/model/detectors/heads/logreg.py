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

"""Implementation of a Logistic Regression detector head.

Loads the pickle written by train_logreg.py (SAE caches or a gram2vec
--use_mmap_cache on a gram2vec cache). The trainer folds any standardisation into the saved
weights, so the head consumes raw features and only applies the interp mask.
Also loads a `.safetensors` file written by save(): the weight with the mask
folded in, the bias, and the threshold and metadata in its header.
"""

import pickle

import safetensors.torch
import torch

from detectorscope.model.detectors.heads.head import DetectorHead

safe_open = safetensors.safe_open  # pytype: disable=module-attr


class LogRegHead(DetectorHead):
  """Logistic Regression detector head (inference and importance scoring only).

  Attributes:
    detector: The torch.nn.Linear layer, on self.device in eval mode.
    interp_mask: Float mask of shape [d_model] (n_layers * d_sae) on
      self.device.
    means: Per-feature means of shape [d_model] if the artifact has them,
      else None.
  """

  def __init__(self, detector_path: str):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    if detector_path.endswith(".safetensors"):
      with safe_open(detector_path, "pt") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
        self.metadata = f.metadata() or {}
      self.threshold = float(self.metadata.pop("threshold", 0.0))
      artifact = {
          "state_dict": state_dict,
          "input_dim": state_dict["weight"].shape[1],
          "interp_mask": torch.ones(state_dict["weight"].shape[1]),
      }
    else:
      with open(detector_path, "rb") as f:
        artifact = pickle.load(f)
    self.detector = torch.nn.Linear(artifact["input_dim"], 1)
    self.detector.load_state_dict(artifact["state_dict"])
    self.detector.to(self.device).eval()
    self.interp_mask = torch.as_tensor(
        artifact["interp_mask"], dtype=torch.float32, device=self.device
    ).flatten()
    self.means = None
    if artifact.get("means") is not None:
      self.means = torch.as_tensor(
          artifact["means"], dtype=torch.float32, device=self.device
      ).flatten()

  def save(self, path: str) -> None:
    """Writes weight (mask folded in) and bias as safetensors with metadata."""
    safetensors.torch.save_file(
        {
            "weight": (self._weight * self.interp_mask)[None].cpu(),
            "bias": self.detector.bias.detach().cpu().clone(),
        },
        path,
        metadata=self.metadata | {"threshold": str(self.threshold)},
    )

  @property
  def _weight(self) -> torch.Tensor:
    """Weight vector of shape [d_model]."""
    return self.detector.weight.detach().squeeze(0)

  def _mask_input(self, x: torch.Tensor) -> torch.Tensor:
    """Apply the interpretability mask: [n, d_model]."""
    return x.detach().to(self.device, dtype=torch.float32) * self.interp_mask

  def predict(self, x: torch.Tensor, logit: bool = True) -> torch.Tensor:
    """Runs an input Tensor through the detector to get a prediction.

    Args:
      x: Input tensor of shape [n, d_model].
      logit: If True, return the log-odds of AI; otherwise P(AI).

    Returns:
      Tensor of shape [n] containing the detector's predictions, on the CPU.
    """
    with torch.no_grad():
      logits = self.detector(self._mask_input(x)).squeeze(-1)
      return (logits if logit else torch.sigmoid(logits)).cpu()

  def get_importance_scores(
      self, x: torch.Tensor, method: str = "counterfactual"
  ) -> torch.Tensor:
    """Computes importance as per-feature contributions to the logit.

    For a linear model, logit(x) = sum_i w_i * x_i + b, so w_i * x_i is both
    the exact additive contribution of feature i and the exact zero-ablation
    effect f(x) - f(x | x_i = 0): no forward passes and no approximation are
    involved.

    NOTE: this differs from XGBoostHead's counterfactual in one way:
    a tree ensemble is piecewise constant, so zeroing an active feature often
    leaves the prediction untouched and yields exactly 0, whereas a linear
    head has no flat regions and every active feature with a non-zero weight
    gets a non-zero score. The same document therefore has far more ablatable
    features under this head than under XGBoost.

    Args:
      x: Input feature vectors of shape [n, d_model].
      method: "counterfactual", the only method a linear head needs.

    Returns:
      Tensor of shape [n, d_model] containing the importance scores, 0.0 at
      inactive and masked-out positions.

    Raises:
      ValueError: For any other method. "shapley" is refused rather than
        aliased, since a linear head's SHAP values are its counterfactuals.
    """
    if method != "counterfactual":
      raise ValueError(f"Supports only 'counterfactual', got {method!r}")
    with torch.no_grad():
      return (self._mask_input(x) * self._weight).cpu()

  def get_global_importance(self) -> torch.Tensor:
    """Returns |w| per feature as a tensor of shape [d_model]."""
    return self._weight.abs().cpu()
