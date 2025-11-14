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

"""Classes for loading and using a trained detector model."""

import torch


class TorchLogisticRegression(torch.nn.Module):
  def __init__(self, num_features):
    super().__init__()
    self.linear = torch.nn.Linear(num_features, 1)

  def forward(self, x):
    return torch.sigmoid(self.linear(x))


class Detector:
  """Loads and uses a trained detector model for importance scoring."""

  def __init__(self, detector_path, use_sae_features=True):
    state_dict = torch.load(detector_path)
    input_features = state_dict["linear.weight"].shape[1]
    self.detector = TorchLogisticRegression(num_features=input_features)
    self.detector.load_state_dict(state_dict)

    self._importance_scorer = (
        self._get_importance_from_sae_activation
        if use_sae_features
        else self._get_importance_from_residual_stream
    )

  def predict(self, x: torch.Tensor):
    """Run an input Tensor through the detector to get a prediction.

    Args:
        x (torch.Tensor): input tensor of shape [n, input_dim]

    Returns:
        torch.Tensor of shape [n, 1] containing the detector's predictions.
    """
    return self.detector.linear(x)

  def get_importance_scores(self, *args, **kwargs):
    """Delegates calculation to the selected importance scoring strategy."""
    return self._importance_scorer(*args, **kwargs)

  def _get_importance_from_sae_activation(
      self, x: torch.Tensor
  ) -> torch.Tensor:
    """Compute importance of SAE features via dot product.

    Used when detector is trained with SAE activations as input.

    Args:
      x (torch.Tensor): input feature vector x of size [1, d_sae]

    Returns:
      importance scores size [n, 1] for each of the n SAE directions
    """
    return torch.mul(x, self.detector.linear.weight.data).T

  def _get_importance_from_residual_stream(
      self, x: torch.Tensor, w_dec: torch.Tensor
  ) -> torch.Tensor:
    """Compute importance of SAE features from residual stream embedding.

    Used when detector is trained with residual stream as input.

    Computed by taking the component of x in each of the SAE feature
    directions (x_proj) and computing the difference in prediction
    logits between x and x - x_proj.

    Args:
      x (torch.Tensor): residual stream vector x of size [1, d_model]
      w_dec (torch.Tensor): SAE decoder matrix of size [n, d_model]

    Returns:
      importance scores size [n, 1] for each of the n SAE directions
    """
    x_proj = (w_dec @ x.T) * w_dec
    return self.predict(x) - self.predict(x - x_proj)
