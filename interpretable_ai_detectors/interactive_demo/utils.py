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

"""Utility functions for the demo app."""

from typing import Any

from interpretable_ai_detectors.model.hf_detector import HFDetector
from interpretable_ai_detectors.model.hf_extractor import HFExtractor
from interpretable_ai_detectors.sae.models.jumprelu_sae import JumpReLUSAE
import pandas as pd
import torch


def run_detector(
    text: str, extractor: HFExtractor, detector: HFDetector, hook_id: str
) -> tuple[torch.Tensor, float]:
  """Run the detector on the text and return the activations and score."""
  activations = extractor.get_activations(text)[hook_id]
  prediction = detector.predict(activations[:, -1, :])
  return activations, torch.sigmoid(prediction).item() * 100.0


def get_important_features(
    activations: torch.Tensor,
    detector: HFDetector,
    sae: JumpReLUSAE,
    feature_df: pd.DataFrame,
    filter_zero_lor: bool = True,
    autointerp_threshold: float = 0.6,
    max_percent_highlighted: float = 0.3
) -> tuple[list[dict[str, Any]], torch.Tensor]:
  """Get all features that were relevant to the detector."""

  # Ablate the SAE features to get their importance scores
  importance = detector.get_importance_scores(
      activations[:, -1, :], sae.W_dec.cpu()
  )

  # Run the SAE to get the SAE feature activations
  _, sae_features = sae.forward(activations)
  max_activation = sae_features.max(dim=1).values.squeeze()

  # Filter out all inactive sae features
  importance = torch.where(max_activation != 0, importance.squeeze(), 0.0)

  # Get the indices of the features with non-zero importance scores
  indices = importance.nonzero().squeeze(1)
  nonzero_tuples = list(zip(indices.tolist(), importance[indices].tolist()))

  feature_list = []
  for i, score in sorted(nonzero_tuples, key=lambda x: abs(x[1]), reverse=True):
    # Retrieve the information for the current feature from the dataframe
    feature_info = (
        feature_df.loc[feature_df["feature_index"] == i].iloc[0].to_dict()
    )

    # If the feature is below the autointerp threshold, skip it
    if feature_info["correlation"] < autointerp_threshold:
      continue

    # If the feature is active on too many tokens, skip it
    num_highlights = sae_features[0, :, i].nonzero().shape[0]
    if (num_highlights / sae_features.shape[1]) > max_percent_highlighted:
      continue

    # If we"re filtering zero LOR and this feature has a zero LOR, skip it
    if filter_zero_lor and feature_info["lor"] == 0.0:
      continue

    # Reassign the description fields of the feature
    # TODO(liamdugan): Refactor this to remove this
    feature_info["description"] = feature_info["terse_description"]
    feature_info["extended_desc"] = feature_info["long_description"]
    feature_info["odds_ratio"] = feature_info["lor"]

    # Add the feature to the list with its associated importance score
    feature_list.append(feature_info | {"importance_score": score})

  return feature_list, sae_features


def get_tokens(extractor: HFExtractor, text: str) -> list[str]:
  """Get the tokens for the text from the tokenizer."""
  token_ids = extractor.tokenizer.encode(
      text, max_length=1024, truncation=True, return_tensors="pt"
  ).T
  raw_tokens = extractor.tokenizer.convert_ids_to_tokens(token_ids)
  return [extractor.tokenizer.convert_tokens_to_string([t]) for t in raw_tokens]


def get_feature_highlights(
    sae_features: torch.Tensor, feature_index: int
) -> list[float]:
  """Given a sae feature index, return the activations for each token."""
  return sae_features[0, :, feature_index].tolist()
