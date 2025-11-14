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

"""Utility functions for interpretable AI detectors."""

import itertools
import json
import random
from typing import Any

import requests
import torch


def visualize_saliency(
    tokens: list[str], scores: torch.Tensor, feature_id: int
) -> str:
  """Generates HTML for a colored token saliency heatmap.

  Args:
    tokens (list): List of token strings.
    scores (torch.Tensor): 1D tensor of saliency scores (e.g., L2 norms).
    feature_id (int): The ID of the feature being visualized.

  Returns:
    str: An HTML string representing the saliency heatmap.
  """
  normalized_scores = scores / (scores.max() + 1e-9)

  html_parts = [f"<h3>Feature #{feature_id}</h3>"]
  container_style = (
      "font-family: monospace; line-height: 2.0; padding: 10px; "
      "border: 1px solid #ddd; border-radius: 5px; background: #f9f9f9; "
      "display: flex; flex-wrap: wrap; gap: 4px;"
  )
  html_parts.append(f"<div style='{container_style}'>")

  for token, score in zip(tokens, normalized_scores):
    token_html = (
        token.replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace(" ", "&nbsp;")
    )
    style = (
        f"background-color: rgba(255, 0, 0, {score.item():.4f});",
        "padding: 2px 4px; margin: 1px; border-radius: 3px;",
    )
    html_parts.append(f"<span style='{style}'>{token_html}</span>")

  html_parts.append("</div>")

  return "".join(html_parts)


def get_explanations_for_sae_feature(feature_index: int) -> str:
  """Query Neuronpedia for the explanation for a given SAE feature."""
  model_id = "gemma-2-2b"
  source_id = "25-gemmascope-res-16k"
  url = f"https://www.neuronpedia.org/api/feature/{model_id}/{source_id}/{feature_index}"
  try:
    response = requests.get(url)
    response.raise_for_status()  # raise exceptions for 4xx or 5xx errors
    feature_data = response.json()
  except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
    return "[[ Connection Error ]]"
  except json.JSONDecodeError as e:
    print(f"Failed to decode response JSON: {e}")
    return "[[ JSON Error ]]"

  # If no explanations for the feature, return "No explanation"
  if not feature_data["explanations"]:
    return "[[ No Explanation ]]"

  return feature_data["explanations"][0]["description"]


def get_sae_feature_density(feature_index: int) -> float:
  """Query Neuronpedia for the density of a given SAE feature."""
  model_id = "gemma-2-2b"
  source_id = "25-gemmascope-res-16k"
  url = f"https://www.neuronpedia.org/api/feature/{model_id}/{source_id}/{feature_index}"
  try:
    response = requests.get(url)
    response.raise_for_status()  # raise exceptions for 4xx or 5xx errors
    feature_data = response.json()
  except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
    return "[[ Connection Error ]]"
  except json.JSONDecodeError as e:
    print(f"Failed to decode response JSON: {e}")
    return "[[ JSON Error ]]"

  return feature_data["frac_nonzero"]


def get_top_k_explanations(activations: torch.Tensor, top_k: int = 5):
  """Get explanations for the top k activations. Expects [1, n] or [n, 1]."""
  activations = torch.flatten(activations)
  _, top_k_indices = torch.topk(torch.abs(activations), k=top_k)
  for i in top_k_indices:
    desc = get_explanations_for_sae_feature(i)
    yield (i, activations[i].item(), desc)


def print_prediction(prediction: torch.Tensor, model: str = "GPT4"):
  """Print the prediction output by the detector. Expects size [1]."""
  pred = torch.sigmoid(prediction).item()
  likelihood = f"{pred:.2%}" if pred > 0.5 else f"{(1.0 - pred):.2%}"
  polarity = "Human" if pred < 0.5 else model
  print(f"Prediction: {prediction.item()} ({polarity}), {likelihood}")
  print("-" * 40)


def print_top_k_sae_activations(activations: torch.Tensor, top_k: int = 5):
  print("Top K SAE Activations:")
  for i, weight, desc in get_top_k_explanations(activations, top_k):
    print(f"{i}: {weight:.4f} -- {desc}")
  print("-" * 40)


def print_top_k_importance_scores(scores, top_k: int = 5, model: str = "GPT4"):
  print("Top K Detector Weights:")
  for i, weight, desc in get_top_k_explanations(scores, top_k):
    polarity = "Human" if weight < 0 else model
    print(f"{i}: {weight:.4f} ({polarity}) -- {desc}")
  print("-" * 40)


def check_memory_alloc_mb(t: torch.Tensor) -> float:
  """Check the amount of memory (in MiB) allocated for a given tensor."""
  if not t.is_sparse:
    return (t.element_size() * t.numel()) / (1024 * 1024)

  indices_memory = t.indices().element_size() * t.indices().numel()
  values_memory = t.values().element_size() * t.values().numel()
  return (indices_memory + values_memory) / (1024 * 1024)


def interleave_lists(lists: list[list[Any]]) -> list[Any]:
  """Interleaves a list of lists to one list."""
  return [
      x for x in itertools.chain.from_iterable(itertools.zip_longest(*lists))
      if x is not None
  ]


def nested_shuffle(lists: list[list[Any]]) -> list[list[Any]]:
  random.shuffle(lists)
  for sublist in lists:
    random.shuffle(sublist)
  return lists


def flatten(lists: list[list[Any]]) -> list[Any]:
  return list(itertools.chain(*lists))
