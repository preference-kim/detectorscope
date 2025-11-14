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
from typing import Any, Iterator

import datasets
import matplotlib.pyplot as plt
import requests
import torch

CSR_KEYS = ["crow_indices", "col_indices", "values", "size"]


def add_values_col(ds: datasets.Dataset, values_col: str) -> datasets.Dataset:
  if not all([key in ds.column_names for key in CSR_KEYS]):
    raise ValueError("Not all CSR columns are present")

  def iterate_slice(ds_slice: datasets.Dataset) -> Iterator[dict[str, Any]]:
    """Iterate over a slice of the dataset and only return the CSR keys."""
    for i in range(len(ds_slice[CSR_KEYS[0]])):
      yield {col: ds_slice[col][i] for col in CSR_KEYS}

  return ds.with_transform(lambda x: {values_col: list(iterate_slice(x))} | x)


def get_distinct_colors_mpl(n: int) -> list[tuple[int, int, int]]:
  """Gets N distinct colors from the 'hsv' colormap in Matplotlib.

  Args:
      n (int): The number of colors to generate.

  Returns:
      A list of N tuples, where each tuple is an (R, G, B) color
      with integer values from 0 to 255.
  """
  # Get N colors from the hsv colormap
  # cmap(i) returns (R, G, B, A) in the range [0.0, 1.0]
  cmap = plt.get_cmap("hsv", n)
  colors_rgba = [cmap(i) for i in range(n)]

  # Convert from [0.0, 1.0] to [0, 255] and ignore the alpha channel
  colors_rgb_255 = []
  for color in colors_rgba:
    rgb_tuple = tuple(int(c * 255) for c in color[:3])
    colors_rgb_255.append(rgb_tuple)

  return colors_rgb_255


def visualize_multi_feature_saliency(
    tokens: list[str], scores: torch.Tensor, feature_id: list[int]
) -> str:
  """Generates HTML for a colored token saliency heatmap with multiple features.

  Args:
      tokens (list): List of token strings.
      scores (torch.Tensor): 2D tensor of activation scores [seq_len, N].
      feature_id (list): The list of IDs of features (len N)

  Returns:
      str: An HTML string representing the saliency heatmap.
  """
  # normalize by each feature's maximum
  normalized_scores = scores / scores.max(dim=0).values

  # Get top feature for each token (for coloring)
  best_feature = torch.argmax(normalized_scores, dim=1)

  # Get max scores for each feature
  max_activations = normalized_scores.max(dim=1).values

  # Get distinct colors
  colors = get_distinct_colors_mpl(len(feature_id))

  html_parts = [
      f"<h3 style='color: rgba({color[0]}, {color[1]}, {color[2]},"
      f" 1.0);'>Feature #{feat}</h3>"
      for feat, color in zip(feature_id, colors)
  ]
  container_style = (
      "font-family: monospace; line-height: 2.0; padding: 10px; "
      "border: 1px solid #ddd; border-radius: 5px; background: #f9f9f9; "
      "display: flex; flex-wrap: wrap; gap: 4px;"
  )
  html_parts.append(f"<div style='{container_style}'>")

  for token, score, feature_index in zip(tokens, max_activations, best_feature):
    token_html = (
        token.replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;")
    )
    color = colors[feature_index]
    style = (
        f"background-color: rgba({color[0]}, {color[1]}, {color[2]},"
        f" {score.item():.4f}); padding: 2px 4px; margin: 1px; border-radius:"
        " 3px;"
    )
    html_parts.append(f"<span style='{style}'>{token_html}</span>")

  html_parts.append("</div>")

  return "".join(html_parts)


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
