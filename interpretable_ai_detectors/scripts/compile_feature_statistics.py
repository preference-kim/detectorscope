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

"""This script compiles LOR, densities, descriptions, correlations for features.

It output a csv that can be read in as a dataframe for use in the interactive
dashboard and in other visualizations
"""

import argparse
import os

from interpretable_ai_detectors.model.activation_cache import MaxActivationCache
from interpretable_ai_detectors.scripts.compute_log_odds_ratio import compute_log_odds
import pandas as pd


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-d",
      "--data_path",
      type=str,
      help="Path to the dataset",
  )
  parser.add_argument(
      "-c",
      "--correlations_path",
      type=str,
      help="Path to the autointerp results file",
  )
  parser.add_argument(
      "-m",
      "--target_model",
      type=str,
      help="Target model to compile LOR for",
  )
  parser.add_argument(
      "-o",
      "--output_path",
      type=str,
      default="feature_statistics.csv",
      help="Path to output csv file to write compiled results to",
  )
  args = parser.parse_args()

  cache_path = os.path.join(args.data_path, "max_activations.pth")
  human_cache_path = os.path.join(args.data_path, "human_max_activations.pth")
  target_cache_path = os.path.join(
      args.data_path, f"{args.target_model}_max_activations.pth"
  )

  cache = MaxActivationCache(cache_path)
  human_cache = MaxActivationCache(human_cache_path)
  target_cache = MaxActivationCache(target_cache_path)

  # Load correlations file as pandas dataframe
  df = pd.read_json(args.correlations_path, lines=True)

  # Add density information from the human-annotated cache
  human_density = dict(enumerate(human_cache.get_feature_density()))
  df["human_density"] = df["feature_index"].map(human_density)

  # Add density information from the target cache
  target_density = dict(enumerate(target_cache.get_feature_density()))
  df["target_density"] = df["feature_index"].map(target_density)

  # Add total density information as a percentage of the entire dataset
  total_size = cache.features.size()[0]
  density_frac = dict(
      enumerate([d / total_size for d in cache.get_feature_density()])
  )
  df["density"] = df["feature_index"].map(density_frac)

  # Add log odds ratio information between the human and target caches
  human_odds = compute_log_odds(human_cache)
  target_odds = compute_log_odds(target_cache)
  lor = dict(enumerate([ho - to for ho, to in zip(human_odds, target_odds)]))
  df["lor"] = df["feature_index"].map(lor)

  # Add each description type as a separate column (e.g. terse, long, etc.)
  def expand_description(row):
    if isinstance(row["description"], dict):
      for k, v in row["description"].items():
        row[f"{k}_description"] = v
      del row["description"]
    return row

  df = df.apply(expand_description, axis=1)

  # Process correlations of features to separate out coeff from p_value
  def process_correlation(row):
    if isinstance(row["correlation"], list):
      correlation, p_value = tuple(row["correlation"])
      row["correlation"] = correlation
      row["p_value"] = p_value
    return row

  df = df.apply(process_correlation, axis=1)

  print(f"Writing dataframe to {args.output_path}...")
  df.to_csv(args.output_path, index=False)
