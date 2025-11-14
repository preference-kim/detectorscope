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

"""Generates feature descriptions using an autointerpretation pipeline.

This script loads a dataset and an activation cache, then uses an SAEExtractor
and FeatureDescriptionValidator to generate and save descriptions for a
specified list of features.
"""

import argparse
import json
import os

import datasets
from interpretable_ai_detectors.model.activation_cache import MaxActivationCache
from interpretable_ai_detectors.model.autointerpret import FeatureDescriptionValidator
from interpretable_ai_detectors.model.sae_extractor import SAEExtractor
import tqdm


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-d",
      "--data_path",
      type=str,
      help="Path to the dataset",
  )
  parser.add_argument(
      "-l",
      "--feature_list",
      type=int,
      nargs="*",
      help="List of features to generate descriptions for",
  )
  parser.add_argument(
      "-t",
      "--num_prompt_examples",
      type=int,
      default=10,
      help="Num examples to include in prompt",
  )
  parser.add_argument(
      "-n",
      "--negative_sampling_method",
      type=str,
      default="uniform",
      choices=["uniform", "semantic"],
      help="Method for sampling negatives in the pipeline",
  )
  parser.add_argument(
      "-q",
      "--quantile_method",
      type=str,
      default="scaled",
      choices=["density", "scaled"],
      help="Method for determining quantiles of activations",
  )
  parser.add_argument(
      "-p",
      "--prompt_sampling_method",
      type=str,
      default="stratified",
      choices=["uniform", "stratified"],
      help="Method for sampling examples to include in the prompt",
  )
  parser.add_argument(
      "-nq",
      "--n_quantiles",
      type=int,
      default=4,
      help="Number of quantiles to split data into (default: 4)",
  )
  parser.add_argument(
      "-o",
      "--output_file",
      type=str,
      default="feature_results.jsonl",
      help="output file for feature correlation results",
  )
  args = parser.parse_args()

  ds = datasets.load_from_disk(args.data_path)
  cache = MaxActivationCache(
      os.path.join(args.data_path, "max_activations.pth")
  )
  extractor = SAEExtractor()

  validator_kwargs = {
      "negative_sampling_method": args.negative_sampling_method,
      "quantile_method": args.quantile_method,
      "prompt_sampling_method": args.prompt_sampling_method,
      "n_quantiles": args.n_quantiles,
  }

  validator = FeatureDescriptionValidator(
      ds, extractor, cache, **validator_kwargs
  )

  print("Running Description Generation...")
  with open(args.output_file, "a") as f:
    for i in tqdm.tqdm(args.feature_list):
      description = validator.generate_new_autointerp_description(
          i, args.num_prompt_examples
      )
      json_line = json.dumps({"feature_index": i} | description)
      f.write(json_line + "\n")
