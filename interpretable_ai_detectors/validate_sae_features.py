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

"""Script to validate SAE features and generate descriptions.

This script provides functionality to:
1.  Identify the k most dense features in an SAE activation cache.
2.  Validate selected features, comparing existing and new descriptions.
3.  Submit batches of validation tasks to a Gemini Batch API.
4.  Save validation results to JSONL files or prepare batch job directories.
"""

import argparse
from collections.abc import Iterator
import datetime
import json
import os
import pickle
import shutil
from typing import Any

import datasets
from interpretable_ai_detectors import utils
from interpretable_ai_detectors.model.activation_cache import MaxActivationCache
from interpretable_ai_detectors.model.autointerpret import FeatureDescriptionValidator
from interpretable_ai_detectors.model.sae_extractor import SAEExtractor
import torch
import tqdm


def _get_k_most_dense_features(
    activation_cache: MaxActivationCache, top_k: int
) -> list[int]:
  """Returns a list of the K most dense features in the dataset."""
  density = torch.tensor(activation_cache.get_feature_density())
  return torch.topk(density, top_k).indices.tolist()


def run_validation_experiment(
    validator: FeatureDescriptionValidator,
    feature_indices: list[int],
    use_neuronpedia_descriptions: bool,
    num_test_examples: int | None,
) -> Iterator[dict[str, Any]]:
  """Computes corr on old and new description for each top_k feature."""
  for i in tqdm.tqdm(feature_indices):
    description = {}
    if use_neuronpedia_descriptions:
      description = {"terse": utils.get_explanations_for_sae_feature(i)}
    else:
      while "terse" not in description:
        description = validator.generate_new_autointerp_description(
            i, validator.n_quantiles * 2
        )

    corr, num_examples = validator.validate_feature_description(
        i, description["terse"], num_test_examples
    )

    yield {
        "feature_index": i,
        "num_examples": num_examples,
        "description": description,
        "correlation": corr,
    }


def run_validation_experiment_batch(
    validator: FeatureDescriptionValidator,
    feature_indices: list[int],
    validator_kwargs: dict[str, Any],
    experiment_dir: str,
    num_test_examples: int | None,
):
  """Submits a batch of features to be processed via the Gemini Batch API."""
  descriptions = [
      utils.get_explanations_for_sae_feature(i) for i in feature_indices
  ]

  # Get the json batches and group dict
  tmpfile, job_name, group_dict = validator.submit_validation_batch_job(
      feature_indices, descriptions, num_test_examples
  )

  # Create a folder for this particular batch job
  full_dir_path = os.path.join(
      experiment_dir,
      (
          f"{validator.validation_method}_{validator.prompt_sampling_method}_"
          f"{validator.quantile_method}_{validator.highlight_method}_"
          f"{validator.use_snippets}_{job_name.split('/')[-1]}"
      ),
  )
  os.makedirs(full_dir_path)

  # Move the temp file to the directory
  shutil.move(tmpfile, os.path.join(full_dir_path, "prompts.jsonl"))

  # Write the group dict to the directory
  with open(os.path.join(full_dir_path, "groups.pkl"), "wb") as f:
    pickle.dump(group_dict, f)

  # Write the metadata to the directory
  with open(os.path.join(full_dir_path, "metadata.json"), "w+") as f:
    output_data = {
        "time_submitted": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "job_name": job_name,
        "validator_kwargs": validator_kwargs,
    }
    json.dump(output_data, f, indent=4)

  return


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
      help="List of features to validate (defaults to top_k densest)",
  )
  parser.add_argument(
      "-k",
      "--top_k",
      type=int,
      default=5,
      help="Num. features to select from densest (when no feature_list)",
  )
  parser.add_argument(
      "--use_neuronpedia_descriptions",
      action="store_true",
      help="If true, query feature explanations from neuronpedia",
  )
  parser.add_argument(
      "-t",
      "--num_test_examples",
      type=int,
      help="Num examples for testing correlation (default: 200)",
  )
  parser.add_argument(
      "-v",
      "--validation_method",
      type=str,
      default="quantile",
      choices=["quantile", "binary", "rank", "pairwise"],
      help="Method for validating the feature description",
  )
  parser.add_argument(
      "-i",
      "--highlight_method",
      type=str,
      default="angled",
      choices=["xml", "angled", "none"],
      help="Method for highlighting large activations in the text",
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
      "-g",
      "--group_size",
      type=int,
      default=5,
      help="Number of examples to run prediction on at once (default: 5)",
  )
  parser.add_argument(
      "-nq",
      "--n_quantiles",
      type=int,
      default=4,
      help="Number of quantiles to split data into (default: 4)",
  )
  parser.add_argument(
      "--use_snippets",
      action="store_true",
      help="If true, use range around max activation rather than full text",
  )
  parser.add_argument(
      "--use_gemini_batch_api",
      action="store_true",
      help="If true, create a gemini batch API request",
  )
  parser.add_argument(
      "--use_all_active_features",
      action="store_true",
      help="If true, evaluate all SAE features that are active on the dataset",
  )
  parser.add_argument(
      "-m",
      "--model_name",
      type=str,
      default="gemini-2.5-flash",
      choices=["gemini-2.5-flash", "gemini-2.5-pro"],
      help="LLM to use for validating features and generating descriptions",
  )
  parser.add_argument(
      "-o",
      "--output_file",
      type=str,
      default="feature_results.jsonl",
      help="output file for feature correlation results",
  )
  parser.add_argument(
      "-e",
      "--experiment_dir",
      type=str,
      help="directory for the experiment",
  )
  args = parser.parse_args()

  ds = datasets.load_from_disk(args.data_path)
  cache = MaxActivationCache(
      os.path.join(args.data_path, "max_activations.pth")
  )
  extractor = SAEExtractor()

  validator_keyword_args = {
      "validation_method": args.validation_method,
      "highlight_method": args.highlight_method,
      "negative_sampling_method": args.negative_sampling_method,
      "quantile_method": args.quantile_method,
      "prompt_sampling_method": args.prompt_sampling_method,
      "n_quantiles": args.n_quantiles,
      "use_snippets": args.use_snippets,
      "group_size": args.group_size,
      "model_name": args.model_name
  }

  description_validator = FeatureDescriptionValidator(
      ds, extractor, cache, **validator_keyword_args
  )

  feature_ids = (
      _get_k_most_dense_features(cache, args.top_k)
      if not args.feature_list
      else args.feature_list
  )

  if args.use_all_active_features:
    feature_ids = [
        i for i, density in enumerate(cache.get_feature_density())
        if density >= args.num_test_examples
    ]

  if args.use_gemini_batch_api:
    run_validation_experiment_batch(
        description_validator,
        feature_ids,
        validator_keyword_args,
        args.experiment_dir,
        args.num_test_examples,
    )
  else:
    with open(args.output_file, "a") as file:
      for output_record in run_validation_experiment(
          description_validator,
          feature_ids,
          args.use_neuronpedia_descriptions,
          args.num_test_examples,
      ):
        json_line = json.dumps(output_record)
        file.write(json_line + "\n")
