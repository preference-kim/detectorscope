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

"""Script to process completed batch jobs from the Generative AI API.

This script queries the Generative AI API for completed batch jobs within a
specified results folder, downloads the generated content, and computes
correlations based on the model responses and group data. The results are
saved in a `correlations.jsonl` file within each job's directory.
"""

import argparse
import json
import os
import pathlib
import pickle
from typing import Any

from google import genai
from interpretable_ai_detectors.model.autointerpret import FeatureDescriptionValidator


COMPLETED_JOB_STATES = [
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
]


def process_content_file(file_content: str):
  json_content = [json.loads(c) for c in file_content.strip().split("\n")]

  def parse_response(c: dict[str, Any]) -> str:
    """Helper function to extract the text response from an output dict."""
    try:
      return c["response"]["candidates"][0]["content"]["parts"][0]["text"]
    except KeyError:
      return ""

  return {c["key"]: parse_response(c) for c in json_content}


def process_completed_jobs(results_folder: str):
  """Query the API for completed jobs and compute correlations."""
  # Construct dict of job ID -> folder name
  subdirs = [x for x in pathlib.Path(results_folder).iterdir() if x.is_dir()]
  job_id_dict = {os.path.basename(x).split("_")[-1]: str(x) for x in subdirs}

  client = genai.Client()

  # Loop and look for completed batch jobs
  batches = client.batches.list(config={"page_size": 50})

  for b in batches.page:
    if b.state.name not in COMPLETED_JOB_STATES:  # pytype: disable=attribute-error
      continue
    if not (b.dest and b.dest.file_name and b.state and b.name):
      continue

    # If the job ID doesn't have a folder, then take note and continue
    if (job_id := b.name.split("/")[-1]) not in job_id_dict:
      print(f"Skipping {job_id} ({b.state.name}) - no folder")
      continue

    job_dir = job_id_dict[job_id]

    # If the correlations.jsonl file already exists, skip
    correlations_path = os.path.join(job_dir, "correlations.jsonl")
    if os.path.exists(correlations_path):
      print(f"Skipping {job_id} ({b.state.name}) - has corr")
      continue

    # Load in the items from the folder
    print(f"Processing job {job_id} ({b.state.name}) at {job_dir}")
    with open(os.path.join(job_dir, "groups.pkl"), "rb") as f:
      group_dict = pickle.load(f)
    with open(os.path.join(job_dir, "metadata.json"), "r") as f:
      metadata = json.load(f)

    # Get the validation method from the validator kwargs
    validation_method = metadata["validator_kwargs"][
        "validation_method"
    ]

    # Download model responses and convert to dict from UUID -> LLM response
    responses_dict = process_content_file(
        client.files.download(file=b.dest.file_name).decode("utf-8")
    )

    # Write output to the correlations file
    print(f"Writing correlations to {correlations_path}")
    with open(correlations_path, "w+") as f:
      for feature_index, mapping in group_dict.items():
        # Get the LLM response (if it exists) for each prompt
        response_tuples = [
            (responses_dict[key], group)
            for key, group in mapping["groups"].items()
            if key in responses_dict
        ]

        # Parse the responses
        prediction_tuples = [
            (
                FeatureDescriptionValidator.parse_response(
                    response, len(group), validation_method
                ),
                group,
            )
            for response, group in response_tuples
        ]

        # Compute the correlation
        predictions, groups = zip(*prediction_tuples)
        correlation = (
            FeatureDescriptionValidator.compute_correlation(
                predictions, groups, validation_method
            )
        )

        # Write the output to the correlations file
        record = {
            "feature_index": feature_index,
            "num_examples": sum(len(g) for g in groups),
            "description": mapping["description"],
            "correlation": correlation,
        }
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-f",
      "--results_folder",
      type=str,
      help="Experiment folder to check batch jobs for",
  )
  args = parser.parse_args()

  process_completed_jobs(args.results_folder)
