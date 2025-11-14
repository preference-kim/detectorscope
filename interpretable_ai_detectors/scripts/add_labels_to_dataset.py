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

"""Script to add a "labels" column to an AI detection dataset."""

import argparse

import datasets
import numpy as np


def add_labels_to_dataset(ds: datasets.Dataset, col: str) -> datasets.Dataset:
  """For all elements in the dataset, add a label column.

  Args:
      ds (Dataset): the dataset to run processing on
      col (str): the column name in the dataset containing the text

  Returns:
      A new dataset object with a labels column
  """
  ds = ds.with_format("numpy")

  def add_label_column(row: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Add a label column to the row."""
    return {"labels": (np.char.lower(row) != "human").astype(int)}

  ds = ds.map(add_label_column, input_columns=[col], batched=True, num_proc=8)

  ds.reset_format()

  return ds


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-d",
      "--data_path",
      type=str,
      default="./activations/detect_gemma_sae",
      help="Path to the dataset",
  )
  parser.add_argument(
      "-c",
      "--column",
      type=str,
      default="condition",
      help="The name of the column containing the model",
  )
  parser.add_argument(
      "-o",
      "--output_path",
      type=str,
      default="./activations/detect_gemma_sae_processed",
      help="The path to the output to save the activations",
  )
  args = parser.parse_args()

  dataset = datasets.load_from_disk(args.data_path)
  processed_ds = add_labels_to_dataset(dataset, args.column)
  processed_ds.save_to_disk(args.output_path)
