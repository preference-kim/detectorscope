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

"""Script to process many tensorboard logs into a single csv to compare runs."""

import argparse
import os
import re

import pandas as pd
from tbparse import SummaryReader
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_accumulator import ScalarEvent
import tqdm


LEGEND = {
    "bw": ("bandwidth", float),
    "bs": ("batch_size", int),
    "lr": ("learning_rate", float),
    "sp": ("sparsity_penalty", float),
    "n": ("num_epochs", int),
    "a": ("accelerator", float),
    "w": ("sparsity_warmup_steps", int),
    "ec": ("e2e_coeff", float)
}

PATTERN = re.compile(r"([a-zA-Z]+)([0-9.e-]+)")


def apply_smoothing(scalars: list[ScalarEvent], w: float) -> list[int]:
  """Applies exponential moving average (EMA) smoothing."""
  value = scalars[0].value
  smoothed_values = []
  for i, event in enumerate(scalars):
    value = w * value + (1 - w) * event.value
    smoothed_values.append(value / (1 - w**(i+1)))
  return smoothed_values


def main(log_dir: str, tag: str, weight: float) -> pd.DataFrame:
  """Parse Tensorboard runs at log_dir to extract min value."""
  # Find all directories that contain tensorboard logs
  dirs_with_logs = [
      root for root, _, files in os.walk(log_dir)
      if any(file.startswith("events.out.tfevents") for file in files)
  ]

  records = []
  for root in tqdm.tqdm(dirs_with_logs):
    ea = EventAccumulator(root, size_guidance={"scalars": 0})
    ea.Reload()

    # Parse file name for hyperparameter info
    record = {
        LEGEND[key][0]: LEGEND[key][1](value)
        for key, value in PATTERN.findall(os.path.basename(root))
    }

    # Find the step where scalar value at "tag" is at a minimum
    scalars = ea.Scalars(tag)
    smoothed = apply_smoothing(scalars, weight)
    min_i = smoothed.index(min(smoothed))

    # Get values for all other scalars at this step in training
    for tag in ea.Tags()["scalars"]:
      record[tag] = apply_smoothing(ea.Scalars(tag), weight)[min_i]

    record["step"] = scalars[min_i].step,
    record["max_step"] = max([s.step for s in scalars])
    record["min_at"] = tag
    record["smoothing"] = weight

    records.append(record)

  return pd.DataFrame.from_records(records)


def main_tbparse(log_dir: str) -> pd.DataFrame:
  reader = SummaryReader(log_dir, pivot=True, extra_columns={"dir_name"})
  scalars_df = reader.scalars.dropna()

  def parse_columns(row: pd.Series) -> pd.Series:
    """Parse the columns of the row to extract hyperparameter info."""
    return pd.Series({
        LEGEND[key][0]: LEGEND[key][1](value)
        for key, value in PATTERN.findall(row["dir_name"])
    })

  return scalars_df.join(scalars_df.apply(parse_columns, axis=1))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-d",
      "--log_dir",
      type=str,
      help="The top-level log directory (will search all subdirectories)."
  )
  parser.add_argument(
      "-t",
      "--tag",
      type=str,
      help="The scalar tag to find (e.g., 'loss', 'accuracy')."
  )
  parser.add_argument(
      "-w",
      "--weight",
      type=float,
      default=0.0,
      help="The weight of EMA smoothing to apply"
  )
  parser.add_argument(
      "--use_tbparse",
      action="store_true",
      help="Use tbparse to get the full output from all runs"
  )
  parser.add_argument(
      "-o",
      "--output_path",
      type=str,
      default="output.csv",
      help="The output path to write the pandas dataframe"
  )
  args = parser.parse_args()

  if args.use_tbparse:
    df = main_tbparse(args.log_dir)
  else:
    df = main(args.log_dir, args.tag, args.weight)

  df.to_csv(args.output_path, index=False)
