# Copyright 2026 Google LLC
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

"""Builds the PRDB dataset from the raw AI Peer Review Detection Benchmark.

Running this file assumes that you've already run the following:
```
git lfs install
git clone huggingface.co/datasets/IntelLabs/AI-Peer-Review-Detection-Benchmark
```

The raw release is one csv per <split>/<model>/<conference>.csv with a
different column layout per conference. Every row is standardised to the
columns id, paper_id, review_id, split, conference, model, review, summary,
confidence, rating, and only review_ids with exactly one human, one gpt4o and
one llama review are kept. The result is a DatasetDict with splits train /
eval / test (SPLIT_MAP) saved with save_to_disk; data/data/PRDB in the paper
is this output.

Usage: python -m detectorscope.data.format_prdb -d <raw dir> -o data/data/PRDB
"""

import argparse
import pathlib
import datasets
import pandas as pd
from titlecase import titlecase

STRUCTURED = [
    "ICLR2022",
    "ICLR2023",
    "ICLR2024",
    "NeurIPS2021",
    "NeurIPS2022",
    "NeurIPS2023",
    "NeurIPS2024",
]

STRUCTURED_COLS = [
    "summary_of_the_paper",
    "main_review",
    "strength_and_weaknesses",
    "summary_of_the_review",
    "clarity_quality_novelty_and_reproducibility",
    "summary",
    "strengths",
    "weaknesses",
    "questions",
    "strengths_and_weaknesses",
    "limitations",
    "limitations_and_societal_impact",
]

KEY_MAPPING = {
    "rating": "parsed_rating",
    "recommendation": "parsed_rating",
    "overall_score": "parsed_rating",
    "overall score": "parsed_rating",
    "confidence": "parsed_confidence",
    "confidence_score": "parsed_confidence",
    "confidence score": "parsed_confidence",
    "summary": "parsed_summary",
    "summary_of_the_paper": "parsed_summary",
    "review": "parsed_text",
}

DATA_COLS = [
    "parsed_text",
    "parsed_summary",
    "parsed_confidence",
    "parsed_rating",
]
INDEX_COLS = [
    "custom_id",
    "PaperId",
    "review-id",
    "split",
    "conference",
    "model",
]

SPLIT_MAP = {"extended": "train", "calibration": "eval", "test": "test"}


def format_structured_review(row: pd.Series) -> str:
  """Joins a row's structured review sections into one review string.

  Each STRUCTURED_COLS section becomes a bold title line followed by its text,
  in the row's column order; NaN sections are skipped.
  """
  structured_text = ""
  for key, value in row.items():
    if key in STRUCTURED_COLS:
      try:
        title = titlecase(str(key).replace("_", " "))

        # value.strip() raises AttributeError when the section is NaN.
        structured_text += f"**{title}**\n{value.strip()}\n\n"
      except AttributeError:
        pass
  return structured_text.strip()


def process_dataset_row(row: pd.Series) -> pd.Series:
  """Standardizes the original dataset's format across conferences.

  Appends the four DATA_COLS to the row. Structured conferences (STRUCTURED)
  get their section columns joined into parsed_text; the others carry a single
  review column that KEY_MAPPING renames.

  Raises:
    ValueError: if a DATA_COLS column could not be filled from the row.
  """
  output = {}
  for key, value in row.items():
    if key in KEY_MAPPING:
      assert KEY_MAPPING[key] not in output
      output[KEY_MAPPING[key]] = str(value).strip()

  if row["conference"] in STRUCTURED:
    output["parsed_text"] = format_structured_review(row)

  # Rows with no mapped summary column get an empty summary.
  if "parsed_summary" not in output:
    output["parsed_summary"] = ""

  if not all([c in output.keys() for c in DATA_COLS]):
    raise ValueError(f"Only found {list(output.keys())} for row {row}")

  # label is authoritative: label 0 rows are the human review, whatever
  # directory they came from.
  row["model"] = "human" if row["label"] == 0 else row["model"]

  return pd.concat([row, pd.Series(output)])


def check_group(s: pd.Series) -> bool:
  """True iff a review_id group has one human, one llama and one gpt4o row."""
  required_models = {"human", "llama", "gpt4o"}
  model_counts = s.value_counts()

  if not required_models.issubset(model_counts.index):
    return False

  for model in required_models:
    if model_counts[model] != 1:
      return False

  return True


def build_dataset(data_path: str) -> datasets.Dataset:
  """Processes the raw data .csv files into a huggingface dataset.

  Args:
    data_path: root of the raw clone, laid out as <split>/<model>/<conf>.csv.

  Returns:
    A DatasetDict keyed by the SPLIT_MAP values (train, eval, test).
  """
  all_dataframes = []
  for file_path in list(pathlib.Path(data_path).rglob("*.csv")):
    split, model, f_name = file_path.relative_to(pathlib.Path(data_path)).parts
    conference = f_name.split(".")[0]

    print(f"Processing: {split}, {model}, {conference}...")
    df = pd.read_csv(file_path)
    df = df.assign(split=split, model=model, conference=conference)
    df = df.apply(process_dataset_row, axis=1, result_type="expand")

    df = df[INDEX_COLS + DATA_COLS].rename(
        columns={
            "custom_id": "id",
            "PaperId": "paper_id",
            "review-id": "review_id",
            "parsed_text": "review",
            "parsed_summary": "summary",
            "parsed_confidence": "confidence",
            "parsed_rating": "rating",
        }
    )

    all_dataframes.append(df)

  # Deduplicate (the human review recurs across generator dirs) and remove
  # null reviews.
  df = pd.concat(all_dataframes, ignore_index=True)
  df = df.drop_duplicates(subset=["review_id", "model"])
  df = df[~df["review"].isnull()]

  # Ensure one AI review from each model for each human review.
  df = df[df.groupby("review_id")["model"].transform(check_group)]

  ds = datasets.DatasetDict()
  for s in df["split"].unique():
    ds[SPLIT_MAP[s]] = datasets.Dataset.from_pandas(df[df["split"] == s])

  return ds


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-d",
      "--data_path",
      type=str,
      help="Location of the raw dataset files",
  )
  parser.add_argument(
      "-o",
      "--output_path",
      type=str,
      help="Output path to save the dataset",
  )
  args = parser.parse_args()

  dataset = build_dataset(args.data_path)
  dataset.save_to_disk(args.output_path)
