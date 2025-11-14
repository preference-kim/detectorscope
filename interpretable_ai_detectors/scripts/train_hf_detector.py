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

"""Fine-tune a Hugging Face model for binary AI-generated text detection.

This script loads a dataset, preprocesses it, and trains a Hugging Face
`AutoModelForSequenceClassification` to distinguish between human-written
and AI-generated text.
"""

import argparse
import datetime
import os

import datasets
from sklearn import metrics
import transformers

HUMAN = "human"


def load_data(
    dataset_id: str,
    tokenizer: transformers.AutoTokenizer,
    processed_dir: str,
    text_col: str,
    model_col: str,
    target_model: str | None,
) -> tuple[datasets.Dataset, datasets.Dataset]:
  """Load Huggingface Dataset and preprocess it for training.

  Load Huggingface Dataset from dataset_id, tokenize it, split train test,
  and convert categorical class labels to binary. If existing processed
  data is found at processed_data_dir, then load that instead.

  Args:
      dataset_id (str): the name of the dataset on huggingface
      tokenizer (AutoTokenizer): the tokenizer to apply to the dataset
      processed_dir (str): the directory to save to & load from
      text_col (str): the column that the text is in
      model_col (str): the column for the model name
      target_model (str | None): the target model we are training to detect

  Returns:
      a tuple of dataset objects (training data, test data)
  """
  if os.path.exists(processed_dir):
    train_ds = datasets.load_from_disk(os.path.join(processed_dir, "train"))
    test_ds = datasets.load_from_disk(os.path.join(processed_dir, "test"))
    return train_ds, test_ds

  ds = datasets.load_dataset(dataset_id)

  if target_model:
    ds = ds.filter(
        lambda x: x[model_col].lower() in {HUMAN, target_model}
    )

  def tokenize_data_row(row):
    """Tokenize, truncate, and pad to 1024."""
    return tokenizer(
        row[text_col],
        truncation=True,
        max_length=1024,
        padding="max_length",
    )

  ds = ds.map(tokenize_data_row, batched=True)

  def add_label_column(row):
    """Convert model name to binary and add as new column 'labels'."""
    return {"labels": 0 if row[model_col].lower() == HUMAN else 1}

  ds = ds.map(add_label_column)

  if "test" not in ds:
    ds = ds["train"].train_test_split(test_size=0.05, seed=42)  # pytype: disable=attribute-error

  print(f"Saving tokenized dataset to {processed_dir}...")
  os.makedirs(processed_dir, exist_ok=True)
  ds["train"].save_to_disk(os.path.join(processed_dir, "train"))
  ds["test"].save_to_disk(os.path.join(processed_dir, "test"))

  return ds["train"], ds["test"]


def train(
    model_name: str,
    dataset_name: str,
    text_col: str,
    model_col: str,
    target_model: str | None,
    processed_data_dir: str,
    output_dir: str,
    lr: float,
    batch_size: int,
):
  """Fine-tune a huggingface model for binary AI-generated text detection.

  Args:
      model_name (str): the name of the base model on huggingface
      dataset_name (str): the dataset to load for training
      text_col (str): the column that the text is in
      model_col (str): the column for the model name
      target_model (str | None): the target model we are training to detect
      processed_data_dir (str): path to save tokenized dataset
      output_dir (str): path to save checkpoints and tensorboard logs
      lr (float): the learning rate for training
      batch_size (int): the per-device batch size to use for training

  Returns:
      Nothing, but saves model to "{output_dir}/final_model"
  """
  tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
  model = transformers.AutoModelForSequenceClassification.from_pretrained(
      model_name, num_labels=2
  )

  train_ds, test_ds = load_data(
      dataset_name,
      tokenizer,
      processed_data_dir,
      text_col,
      model_col,
      target_model,
  )

  def compute_metrics(pred):
    acc = metrics.accuracy_score(pred.label_ids, pred.predictions.argmax(-1))
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        pred.label_ids, pred.predictions.argmax(-1), average="binary"
    )
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

  training_args = transformers.TrainingArguments(
      output_dir=f"{output_dir}/checkpoints",
      num_train_epochs=1,
      per_device_train_batch_size=batch_size,
      per_device_eval_batch_size=batch_size,
      warmup_steps=500,
      weight_decay=0.01,
      logging_dir=f"{output_dir}/tensorboard",
      logging_steps=100,
      eval_steps=1000,
      save_steps=1000,
      save_strategy="steps",
      eval_strategy="steps",
      load_best_model_at_end=True,
      report_to="tensorboard",
      learning_rate=lr,
      save_total_limit=5,
      metric_for_best_model="eval_loss",
  )

  trainer = transformers.Trainer(
      model=model,
      args=training_args,
      train_dataset=train_ds,
      eval_dataset=test_ds,
      compute_metrics=compute_metrics,
  )

  trainer.train()

  results = trainer.evaluate()
  print(f"Final Results {results}")

  trainer.save_model(f"{output_dir}/final_model")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-m",
      "--model_name",
      type=str,
      default="google/gemma-2-2b",
      help="Name of the Huggingface Model to train",
  )
  parser.add_argument(
      "-d",
      "--data",
      type=str,
      default="imone/OpenOrca_FLAN",
      help="Name of Huggingface dataset to use for training",
  )
  parser.add_argument(
      "-t",
      "--text_col",
      type=str,
      default="response",
      help="Name of text column in Huggingface dataset",
  )
  parser.add_argument(
      "-c",
      "--model_col",
      type=str,
      default="condition",
      help="Name of model column in Huggingface dataset",
  )
  parser.add_argument(
      "-r",
      "--target_model",
      type=str,
      default=None,
      help="Name of model to train detector against",
  )
  parser.add_argument(
      "-p",
      "--processed_data_dir",
      type=str,
      default="data/openflan_processed",
      help="Directory to use to cache tokenized dataset",
  )
  parser.add_argument(
      "-l",
      "--learning_rate",
      type=float,
      default=5e-5,
      help="Learning Rate for training model",
  )
  parser.add_argument(
      "-b",
      "--batch_size",
      type=int,
      default=6,
      help="Batch size to use for training",
  )
  args = parser.parse_args()

  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  out_dir = f"outputs/{timestamp}_LR{args.learning_rate}"

  train(
      args.model_name,
      args.data,
      args.text_col,
      args.model_col,
      args.target_model,
      args.processed_data_dir,
      out_dir,
      args.learning_rate,
      args.batch_size,
  )
