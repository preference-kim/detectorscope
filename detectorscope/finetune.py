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

"""Fine-tune a Hugging Face model for binary AI-generated text detection.

Pipeline stage 0. Loads a dataset, preprocesses it, and trains a Hugging Face
`AutoModelForSequenceClassification` to distinguish between human-written
and AI-generated text. The checkpoint is the paper's neural baseline
(`evaluate -dt huggingface`) and the extractor of the FT-Model+SAE setting;
the CLI defaults are the fine-tuning hyper-parameters the paper reports.

Usage:
  python -m detectorscope.finetune -m google/gemma-2-2b \
      -d data/data/PRDB/train -t review -c model -r gpt4o -o gpt4o_detect_gemma
"""

import argparse
import datetime
import os

import datasets
from sklearn import metrics
import transformers

# model_col value (lower-cased) for human text; every other value is AI.
HUMAN = "human"
SEED = 42


def load_data(
    dataset_id: str,
    tokenizer: transformers.AutoTokenizer,
    processed_dir: str,
    text_col: str,
    model_col: str,
    target_model: str | None,
) -> tuple[datasets.Dataset, datasets.Dataset]:
  """Load a Huggingface dataset and preprocess it for training.

  Loads dataset_id, keeps only human and target_model rows when a target is
  given, tokenizes to 1024 tokens (truncated and padded), converts the model
  names to binary labels (0 human, 1 AI) and picks the test split: the
  dataset's test, dev or validation split, else a 5% seed-42 holdout of
  train. The result is saved under processed_dir and, when that directory
  already exists, loaded from there instead, whatever the other arguments say.

  Args:
    dataset_id: the name of the dataset on huggingface.
    tokenizer: the tokenizer to apply to the dataset.
    processed_dir: the directory to save to and load from.
    text_col: the column that the text is in.
    model_col: the column for the model name.
    target_model: the target model we are training to detect; None keeps
      every generator.

  Returns:
    A tuple of dataset objects (training data, test data).
  """
  if os.path.exists(processed_dir):
    train_ds = datasets.load_from_disk(os.path.join(processed_dir, "train"))
    test_ds = datasets.load_from_disk(os.path.join(processed_dir, "test"))
    return train_ds, test_ds

  ds = datasets.load_dataset(dataset_id)

  if target_model:
    ds = ds.filter(lambda x: x[model_col].lower() in {HUMAN, target_model})

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

  if "label" not in ds:
    ds = ds.map(add_label_column)

  if "test" not in ds:
    if "dev" in ds:
      ds["test"] = ds.pop("dev")
    elif "validation" in ds:
      ds["test"] = ds.pop("validation")
    else:
      ds = ds["train"].train_test_split(
          test_size=0.05, seed=42
      )  # pytype: disable=attribute-error

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
    eval_every: int,
    test_size: int | None,
):
  """Fine-tune a huggingface model for binary AI-generated text detection.

  Trains for one epoch, evaluating and checkpointing every eval_every steps,
  and keeps the checkpoint with the lowest eval loss. Writes checkpoints to
  {output_dir}/checkpoints, TensorBoard logs to {output_dir}/tensorboard and
  the final model to {output_dir}/final_model.

  Args:
    model_name: the name of the base model on huggingface.
    dataset_name: the dataset to load for training.
    text_col: the column that the text is in.
    model_col: the column for the model name.
    target_model: the target model we are training to detect.
    processed_data_dir: path to save the tokenized dataset.
    output_dir: path to save checkpoints and tensorboard logs.
    lr: the learning rate for training.
    batch_size: the per-device batch size for training and eval.
    eval_every: number of train steps between evals.
    test_size: the size to downsample the test set to (seeded shuffle).
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

  if test_size is not None and test_size < len(test_ds):
    print(
        f"Downsampling test set from {len(test_ds)} to {test_size} examples..."
    )
    test_ds = test_ds.shuffle(seed=SEED).select(range(test_size))

  def compute_metrics(pred):
    """Accuracy and binary-average precision, recall and F1 from argmax."""
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
      logging_steps=100,
      eval_steps=eval_every,
      save_steps=eval_every,
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
  parser.add_argument(
      "-o",
      "--output_tag",
      type=str,
      default="run",
      help="Name of the training run",
  )
  parser.add_argument(
      "-e",
      "--eval_every",
      type=int,
      default=1000,
      help="Number of steps after which we run eval and save a checkpoint",
  )
  parser.add_argument(
      "-s",
      "--test_size",
      type=int,
      default=None,
      help="Size to downsample test data to",
  )
  args = parser.parse_args()

  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  out_dir = f"outputs/{args.output_tag}_{timestamp}_LR{args.learning_rate}"

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
      args.eval_every,
      args.test_size,
  )
