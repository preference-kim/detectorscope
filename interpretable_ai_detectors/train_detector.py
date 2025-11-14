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

"""Trains a logistic regression model to detect AI-generated text."""

import argparse
import datetime

import datasets
from interpretable_ai_detectors.model.detector import TorchLogisticRegression
import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.utils import tensorboard
import torchmetrics
import tqdm


def load_dataset(
    path: str, batch_size: int, num_examples: int | None, test_size: int
) -> tuple[data.DataLoader, data.DataLoader, int]:
  """Load dataset from local path and process for training."""
  ds = datasets.load_from_disk(path)

  if num_examples:
    ds = ds.select(range(num_examples))  # pytype: disable=attribute-error

  # Set human = 0 and AI = 1
  ds = ds.select_columns(["condition", "hook_sae_input"])
  ds = ds.cast_column("condition", datasets.ClassLabel(names=["Human", "GPT4"]))
  ds = ds.with_format("torch", dtype=torch.float32)

  # Split into train and test
  split = ds.train_test_split(test_size=test_size, shuffle=True)
  train_loader = data.DataLoader(
      split["train"], batch_size=batch_size, num_workers=4, shuffle=True
  )
  test_loader = data.DataLoader(
      split["test"], batch_size=batch_size, num_workers=4, shuffle=False
  )

  # Set the feature dimension to the maximum size element to account for
  # different shapes of token activations (This is necessary due to changes
  # made in data processing script -- and me not wanting to recalculate)
  input_dim = torch.max(torch.tensor(ds[0]["hook_sae_input"].size()))

  return train_loader, test_loader, input_dim.item()


def evaluate(
    model: torch.nn.Module, data_loader: data.DataLoader, device: torch.device
) -> float:
  """Calculates model accuracy on a given dataset."""
  model.eval()
  metric = torchmetrics.Accuracy("binary")
  with torch.no_grad():
    for batch in data_loader:
      x_out = model(batch["hook_sae_input"].squeeze().to(device))
      metric.update(x_out.round().cpu(), batch["condition"].unsqueeze(-1))
  return metric.compute().item()  # pytype: disable=attribute-error


def train(
    data_path: str,
    num_examples: int,
    learning_rate: float,
    batch_size: int,
    test_size: int,
    eval_every: int,
    tensorboard_dir: str,
    model_out_prefix: str,
):
  """Train logistic regression on the data from data_path."""
  device = "cuda" if torch.cuda.is_available() else "cpu"
  train_loader, test_loader, input_features = load_dataset(
      data_path, batch_size, num_examples, test_size
  )

  model = TorchLogisticRegression(input_features).to(device).train()
  criterion = nn.BCELoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  writer = tensorboard.SummaryWriter(f"{tensorboard_dir}_{timestamp}")

  with tqdm.tqdm(train_loader, desc="Training Model") as pbar:
    for i, batch in enumerate(pbar):
      model.train()

      batch_x = batch["hook_sae_input"].squeeze().to(device)
      batch_y = batch["condition"].unsqueeze(-1).to(device)

      outputs = model(batch_x)
      loss = criterion(outputs, batch_y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      writer.add_scalar("Loss/train", loss.item(), global_step=i)

      if i % eval_every == 0:
        acc = evaluate(model, test_loader, device)
        writer.add_scalar("Accuracy/val", acc, global_step=i)
        pbar.set_postfix(test_acc=f"{acc:.2%}", train_loss=loss.item())

  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  torch.save(model.state_dict(), f"{model_out_prefix}_{timestamp}.pth")
  writer.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-d",
      "--data_path",
      type=str,
      default="/home/liamdugan_google_com/sae_activations_all",
      help="Path to the dataset",
  )
  parser.add_argument(
      "-n",
      "--num_examples",
      type=int,
      help="Num examples to train on (Optional, defaults to full dataset)",
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
      default=32,
      help="Batch Size for training model",
  )
  parser.add_argument(
      "-t",
      "--test_size",
      type=int,
      default=20000,
      help="Number of examples to use for test set.",
  )
  parser.add_argument(
      "-e",
      "--eval_every",
      type=int,
      default=100,
      help="Evaluate on test data every X batches",
  )
  parser.add_argument(
      "-s",
      "--tensorboard_dir",
      type=str,
      default="tensorboard/logreg_train",
      help="Parent directory to write tensorboard logs to",
  )
  parser.add_argument(
      "-m",
      "--model_out_prefix",
      type=str,
      default="detectors/logreg",
      help="Name for output model",
  )
  args = parser.parse_args()

  # Disable caching to ensure that huggingface doesn't save the intermediate
  # steps of computation. This takes up twice as much disk space and can
  # cause OOM errors on large datasets.
  # NOTE: This is not necessary for machines with lots of extra disk space.
  datasets.disable_caching()

  train(
      args.data_path,
      args.num_examples,
      args.learning_rate,
      args.batch_size,
      args.test_size,
      args.eval_every,
      args.tensorboard_dir,
      args.model_out_prefix,
  )
