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

"""This script trains a Sparse Autoencoder (SAE) model on a given dataset.

It includes functionalities for loading and preprocessing datasets, training the
SAE with various loss components and regularization, and logging metrics to
TensorBoard. The trained model can be saved to a specified output path.
"""

import argparse
import collections
import datetime
import math
import random

import datasets
from interpretable_ai_detectors import utils
from interpretable_ai_detectors.sae.sae import AutoSAE
import numpy as np
import sae_lens
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils import data
from torch.utils import tensorboard
import tqdm
from typing_extensions import override


class TokenShuffleDataset(data.IterableDataset):
  """Dataset that shuffles tokens for training full sequence SAEs."""

  def __init__(
      self,
      ds: datasets.Dataset,
      col: str,
      buf_size: int = 1000000,
      seed: int = 42,
  ):
    super().__init__()
    self.ds = ds
    self.col = col
    self.buf_size = buf_size
    self.seed = seed
    self.epoch = 0

  @override
  def __len__(self):
    # HACK: 650 is roughly the average number of tokens per example in the
    # dataset. This is used to set the tqdm bar for training.
    # TODO(liamdugan): Replace with a more accurate estimate.
    return 650 * len(self.ds)

  def _get_row_iterator(self, worker_id, num_workers, epoch):
    shuffled_ds = self.ds.shuffle(seed=self.seed + epoch, keep_in_memory=False)

    indices_per_worker = math.ceil(len(self.ds) / num_workers)
    start_idx = worker_id * indices_per_worker
    end_idx = min(start_idx + indices_per_worker, len(self.ds))

    for i in range(start_idx, end_idx):
      yield shuffled_ds[i]

  @override
  def __iter__(self):
    worker_info = data.get_worker_info()
    worker_id = worker_info.id if worker_info else 0
    num_workers = worker_info.num_workers if worker_info else 1

    # TODO(myuzaki): switch to using a single rng for all workers and split seed
    self.epoch += 1
    rng = random.Random(self.seed * (self.epoch + 1) + worker_id)
    buffer = []

    for row in self._get_row_iterator(worker_id, num_workers, self.epoch):
      for i in range(1, row[self.col].shape[1]):
        buffer.append(row[self.col][0, i, :])
        if len(buffer) >= self.buf_size:
          rng.shuffle(buffer)
          while buffer:
            yield {self.col: buffer.pop()}

    rng.shuffle(buffer)
    while buffer:
      yield {self.col: buffer.pop()}


def deduplicate_dataset(ds: datasets.Dataset, col: str) -> datasets.Dataset:
  """Deduplicate dataset across column name (col)."""
  df = ds.select_columns([col]).to_pandas()
  unique_indices = df.drop_duplicates(subset=[col], keep="first").index
  return ds.select(unique_indices)


def balance_dataset(
    ds: datasets.Dataset, col: str, seed: int = 42
) -> datasets.Dataset:
  """Balance dataset on values in the input column (col)."""
  random.seed(seed)
  df = ds.select_columns([col]).to_pandas()
  grouped_indices = df.groupby(col).groups.values()
  min_size = min([len(group) for group in grouped_indices])
  sampled = [random.sample(list(i), min_size) for i in grouped_indices]
  return ds.select(utils.flatten(sampled))


def get_avg_activation_norm(train_loader: data.DataLoader, col: str) -> float:
  """Get average L2 norm of activations in the training data."""
  return torch.cat([b[col].norm(dim=1) for b in train_loader]).mean().item()


def store_gradients(model: nn.Module) -> list[torch.Tensor]:
  """Return a copy of the gradients attached to model."""
  return [
      p.grad.clone() if p.grad is not None else None
      for p in model.parameters()
  ]


def get_total_gradient_norm(model: nn.Module) -> float:
  """Compute total L2 norm of all gradients attached to model."""
  gradients = [
      g.data.flatten() for g in store_gradients(model) if g is not None
  ]
  return torch.cat(gradients).norm().item()


def recombine_gradients(
    model: nn.Module, stored_gradients: list[torch.Tensor]
) -> nn.Module:
  """Add all stored gradients to existing model gradients."""
  for stored_grad, param in zip(stored_gradients, model.parameters()):
    if stored_grad is None:
      continue

    param.grad = (
        stored_grad if param.grad is None else param.grad + stored_grad
    )
  return model


def log_metrics(
    loss: dict[torch.Tensor],
    split: str,
    step: int,
    coeff: float,
    e2e_coeff: float,
    writer: tensorboard.SummaryWriter,
):
  """Log loss metrics to tensorboard."""
  for key, value in loss.items():
    writer.add_scalar(f"{key}_loss/{split}", value.mean(), step)
    if key in {"l0", "l1"}:
      scaled_value = (value / coeff).mean()
      writer.add_scalar(f"{key}/{split}", scaled_value, step)
    elif key in {"e2e"}:
      scaled_value = (value / e2e_coeff).mean()
      writer.add_scalar(f"{key}/{split}", scaled_value, step)


def load_dataset(
    data_path: str,
    col_name: str,
    batch_size: int,
    num_examples: int | None,
    test_size: int,
    dedup: bool,
    dedup_col: str,
    balance: bool,
    balance_col: str,
    use_labels: bool,
    label_col: str,
    target_model: str,
    do_token_shuffle: bool,
) -> tuple[data.DataLoader, data.DataLoader]:
  """Loads dataset from local path, and creates the train and test dataloaders.

  Args:
      data_path (str): the local path to load the dataset from
      col_name (str): the name of the column that contains the activations
      batch_size (int): the size of the batches to load
      num_examples (int): size of the dataset (if none, default to full)
      test_size (int): number of examples to leave out as test data
      dedup (bool): if True, deduplicate the dataset on dedup_col
      dedup_col (str): column name to deduplicate on
      balance (bool): if True, balance the dataset on balance_col
      balance_col (str): column to balance the dataset across
      use_labels (bool): if True, include binary labels in the dataset
      label_col (str): column to use to compute labels
      target_model (str): if non-empty, filter to only include target model
      do_token_shuffle (bool): if true, shuffle tokens from different sequences

  Returns:
      a tuple of Dataloader objects (train_loader, test_loader)
  """
  ds = datasets.load_from_disk(data_path)

  if target_model:
    ds = ds.filter(lambda x: x[label_col].lower() in {"human", target_model})

  def add_label_column(row):
    """Convert model name to binary and add as new column 'labels'."""
    return {"labels": (np.char.lower(row) != "human").astype(int)}

  if dedup:
    ds = deduplicate_dataset(ds, dedup_col)
  if balance:
    ds = balance_dataset(ds, balance_col)
  if num_examples:
    ds = ds.select(range(num_examples))  # pytype: disable=attribute-error

  if use_labels:
    ds = ds.with_format("numpy")
    ds = ds.map(
        add_label_column,
        input_columns=[label_col],
        batched=True,
        num_proc=8,
    )
    ds = ds.select_columns([col_name, "labels"])
  else:
    ds = ds.select_columns([col_name])

  ds = ds.with_format("torch", dtype=torch.float32)
  split = ds.train_test_split(test_size=test_size, shuffle=True, seed=42)

  if do_token_shuffle:
    train_loader = data.DataLoader(
        TokenShuffleDataset(split["train"], col_name),
        batch_size=batch_size,
        num_workers=8,
    )
    test_loader = data.DataLoader(
        TokenShuffleDataset(split["test"], col_name),
        batch_size=batch_size,
        num_workers=8,
    )
  else:
    train_loader = data.DataLoader(
        split["train"], batch_size=batch_size, num_workers=4, shuffle=True
    )
    test_loader = data.DataLoader(
        split["test"], batch_size=batch_size, num_workers=4, shuffle=False
    )

  return train_loader, test_loader


def evaluate(
    model: nn.Module,
    data_loader: data.DataLoader,
    col_name: str,
    sparsity_coeff: float,
    e2e_coeff: float,
) -> dict[str, torch.Tensor]:
  """Evaluates the SAE on test data. Returns dictionary of losses."""
  model.eval()
  losses = collections.defaultdict(list)
  with torch.no_grad():
    with tqdm.tqdm(total=len(data_loader), desc="Evaluating Model") as pbar:
      for batch in data_loader:
        labels = batch["labels"].to(model.device) if "labels" in batch else None
        batch = batch[col_name].to(model.device)

        reconst, features = model.forward(batch)
        loss = model.loss_fn(
            batch, reconst, features, sparsity_coeff, e2e_coeff, labels
        )

        mean_mse = ((batch - batch.mean(dim=0)) ** 2).mean(dim=-1)
        loss["total"] = sum(loss.values())
        loss["fvu"] = loss["mse"] / mean_mse

        for key in loss.keys():
          losses[key].append(loss[key])

        pbar.update(1)

  return {k: torch.cat(v) for k, v in losses.items()}


def train(
    model: AutoSAE,
    train_loader: data.DataLoader,
    test_loader: data.DataLoader,
    col: str,
    tensorboard_dir: str,
    num_epochs: int,
    eval_every: int,
    sparsity_coeff: float,
    e2e_coeff: float,
    learning_rate: float,
    adam_beta1: float,
    sparsity_warmup_steps: int = 10000,
    lr_warmup_steps: int = 1000,
    gradient_clip_max_norm: float = 6.0,
    freeze_encoder: bool = False,
    separate_inference: bool = False,
    clip_gradients: bool = False,
) -> AutoSAE:
  """Train an SAE model.

  Args:
      model (AutoSAE): an SAE initialized to pre-trained weights
      train_loader (str): dataloader for training data
      test_loader (str): dataloader for test data
      col (str): the name of the dataset column where activations are located
      tensorboard_dir (str): directory to save tensorboard plots
      num_epochs (int): number of epochs to train for
      eval_every (int): num batches before we run eval
      sparsity_coeff (float): coefficient for the sparsity loss penalty
      e2e_coeff (float): coefficient for the end-to-end loss term
      learning_rate (float): learning rate to use in training
      adam_beta1 (float): value for beta1 term in Adam optimizer
      sparsity_warmup_steps (int): number of steps to use for sparsity warmup
      lr_warmup_steps (int): number of steps to use for learning rate warmup
      gradient_clip_max_norm (float): max norm when clip_gradients is True
      freeze_encoder (bool): if true, freeze encoder weights during training
      separate_inference (bool): if true, separate backward pass per loss term
      clip_gradients (bool): if true, clip gradients during training

  Returns:
      the final trained model (AutoSAE)
  """
  if freeze_encoder:
    for param in model.encoder.parameters():
      param.requires_grad = False

  optimizer = optim.Adam(
      filter(lambda p: p.requires_grad, model.parameters()),
      lr=learning_rate,
      betas=(adam_beta1, 0.999),
  )

  def cosine_lr_schedule(step: float, start_factor: float = 0.1) -> float:
    """Cosine warmup scheduler from start_factor -> 1.0."""
    if step >= lr_warmup_steps:
      return 1.0
    progress = step / lr_warmup_steps
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * (progress - 1.0)))
    return start_factor + (1.0 - start_factor) * cos_factor

  scheduler = optim.lr_scheduler.LambdaLR(
      optimizer, lr_lambda=cosine_lr_schedule
  )

  writer = tensorboard.SummaryWriter(tensorboard_dir)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)

  max_sparsity_coeff = sparsity_coeff
  total_size = len(train_loader) * num_epochs
  batch_size = train_loader.batch_size
  with tqdm.tqdm(total=total_size, desc="Training Model") as pbar:
    for epoch in range(num_epochs):
      for i, batch in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()

        labels = batch["labels"].to(device) if "labels" in batch else None
        batch = batch[col].to(device)
        step = (epoch * len(train_loader) + i) * batch_size

        # Linear warmup for sparsity coefficient
        if (s := int(step / batch_size)) <= sparsity_warmup_steps:
          progress_frac = float(s) / float(sparsity_warmup_steps)
          sparsity_coeff = progress_frac * max_sparsity_coeff

        writer.add_scalar("Sparsity_coeff", sparsity_coeff, step)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], step)

        reconst, features = model.forward(batch)
        loss = model.loss_fn(
            batch, reconst, features, sparsity_coeff, e2e_coeff, labels
        )

        # Apply separate backward passes for each loss component
        # (Used for debug purposes to analyze norm of gradients on
        # individual loss components)
        if separate_inference:
          stored_grads = []
          for key, loss_value in loss.items():
            loss_value.mean().backward(retain_graph=True)
            norm = get_total_gradient_norm(model)
            stored_grads.append(store_gradients(model))
            writer.add_scalar(f"grad_norm/{key}", norm, step)
            optimizer.zero_grad()

          for grad in stored_grads:
            model = recombine_gradients(model, grad)
        else:
          sum(loss.values()).mean().backward()  # pytype: disable=attribute-error

        # Project out gradient components parallel to decoder rows
        if model.W_dec.grad is not None:
          with torch.no_grad():
            dot_products = model.W_dec.grad * model.W_dec
            sum_dot = dot_products.sum(dim=1, keepdim=True)
            model.W_dec.grad.sub_(sum_dot * model.W_dec)

        # Clip gradients to specified max norm
        if clip_gradients:
          torch.nn.utils.clip_grad_norm_(
              model.parameters(), gradient_clip_max_norm
          )

        # Log total gradient norm
        total_norm = get_total_gradient_norm(model)
        writer.add_scalar("grad_norm/Total", total_norm, step)

        optimizer.step()
        scheduler.step()

        # Renormalize decoder rows
        model.W_dec.data = F.normalize(model.W_dec.data, p=2, dim=1)

        # Compute fraction of variance unexplained (FVU)
        mean_mse = ((batch - batch.mean(dim=0)) ** 2).mean(dim=-1)
        loss["total"] = sum(loss.values())
        loss["fvu"] = loss["mse"] / mean_mse

        # Log metrics to tensorboard
        log_metrics(
            loss, "train", step, sparsity_coeff, e2e_coeff, writer
        )

        # Evaluate on test data every N batches
        if i % eval_every == 1:
          val_loss = evaluate(
              model, test_loader, col, sparsity_coeff, e2e_coeff
          )
          log_metrics(
              val_loss, "val", step, sparsity_coeff, e2e_coeff, writer
          )

          pbar.set_postfix(
              epoch=f"{epoch + 1}/{num_epochs}",
              test_loss=f"{val_loss['total'].mean().item():.2f}",
              train_loss=f"{loss['total'].mean().item():.2f}",
          )

        pbar.update(1)

  return model


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-s",
      "--base_sae_release",
      type=str,
      default="gemma-scope-2b-pt-res-canonical",
      help="Name of the base SAE release to fine-tune",
  )
  parser.add_argument(
      "-i",
      "--base_sae_id",
      type=str,
      default="layer_25/width_16k/canonical",
      help="ID of the base SAE to fine-tune",
  )
  parser.add_argument(
      "-d",
      "--data_path",
      type=str,
      default="./activations/all_activations",
      help="Dataset of model activations to use for training",
  )
  parser.add_argument(
      "-c",
      "--column_name",
      type=str,
      default="hook_sae_input",
      help="Column name in the dataset that holds the cached activations",
  )
  parser.add_argument(
      "-n",
      "--num_examples",
      type=int,
      default=None,
      help="Number of examples to use in training",
  )
  parser.add_argument(
      "-z",
      "--test_size",
      type=int,
      default=10000,
      help="Number of examples to use for testing",
  )
  parser.add_argument(
      "-p",
      "--num_epochs",
      type=int,
      default=1,
      help="Number of epochs to run training over",
  )
  parser.add_argument(
      "-f",
      "--sparsity_coeff",
      type=float,
      default=6e-4,
      help="Coefficient for sparsity penalty (L0 or L1 depending on the SAE)",
  )
  parser.add_argument(
      "-e",
      "--eval_every",
      type=int,
      default=500,
      help="Eval every N batches",
  )
  parser.add_argument(
      "-l",
      "--learning_rate",
      type=float,
      default=7e-5,
      help="Learning Rate for training model",
  )
  parser.add_argument(
      "-b",
      "--batch_size",
      type=int,
      default=1024,
      help="Batch size to use for training",
  )
  parser.add_argument(
      "-w",
      "--bandwidth",
      type=float,
      default=0.001,
      help="Bandwidth to use for KDEs in JumpReLU training",
  )
  parser.add_argument(
      "-a",
      "--accelerator",
      type=float,
      default=1.0,
      help="Accelerator to use for threshold gradients in JumpReLU",
  )
  parser.add_argument(
      "-sw",
      "--sparsity_warmup_steps",
      type=int,
      default=10000,
      help="Number of steps to use for the sparsity penalty warmup",
  )
  parser.add_argument(
      "-lw",
      "--lr_warmup_steps",
      type=int,
      default=1000,
      help="Number of steps to use for the learning rate warmup",
  )
  parser.add_argument(
      "-gc",
      "--gradient_clip_max_norm",
      type=float,
      default=6.0,
      help="Max norm to use when clipping gradients",
  )
  parser.add_argument(
      "-t",
      "--tensorboard_dir",
      type=str,
      default="./tensorboard/sae_indep_search",
      help="Directory to save tensorboard plots",
  )
  parser.add_argument(
      "-o",
      "--output_path",
      type=str,
      default="finetuned_sae.pth",
      help="Path to save the fine-tuned SAE model",
  )
  parser.add_argument(
      "-beta",
      "--adam_beta1",
      type=float,
      default=0.0,
      help="Value to use for adam beta1",
  )
  parser.add_argument(
      "--freeze_encoder",
      action="store_true",
      help="If true, freeze the SAE encoder during fine-tuning",
  )
  parser.add_argument(
      "--rescale_model",
      action="store_true",
      help="If true, rescale model during training to operate on unit norm",
  )
  parser.add_argument(
      "--separate_inference",
      action="store_true",
      help="If true, do separate backward passes for loss components",
  )
  parser.add_argument(
      "--dedup",
      action="store_true",
      help="If true, deduplicate training data",
  )
  parser.add_argument(
      "-dc",
      "--dedup_col",
      type=str,
      default="response",
      help="Column to deduplicate dataset across",
  )
  parser.add_argument(
      "--balance",
      action="store_true",
      help="If true, balance training data across Human vs. AI",
  )
  parser.add_argument(
      "-bc",
      "--balance_col",
      type=str,
      default="condition",
      help="Column to balance dataset across",
  )
  parser.add_argument(
      "--clip_gradients",
      action="store_true",
      help="If true, clip gradients",
  )
  parser.add_argument(
      "--use_pre_enc_bias",
      action="store_true",
      help="If true, use a pre-encoder bias",
  )
  parser.add_argument(
      "--normalize_input",
      action="store_true",
      help="If true, scale input activations to unit norm during training",
  )
  parser.add_argument(
      "--init_from_scratch",
      action="store_true",
      help="If true, initialize the model from scratch",
  )
  parser.add_argument(
      "--reset_thresholds",
      action="store_true",
      help="If true, set all thresholds to 0.001 before training",
  )
  parser.add_argument(
      "-ep",
      "--e2e_model_path",
      type=str,
      default=None,
      help="If passed in, apply e2e loss to the SAE",
  )
  parser.add_argument(
      "-ec",
      "--e2e_coeff",
      type=float,
      default=1.0,
      help="Coefficient to use for E2E loss term",
  )
  parser.add_argument(
      "-tm",
      "--target_model",
      type=str,
      default="",
      help="Target model to train on",
  )
  parser.add_argument(
      "--do_token_shuffle",
      action="store_true",
      help="If true, shuffle all token activations in the dataset",
  )
  args = parser.parse_args()

  if args.rescale_model:
    args.use_pre_enc_bias = True
    args.normalize_input = True

  pretrained_sae = sae_lens.SAE.from_pretrained(
      release=args.base_sae_release,
      sae_id=args.base_sae_id,
  )

  sae_model = AutoSAE.from_pretrained(
      pretrained_sae, bandwidth=args.bandwidth, acc=args.accelerator
  )

  # Index tensorboard dir by training parameters and current time
  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  tensorboard_dir_path = (
      f"{args.tensorboard_dir}_{timestamp}_bs{args.batch_size}"
      f"_bw{args.bandwidth}_lr{args.learning_rate}_sp{args.sparsity_coeff}"
      f"_n{args.num_epochs}_a{args.accelerator}_w{args.sparsity_warmup_steps}"
      f"_ec{args.e2e_coeff}"
  )

  print(f"Loading data from {args.data_path}...")
  train_data_loader, test_data_loader = load_dataset(
      args.data_path,
      col_name=args.column_name,
      batch_size=args.batch_size,
      num_examples=args.num_examples,
      test_size=args.test_size,
      dedup=args.dedup,
      dedup_col=args.dedup_col,
      balance=args.balance,
      balance_col=args.balance_col,
      use_labels=bool(args.e2e_model_path is not None),
      label_col=args.balance_col,
      target_model=args.target_model,
      do_token_shuffle=args.do_token_shuffle,
  )

  if args.use_pre_enc_bias:
    sae_model.use_pre_enc_bias = True
  if args.normalize_input:
    sae_model.C = get_avg_activation_norm(train_data_loader, args.column_name)
  if args.rescale_model:
    sae_model.rescale_to_unit_norm()
  if args.init_from_scratch:
    sae_model.initialize()
  if args.reset_thresholds:
    sae_model.reset_thresholds()
  if args.e2e_model_path:
    sae_model.activate_e2e_loss(args.e2e_model_path)

  trained_model = train(
      sae_model,
      train_data_loader,
      test_data_loader,
      args.column_name,
      tensorboard_dir_path,
      args.num_epochs,
      args.eval_every,
      args.sparsity_coeff,
      args.e2e_coeff,
      args.learning_rate,
      args.adam_beta1,
      args.sparsity_warmup_steps,
      args.lr_warmup_steps,
      args.gradient_clip_max_norm,
      args.freeze_encoder,
      args.separate_inference,
      args.clip_gradients,
  )

  if args.rescale_model:
    trained_model.undo_rescaling()

  output_path = f"{tensorboard_dir_path}/{args.output_path}"
  print(f"Saving model to {output_path}...")
  torch.save(trained_model.state_dict(), output_path)
