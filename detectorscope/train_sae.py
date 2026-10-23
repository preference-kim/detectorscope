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

"""Fine-tunes Sparse Autoencoders (SAEs) on model activations.

Pipeline stage 1. Loads one pretrained SAE per hook point from a Gemma Scope
repo on the Hub (the paper uses gemma-scope-2b-pt-res), wraps each in
JumpReLUSAE and trains them side by side on activations read from a dataset of
cached activations or, with --do_extract_activations, extracted live from the
extractor model. Losses (MSE and the SAE's sparsity penalty) and statistics are
logged to TensorBoard under a run directory named after the hyper-parameters and
the start time; each trained SAE's state_dict is saved there as
``<hook>_<output_path>``.

Usage (README section 3 has the full paper command):
  python -m detectorscope.train_sae -s gemma-scope-2b-pt-res-canonical \
      -i layer_7/width_16k/canonical -hp model.layers.8 \
      -d data/data/PRDB/train -c review \
      -em models/detectors/gpt4o_detect_gemma --do_extract_activations
"""

import argparse
import collections
import datetime
import math
import random

import datasets
from detectorscope.model.extractor import HFExtractor
from detectorscope.model.sae import JumpReLUSAE
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils import data
from torch.utils import tensorboard
import tqdm
from typing_extensions import override


class TokenShuffleDataset(data.IterableDataset):
  """Dataset that shuffles tokens for training full sequence SAEs.

  Yields one token at a time as a dict ``{hook: activation}`` so that a
  DataLoader batch mixes tokens from many documents. Rows are pooled, sorted by
  text length into uniform batches and run through the extractor; the tokens
  collect in a buffer of ``buf_size`` that is shuffled before it is drained.
  Position 0 (the BOS token) is skipped.

  Args:
    ds: Dataset holding the text column and, without an extractor, one
      activation column per hook.
    text_col: Text column when extracting live; with cached activations, an
      activation column whose second dimension is the token count.
    extractor_path: Model whose activations are extracted live; when None the
      activations are read from the hook columns of ``ds``.
    hooks: Hook names, used as the extractor's hook points and as output keys.
    buf_size: Number of tokens held before shuffling and yielding.
    batch_size: Documents per extractor forward pass (not the DataLoader
      batch size).
    pool_multiplier: Rows pooled before length sorting, as a multiple of
      ``batch_size``.
    seed: Base seed; each epoch and worker derives its own generator.
  """

  def __init__(
      self,
      ds: datasets.Dataset,
      text_col: str,
      extractor_path: str | None = None,
      hooks: list[str] = [],
      buf_size: int = 1000000,
      batch_size: int = 8,
      pool_multiplier: int = 1000,
      seed: int = 42,
  ):
    super().__init__()
    self.ds = ds
    self.text_col = text_col
    self.buf_size = buf_size
    self.batch_size = batch_size
    self.pool_multiplier = pool_multiplier
    self.seed = seed
    self.epoch = 0
    self.hooks = hooks

    if extractor_path is not None:
      self.extractor = HFExtractor(extractor_path, default_hooks=self.hooks)

  @override
  def __len__(self):
    """Approximate token count, used only to size the tqdm bar."""
    # 650 is roughly the average number of tokens per example.
    return 650 * len(self.ds)

  def _create_length_grouped_batches(
      self, pool: list[dict], rng: random.Random
  ) -> list[list[dict]]:
    """Sorts a pool of rows by length and chunks them into uniform batches."""
    # Sort by string length (proxy for token length)
    pool.sort(key=lambda x: len(x[self.text_col]))

    batches = [
        pool[i : i + self.batch_size]
        for i in range(0, len(pool), self.batch_size)
    ]

    rng.shuffle(batches)
    return batches

  def _get_row_iterator(self, worker_id, num_workers, epoch):
    """Yields this worker's contiguous slice of the epoch-shuffled dataset."""
    shuffled_ds = self.ds.shuffle(seed=self.seed + epoch, keep_in_memory=False)

    indices_per_worker = math.ceil(len(self.ds) / num_workers)
    start_idx = worker_id * indices_per_worker
    end_idx = min(start_idx + indices_per_worker, len(self.ds))

    for i in range(start_idx, end_idx):
      yield shuffled_ds[i]

  def _process_and_buffer(
      self, batch: list[dict], buffer: list[dict[str, torch.Tensor]]
  ):
    """Appends every non-BOS, non-padding token of the batch to the buffer."""
    if self.extractor:
      texts = [row[self.text_col] for row in batch]
      acts, mask = self.extractor.get_activations(texts, return_mask=True)
      for b in range(mask.shape[0]):
        seq_len = int(mask[b].sum().item())
        for i in range(1, seq_len):
          buffer.append({k: v[b, i, :] for k, v in acts.items()})
    else:
      for row in batch:
        for i in range(1, row[self.text_col].shape[1]):
          buffer.append({k: v[0, i, :] for k, v in row.items()})

  @override
  def __iter__(self):
    """Yields shuffled single-token activation dicts for this worker's shard."""
    worker_info = data.get_worker_info()
    worker_id = worker_info.id if worker_info else 0
    num_workers = worker_info.num_workers if worker_info else 1

    self.epoch += 1
    rng = random.Random(self.seed * (self.epoch + 1) + worker_id)
    pool_capacity = self.batch_size * self.pool_multiplier
    buffer = []
    pool = []

    for row in self._get_row_iterator(worker_id, num_workers, self.epoch):
      pool.append(row)
      if len(pool) >= pool_capacity:
        for batch in self._create_length_grouped_batches(pool, rng):
          self._process_and_buffer(batch, buffer)
          if len(buffer) >= self.buf_size:
            rng.shuffle(buffer)
            while buffer:
              yield buffer.pop()
        pool = []

    if pool:
      for batch in self._create_length_grouped_batches(pool, rng):
        self._process_and_buffer(batch, buffer)

    rng.shuffle(buffer)
    while buffer:
      yield buffer.pop()


def get_rms_activation_norms(
    train_loader: data.DataLoader, cols: list[str]
) -> list[float]:
  """Returns the Root Mean Square (RMS) of the L2 norms of activations.

  One value per column in `cols`, computed over the whole loader; used as the
  normalising constant ``C`` when inputs are scaled to unit norm.
  """
  return (
      torch.cat(
          [
              torch.stack([(b[c] ** 2).sum(dim=1) for c in cols], dim=1)
              for b in train_loader
          ]
      )
      .mean(dim=0)
      .sqrt()
      .tolist()
  )


def store_gradients(model: nn.Module) -> list[torch.Tensor]:
  """Returns a copy of the gradients attached to model (None where absent)."""
  return [
      p.grad.clone() if p.grad is not None else None for p in model.parameters()
  ]


def get_total_gradient_norm(model: nn.Module) -> float:
  """Computes the total L2 norm of all gradients attached to model."""
  gradients = [
      g.data.flatten() for g in store_gradients(model) if g is not None
  ]
  return torch.cat(gradients).norm().item()


def log_metrics(
    loss: dict[torch.Tensor],
    stats: dict[torch.Tensor],
    hook: str,
    split: str,
    step: int,
    writer: tensorboard.SummaryWriter,
):
  """Logs the mean of every loss and stat tensor to TensorBoard.

  Tags are ``{hook}/{key}_loss/{split}`` and ``{hook}/{key}/{split}``.
  """
  for key, value in loss.items():
    writer.add_scalar(f"{hook}/{key}_loss/{split}", value.mean(), step)
  for key, value in stats.items():
    writer.add_scalar(f"{hook}/{key}/{split}", value.mean(), step)


def load_dataset(
    data_path: str,
    hooks: list[str],
    col_name: str,
    batch_size: int,
    num_examples: int | None,
    test_size: int,
    model_col: str,
    target_model: str,
    do_token_shuffle: bool,
    do_extract: bool,
    extractor_model_path: str,
) -> tuple[data.DataLoader, data.DataLoader]:
  """Loads a dataset from a local path and creates the train and test loaders.

  Args:
    data_path: Local path of a ``datasets`` dataset saved with save_to_disk.
    hooks: Activations to train on: dataset columns when reading cached
      activations, extractor hook points when extracting live.
    col_name: Name of the column that contains the text.
    batch_size: Batch size of both loaders.
    num_examples: Rows kept after the target filter (None keeps all).
    test_size: Number of examples left out as test data (split seeded at 42).
    model_col: Column holding the generator name, read by target_model.
    target_model: If non-empty, keep only "human" rows and rows of this model.
    do_token_shuffle: If True, wrap both splits in TokenShuffleDataset.
    do_extract: If True, extract activations live (single-process loaders,
      40k-token shuffle buffer).
    extractor_model_path: Path to the model used for extracting.

  Returns:
    A tuple ``(train_loader, test_loader)``.
  """
  ds = datasets.load_from_disk(data_path)

  if target_model:
    ds = ds.filter(lambda x: x[model_col].lower() in {"human", target_model})

  if num_examples:
    ds = ds.select(range(num_examples))  # pytype: disable=attribute-error

  cols_to_keep = [col_name]
  if extractor_model_path is None:
    cols_to_keep.extend(hooks)

  ds = ds.select_columns(cols_to_keep)
  ds = ds.with_format("torch", dtype=torch.float32)
  split = ds.train_test_split(test_size=test_size, shuffle=True, seed=42)

  dataset_kwargs = {"text_col": col_name, "hooks": hooks}
  if do_extract:
    dataset_kwargs["extractor_path"] = extractor_model_path
    dataset_kwargs["buf_size"] = 40000

  if do_token_shuffle:
    train_loader = data.DataLoader(
        TokenShuffleDataset(split["train"], **dataset_kwargs),
        batch_size=batch_size,
        num_workers=(0 if do_extract else 8),
    )
    test_loader = data.DataLoader(
        TokenShuffleDataset(split["test"], **dataset_kwargs),
        batch_size=batch_size,
        num_workers=(0 if do_extract else 8),
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
    models: list[nn.Module],
    data_loader: data.DataLoader,
    hooks: list[str],
    sparsity_coeff: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
  """Evaluates the SAEs on test data.

  Args:
    models: One SAE per entry of `hooks`, in the same order.
    data_loader: Test loader yielding ``{hook: activations}`` batches.
    hooks: Batch keys to evaluate each model on.
    sparsity_coeff: Current sparsity coefficient.

  Returns:
    ``(losses, stats)``, each a dict ``hook -> name -> tensor`` of per-example
    values concatenated over the loader. Losses hold the SAE's own terms plus
    ``total``; stats add ``fvu``, the fraction of variance unexplained.
  """
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  losses = collections.defaultdict(lambda: collections.defaultdict(list))
  statistics = collections.defaultdict(lambda: collections.defaultdict(list))
  with torch.no_grad():
    with tqdm.tqdm(total=len(data_loader), desc="Evaluating Model") as pbar:
      for full_batch in data_loader:

        for model, hook in zip(models, hooks):
          model.eval()
          batch = full_batch[hook].to(device)

          if model.is_unit_norm_scaled:
            batch = batch / model.C

          reconst, features = model.forward(batch)
          loss, stats = model.loss_fn(batch, reconst, features, sparsity_coeff)

          mean_mse = ((batch - batch.mean(dim=0)) ** 2).sum(dim=-1)
          loss["total"] = sum(loss.values())
          stats["fvu"] = loss["mse"] / mean_mse

          for key in loss.keys():
            losses[hook][key].append(loss[key])
          for key in stats.keys():
            statistics[hook][key].append(stats[key])

        pbar.update(1)

  total_losses = {}
  total_stats = {}
  for hook in hooks:
    total_losses[hook] = {k: torch.cat(v) for k, v in losses[hook].items()}
    total_stats[hook] = {k: torch.cat(v) for k, v in statistics[hook].items()}

  return total_losses, total_stats


def train(
    models: list[JumpReLUSAE],
    train_loader: data.DataLoader,
    test_loader: data.DataLoader,
    text_col: str,
    hooks: list[str],
    tensorboard_dir: str,
    num_epochs: int,
    eval_every: int,
    sparsity_coeff: float,
    learning_rate: float,
    adam_beta1: float,
    sparsity_warmup_steps: int = 10000,
    lr_warmup_steps: int = 1000,
    dead_latent_check_freq: int = 10000,
) -> JumpReLUSAE:
  """Trains one SAE per hook point.

  The models share the loader: every batch carries all hooks' activations and
  the models are stepped in turn. TensorBoard steps count samples (batch index
  times batch size; tokens under token shuffling). Decoder rows are kept at
  unit norm by projecting out the parallel gradient component and
  renormalising after every step. Every `dead_latent_check_freq` batches the
  latents that never fired since the last check are counted and logged.

  Args:
    models: SAEs initialised to pretrained weights, in the order of `hooks`.
    train_loader: Dataloader for training data.
    test_loader: Dataloader for test data.
    text_col: Column that locates each row's activations (the text column when
      extracting live); unused by train().
    hooks: Names of the activations to train on.
    tensorboard_dir: Directory to save TensorBoard logs.
    num_epochs: Number of epochs to train for.
    eval_every: Number of batches between evaluations on `test_loader`.
    sparsity_coeff: Final coefficient of the sparsity penalty, reached by
      linear warmup over `sparsity_warmup_steps` batches.
    learning_rate: Peak learning rate (cosine warmup over `lr_warmup_steps`).
    adam_beta1: Beta1 of the Adam optimizer.
    sparsity_warmup_steps: Number of batches of sparsity warmup.
    lr_warmup_steps: Number of batches of learning rate warmup.
    dead_latent_check_freq: Number of batches between dead-latent counts.

  Returns:
    The trained models, in the order of `hooks`.
  """

  writer = tensorboard.SummaryWriter(tensorboard_dir)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def cosine_lr_schedule(step: float, start_factor: float = 0.1) -> float:
    """Cosine warmup from start_factor to 1.0 over lr_warmup_steps, then 1.0."""
    if step >= lr_warmup_steps:
      return 1.0
    progress = step / lr_warmup_steps
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * (progress - 1.0)))
    return start_factor + (1.0 - start_factor) * cos_factor

  # Initialize training state for all models
  training_state = []
  for model in tqdm.tqdm(models, desc="Initializing Model State"):
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        betas=(adam_beta1, 0.999),
    )

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=cosine_lr_schedule
    )

    activation_counts = torch.zeros(model.d_hidden, device=device)
    training_state.append((optimizer, scheduler, activation_counts))

    model = model.to(device)

  steps_since_dead_check = 0
  count_dead_this_step = False
  max_sparsity_coeff = sparsity_coeff
  total_size = len(train_loader) * num_epochs
  batch_size = train_loader.batch_size
  with tqdm.tqdm(total=total_size, desc="Training Model") as pbar:
    for epoch in range(num_epochs):
      pbar.set_postfix(epoch=f"{epoch + 1}/{num_epochs}")
      for i, full_batch in enumerate(train_loader):
        step = (epoch * len(train_loader) + i) * batch_size

        # Determine whether to count dead latents this step
        steps_since_dead_check += 1
        if steps_since_dead_check >= dead_latent_check_freq:
          count_dead_this_step = not count_dead_this_step
          if not count_dead_this_step:
            steps_since_dead_check = 0

        # Linear warmup for sparsity coefficient
        if (s := int(step / batch_size)) <= sparsity_warmup_steps:
          progress_frac = float(s) / float(sparsity_warmup_steps)
          sparsity_coeff = progress_frac * max_sparsity_coeff

        for model, state, hook in zip(models, training_state, hooks):
          optimizer, scheduler, activation_counts = state
          batch = full_batch[hook].to(device)

          writer.add_scalar(f"{hook}/Sparsity_coeff", sparsity_coeff, step)
          writer.add_scalar(f"{hook}/LR", optimizer.param_groups[0]["lr"], step)

          model.train()
          optimizer.zero_grad()

          if model.is_unit_norm_scaled:
            batch = (batch / model.C).detach()

          reconst, features = model.forward(batch)

          # Track active features (for the dead-latent check below)
          with torch.no_grad():
            activation_counts += (features > 0).sum(dim=0)

          loss, stats = model.loss_fn(batch, reconst, features, sparsity_coeff)

          sum(
              loss.values()
          ).mean().backward()  # pytype: disable=attribute-error

          # Project out gradient components parallel to decoder rows
          if model.W_dec.grad is not None:
            with torch.no_grad():
              dot_products = model.W_dec.grad * model.W_dec
              sum_dot = dot_products.sum(dim=1, keepdim=True)
              model.W_dec.grad.sub_(sum_dot * model.W_dec)

          # Log total gradient norm
          total_norm = get_total_gradient_norm(model)
          writer.add_scalar(f"{hook}/grad_norm/Total", total_norm, step)

          optimizer.step()
          scheduler.step()

          # Renormalize decoder rows
          model.W_dec.data = F.normalize(model.W_dec.data, p=2, dim=1)

          # Compute fraction of variance unexplained (FVU)
          mean_mse = ((batch - batch.mean(dim=0)) ** 2).sum(dim=-1)
          loss["total"] = sum(loss.values())
          stats["fvu"] = loss["mse"] / mean_mse

          if count_dead_this_step:
            # Log total number of dead features
            dead_indices = torch.where(activation_counts == 0)[0]
            writer.add_scalar(f"{hook}/dead_indices", len(dead_indices), step)
            activation_counts.zero_()

          # Log metrics to tensorboard
          log_metrics(loss, stats, hook, "train", step, writer)

        # Evaluate on test data every N batches
        if i % eval_every == (eval_every - 1):
          val_loss, val_stats = evaluate(
              models, test_loader, hooks, sparsity_coeff
          )
          for hook in hooks:
            log_metrics(
                val_loss[hook], val_stats[hook], hook, "val", step, writer
            )

        pbar.update(1)

  return models


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-s",
      "--gemma_scope_repo",
      type=str,
      default="google/gemma-scope-2b-pt-res",
      help="Hub repo of the Gemma Scope SAEs to fine-tune",
  )
  parser.add_argument(
      "-i",
      "--sae_ids",
      type=str,
      nargs="+",
      default="layer_25/width_16k/canonical",
      help="List of SAEs to fine-tune. Must be same order as hooks",
  )
  parser.add_argument(
      "-hp",
      "--hooks",
      type=str,
      nargs="+",
      default="norm",
      help="The hooks to fine-tune SAEs on. Order must match sae_ids",
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
      default="text",
      help="Column name in the dataset that holds the text",
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
      default=5000,
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
      "-rf",
      "--resample_frequency",
      type=int,
      default=10000,
      help="Number of batches between dead-latent counts",
  )
  parser.add_argument(
      "-lw",
      "--lr_warmup_steps",
      type=int,
      default=1000,
      help="Number of steps to use for the learning rate warmup",
  )
  parser.add_argument(
      "-t",
      "--tensorboard_dir",
      type=str,
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
      "--rescale_model",
      action="store_true",
      help="If true, rescale model during training to operate on unit norm",
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
      "-mc",
      "--model_col",
      type=str,
      default="model",
      help="Column holding the generator name (read by --target_model)",
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
  parser.add_argument(
      "--do_extract_activations",
      action="store_true",
      help="If true, extract activations live during training",
  )
  parser.add_argument(
      "-em",
      "--extractor_model_path",
      type=str,
      default="",
      help="Model to use when extracting activations",
  )
  args = parser.parse_args()

  if args.rescale_model:
    args.use_pre_enc_bias = True
    args.normalize_input = True

  # Index tensorboard dir by training parameters and current time
  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  tensorboard_dir_path = (
      f"{args.tensorboard_dir}_{timestamp}_bs{args.batch_size}"
      f"_bw{args.bandwidth}_lr{args.learning_rate}_sp{args.sparsity_coeff}"
      f"_n{args.num_epochs}_a{args.accelerator}_w{args.sparsity_warmup_steps}"
  )

  print(f"Loading data from {args.data_path}...")
  train_data_loader, test_data_loader = load_dataset(
      args.data_path,
      hooks=args.hooks,
      col_name=args.column_name,
      batch_size=args.batch_size,
      num_examples=args.num_examples,
      test_size=args.test_size,
      model_col=args.model_col,
      target_model=args.target_model,
      do_token_shuffle=args.do_token_shuffle,
      do_extract=args.do_extract_activations,
      extractor_model_path=args.extractor_model_path,
  )

  # If we're extracting activations, append _input to all hooks
  hook_points = args.hooks
  if args.do_extract_activations:
    hook_points = [f"{h}_input" for h in hook_points]

  # Get the normalizing constants if we're normalizing
  rms_norms = [0] * len(hook_points)
  if args.normalize_input:
    rms_norms = get_rms_activation_norms(test_data_loader, hook_points)

  # Load in the pretrained SAE for each hook point
  sae_models = []
  for sae_id, C in tqdm.tqdm(
      zip(args.sae_ids, rms_norms), desc="Loading Pretrained SAEs"
  ):
    sae_model = JumpReLUSAE.from_gemma_scope(
        sae_id,
        args.gemma_scope_repo,
        bandwidth=args.bandwidth,
        acc=args.accelerator,
    )

    if args.use_pre_enc_bias:
      sae_model.use_pre_enc_bias = True
    if args.normalize_input:
      sae_model.C = C
    if args.rescale_model:
      sae_model.rescale_to_unit_norm()

    sae_models.append(sae_model)

  trained_models = train(
      sae_models,
      train_data_loader,
      test_data_loader,
      text_col=args.column_name,
      hooks=hook_points,
      tensorboard_dir=tensorboard_dir_path,
      num_epochs=args.num_epochs,
      eval_every=args.eval_every,
      sparsity_coeff=args.sparsity_coeff,
      learning_rate=args.learning_rate,
      adam_beta1=args.adam_beta1,
      sparsity_warmup_steps=args.sparsity_warmup_steps,
      lr_warmup_steps=args.lr_warmup_steps,
      dead_latent_check_freq=args.resample_frequency,
  )

  for model, hook in zip(trained_models, hook_points):
    if args.rescale_model:
      model.undo_rescaling()

    output_path = f"{tensorboard_dir_path}/{hook}_{args.output_path}"
    print(f"Saving model to {output_path}...")
    torch.save(model.state_dict(), output_path)
