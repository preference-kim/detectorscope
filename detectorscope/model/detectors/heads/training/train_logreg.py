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

"""Trains a logistic regression head with SGD + proximal L1.

The head classifies AI vs human text from a document's feature vector: the
interp-masked max SAE activations of a split, or the gram2vec linguistic
features of the baseline. The L1 penalty is applied via a proximal
(soft-thresholding) update after each optimizer step, which produces exact
zeros in the learned weights. An optional L2 penalty can be added to the loss
for elastic-net style regularization.

Both feature kinds go through the same loop, optimiser, proximal step, split
and seeds, so the baseline is tuned under exactly the protocol of the SAE
heads. The one deliberate difference is the standardisation: SAE activations
are divided by their per-feature RMS (`--feature_scale`, from process.py
feature-scale), which keeps zeros at zero; the dense gram2vec rates are
z-scored over the training rows (`--standardize zscore`). For sparse
features RMS ~= std, so lambda keeps the same
"penalty per standard deviation" meaning across both. Either transform is
folded into the saved weights and bias, so every artifact consumes raw
features and loads through LogRegHead unchanged.

The output pickle holds ``state_dict`` (weight and bias in raw-feature
units), ``input_dim``, ``interp_mask`` and ``epochs_trained``; heads trained
on a cache that names its columns (gram2vec) add ``feature_names``,
``feature_cache`` and ``target_model``, and z-scored ones ``feature_mean``
and ``feature_std``. Usage:

  python -m detectorscope.model.detectors.heads.training.train_logreg \
      -d DATA_PATH --use_mmap_cache MMAP_DIR \
      --feature_scale MMAP_DIR/feature_scale.npy \
      --lr 1e-2 --l1_lambda 1e-1 --epochs 3 -t 0.7 -o OUT.pkl
  python -m detectorscope.model.detectors.heads.training.train_logreg \
      -d data/data/PRDB/train --use_mmap_cache outputs/gram2vec_features/PRDB_train \
      --standardize zscore -tm gpt4o --lr 3e-2 --l1_lambda 1e-2 --epochs 10 \
      --checkpoint_epochs 3,10 -o OUT.pkl
"""

import argparse
import os
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from detectorscope.data import loading
from detectorscope.data.cache import LayerStack

# --standardize zscore reads the training rows into memory; refuse above this.
ZSCORE_MAX_BYTES = 16 * 2**30


class LogisticRegressionModel(nn.Module):
  """Single-layer logistic regression over (masked) feature vectors."""

  def __init__(self, input_dim: int):
    super().__init__()
    self.linear = nn.Linear(input_dim, 1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Returns logits of shape [n] for inputs of shape [n, input_dim]."""
    return self.linear(x).squeeze(-1)


@torch.no_grad()
def soft_threshold_(weight: torch.Tensor, threshold: float):
  """In-place proximal update for the L1 penalty (soft-thresholding).

  Applies w <- sign(w) * max(|w| - threshold, 0) elementwise. This is the
  closed-form solution to min_w [ 0.5 * (w - a)^2 + threshold * |w| ] and is
  what produces exact zeros in the learned weights (rather than the near-zero
  dither produced by subgradient descent on the L1 term).

  Args:
    weight: Parameter tensor, modified in place.
    threshold: Shrinkage amount; the training loop passes lr * l1_lambda.
  """
  weight.copy_(
      torch.sign(weight) * torch.clamp(weight.abs() - threshold, min=0.0)
  )


@torch.no_grad()
def fold_standardisation(
    weight: torch.Tensor,
    bias: torch.Tensor,
    feature_scale: np.ndarray | None,
    feature_shift: np.ndarray | None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Re-expresses a head trained on standardised inputs in raw-input terms.

  Training sees z_j = (x_j - mu_j) / s_j, so
    w'·z + b' = sum_j (w'_j / s_j) x_j + (b' - sum_j (w'_j / s_j) mu_j).
  With no shift this is the plain w' / s of the SAE heads and the bias is
  untouched.

  Args:
    weight: Learned weight of shape [1, input_dim] (not modified).
    bias: Learned bias of shape [1] (not modified).
    feature_scale: Per-feature divisor s of shape [input_dim], or None.
    feature_shift: Per-feature offset mu of shape [input_dim], or None.

  Returns:
    (weight, bias) copies on the same device as the inputs.
  """
  w = weight.detach().clone()
  b = bias.detach().clone()
  if feature_scale is not None:
    w = w / torch.as_tensor(feature_scale, dtype=torch.float32, device=w.device)
  if feature_shift is not None:
    shift = torch.as_tensor(feature_shift, dtype=torch.float32, device=w.device)
    b = b - (w * shift).sum(dim=-1)
  return w, b


def save_head(
    path: str,
    weight: torch.Tensor,
    bias: torch.Tensor,
    interp_mask: np.ndarray,
    epochs_trained: int,
    extra_artifact: dict | None = None,
) -> None:
  """Pickles a head in the layout LogRegHead loads.

  Args:
    path: output pickle.
    weight: [1, input_dim] weight in raw-feature units (standardisation
      already folded in).
    bias: [1] bias in raw-feature units.
    interp_mask: the mask the head was trained under, stored as is.
    epochs_trained: epochs completed when the weights were taken.
    extra_artifact: extra keys merged into the pickle (feature names, ...).
  """
  with open(path, "wb") as f:
    pickle.dump(
        {
            "state_dict": {
                "weight": weight.detach().cpu(),
                "bias": bias.detach().cpu(),
            },
            "input_dim": weight.shape[-1],
            "interp_mask": interp_mask,
            "epochs_trained": epochs_trained,
            **(extra_artifact or {}),
        },
        f,
    )


def train_logreg(
    y: np.ndarray,
    interp_mask: np.ndarray,
    features: LayerStack,
    row_map: np.ndarray,
    l1_lambda: float = 1e-1,
    l2_lambda: float = 0.0,
    epochs: int = 3,
    batch_size: int = 1024,
    lr: float = 1e-2,
    cpu_only: bool = False,
    optimizer_name: str = "sgd",
    momentum: float = 0.9,
    feature_scale: np.ndarray | None = None,
    feature_shift: np.ndarray | None = None,
    checkpoint_epochs: tuple = (),
    checkpoint_path_fn=None,
    extra_artifact: dict | None = None,
) -> LogisticRegressionModel:
  """Trains a logistic regression head over a feature source.

  The objective is BCE(x, y) + l1_lambda * ||w||_1 + l2_lambda * ||w||_2^2.
  The L1 term is handled with a proximal step (exact zeros); the L2 term is
  added directly to the loss. A stratified 10% split (seed 42) is held out
  and its accuracy printed; the split, torch init and batch order are seeded
  (42). Each batch gathers only the rows it needs out of the feature source,
  so no layer is ever resident in full when the caches are memory-mapped.

  Args:
    y: float32 0/1 label per training row (1 = AI).
    interp_mask: Mask of shape [input_dim] for interpretability.
    features: the cache's LayerStack (SAE max activations or gram2vec).
    row_map: Cache row of each training row, so a filtered subset keeps
      contiguous indices.
    l1_lambda: Coefficient for the L1 penalty (proximal, default on).
    l2_lambda: Coefficient for the L2 penalty (default off).
    epochs: Number of passes over the training data.
    batch_size: Minibatch size (each batch is densified on the fly).
    lr: Learning rate for the optimizer.
    cpu_only: If true, train only on cpu regardless of device availability.
    optimizer_name: "adam" or "sgd". SGD is the correct pairing for the
      proximal L1 step: its update is lr * grad, so the lr cancels against the
      threshold and a weight survives iff its gradient exceeds l1_lambda.
      Adam normalises the update to roughly lr regardless of gradient size,
      which removes exactly the signal the threshold needs to rank features.
    momentum: Momentum for SGD. Averages gradients without normalising them,
      so the selection above is preserved, but it does inflate the effective
      step by ~1/(1-momentum) -- l1_lambda is not comparable across optimizers
      or across momentum settings.
    feature_scale: Optional per-feature divisor of shape [input_dim]. SAE
      activations are unnormalised and span orders of magnitude, so plain SGD
      diverges at any lr that suits the small features; dividing by the
      per-feature RMS (process.py feature-scale) equalises gradient
      scale. Scaling rather than centring keeps zeros at zero, preserving
      sparsity. The learned weights are divided by the same scale before
      returning, so the saved model consumes raw features and nothing
      downstream needs to know. L1 therefore acts in the standardised space,
      the usual lasso convention.
    feature_shift: Optional per-feature offset subtracted before
      feature_scale divides, i.e. full z-scoring (x - mu) / sigma. Only
      sensible for dense features (centring would densify the SAE caches).
      Folded into the saved bias alongside the weight rescaling, so the
      artifact still consumes raw features.
    checkpoint_epochs: 1-based epoch numbers at which to also save the model,
      so a single run yields several epoch budgets instead of guessing one.
    checkpoint_path_fn: callable(epoch) -> path, used with checkpoint_epochs.
      Each checkpoint pickle has the same layout as the final artifact.
    extra_artifact: Extra keys merged into every checkpoint pickle (e.g.
      feature names), so an artifact is self-describing.

  Returns:
    The trained LogisticRegressionModel, with any standardisation folded back
    into its weights and bias so it consumes raw features.
  """
  mask_1d = np.asarray(interp_mask).ravel()
  n_rows = len(row_map)
  input_dim = features.dim

  # The mask is applied per batch rather than once over the whole matrix,
  # which is equivalent since masked features end up zero either way. One
  # multiply already exists for the mask, so folding the reciprocal scale in
  # here makes normalisation free.
  scale_f32 = mask_1d.astype(np.float32)
  if feature_scale is not None:
    scale_f32 = scale_f32 / feature_scale.astype(np.float32)
  shift_f32 = (
      None if feature_shift is None else feature_shift.astype(np.float32)
  )

  def fetch_rows(global_idx: np.ndarray) -> torch.Tensor:
    batch = features.fetch(row_map[global_idx])
    if shift_f32 is not None:
      batch -= shift_f32
    batch *= scale_f32
    return torch.from_numpy(batch)

  # Split on row indices rather than materializing X_train / X_test. Passing
  # np.arange(n) through train_test_split with the same test_size, random_state,
  # and stratify labels yields the exact same partition and ordering as
  # splitting the data directly, so fetch_rows(train_idx) is the training
  # matrix row-for-row without ever holding it in full.
  train_idx, test_idx, y_train, y_test = train_test_split(
      np.arange(n_rows), y, test_size=0.1, random_state=42, stratify=y
  )

  # Train the model
  device = "cuda" if torch.cuda.is_available() else "cpu"
  device = "cpu" if cpu_only else device

  torch.manual_seed(42)
  model = LogisticRegressionModel(input_dim).to(device)
  if optimizer_name == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
  else:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  loss_fn = nn.BCEWithLogitsLoss()

  y_train_t = torch.from_numpy(y_train)
  n_train = len(train_idx)
  rng = np.random.default_rng(42)

  n_batches = (n_train + batch_size - 1) // batch_size
  for epoch in tqdm(range(epochs), desc="Training LogReg"):
    perm = rng.permutation(n_train)
    epoch_loss = 0.0
    # Inner progress bar tracks progress batch by batch within the epoch.
    # leave=False so each epoch's bar is cleared once it completes.
    batch_bar = tqdm(
        range(0, n_train, batch_size),
        total=n_batches,
        desc=f"Epoch {epoch + 1}/{epochs}",
        leave=False,
    )
    for start in batch_bar:
      idx = perm[start : start + batch_size]
      xb = fetch_rows(train_idx[idx]).to(device)
      yb = y_train_t[idx].to(device)

      optimizer.zero_grad()
      loss = loss_fn(model(xb), yb)
      if l2_lambda > 0:
        loss = loss + l2_lambda * model.linear.weight.pow(2).sum()
      loss.backward()
      optimizer.step()

      # Proximal step for the L1 penalty. Note the L1 term is deliberately
      # NOT in the loss; this step accounts for it exactly.
      if l1_lambda > 0:
        soft_threshold_(model.linear.weight, lr * l1_lambda)

      epoch_loss += loss.item() * len(idx)
      batch_bar.set_postfix(loss=loss.item())
    batch_bar.close()
    # Sparsity evolves over training, so log it per epoch: a single end-of-run
    # count cannot show whether it had converged or was still falling.
    epoch_nnz = int((model.linear.weight != 0).sum().item())
    epoch_wmax = float(model.linear.weight.abs().max().item())
    tqdm.write(
        f"Epoch {epoch + 1}/{epochs}: loss={epoch_loss / n_train:.4f} "
        f"nonzero={epoch_nnz} |w|max={epoch_wmax:.5f}"
    )

    if epoch + 1 in checkpoint_epochs and checkpoint_path_fn is not None:
      # Weights are unscaled on the way out so every checkpoint, intermediate or
      # final, consumes raw activations.
      ckpt_w, ckpt_b = fold_standardisation(
          model.linear.weight, model.linear.bias, feature_scale, feature_shift
      )
      path = checkpoint_path_fn(epoch + 1)
      save_head(path, ckpt_w, ckpt_b, interp_mask, epoch + 1, extra_artifact)
      tqdm.write(f"  checkpoint @ epoch {epoch + 1} -> {path}")

  # Evaluate
  model.eval()
  preds = []
  with torch.no_grad():
    for start in range(0, len(test_idx), batch_size):
      xb = fetch_rows(test_idx[start : start + batch_size]).to(device)
      preds.append((torch.sigmoid(model(xb)) > 0.5).long().cpu())
  preds = torch.cat(preds).numpy()

  if feature_scale is not None or feature_shift is not None:
    # Undo the standardisation so the artifact operates on raw activations:
    # z = sum_j (x_j / s_j) * w'_j == sum_j x_j * (w'_j / s_j), plus the bias
    # correction when the inputs were also centred.
    with torch.no_grad():
      w, b = fold_standardisation(
          model.linear.weight, model.linear.bias, feature_scale, feature_shift
      )
      model.linear.weight.copy_(w)
      model.linear.bias.copy_(b)

  nnz = int((model.linear.weight != 0).sum().item())
  print(
      f"Accuracy: {accuracy_score(y_test, preds):.4f}\n"
      f"Nonzero weights: {nnz} / {input_dim}\n"
      f"{classification_report(y_test, preds)}"
  )

  return model


def run(args: argparse.Namespace) -> None:
  """Trains one head from the command line and pickles it to --output_path.

  With --target_model only "human" rows and that generator's rows are kept
  (the PRDB per-generator heads); --interp_threshold tau builds the interp
  mask from the description files in --data_path, otherwise every feature
  is kept. Checkpoints are written beside the final model during training.
  """
  if args.exit_if_trained_model_exists and os.path.exists(args.output_path):
    print(f"Found model at {args.output_path}, exiting")
    return
  if args.threads:
    torch.set_num_threads(args.threads)

  standardize = args.standardize
  if standardize == "zscore" and args.feature_scale:
    raise ValueError("--standardize zscore and --feature_scale are exclusive")

  # A cache that is z-scored is read into memory: z-scoring needs every
  # training row once and training then gathers random rows of it.
  source = loading.load_source(args, mmap=standardize != "zscore")
  y = (source.labels != "human").astype(np.float32)
  n_ai = int(y.sum())
  print(
      f"{len(y):,} training rows ({len(y) - n_ai:,} human, {n_ai:,} AI) "
      f"x {source.dim} features"
  )

  feature_scale = None
  feature_shift = None
  extra = dict(source.extra_artifact)
  if standardize == "zscore":
    # Statistics over every kept row (not just the 90% split), mirroring
    # feature_scale.npy, which is computed over the whole cache. The rows
    # are materialised, which is what makes this a dense-cache option.
    n_bytes = len(source.row_map) * source.dim * 4
    if n_bytes > ZSCORE_MAX_BYTES:
      raise ValueError(
          f"--standardize zscore would materialise {n_bytes / 2**30:.1f} GiB "
          "of training rows; it is meant for dense caches such as gram2vec, "
          "SAE caches use --feature_scale"
      )
    mu, sigma = loading.standardisation(source.features.fetch(source.row_map))
    feature_scale, feature_shift = sigma, mu
    extra |= {"feature_mean": mu, "feature_std": sigma}
  elif args.feature_scale:
    feature_scale = np.load(args.feature_scale)

  os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
  model = train_logreg(
      y,
      source.interp_mask,
      source.features,
      source.row_map,
      l1_lambda=args.l1_lambda,
      l2_lambda=args.l2_lambda,
      epochs=args.epochs,
      batch_size=args.batch_size,
      lr=args.lr,
      cpu_only=args.cpu_only,
      optimizer_name=args.optimizer,
      momentum=args.momentum,
      feature_scale=feature_scale,
      feature_shift=feature_shift,
      checkpoint_epochs=tuple(
          int(e) for e in args.checkpoint_epochs.split(",") if e.strip()
      ),
      checkpoint_path_fn=lambda ep: args.output_path.replace(
          ".pkl", f"_ep{ep}.pkl"
      ),
      extra_artifact=extra,
  )

  print(f"Saving model to {args.output_path}")
  save_head(
      args.output_path,
      model.linear.weight,
      model.linear.bias,
      source.interp_mask,
      args.epochs,
      extra,
  )


if __name__ == "__main__":
  # The defaults are the paper protocol: SGD with momentum 0.9, lr 1e-2,
  # lambda 1e-1, 3 epochs, batch 1024.
  parser = argparse.ArgumentParser(
      description="Train an SGD + proximal-L1 logistic regression head."
  )
  parser.add_argument(
      "-d",
      "--data_path",
      type=str,
      default=None,
      help=(
          "Split directory: the Hugging Face dataset with the label column, "
          "its layer{N}_max_activations.pth caches and the description files"
      ),
  )
  parser.add_argument(
      "--use_mmap_cache",
      type=str,
      default=None,
      help=(
          "Directory of layer{N}.*.npy caches (process.py). Batches are "
          "gathered from the memory-mapped arrays instead of the resident "
          ".pth layers in --data_path; same training data"
      ),
  )
  parser.add_argument(
      "-n",
      "--num_examples",
      type=int,
      default=None,
      help="Keep only the first N rows (after the target filter)",
  )
  parser.add_argument(
      "-l",
      "--label_col",
      type=str,
      default="model",
      help=(
          'Dataset column holding the generator name; "human" is the '
          "negative class and every other value is AI"
      ),
  )
  parser.add_argument(
      "-t",
      "--interp_threshold",
      type=float,
      default=None,
      help=(
          "tau: train only on features whose description simulation score is "
          ">= tau (SAE caches only). Unset or 0.0 keeps every feature"
      ),
  )
  parser.add_argument(
      "-m",
      "--desc_gen_model",
      type=str,
      default="gemini-3.1-flash-lite-preview",
      help="Tag of the layer{N}_{tag}_feature_results.jsonl description files",
  )
  parser.add_argument(
      "-tm",
      "--target_model",
      type=str,
      default=None,
      help=(
          'Keep only "human" rows and this generator\'s rows (the PRDB '
          "per-generator heads); default trains on every row"
      ),
  )
  parser.add_argument(
      "--feature_scale",
      type=str,
      default=None,
      help=(
          "Path to feature_scale.npy (process.py feature-scale). "
          "Divides each feature by its corpus RMS during training; required "
          "for SGD to converge on the unnormalised SAE activations"
      ),
  )
  parser.add_argument(
      "--standardize",
      type=str,
      default="none",
      choices=["none", "zscore"],
      help=(
          "zscore: centre and scale every feature by its mean and std over "
          "the training rows, which are read into memory once (dense caches "
          "such as gram2vec). none: raw features, or --feature_scale"
      ),
  )
  parser.add_argument(
      "--optimizer",
      type=str,
      default="sgd",
      choices=["adam", "sgd"],
      help=(
          "Optimizer for the head. SGD pairs correctly with the proximal L1 "
          "step; Adam's normalised update does not"
      ),
  )
  parser.add_argument(
      "--momentum",
      type=float,
      default=0.9,
      help="Momentum for SGD (ignored by adam)",
  )
  parser.add_argument(
      "--l1_lambda",
      type=float,
      default=1e-1,
      help="Coefficient for the L1 penalty (applied via proximal step)",
  )
  parser.add_argument(
      "--l2_lambda",
      type=float,
      default=0.0,
      help="Coefficient for the L2 penalty (set >0 for elastic net)",
  )
  parser.add_argument(
      "--epochs",
      type=int,
      default=3,
      help="Number of training epochs",
  )
  parser.add_argument(
      "-b",
      "--batch_size",
      type=int,
      default=1024,
      help="Minibatch size",
  )
  parser.add_argument(
      "--lr",
      type=float,
      default=1e-2,
      help="Learning rate",
  )
  parser.add_argument(
      "--checkpoint_epochs",
      type=str,
      default="",
      help=(
          "Comma-separated 1-based epochs to also save at, e.g. '3,10'. Each "
          "is written beside --output_path with an _ep{N} suffix"
      ),
  )
  parser.add_argument(
      "--threads",
      type=int,
      default=None,
      help="torch intra-op threads (default: torch's own choice)",
  )
  parser.add_argument(
      "--cpu_only",
      action="store_true",
      help="If true, run on cpu regardless of device",
  )
  parser.add_argument(
      "--exit_if_trained_model_exists",
      action="store_true",
      help="If true, check if a model exists at output_path and if so, exit",
  )
  parser.add_argument(
      "-o",
      "--output_path",
      type=str,
      default="logregmodel.pkl",
      help="Output path to save the model to",
  )
  args = parser.parse_args()
  run(args)
