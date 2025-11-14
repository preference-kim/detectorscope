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

"""Applies a trained SAE model to a dataset and saves the activations.

This script loads a dataset and a trained JumpReLUSAE model, computes the SAE
activations for a specified column in the dataset, and saves the resulting
dataset with the added 'sae_activation' column.
"""

import argparse

import datasets
from interpretable_ai_detectors.sae.models.jumprelu_sae import JumpReLUSAE
import torch


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-d",
      "--data_path",
      type=str,
      help="Path to the dataset",
  )
  parser.add_argument(
      "-s",
      "--sae_path",
      type=str,
      help="Path to the .pth file to load the sae from",
  )
  parser.add_argument(
      "-c",
      "--column_name",
      type=str,
      default="norm_input",
      help="Name of column to apply the SAE to in the source dataset",
  )
  parser.add_argument(
      "-o",
      "--output_path",
      type=str,
      help="Name of the location to save the new dataset to",
  )
  args = parser.parse_args()

  print("Loading dataset...")
  ds = datasets.load_from_disk(args.data_path)
  ds = ds.with_format("torch", dtype=torch.float32)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  print("Loading SAE...")
  state_dict = torch.load(args.sae_path)
  sae_model = JumpReLUSAE(
      d_input=state_dict["W_enc"].shape[0],
      d_hidden=state_dict["W_enc"].shape[1],
  )
  sae_model.load_state_dict(state_dict)
  sae_model = sae_model.to(device).eval()

  def run_sae_batched(
      batch: dict[str, torch.Tensor],
  ) -> dict[str, torch.Tensor]:
    _, features = sae_model.forward(batch[args.column_name].to(device))
    return {"sae_activation": features.detach().cpu()}

  print("Applying SAE to dataset...")
  processed_ds = ds.map(run_sae_batched, batched=True, batch_size=128)
  processed_ds.reset_format()

  processed_ds.save_to_disk(args.output_path)
