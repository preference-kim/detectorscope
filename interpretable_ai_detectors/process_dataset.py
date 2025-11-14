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

"""This script processes a dataset to extract activations.

It can handle both sparse and dense activations, and allows saving activations
for either only the last token or all tokens. The processed dataset is saved
to disk.
"""
import argparse

import accelerate
import datasets
from interpretable_ai_detectors.model.hf_extractor import HFExtractor
from interpretable_ai_detectors.model.sae_extractor import SAEExtractor
import torch


def process_dataset(
    extractor: SAEExtractor | HFExtractor,
    ds: datasets.Dataset,
    col: str,
    save_all_tokens: bool = False,
) -> datasets.Dataset:
  """Extract activations from a dataset.

  For all elements in a dataset, extract their SAE activations
  and output a new dataset with these activations saved as new columns.
  For activations that are meant to be sparse, convert to sparse CSR format.
  For dense activations, only save the last token activation.

  Args:
    extractor (SAEExtractor): a wrapper around the LLM + SAE
    ds (Dataset): the dataset to run processing on
    col (str): the column name in the dataset containing the text
    save_all_tokens (bool): if True, save acts for all tokens not just last

  Returns:
    A new dataset object with additional columns, one for each SAE
    activation saved.
  """

  def sparsify(x: torch.Tensor):
    x = x.to_sparse(layout=torch.sparse_csr)
    return {
        "crow_indices": x.crow_indices(),
        "col_indices": x.col_indices(),
        "values": x.values(),
        "size": list(x.size()),
    }

  def is_sparse(hook_point: str) -> bool:
    return hook_point == "hook_sae_acts_post"

  # Save only the last token if save_all_tokens == False
  process = (lambda x: x[0, -1, :]) if not save_all_tokens else (lambda x: x)

  return ds.map(
      lambda x: {
          key: sparsify(value) if is_sparse(key) else process(value)
          for key, value in extractor.get_activations(x[col]).items()
      }
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-d",
      "--data_path",
      type=str,
      default="imone/OpenOrca_FLAN",
      help="Path to the dataset",
  )
  parser.add_argument(
      "-t",
      "--text_column",
      type=str,
      default="response",
      help="The name of the column containing the text",
  )
  parser.add_argument(
      "-n",
      "--n_examples",
      type=int,
      help="The number of examples to process (opt. defaults to all)",
  )
  parser.add_argument(
      "-e",
      "--extractor_type",
      type=str,
      default="sae_lens",
      choices=["sae_lens", "huggingface"],
      help="The extractor to use (either sae_lens or huggingface)",
  )
  parser.add_argument(
      "-m",
      "--model_path",
      type=str,
      default=None,
      help="The local path to the model you want to use",
  )
  parser.add_argument(
      "-k",
      "--hook_points",
      type=str,
      nargs="+",
      default="hook_sae_acts_post",
      help="The hook points you want to extract activations for",
  )
  parser.add_argument(
      "--save_all_tokens",
      action="store_true",
      help="If true, save all token activations",
  )
  parser.add_argument(
      "--do_shard",
      action="store_true",
      help="If true, shard dataset (used with accelerate launch)",
  )
  parser.add_argument(
      "-o",
      "--output_path",
      type=str,
      default="./activations/pre_activations_all",
      help="The path to the output to save the activations",
  )
  args = parser.parse_args()

  # Disable caching to ensure that huggingface doesn't save the intermediate
  # steps of computation. This takes up twice as much disk space and can
  # cause OOM errors on large datasets.
  # NOTE: This is not necessary for machines with lots of extra disk space.
  datasets.disable_caching()

  # Load dataset
  split_value = f"train[:{args.n_examples}]" if args.n_examples else "train"
  dataset = datasets.load_dataset(args.data_path, split=split_value)

  # If we're sharding, initialize accelerator and shard dataset
  if args.do_shard:
    accelerator = accelerate.Accelerator()
    dataset = dataset.shard(  # pytype: disable=attribute-error
        num_shards=accelerator.num_processes,
        index=accelerator.process_index,
    )
    output_path = f"{args.output_path}_shard{accelerator.process_index}"
  else:
    output_path = args.output_path

  # Load SAE extractor
  if args.extractor_type == "sae_lens":
    extractor_model = SAEExtractor()
  elif args.extractor_type == "huggingface":
    extractor_model = HFExtractor(
        model_path=args.model_path, default_hooks=args.hook_points
    )
  else:
    raise ValueError(f"Unknown extractor_type {args.extractor_type}")

  # Extract activations for each element of the dataset
  processed_ds = process_dataset(
      extractor_model, dataset, args.text_column, args.save_all_tokens
  )

  # Write to output path
  processed_ds.save_to_disk(output_path)
