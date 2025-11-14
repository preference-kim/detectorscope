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

"""Script to combine shards of a dataset into a single dataset."""

import argparse

import datasets


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-n",
      "--n_shards",
      type=int,
      help="The number of shards to combine",
  )
  parser.add_argument(
      "-o",
      "--output_path",
      type=str,
      help="The path to the location of the combined shards",
  )
  args = parser.parse_args()

  shard_paths = [args.output_path + f"_shard{i}" for i in range(args.n_shards)]
  print(f"Loading {shard_paths}...")
  shards_to_combine = [datasets.load_from_disk(path) for path in shard_paths]

  print("Combining Datasets...")
  combined_ds = datasets.concatenate_datasets(shards_to_combine)

  print(f"Saving to {args.output_path}...")
  combined_ds.save_to_disk(args.output_path, num_proc=8)
