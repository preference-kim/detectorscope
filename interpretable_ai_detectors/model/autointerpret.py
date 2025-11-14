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

"""A module for automatically interpreting and validating SAE features.

This module provides the `FeatureDescriptionValidator` class, which is used to
generate and validate natural language descriptions of Sparse Autoencoder (SAE)
features based on their activation patterns across a dataset. It uses a Gemini
model to perform the validation and generation tasks.
"""

import argparse
import bisect
import collections
import itertools
import os
import random
import re
from typing import Any
import uuid

import datasets
from google import genai
from interpretable_ai_detectors import utils
from interpretable_ai_detectors.model.activation_cache import MaxActivationCache
from interpretable_ai_detectors.model.gemini import GeminiGenerator
from interpretable_ai_detectors.model.sae_extractor import SAEExtractor
import more_itertools
import numpy as np
import scipy
import torch
import tqdm


HIGHLIGHTS = {
    "xml": ("<highlight>", "</highlight>"),
    "angled": ("<<", ">>"),
    "none": ("", ""),
}

HIGHLIGHT_THRESHOLD = 0.1

HIGHLIGHT_EXPLANATION_TEMPLATE = (
    "For each sequence, the strongest activating words are indicated with"
    ' {pre}...{post} - note that these tags "{pre}" and "{post}" are'
    " not part of the actual sentence, they're just used to highlight the"
    " places where the neuron is most active.\n"
)

FAILURE = "[[ Failure ]]"

GROUPING_EXPLANATIONS = {
    "ungrouped": (
        "You will be given a list of documents, sorted from most to least "
        "activating (according to the maximum activation across tokens "
        "in that document)."
    ),
    "quantile_grouped": (
        "You will be given a list of documents, sorted into groups "
        "according to the maximum strength of the activation across "
        "tokens in that document."
    ),
}

EXPLANATION_TYPES = {
    "terse": (
        "A short explanation, limited to 20 words or less. This explanation "
        "should be shorthand, omitting punctuation and formatting and "
        "capitalization. You can omit prefixes like 'This neuron fires on', "
        "instead describe it directly."
    ),
    "keywords": (
        "A comma-separated list of up to 3 key words or short phrases that "
        "activate this neuron, in quotation marks '...'. Note that this "
        "should be actual examples where the neuron might fire, not "
        "descriptions of those examples (so it doesn't have to literally "
        "be single words, although often it will be)."
    ),
    "long": (
        "A comprehensive but concise explanation of what this neuron "
        "activates on. This should be 3 sentences: the first describes "
        "the concept briefly, the second describes what words or phrases "
        "most commonly activate it in the top activating documents, and "
        "the third describes what words or phrases cause it to also activate "
        "(but weaker) in some lower quantiles. If only one word or phrase "
        "activates the neuron within the document, make sure you specify "
        "which one (rather than just saying what kinds of documents activate "
        "it). Don't use overly formal language, be as simple and concise as "
        "possible. Don't include lists of more than 5 words or phrases in a "
        "row. Note that you don't have to list words if it's not relevant "
        "for concisely explaining the neuron."
    ),
}

QUANTILE_DESCRIPTIONS = {
    "top": "These are the highest-activating documents in the dataset:",
    "bottom": (
        "These are some of the lowest-activating documents in the dataset:"
    ),
    "middle": "These are some documents from the group with score ",
}


class FeatureDescriptionValidator:
  """Validates feature descriptions using a Gemini model.

  This class handles the process of generating prompts with highlighted
  examples of feature activations, querying a Gemini model to validate
  a given description, and computing a correlation score based on the
  model's responses.
  """

  def __init__(
      self,
      dataset: datasets.Dataset,
      sae_extractor: SAEExtractor,
      activation_cache: MaxActivationCache,
      validation_method: str = "quantile",
      highlight_method: str = "xml",
      negative_sampling_method: str = "uniform",
      quantile_method: str = "density",
      prompt_sampling_method: str = "stratified",
      n_quantiles: int = 4,
      model_name: str = "gemini-2.5-pro",
      use_snippets: bool = True,
      group_size: int = 5,
      text_col: str = "review",
      activation_col: str = "sae_activation",
  ):
    self.data = dataset
    self.cache = activation_cache
    self.gemini_generator = GeminiGenerator(model_name)
    self.extractor = sae_extractor
    self.text_col = text_col
    self.activation_col = activation_col

    self.deterministic_group_sampling = True
    self.group_sample_seed = 42

    self.validation_method = validation_method
    self.highlight_method = highlight_method
    self.negative_sampling_method = negative_sampling_method
    self.prompt_sampling_method = prompt_sampling_method
    self.quantile_method = quantile_method
    self.n_quantiles = n_quantiles

    self.use_snippets = use_snippets
    self.snippet_len = 20

    # group size is set to 2 if we're doing pairwise
    self.group_size = (
        group_size if self.validation_method != "pairwise" else 2
    )

    # Add activation column to the dataset if it is not present
    if self.activation_col not in self.data.column_names:
      self.data = utils.add_values_col(self.data, self.activation_col)

    # TODO(liamdugan): Make this relative import to __file__ location
    #                  or take it in as a CLI argument
    with open(f"model/validation_prompts/{validation_method}.txt") as f:
      self.validation_prompt_template = f.read()

    with open("model/validation_prompts/generation.txt") as f:
      self.generation_prompt_template = f.read()

    self.highlight_explanation = ""
    if self.highlight_method != "none":
      self.highlight_explanation = HIGHLIGHT_EXPLANATION_TEMPLATE.format(
          pre=HIGHLIGHTS[self.highlight_method][0],
          post=HIGHLIGHTS[self.highlight_method][1],
      )

    self.generation_config = genai.types.GenerateContentConfig(
        temperature=1.0, top_p=0.95, top_k=64
    )
    self.validation_config = genai.types.GenerateContentConfig(
        temperature=0
    )

  def _get_quantiles(
      self, activations: dict[int, float]
  ) -> dict[int, list[int]]:
    """Bin documents by max activation quantiles."""
    # construct the [0, 1] quantiles
    quantile_ranges = np.linspace(0, 1, num=self.n_quantiles + 1)

    # get the activation ranges for each quantile
    if self.quantile_method == "scaled":
      quantiles = [max(activations.values()) * i for i in quantile_ranges]
    elif self.quantile_method == "density":
      quantiles = np.quantile(list(activations.values()), quantile_ranges)
    else:
      raise ValueError(f"Unknown quantile method: {self.quantile_method}")

    # assign each document a quantile based on its max activation
    quantile_dict = collections.defaultdict(list)
    for doc_id, max_activation in activations.items():
      qid = max(bisect.bisect_left(quantiles, max_activation) - 1, 0)
      quantile_dict[qid].append(doc_id)

    return quantile_dict

  def _sample_negatives(
      self, feature_index: int, quantile_dict: dict[int, list[int]]
  ) -> list[int]:
    """Sample negative examples for the autointerp pipeline."""
    # Get the set of all active and inactive documents
    # TODO(liamdugan): Refactor to not call get_active_docs here
    # and instead compute the active docs from quantile_dict
    active_docs = set(self.cache.get_active_docs(feature_index))
    inactive_docs = set(range(len(self.data))) - active_docs

    # Get the average quantile length
    quantile_lengths = [len(bin) for _, bin in quantile_dict.items()]
    avg_quantile_len = int(sum(quantile_lengths) / len(quantile_lengths))

    # Sample negative examples according to the strategy
    if self.negative_sampling_method == "uniform":
      n_sample = min(avg_quantile_len, len(inactive_docs))
      negative_docs = random.sample(list(inactive_docs), n_sample)
    elif self.negative_sampling_method == "semantic":
      raise NotImplementedError("Sampling negatives w/ semantic sim.")
    else:
      raise ValueError(
          f"Unknown neg sampling method: {self.negative_sampling_method}"
      )

    return negative_docs

  def _equalize_quantiles(
      self, quantile_dict: dict[int, list[int]], n: int
  ) -> dict[int, list[int]]:
    """Selects N docs from quantile_dict to maximally equalize classes."""
    # Get the number of documents in each quantiles
    q_sizes = {qid: len(bin) for qid, bin in quantile_dict.items()}

    # clamp n to be at most len(dataset)
    n = min(n, sum(q_sizes.values())) if n else sum(q_sizes.values())

    # initialize num sampled per quantile to 0
    sampled = {qid: 0 for qid in quantile_dict.keys()}

    while items_left := n - sum(sampled.values()) > 0:
      # compute the number of items to get from each quantile
      target = max(int(items_left / len(quantile_dict)), 1)

      # For each quantile, sample at most 'target' examples
      for qid in quantile_dict.keys():
        sampled[qid] += min(target, q_sizes[qid] - sampled[qid])
        if sum(sampled.values()) >= n:
          break

    # Return N items from each quantile
    return {
        qid: random.sample(bin, sampled[qid])
        for qid, bin in quantile_dict.items()
    }

  def _sample_groups(
      self, quantile_dict: dict[int, list[int]], num_test_examples: int
  ) -> list[list[int]]:
    """Constructs a list of chunks (list of tuples) for use in prompts."""
    # Select num_test_examples from the full quantile dict
    # in a way that maximally balances classes
    subsampled_quantile_dict = self._equalize_quantiles(
        quantile_dict, num_test_examples
    )

    # Convert quantile dict to list of lists of tuples
    q_lists = [
        [(qid, doc) for doc in subsampled_quantile_dict[qid]]
        for qid in subsampled_quantile_dict.keys()
    ]

    # Flatten & shuffle for uniform, interleave for stratified
    if self.prompt_sampling_method == "uniform":
      q_list = list(itertools.chain.from_iterable(q_lists))
      random.shuffle(q_list)
    elif self.prompt_sampling_method == "stratified":
      q_list = utils.interleave_lists(q_lists)
    else:
      raise ValueError(
          f"Unknown prompt sampling method: {self.prompt_sampling_method}"
      )

    # Chunk the list into groups of len group_size
    groups = [
        g[: self.group_size]
        for g in more_itertools.chunked(q_list, self.group_size)
        if len(g) >= self.group_size
    ]

    # shuffle the list of lists and return
    return utils.nested_shuffle(groups)

  def _apply_highlights(
      self, doc_id: int, feature_id: int, act_threshold: float
  ) -> str:
    """Retrieve document and add feature activation highlights."""
    # Get the tokens of the document
    tokens = self.extractor.model.to_str_tokens(
        self.data[doc_id][self.text_col],
        prepend_bos=self.extractor.prepend_bos,
    )

    # Get cached SAE activations of all features for the particular document
    activations = torch.sparse_csr_tensor(
        **self.data[doc_id][self.activation_col]
    )

    # Extract the activations of the particular feature on the document
    a_list = [
        activations[0, i, feature_id].item() for i in range(len(tokens))
    ]

    # If all activations are 0, randomly sample a spot to highlight
    if not any(a_list):
      a_list[random.choice(range(len(tokens)))] = act_threshold

    # If using snippets, crop document to +/- snippet len around max activ.
    if self.use_snippets:
      max_i = a_list.index(max(a_list))
      tokens = tokens[
          max(max_i - self.snippet_len, 0) : max_i + self.snippet_len
      ]
      a_list = a_list[
          max(max_i - self.snippet_len, 0) : max_i + self.snippet_len
      ]

    # Add highlights to the token if activation is over the threshold
    pre, post = HIGHLIGHTS[self.highlight_method]
    formatted_str = "".join(
        f"{pre}{tok}{post}" if a_list[i] >= act_threshold else tok
        for i, tok in enumerate(tokens)
    )

    # Join together adjacent highlighted tokens
    return formatted_str.replace(f"{post}{pre}", "")

  def _format_validation_prompt(
      self,
      group: list[list[int]],
      feature_index: int,
      feature_description: str,
      max_activation: float,
  ) -> str:
    """Formats the validation prompt for the Gemini model.

    This method takes a group of documents, applies highlighting based on
    feature activations, and formats them into a prompt using the
    `validation_prompt_template`.

    Args:
      group: A list of lists, where each inner list contains `[quantile_id,
        doc_id]`.
      feature_index: The index of the feature being validated.
      feature_description: The description of the feature to be validated.
      max_activation: The maximum activation value for this feature across the
        dataset.

    Returns:
      A string containing the formatted validation prompt.
    """
    # Apply highlights for all documents in this group
    threshold = max_activation * HIGHLIGHT_THRESHOLD
    examples = [
        self._apply_highlights(doc_id, feature_index, threshold)
        for _, doc_id in group
    ]

    # Format the examples into a numbered list
    example_text = "".join(
        [f"{i+1}. {repr(example)}\n" for i, example in enumerate(examples)]
    )

    return self.validation_prompt_template.format(
        examples=example_text,
        group_size=len(group),
        max_quantile=self.n_quantiles,
        explanation=feature_description.strip(),
        highlight_explanation=self.highlight_explanation,
    )

  @classmethod
  def parse_response(
      cls, text: str, size: int, validation_method: str
  ) -> list[int]:
    """Check if the string is a list of integers of the correct size."""
    if validation_method == "binary" and text.strip() == "None":
      return [0] * size

    if not re.fullmatch(r"\s*\d+(\s*,\s*\d+)*\s*", text):
      return FAILURE

    outputs = [int(n) for n in re.findall(r"\d+", text)]

    if validation_method in ["rank", "quantile"]:
      if len(outputs) != size:
        return FAILURE

    if validation_method == "pairwise":
      if len(outputs) != 1 or outputs[0] not in [1, 2]:
        return FAILURE

    match validation_method:
      case "quantile":
        return outputs
      case "rank":
        return [outputs.index(i) for i in range(1, size + 1)]
      case "pairwise":
        return [1, 0] if outputs[0] == 1 else [0, 1]
      case "binary":
        return [1 if i in outputs else 0 for i in range(1, size + 1)]

    raise ValueError(f"Unknown validation method: {validation_method}")

  @classmethod
  def _kendalls_tau(
      cls, pairs: list[tuple[list[int, int], list[int, int]]],
  ) -> float:
    """Manually computes Kendall's Tau-a for incomplete ranking."""
    # Has 1 if both pairs have matching rankings else 0
    matches = [int((a[0] > a[1]) == (b[0] > b[1])) for a, b in pairs]

    # get the number of concordant and discordant pairs
    c = sum(matches)
    d = len(matches) - c

    # Kendall's Tau-a is concordant - discordant / total pairs
    # https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
    #
    # NOTE: We have to compute this manually because none of the libraries
    # that implement this statistic support pairwise incomplete rankings
    #
    # TODO(liamdugan): Swap to Tau-b to account for ties when
    #                  using uniform for prompt_sampling_method
    # TODO(liamdugan): Error check for dividebyzero
    return (c - d) / (c + d)

  @classmethod
  def compute_correlation(
      cls,
      outputs: list[list[int]],
      groups: list[list[tuple[int, int]]],
      validation_method: str,
  ) -> float:
    """Filter out failed examples and compute correlations."""
    # Pair outputs w/ true quantile values and filter out parsing failures
    groups = [[qid for qid, _ in g] for g in groups]
    filtered = [(o, g) for o, g in zip(outputs, groups) if o != FAILURE]

    # If pairwise rankings, return kendall's tau
    if validation_method == "pairwise":
      return FeatureDescriptionValidator._kendalls_tau(filtered)

    # Convert paired groups to y_pred and y_true for scipy
    pairs = itertools.chain.from_iterable([zip(*p) for p in filtered])
    y_pred, y_true = zip(*pairs)

    # compute spearman correlation and return
    return scipy.stats.spearmanr(y_true, y_pred)

  @classmethod
  def _debug_print_examples(
      cls,
      prompts: list[str],
      responses: list[str],
      parsed_outputs: list[list[int]],
      groups: list[list[int]],
  ):
    """Prints out a group of examples for debugging purposes."""
    for p, r, o, g in zip(prompts, responses, parsed_outputs, groups):
      for item in (p, r, o, g):
        print("-" * 20)
        print(item)

  def validate_feature_description(
      self,
      feature_index: int,
      feature_description: str,
      num_test_examples: int,
  ) -> tuple[float, int]:
    """Validates a feature description using the Gemini model.

    This method generates validation prompts, queries the Gemini model,
    parses the responses, and computes a correlation score to assess how well
    the provided description aligns with the feature's activations.

    Args:
      feature_index: The index of the feature to validate.
      feature_description: The description string to be validated.
      num_test_examples: The number of examples to use for validation.

    Returns:
      A tuple containing:
        - The correlation score between the model's predictions and the true
          activation quantiles.
        - The total number of examples used in the validation prompts
          (number of prompts * group_size).
    """
    # Sample the documents and generate the prompts
    prompts, groups = self.generate_validation_prompts(
        feature_index, feature_description, num_test_examples
    )

    # Generate responses using gemini
    responses = self.gemini_generator.async_generate(
        prompts, self.validation_config
    )

    # Parse responses for lists of integer predictions
    parsed_outputs = [
        FeatureDescriptionValidator.parse_response(
            r, len(g), self.validation_method
        )
        for r, g in zip(responses, groups)
    ]

    # Compute the correlation on the feature
    return (
        FeatureDescriptionValidator.compute_correlation(
            parsed_outputs, groups, self.validation_method
        ),
        len(prompts) * self.group_size,
    )

  def generate_validation_prompts(
      self,
      feature_index: int,
      feature_description: str,
      num_test_examples: int
  ) -> tuple[list[str], list[tuple[int, int]]]:
    """Create the list of prompts for validation but do not request them."""
    # Get the dataset-wide distribution of max_activations for the feature
    feats = dict(self.cache.get_max_activations(feature_index))
    max_activation = max(feats.values())

    # Bin documents into quantiles
    quantile_dict = self._get_quantiles(feats)

    # If in deterministic mode, set seed for sampling here
    if self.deterministic_group_sampling:
      random.seed(self.group_sample_seed)

    # Sample negative examples and give them quantile -1
    quantile_dict[-1] = self._sample_negatives(feature_index, quantile_dict)

    # Sample the groups of documents that will be together in each prompt
    groups = self._sample_groups(quantile_dict, num_test_examples)

    # Format the prompts, add highlights, snip to max_activation
    prompts = [
        self._format_validation_prompt(
            g, feature_index, feature_description, max_activation
        ).strip()
        for g in groups
    ]

    return prompts, groups

  def submit_validation_batch_job(
      self,
      feature_indices: list[int],
      descriptions: list[str],
      num_test_examples: int,
  ) -> tuple[str, str, dict[int, dict[str, Any]]]:
    """Validate feature description using the batch API."""
    config = self.validation_config
    group_dict = {}
    batch = []
    for i, desc in tqdm.tqdm(list(zip(feature_indices, descriptions))):
      # If feature has no activations on the dataset whatsoever, continue
      if not self.cache.get_active_docs(i):
        continue

      # Get prompts for the model
      prompts, groups = self.generate_validation_prompts(
          i, desc, num_test_examples
      )

      # create a set of ids to track which prompts have which labels
      ids = [str(uuid.uuid4()) for _ in range(len(prompts))]

      # Format prompts in the Gemini batch api format and add to batch
      batch.extend(
          [
              self.gemini_generator.format_prompt(prompt, id, config)
              for prompt, id in zip(prompts, ids)
          ]
      )

      # Create the mapping from feature id to uuid4 and group
      group_dict |= {
          i: {
              "groups": {id: group for id, group in zip(ids, groups)},
              "description": desc,
          }
      }

    # Submit the batch job
    tempfile, job_name = self.gemini_generator.submit_batch_job(batch)

    # return the filepaths and the group dict for the batch
    return tempfile, job_name, group_dict

  def _get_quantile_desc(self, quantile: int):
    if quantile == self.n_quantiles - 1:
      return QUANTILE_DESCRIPTIONS["top"]
    elif quantile == -1:
      return QUANTILE_DESCRIPTIONS["bottom"]
    quantile_str = f"{quantile+1}/{self.n_quantiles}:"
    return QUANTILE_DESCRIPTIONS["middle"] + quantile_str

  def _format_generation_prompt(
      self,
      feature_index: int,
      group: list[tuple[int, int]],
      max_activation: float
  ) -> str:
    """Formats the prompt for generating feature descriptions.

    This method takes a group of documents, grouped by quantile, and formats
    them into a prompt using the `generation_prompt_template`. The prompt
    includes examples of documents and instructions for the model on how to
    generate different types of explanations (terse, keywords, long).

    Args:
      feature_index: the index of the feature to generate description for
      group: A list of tuples, where each contains `(quantile_id, doc_id)`.
      max_activation: the maximum activation of that feature on the dataset

    Returns:
      A string containing the formatted generation prompt.
    """
    threshold = max_activation * HIGHLIGHT_THRESHOLD

    # Convert group to dict: quantile_id -> list of docs
    quantile_dict = collections.defaultdict(list)
    for key, value in group:
      quantile_dict[key].append(value)

    # Convert the documents to string format grouped by quantile
    example_lines = []
    for quantile, doc_ids in quantile_dict.items():
      example_lines.append(self._get_quantile_desc(quantile) + "\n")
      for i, doc_id in enumerate(doc_ids):
        example = self._apply_highlights(doc_id, feature_index, threshold)
        example_lines.append(f"{i+1}. {repr(example)}\n")
      example_lines.append("\n")

    # Convert the explanation types to strings
    explanation_types_str = "\n".join(
        [f"- <{k}>: {v}" for k, v in EXPLANATION_TYPES.items()]
    )
    explanation_types_structure_str = "\n" + "\n".join(
        [f"<{k}>...</{k}>" for k in EXPLANATION_TYPES.keys()]
    )
    model_prefix_str = f"<{list(EXPLANATION_TYPES.keys())[0]}>"

    # Return the completed generation prompt
    return self.generation_prompt_template.format(
        grouping_explanation=GROUPING_EXPLANATIONS["quantile_grouped"],
        explanation_types=explanation_types_str,
        explanation_types_structure=explanation_types_structure_str,
        examples="".join(example_lines),
        model_prefix=model_prefix_str,
        highlight_explanation=self.highlight_explanation
    )

  def _parse_tagged_text(self, text: str) -> dict[str, str]:
    """Parse text for explanations within tags e.g. <terse>...</terse>."""
    pattern = re.compile(r"<(terse|keywords|long)>(.*?)</\1>", re.DOTALL)
    return {tag: content.strip() for tag, content in pattern.findall(text)}

  def generate_new_autointerp_description(
      self, feature_index: int, num_test_examples: int
  ) -> dict[str, str]:
    """Generate a new description for the given feature_index."""
    # Get the dataset-wide distribution of max_activations for the feature
    feats = dict(self.cache.get_max_activations(feature_index))
    max_activation = max(feats.values())

    # Bin documents into quantiles
    quantile_dict = self._get_quantiles(feats)

    # If in deterministic mode, set seed for sampling here
    if self.deterministic_group_sampling:
      random.seed(self.group_sample_seed)

    # Sample negative examples and give them quantile -1
    quantile_dict[-1] = self._sample_negatives(feature_index, quantile_dict)

    # Sample the group of documents that will be together in the prompt
    groups = self._sample_groups(quantile_dict, num_test_examples)

    # Create the generation prompt from the sampled documents
    group = list(itertools.chain.from_iterable(groups))
    prompt = self._format_generation_prompt(
        feature_index, group, max_activation
    )

    # Generate feature description using gemini
    response = self.gemini_generator.generate(
        [prompt.strip()], self.generation_config
    )

    # Return the parsed description
    return self._parse_tagged_text(response[0].text)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-d",
      "--data_path",
      type=str,
      default="./activations/all_activations_10k_csr",
      help="Path to the dataset on disk",
  )
  parser.add_argument(
      "-f",
      "--feature_index",
      type=int,
      default=3644,
      help="Index of feature to validate",
  )
  parser.add_argument(
      "-t",
      "--num_test_examples",
      type=int,
      default=20,
      help="Number of examples to use when testing correlation (default: 20)",
  )
  parser.add_argument(
      "-v",
      "--validation_method",
      type=str,
      default="quantile",
      choices=["quantile", "binary", "rank", "pairwise"],
      help="Method for validating the feature description",
  )
  parser.add_argument(
      "-i",
      "--highlight_method",
      type=str,
      default="xml",
      choices=["xml", "angled", "none"],
      help="Method for highlighting large activations in the text",
  )
  parser.add_argument(
      "-n",
      "--negative_sampling_method",
      type=str,
      default="uniform",
      choices=["uniform", "semantic"],
      help="Method for sampling negatives in the pipeline",
  )
  parser.add_argument(
      "-q",
      "--quantile_method",
      type=str,
      default="density",
      choices=["density", "scaled"],
      help="Method for determining quantiles of activations",
  )
  parser.add_argument(
      "-p",
      "--prompt_sampling_method",
      type=str,
      default="stratified",
      choices=["uniform", "stratified"],
      help="Method for sampling examples to include in the prompt",
  )
  parser.add_argument(
      "-g",
      "--group_size",
      type=int,
      default=5,
      help="Number of examples to run prediction on at once (default: 5)",
  )
  parser.add_argument(
      "-nq",
      "--n_quantiles",
      type=int,
      default=4,
      help="Number of quantiles to split data into (default: 4)",
  )
  parser.add_argument(
      "--use_snippets",
      action="store_true",
      help="If true, use range around max activation rather than full text",
  )
  args = parser.parse_args()

  # Loading in components for the validator
  print("Loading Dataset and Tokenizer...")
  ds = datasets.load_from_disk(args.data_path)
  cache = MaxActivationCache(
      os.path.join(args.data_path, "max_activations.pth")
  )
  extractor = SAEExtractor()

  # TODO(liamdugan): Turn this into a frozen config dataclass
  validator_kwargs = {
      "validation_method": args.validation_method,
      "highlight_method": args.highlight_method,
      "negative_sampling_method": args.negative_sampling_method,
      "quantile_method": args.quantile_method,
      "prompt_sampling_method": args.prompt_sampling_method,
      "n_quantiles": args.n_quantiles,
      "use_snippets": args.use_snippets,
      "group_size": args.group_size,
  }

  # Initialize the validator
  print("Creating validator...")
  validator = FeatureDescriptionValidator(
      ds, extractor, cache, **validator_kwargs
  )

  # Get autointerp description from neuronpedia
  print("Getting Description...")
  description = utils.get_explanations_for_sae_feature(args.feature_index)

  # Compute the correlation
  print("Running Validation...")
  corr, num_examples = validator.validate_feature_description(
      args.feature_index, description, args.num_test_examples
  )

  print(f"Correlation: {corr}, Num examples: {num_examples}")
