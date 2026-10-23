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

"""Describes SAE features with Gemini and validates the descriptions.

`FeatureDescriptionValidator` generates and validates natural-language
descriptions of Sparse Autoencoder (SAE) features from their activation
patterns across a dataset, using a Gemini model for both tasks. Validation
scores a description by how well Gemini can recover each document's activation
quantile from it; the resulting Spearman correlation is the simulation score
stored as `correlation` in the `*_feature_results.jsonl` files, which
`data.cache.get_interp_mask` thresholds with tau to build the interp mask.

Run as a script (README step 5), for each selected feature it:
1.  Asks Gemini for a description from the feature's highest-activating
    documents in the activation cache.
2.  Validates it with the quantile method: Gemini is shown the description
    and guesses the activation quantile of documents sampled across
    quantiles; the Spearman correlation with the true quantiles is recorded.
    This is the simulation score the heads threshold on (tau).
3.  Appends one JSON record per feature to a JSONL file.

Features are an explicit ``--feature_list``; otherwise, when the split has a
``layer<N>_log_odds_ratios.csv`` (``process lor``), the features with the
highest |log odds ratio| seen often enough, which is how the paper's features
were chosen (``--use_highest_lor_features`` forces this, ``-k`` caps it);
otherwise the ``-k`` densest features in the cache.

Reads from ``data_path``: the saved dataset, ``layer<N>_max_activations.pth``
and, when needed, ``layer<N>_log_odds_ratios.csv`` and ``embeddings.npy``
(computed and saved on first use for semantic negative sampling). Writes
``layer<N>_<output_file>`` (``_partition<i>`` suffix with ``--do_partition``).
The heads look for ``layer<N>_<model>_feature_results.jsonl``, so pass
``-o <model>_feature_results.jsonl``. Interrupted runs resume with
``--check_for_existing_features``.

Usage (the paper's settings):
  python -m detectorscope.autointerpret \
      -d data/sae_activations/PRDB_train_gemma2_gpt4o \
      -em models/detectors/gpt4o_detect_gemma -l 7 -x review \
      -t 50 -tg 10 --use_snippets -m gemini-3.1-flash-lite-preview \
      -o gemini-3.1-flash-lite-preview_feature_results.jsonl
"""

import argparse
import bisect
import collections
from collections.abc import Iterator
import itertools
import json
import math
import os
import random
import re
from typing import Any

import datasets
from google import genai
from detectorscope.data.cache import CSR_KEYS
from detectorscope.data.cache import sae_columns
from detectorscope.data.cache import sae_hook
from detectorscope.data.cache import MaxActivationCache
from detectorscope.model.gemini import GeminiGenerator
from detectorscope.model.extractor import HFExtractor
import more_itertools
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
import torch
import tqdm

# The two prompt templates ship with the package under prompts/.
PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "prompts"
)

HIGHLIGHTS = {
    "xml": ("<highlight>", "</highlight>"),
    "angled": ("<<", ">>"),
    "none": ("", ""),
}

# A token is highlighted when its activation is at least this fraction of the
# feature's dataset-wide max activation.
HIGHLIGHT_THRESHOLD = 0.1

HIGHLIGHT_EXPLANATION_TEMPLATE = (
    "For each sequence, the strongest activating words are indicated with"
    ' {pre}...{post} - note that these tags "{pre}" and "{post}" are'
    " not part of the actual sentence, they're just used to highlight the"
    " places where the neuron is most active.\n"
)

# Sentinel returned by parse_response for an unparseable Gemini reply.
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
    "negative": "These are documents that do not activate the feature at all:",
}


def add_values_col(
    ds: datasets.Dataset, values_col: str, layer: int | None = None
) -> datasets.Dataset:
  """Adds a column that packs a layer's CSR activation columns into dicts.

  Used by the auto-interp validator to index one document's activations at a
  time. ``layer`` is the GemmaScope layer index; `cache.sae_hook` maps it to
  the hook whose columns are read (layer 7 reads ``model.layers.8_input_*``).
  A falsy layer (None or 0) reads the unprefixed CSR columns instead.

  Args:
    ds: activation dataset written by extract.py sae.
    values_col: name of the new column; each entry is a dict over CSR_KEYS.
    layer: GemmaScope layer index, or None for unprefixed columns.

  Returns:
    ds with a lazy format transform; nothing is written to disk.

  Raises:
    ValueError: if any of the four CSR columns is missing.
  """
  columns = sae_columns(sae_hook(layer)) if layer else {k: k for k in CSR_KEYS}

  if not all(c in ds.column_names for c in columns.values()):
    raise ValueError("Not all CSR columns are present")

  def iterate_slice(ds_slice: datasets.Dataset) -> Iterator[dict[str, Any]]:
    """Iterate over a slice of the dataset and only return the CSR keys."""
    for i in range(len(ds_slice[columns[CSR_KEYS[0]]])):
      yield {key: ds_slice[col][i] for key, col in columns.items()}

  return ds.with_transform(lambda x: {values_col: list(iterate_slice(x))} | x)


def interleave_lists(lists: list[list[Any]]) -> list[Any]:
  """Interleaves lists round-robin into one list, dropping None items."""
  return [
      x
      for x in itertools.chain.from_iterable(itertools.zip_longest(*lists))
      if x is not None
  ]


def nested_shuffle(lists: list[list[Any]]) -> list[list[Any]]:
  """Shuffles the outer list and every inner list in place; returns lists."""
  random.shuffle(lists)
  for sublist in lists:
    random.shuffle(sublist)
  return lists


def flatten(lists: list[list[Any]]) -> list[Any]:
  """Concatenates a list of lists into one list."""
  return list(itertools.chain(*lists))


def compute_embedding_matrix(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> np.ndarray:
  """Converts a list of texts into an embedding matrix [n_texts, model_dim].

  Rows are L2-normalised float32, so a row dot product is a cosine
  similarity. run() caches the result as embeddings.npy for semantic
  negative sampling; the paraphrase analysis scores rewrite pairs.
  """
  # Imported here: nothing else in the package needs sentence_transformers.
  from sentence_transformers import SentenceTransformer  # pylint: disable=import-outside-toplevel

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = SentenceTransformer(model_name, device=device)
  return model.encode(
      texts,
      batch_size=batch_size,
      show_progress_bar=True,
      normalize_embeddings=True,
  ).astype(np.float32)


class FeatureDescriptionValidator:
  """Generates and validates feature descriptions with a Gemini model.

  Documents are binned by their max activation into `n_quantiles` quantiles
  (inactive documents form quantile -1), sampled into groups of `group_size`,
  and rendered with the activating tokens highlighted. Generation shows one
  prompt grouped by quantile and asks for a description; validation shows a
  description and asks Gemini to guess each document's quantile, scored by
  Spearman correlation with the truth.

  Args:
    dataset: activation dataset written by extract.py sae, with the text in
      `text_col` and per-token CSR activation columns.
    sae_extractor: extractor whose tokenizer renders documents into tokens;
      the model itself is not needed.
    activation_cache: max activation cache of the same layer and split.
    highlight_method: key of HIGHLIGHTS; "none" disables highlighting.
    negative_sampling_method: "uniform", or "semantic" to refill each group's
      negative slots with inactive documents assigned to that group's
      positives by a global max-cosine matching (needs `embeddings`).
    quantile_method: "density" for equal-count bins, "scaled" for equal-width
      bins over [0, max activation].
    prompt_sampling_method: "stratified" interleaves quantiles so every group
      spans them; "uniform" shuffles freely.
    n_quantiles: number of active quantiles; -1 is added for inactive docs.
    snippet_len: tokens kept on either side of the max-activating token.
    model_name: Gemini model name.
    use_snippets: crop each document to a snippet around its max activation.
    group_size: documents per validation prompt.
    text_col: dataset column holding the document text.
    activation_col: column with the packed CSR activations; added via
      `add_values_col` when absent.
    layer_number: GemmaScope layer whose CSR columns to pack; None reads the
      unprefixed columns.
    embeddings: [n_docs, dim] unit-normalised document embeddings, used only
      for semantic negative sampling.
  """

  def __init__(
      self,
      dataset: datasets.Dataset,
      sae_extractor: HFExtractor,
      activation_cache: MaxActivationCache,
      highlight_method: str = "xml",
      negative_sampling_method: str = "uniform",
      quantile_method: str = "density",
      prompt_sampling_method: str = "stratified",
      n_quantiles: int = 4,
      snippet_len: int = 20,
      model_name: str = "gemini-2.5-pro",
      use_snippets: bool = True,
      group_size: int = 5,
      text_col: str = "review",
      activation_col: str = "sae_activation",
      layer_number: int | None = None,
      embeddings: np.ndarray | None = None,
  ):
    self.data = dataset
    self.cache = activation_cache
    self.model_name = model_name
    self.gemini_generator = GeminiGenerator(model_name)
    self.extractor = sae_extractor
    self.text_col = text_col
    self.activation_col = activation_col
    self.embed = embeddings

    self.deterministic_group_sampling = True
    self.group_sample_seed = 42

    # Only the quantile validation method survives; the name is kept because
    # it selects the prompt template and labels the output records.
    self.validation_method = "quantile"
    self.highlight_method = highlight_method
    self.negative_sampling_method = negative_sampling_method
    self.prompt_sampling_method = prompt_sampling_method
    self.quantile_method = quantile_method
    self.n_quantiles = n_quantiles

    self.use_snippets = use_snippets
    self.snippet_len = snippet_len
    self.layer_number = layer_number

    self.group_size = group_size

    # Pack the layer's CSR columns into one dict column if not already present
    if self.activation_col not in self.data.column_names:
      self.data = add_values_col(
          self.data, self.activation_col, self.layer_number
      )

    with open(os.path.join(PROMPTS_DIR, f"{self.validation_method}.txt")) as f:
      self.validation_prompt_template = f.read()

    with open(os.path.join(PROMPTS_DIR, "generation.txt")) as f:
      self.generation_prompt_template = f.read()

    self.highlight_explanation = ""
    if self.highlight_method != "none":
      self.highlight_explanation = HIGHLIGHT_EXPLANATION_TEMPLATE.format(
          pre=HIGHLIGHTS[self.highlight_method][0],
          post=HIGHLIGHTS[self.highlight_method][1],
      )

    self.generation_config = genai.types.GenerateContentConfig(
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        thinking_config=genai.types.ThinkingConfig(thinking_level="low"),
    )
    self.validation_config = genai.types.GenerateContentConfig(
        temperature=0,
        thinking_config=genai.types.ThinkingConfig(thinking_level="low"),
    )

  def _get_quantiles(
      self, activations: dict[int, float]
  ) -> dict[int, list[int]]:
    """Bins documents by max activation quantile.

    Args:
      activations: doc_id -> max activation, active documents only.

    Returns:
      quantile id -> doc ids, with ids 0..n_quantiles-1 from lowest to highest
      activation and -1 holding every document absent from `activations`.
    """
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

    # Assign all inactive documents quantile -1
    active_docs = set(activations.keys())
    quantile_dict[-1] = list(set(range(len(self.data))) - active_docs)

    return quantile_dict

  def _equalize_quantiles(
      self, quantile_dict: dict[int, list[int]], n: int
  ) -> dict[int, int]:
    """Decides how many docs to draw per quantile, as equally as n allows.

    Quantiles smaller than their share are exhausted and the remainder is
    redistributed over the others. A falsy n means every document.

    Returns:
      quantile id -> number of documents to sample from it.
    """
    q_sizes = {qid: len(bin) for qid, bin in quantile_dict.items()}

    # clamp n to be at most len(dataset)
    n = min(n, sum(q_sizes.values())) if n else sum(q_sizes.values())

    sampled = {qid: 0 for qid in quantile_dict.keys()}

    while (items_left := n - sum(sampled.values())) > 0:
      # compute the number of items to get from each quantile
      target = max(int(items_left / len(quantile_dict)), 1)

      # For each quantile, sample at most 'target' examples
      for qid in quantile_dict.keys():
        sampled[qid] += min(target, q_sizes[qid] - sampled[qid])
        if sum(sampled.values()) >= n:
          break

    # Return N items from each quantile
    return sampled

  def _compute_pairwise_semantic(
      self, pos_docs: list[int], neg_docs: list[int]
  ) -> dict[int, int]:
    """Matches each positive doc to a distinct negative doc.

    The one-to-one assignment maximises the global sum of cosine similarity
    (embeddings are assumed unit-normalised, so the dot product is cosine).

    Returns:
      pos_doc -> (neg_doc, cosine_sim) for every matched positive.
    """
    # Similarity matrix [total_positives, n_inactive].
    # Entry (i, j) = cosine_similarity(positive_i, candidate_j).
    sim_matrix = self.embed[pos_docs] @ self.embed[neg_docs].T

    # Find the globally most-similar assignment between columns
    # (note: linear_sum_assignment minimises, so we pass -sim)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-sim_matrix)

    return {
        pos_docs[i]: (neg_docs[j], sim_matrix[i, j])
        for i, j in zip(row_ind, col_ind)
    }

  def _inject_semantic_negatives(
      self,
      groups: list[list[tuple[int, int]]],
      quantile_dict: dict[int, list[int]],
  ) -> list[list[tuple[int, int]]]:
    """Replaces uniformly sampled negatives with semantic ones.

    Each negative slot in a group is refilled with the inactive document
    matched to one of that group's positives, highest similarity first, so a
    negative cannot be told apart from the positives by topic alone.
    """
    # Get all positive and negative docs in the current pool
    pos_docs = [doc for qid, doc in flatten(groups) if qid != -1]
    neg_docs = quantile_dict[-1]

    # Compute the matching between positive and negative
    # that maximizes cosine similarity globally
    pairs = self._compute_pairwise_semantic(pos_docs, neg_docs)

    # replace existing negative documents with ones that have
    # high similarity to another positive example in the group
    resampled_groups = []
    for group in groups:
      paired_negatives = sorted(
          [pairs[doc] for qid, doc in group if qid != -1 and doc in pairs],
          key=lambda x: x[1],
          reverse=True,
      )

      resampled_group = []
      i = 0
      for qid, doc_id in group:
        if qid == -1:
          # resample negative documents (highest sim first)
          if i < len(paired_negatives):
            doc_id = paired_negatives[i][0]
            i += 1
        resampled_group.append((qid, doc_id))

      resampled_groups.append(resampled_group)

    return resampled_groups

  def _sample_examples(
      self, grouped_qids: list[list[int]], quantile_dict: dict[int, list[int]]
  ) -> list[list[tuple[int, int]]]:
    """Draws a distinct document for every quantile slot in grouped_qids."""
    # First sample positives. Initialize counters
    counters = {k: 0 for k in quantile_dict.keys()}

    # shuffle all quantile dict lists
    shuffled_qdict = {
        qid: random.sample(docs, len(docs))
        for qid, docs in quantile_dict.items()
    }

    # Sample documents for each qid such that we now have
    # a list of lists of pairs (qid, sampled_doc_id)
    # [[-1, 3], [0, 2]] --> [[(-1, 343), (3, 488)], [(0, 34), (2, 32)]]
    groups = []
    for qid_group in grouped_qids:
      sampled_group = []
      for qid in qid_group:
        sampled_doc = shuffled_qdict[qid][counters[qid]]
        sampled_group.append((qid, sampled_doc))
        counters[qid] += 1
      groups.append(sampled_group)

    # If we're doing semantic negative sampling, replace
    # uniformly sampled negatives with semantic negatives
    if self.negative_sampling_method == "semantic":
      groups = self._inject_semantic_negatives(groups, quantile_dict)

    return groups

  def _sample_groups(
      self, quantile_dict: dict[int, list[int]], num_test_examples: int
  ) -> list[list[tuple[int, int]]]:
    """Samples the (quantile id, doc id) groups that share one prompt.

    A trailing chunk shorter than group_size is dropped, so the number of
    documents used can be below num_test_examples.
    """
    # Decide num examples per quantile to sample
    num_docs_per_quantile = self._equalize_quantiles(
        quantile_dict, num_test_examples
    )

    # Convert quantile dict to list of lists of qid
    q_lists = [
        [qid] * num_docs for qid, num_docs in num_docs_per_quantile.items()
    ]

    # Flatten & shuffle for uniform, interleave for stratified
    if self.prompt_sampling_method == "uniform":
      q_list = list(itertools.chain.from_iterable(q_lists))
      random.shuffle(q_list)
    elif self.prompt_sampling_method == "stratified":
      q_list = interleave_lists(q_lists)
    else:
      raise ValueError(
          f"Unknown prompt sampling method: {self.prompt_sampling_method}"
      )

    # Chunk the list into groups of len group_size
    grouped_qids = [
        g[: self.group_size]
        for g in more_itertools.chunked(q_list, self.group_size)
        if len(g) >= self.group_size
    ]

    # Sample documents from the quantile dict for each qid
    groups = self._sample_examples(grouped_qids, quantile_dict)

    # shuffle the list of lists and return
    return nested_shuffle(groups)

  def _apply_highlights(
      self,
      doc_id: int,
      feature_id: int,
      act_threshold: float,
      do_random_highlight: bool = True,
  ) -> str:
    """Renders a document with the feature's activating tokens highlighted.

    Tokens at or above act_threshold are wrapped in the highlight markers and
    adjacent highlights are merged. An inactive document is centred on a
    random token; with do_random_highlight that token is also highlighted so a
    validation prompt does not give negatives away by their missing marker.
    """
    tokens = self.extractor.get_token_strings(self.data[doc_id][self.text_col])

    # Get cached SAE activations of all features for the particular document
    activations = torch.sparse_csr_tensor(
        **self.data[doc_id][self.activation_col]
    )

    # Extract the activations of the particular feature on the document
    a_list = [activations[0, i, feature_id].item() for i in range(len(tokens))]

    # If all activations are 0, pick a random center (and optionally highlight)
    center = None
    if not any(a_list):
      center = random.choice(range(len(tokens)))
      if do_random_highlight:
        a_list[center] = act_threshold

    # If using snippets, crop document to +/- snippet len around max activ.
    if self.use_snippets:
      max_i = center if center is not None else a_list.index(max(a_list))
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
    """Formats one group of highlighted documents into a validation prompt.

    Args:
      group: (quantile_id, doc_id) pairs, in prompt order.
      feature_index: feature being validated.
      feature_description: description Gemini is asked to simulate.
      max_activation: dataset-wide max activation of the feature; sets the
        highlight threshold.

    Returns:
      The filled `prompts/quantile.txt` template.
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
    """Parses a Gemini reply into the predicted quantile per document.

    Returns:
      The list of `size` integers, or FAILURE when the reply is not exactly a
      comma-separated list of that many integers.
    """
    if validation_method != "quantile":
      raise ValueError(f"Unknown validation method: {validation_method}")

    if not re.fullmatch(r"\s*\d+(\s*,\s*\d+)*\s*", text):
      return FAILURE

    outputs = [int(n) for n in re.findall(r"\d+", text)]

    if len(outputs) != size:
      return FAILURE

    return outputs

  @classmethod
  def compute_correlation(
      cls,
      outputs: list[list[int]],
      groups: list[list[tuple[int, int]]],
      validation_method: str,
  ) -> float:
    """Computes the Spearman correlation of predicted and true quantiles.

    Groups whose reply failed to parse are dropped; the rest are pooled over
    all documents before correlating.

    Returns:
      scipy's spearmanr result, a (rho, p_value) pair. This is what the
      feature results files store under `correlation`; `cache.get_interp_mask`
      compares rho against tau.
    """
    # Pair outputs w/ true quantile values and filter out parsing failures
    groups = [[qid for qid, _ in g] for g in groups]
    filtered = [(o, g) for o, g in zip(outputs, groups) if o != FAILURE]

    if validation_method != "quantile":
      raise ValueError(f"Unknown validation method: {validation_method}")

    # Convert paired groups to y_pred and y_true for scipy
    pairs = itertools.chain.from_iterable([zip(*p) for p in filtered])
    y_pred, y_true = zip(*pairs)

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
    """Scores how well a description predicts the feature's activations.

    Builds the validation prompts, queries Gemini asynchronously, parses the
    replies and correlates the predicted quantiles with the true ones.

    Args:
      feature_index: feature to validate.
      feature_description: description to be validated.
      num_test_examples: documents to sample across quantiles.

    Returns:
      (correlation, n_examples): the spearmanr result of compute_correlation
      and the number of documents actually prompted (prompts * group_size).
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
      self, feature_index: int, feature_description: str, num_test_examples: int
  ) -> tuple[list[str], list[tuple[int, int]]]:
    """Builds the validation prompts without sending them.

    Returns:
      (prompts, groups): one prompt per group, where each group is the list
      of (quantile_id, doc_id) pairs it shows, in the same order.
    """
    # Get the dataset-wide distribution of max_activations for the feature
    feats = dict(self.cache.get_max_activations(feature_index))
    max_activation = max(feats.values())

    # Bin documents into quantiles
    quantile_dict = self._get_quantiles(feats)

    # In deterministic mode, reseed so the same feature always yields the same
    # groups
    if self.deterministic_group_sampling:
      random.seed(self.group_sample_seed)

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

  def _get_quantile_desc(self, quantile: int):
    """Header introducing one quantile's examples in a generation prompt."""
    if quantile == self.n_quantiles - 1:
      return QUANTILE_DESCRIPTIONS["top"]
    elif quantile == 0:
      return QUANTILE_DESCRIPTIONS["bottom"]
    elif quantile == -1:
      return QUANTILE_DESCRIPTIONS["negative"]
    quantile_str = f"{quantile+1}/{self.n_quantiles}:"
    return QUANTILE_DESCRIPTIONS["middle"] + quantile_str

  def _format_generation_prompt(
      self,
      feature_index: int,
      group: list[tuple[int, int]],
      max_activation: float,
  ) -> str:
    """Formats the prompt for generating a feature description.

    Documents are listed by quantile, highest activating first, followed by
    instructions for each EXPLANATION_TYPES entry (terse, keywords, long).
    Inactive documents are never given a random highlight here.

    Args:
      feature_index: feature to describe.
      group: (quantile_id, doc_id) pairs; grouped by quantile internally.
      max_activation: dataset-wide max activation of the feature; sets the
        highlight threshold.

    Returns:
      The filled `prompts/generation.txt` template.
    """
    threshold = max_activation * HIGHLIGHT_THRESHOLD

    # Convert group to dict: quantile_id -> list of docs
    quantile_dict = collections.defaultdict(list)
    for key, value in group:
      quantile_dict[key].append(value)

    # Sort quantile dict by key (highest activating documents first)
    quantile_dict = dict(sorted(quantile_dict.items(), reverse=True))

    # Convert the documents to string format grouped by quantile
    example_lines = []
    for quantile, doc_ids in quantile_dict.items():
      example_lines.append(self._get_quantile_desc(quantile) + "\n")
      for i, doc_id in enumerate(doc_ids):
        example = self._apply_highlights(
            doc_id, feature_index, threshold, do_random_highlight=False
        )
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
        highlight_explanation=self.highlight_explanation,
    )

  def _parse_tagged_text(self, text: str) -> dict[str, str]:
    """Parse text for explanations within tags e.g. <terse>...</terse>."""
    pattern = re.compile(r"<(terse|keywords|long)>(.*?)</\1>", re.DOTALL)
    return {tag: content.strip() for tag, content in pattern.findall(text)}

  def generate_new_autointerp_description(
      self, feature_index: int, num_test_examples: int
  ) -> dict[str, str]:
    """Asks Gemini for a new description of the given feature.

    Samples the same groups as validation would and flattens them into a
    single generation prompt.

    Returns:
      explanation type -> text for each of terse, keywords and long that the
      reply contained; a type Gemini omitted is absent from the dict.
    """
    # Get the dataset-wide distribution of max_activations for the feature
    feats = dict(self.cache.get_max_activations(feature_index))
    max_activation = max(feats.values())

    # Bin documents into quantiles
    quantile_dict = self._get_quantiles(feats)

    # In deterministic mode, reseed so the same feature always yields the same
    # groups
    if self.deterministic_group_sampling:
      random.seed(self.group_sample_seed)

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

    return self._parse_tagged_text(response[0].text)


def _get_k_most_dense_features(
    activation_cache: MaxActivationCache, top_k: int
) -> list[int]:
  """Returns the top_k features with the highest density in the cache."""
  density = torch.tensor(activation_cache.get_feature_density())
  return torch.topk(density, top_k).indices.tolist()


def _get_k_highest_lor_features(
    lor_path: str, top_k: int | None = None, min_occ: int = 25
) -> list[int]:
  """Returns the top_k highest |log odds ratio| features seen > min_occ times.

  Occurrence is ``human_density + target_density`` from the LOR csv (document
  counts rescaled to the smaller class); run() passes num_test_examples so
  each feature has enough documents to validate.
  """
  lor_df = pd.read_csv(lor_path)
  feats = (
      lor_df.loc[
          lambda x: (x["human_density"] + x["target_density"]) > min_occ
      ].reindex(lor_df["lor"].abs().sort_values(ascending=False).index)
  ).dropna()
  return (
      (feats.head(top_k) if top_k else feats)["feature_id"].astype(int).tolist()
  )


def run_validation_experiment(
    validator: FeatureDescriptionValidator,
    feature_indices: list[int],
    num_test_examples: int,
    num_gen_examples: int,
) -> Iterator[dict[str, Any]]:
  """Generates and validates a description for each feature.

  Generation is retried until the response contains a "terse" description.

  Args:
    validator: The validator holding the dataset, extractor and cache.
    feature_indices: Features to describe and validate.
    num_test_examples: Number of examples used for the validation correlation.
    num_gen_examples: Number of examples shown when generating a description.

  Yields:
    One record per feature with the description and its correlation.
  """
  for i in feature_indices:
    description = {}
    while "terse" not in description:
      description = validator.generate_new_autointerp_description(
          i, num_gen_examples
      )

    corr, num_examples = validator.validate_feature_description(
        i, description["terse"], num_test_examples
    )

    yield {
        "feature_index": i,
        "num_examples": num_examples,
        "description": description,
        "correlation": corr,
        "model_name": validator.model_name,
    }


def run(args: argparse.Namespace) -> None:
  """Selects features, describes and validates them, and appends JSONL.

  Records are flushed one at a time, so a killed job loses at most one.
  """
  prefix = f"layer{args.layer_number}_" if args.layer_number else ""
  cache_path = os.path.join(args.data_path, f"{prefix}max_activations.pth")
  lor_path = os.path.join(args.data_path, f"{prefix}log_odds_ratios.csv")

  ds = datasets.load_from_disk(args.data_path)
  cache = MaxActivationCache.from_pth(cache_path)
  extractor = HFExtractor(args.extractor_path, tokenizer_only=True)

  embeddings = None
  if args.negative_sampling_method == "semantic":
    try:
      embeddings_path = os.path.join(args.data_path, "embeddings.npy")
      embeddings = np.load(embeddings_path).astype(np.float32)
      print(f"Loaded embeddings from {embeddings_path}")
    except FileNotFoundError:
      print("Embedding matrix not found. Caching embeddings...")
      embeddings = compute_embedding_matrix(ds[args.text_col])
      print(f"Saving embeddings to {embeddings_path}...")
      np.save(embeddings_path, embeddings)

  validator_keyword_args = {
      "highlight_method": args.highlight_method,
      "negative_sampling_method": args.negative_sampling_method,
      "quantile_method": args.quantile_method,
      "prompt_sampling_method": args.prompt_sampling_method,
      "n_quantiles": args.n_quantiles,
      "use_snippets": args.use_snippets,
      "snippet_len": args.snippet_len,
      "group_size": args.group_size,
      "model_name": args.model_name,
      "layer_number": args.layer_number,
      "text_col": args.text_col,
      "embeddings": embeddings,
  }

  description_validator = FeatureDescriptionValidator(
      ds, extractor, cache, **validator_keyword_args
  )

  if not (feature_ids := args.feature_list):
    if args.use_highest_lor_features or (
        args.top_k is None and os.path.exists(lor_path)
    ):
      feature_ids = _get_k_highest_lor_features(
          lor_path, args.top_k, args.num_test_examples
      )
    elif args.top_k is not None:
      feature_ids = _get_k_most_dense_features(cache, args.top_k)
    else:
      raise ValueError(
          f"{lor_path} not found (process lor): pass -k to select the "
          "densest features, or -f an explicit feature list"
      )

  output_descriptions_path = os.path.join(
      args.data_path, f"{prefix}{args.output_file}"
  )

  if args.check_for_existing_features:
    if os.path.exists(output_descriptions_path):
      df = pd.read_json(output_descriptions_path, lines=True)
      existing_feats = (
          df["feature_index"].unique().tolist() if not df.empty else []
      )
      num_old, num_existing = len(feature_ids), len(existing_feats)
      feature_ids = [i for i in feature_ids if i not in existing_feats]
      print(
          f"Found {num_existing} existing features...\n"
          f"Removed {num_old - len(feature_ids)} features "
          f"({num_old} -> {len(feature_ids)})"
      )

  if args.do_partition:
    rng = random.Random(args.partition_seed)
    shuffled = list(feature_ids)
    rng.shuffle(shuffled)
    chunk_size = math.ceil(len(shuffled) / args.num_partitions)
    start = args.partition_index * chunk_size
    feature_ids = shuffled[start : start + chunk_size]
    print(
        f"Partition {args.partition_index}/{args.num_partitions}: "
        f"processing {len(feature_ids)} features"
    )

    # Give each partition its own output file so parallel jobs don't collide
    base, ext = os.path.splitext(output_descriptions_path)
    output_descriptions_path = f"{base}_partition{args.partition_index}{ext}"

    # If we're also checking for existing features, remove any that have
    # already been written to this partition's output file. Do this AFTER
    # partitioning so the partition assignment stays consistent across restarts.
    if args.check_for_existing_features and os.path.exists(
        output_descriptions_path
    ):
      part_df = pd.read_json(output_descriptions_path, lines=True)
      existing_in_partition = (
          part_df["feature_index"].unique().tolist()
          if not part_df.empty
          else []
      )
      num_before = len(feature_ids)
      feature_ids = [i for i in feature_ids if i not in existing_in_partition]
      print(
          f"Partition {args.partition_index}: skipping "
          f"{num_before - len(feature_ids)} already-completed features "
          f"({num_before} -> {len(feature_ids)})"
      )

  # If num gen examples is not specified, set to group_size * 2
  # for consistency with legacy implementation
  if (num_gen_examples := args.num_gen_examples) is None:
    num_gen_examples = description_validator.group_size * 2

  with open(output_descriptions_path, "a") as file:
    for output_record in tqdm.tqdm(
        run_validation_experiment(
            description_validator,
            feature_ids,
            args.num_test_examples,
            num_gen_examples,
        ),
        total=len(feature_ids),
        desc="Validating Features",
    ):
      json_line = json.dumps(output_record)
      file.write(json_line + "\n")
      file.flush()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-d",
      "--data_path",
      type=str,
      help="Path to the dataset",
  )
  parser.add_argument(
      "-f",
      "--feature_list",
      type=int,
      nargs="*",
      help="Features to describe and validate (default: by log odds ratio)",
  )
  parser.add_argument(
      "-k",
      "--top_k",
      type=int,
      default=None,
      help="Select the k densest features; caps the LOR selection instead "
      "when it applies",
  )
  parser.add_argument(
      "-t",
      "--num_test_examples",
      type=int,
      help="Num examples for testing correlation",
  )
  parser.add_argument(
      "-tg",
      "--num_gen_examples",
      type=int,
      help="Num examples for generating descriptions",
  )
  parser.add_argument(
      "-i",
      "--highlight_method",
      type=str,
      default="angled",
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
      default="scaled",
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
      "-sl",
      "--snippet_len",
      type=int,
      default=20,
      help="Length of snippets to apply to the documents",
  )
  parser.add_argument(
      "-x",
      "--text_col",
      type=str,
      default="review",
      help="Name of the text column to use for extraction",
  )
  parser.add_argument(
      "--use_snippets",
      action="store_true",
      help="If true, use range around max activation rather than full text",
  )
  parser.add_argument(
      "--use_highest_lor_features",
      action="store_true",
      help="Select by |log odds ratio| even when -k is given",
  )
  parser.add_argument(
      "--check_for_existing_features",
      action="store_true",
      help="If true, dont regenerate descriptions for existing feats",
  )
  parser.add_argument(
      "-m",
      "--model_name",
      type=str,
      default="gemini-2.5-flash",
      choices=[
          "gemini-2.5-flash-lite",
          "gemini-2.5-flash",
          "gemini-2.5-pro",
          "gemini-3-pro-preview",
          "gemini-3-flash-preview",
          "gemini-3.1-pro-preview",
          "gemini-3.1-flash-lite-preview",
      ],
      help="LLM to use for validating features and generating descriptions",
  )
  parser.add_argument(
      "-o",
      "--output_file",
      type=str,
      default="feature_results.jsonl",
      help="output file for feature correlation results",
  )
  parser.add_argument(
      "-em",
      "--extractor_path",
      type=str,
      required=True,
      help="Path to the HF model (or hub id) whose tokenizer is used",
  )
  parser.add_argument(
      "-l",
      "--layer_number",
      type=int,
      default=None,
      help="(Optional) Name for the layer. Defaults to empty str",
  )
  parser.add_argument(
      "--do_partition",
      action="store_true",
      help="If true, partition features across parallel jobs",
  )
  parser.add_argument(
      "--num_partitions",
      type=int,
      default=1,
      help="Total number of partitions to split features into (default: 1)",
  )
  parser.add_argument(
      "--partition_index",
      type=int,
      default=0,
      help="Index of the partition this job should process (0-based)",
  )
  parser.add_argument(
      "--partition_seed",
      type=int,
      default=42,
      help="Random seed for partitioning features (default: 42)",
  )
  args = parser.parse_args()
  run(args)
