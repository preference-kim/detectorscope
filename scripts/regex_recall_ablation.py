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

"""SAE feature recall evaluator: score auto-interp descriptions with regexes.

Checks whether SAE feature descriptions have high recall by converting them to
regular expressions and measuring how well those regexes agree with feature
activations across the dataset. Also computes log-odds ratios (LOR) for each
candidate regex against the AI-detection task, so a regex's LOR can be compared
with the precomputed LOR of the feature it imitates.

Pipeline per feature:
  1. Screen: ask Gemini whether the feature is amenable to a regex.
  2. Generate: ask Gemini to produce `--num_candidates` Python regex patterns.
  3. Apply: run each regex on every document in the dataset.
  4. Score: compute confusion matrix, precision, recall, F1, and LOR for each
     candidate. Select the best candidate by F1 against the feature.
  5. Write: append a result record to the output JSONL file.

If `--refine_with_misclassified_examples` is set, candidates are generated
iteratively: each subsequent regex is conditioned on the misclassified examples
of the previous one (or of the best so far; see `--refine_from_last`).

Inputs are read from `--data_path`: the HF dataset saved with
`load_from_disk`, `layer<N>_max_activations.pth` (MaxActivationCache),
`layer<N>_<desc_model>_feature_results.jsonl` (auto-interp descriptions) and
optionally `layer<N>_log_odds_ratios.csv`. The output JSONL is appended to,
so reruns accumulate records.

Usage:
    python scripts/regex_recall_ablation.py \\
        --data_path /path/to/dataset \\
        --layer 7 \\
        --output results.jsonl \\
        --num_candidates 5 \\
        --model_column model \\
        --text_column text

    # With iterative refinement:
    python scripts/regex_recall_ablation.py \\
        --data_path /path/to/dataset \\
        --layer 7 \\
        --num_candidates 5 \\
        --refine_with_misclassified_examples \\
        --num_refinement_examples 10

    # With activation examples in the prompt:
    python scripts/regex_recall_ablation.py \\
        --data_path /path/to/dataset \\
        --layer 7 \\
        --include_activation_examples \\
        --num_activation_examples 5 \\
        --extractor_path /path/to/hf/model
"""

import argparse
import glob
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import datasets as hf_datasets
import pandas as pd
import torch
from google import genai
from tqdm import tqdm

from detectorscope.data.cache import MaxActivationCache
from detectorscope.data.cache import SAE_COLUMN_INFIX
from detectorscope.data.cache import sae_hook
from detectorscope.model.gemini import GeminiGenerator

from detectorscope.autointerpret import HIGHLIGHTS, HIGHLIGHT_THRESHOLD
from detectorscope.model.extractor import HFExtractor

# ─── Prompt templates ─────────────────────────────────────────────────────────
# {keywords_line} is "Keywords: ..." or "" — populated by _keywords_line().
# The << >> markers in the examples are HIGHLIGHTS["angled"] from autointerpret.

_SCREEN_PROMPT = """\
You are evaluating whether a sparse autoencoder (SAE) feature can be accurately \
captured by a Python regular expression applied to raw document text.

A feature IS amenable to a regex if it fires on specific, syntactically \
identifiable string patterns, for example:
  - Currency amounts: "$16,339", "£450"
  - Punctuation sequences: commas, colons, closing guillemets
  - LaTeX notation: "\\pm", "_{{", "$p_{{c,n}}$"
  - Numeric patterns: percentages, dates, measurement values with units

A feature is NOT amenable if it fires on semantic meaning, discourse structure, \
or context that cannot be inferred from the local character sequence, for example:
  - "academic peer-review feedback"
  - "legal proceeding terminology"
  - "tokens that follow a topic shift"

Feature description: {description}
{keywords_line}
Respond with a JSON object and nothing else:
{{"amenable": true, "reason": "brief explanation"}}
or
{{"amenable": false, "reason": "brief explanation"}}"""

_REGEX_PROMPT = """\
Generate a Python regular expression pattern that identifies documents containing \
the following SAE (sparse autoencoder) feature.

Feature description: {description}
{keywords_line}
{activation_section}
Requirements:
  - Return the raw regex pattern wrapped in <pattern> tags, e.g. <pattern>\\$[\\d,]+</pattern>
  - No Python code, no re.compile(), no surrounding quotes or markdown
  - The pattern will be used with re.search() on the full document text (re.IGNORECASE)
  - Aim for high F1: balance precision and recall — avoid patterns so broad they match \
unrelated documents, and avoid patterns so narrow they miss many true positives
  - Avoid catastrophic backtracking (no nested quantifiers over open-ended classes)
  - << >> markers in the examples show the token spans where the feature fired most strongly \
they are annotation only and do not appear in real text; do not match them

<pattern>YOUR_REGEX_HERE</pattern>"""

_REGEX_PROMPT_MULTI = """\
Generate {n} distinct Python regular expression patterns that each identify \
documents containing the following SAE (sparse autoencoder) feature. Make each \
pattern meaningfully different from the others — vary the syntactic approach, \
level of specificity, anchoring, or character classes used.

Feature description: {description}
{keywords_line}
{activation_section}
Requirements:
  - Wrap each pattern in its own <pattern> tags, e.g. <pattern>\\$[\\d,]+</pattern>
  - Provide exactly {n} patterns in sequence
  - No Python code, no re.compile(), no surrounding quotes or markdown
  - Each pattern will be used with re.search() on the full document text (re.IGNORECASE)
  - Aim for high F1: balance precision and recall — avoid patterns so broad they match \
unrelated documents, and avoid patterns so narrow they miss many true positives
  - Avoid catastrophic backtracking (no nested quantifiers over open-ended classes)
  - << >> markers in the examples show the token spans where the feature fired most strongly \
they are annotation only and do not appear in real text; do not match them

<pattern>PATTERN_1_HERE</pattern>
<pattern>PATTERN_2_HERE</pattern>"""

_REFINE_PROMPT = """\
You previously generated a Python regular expression to identify documents \
containing the following SAE feature.

Feature description: {description}
{keywords_line}
Your previous regex: {pattern}

This regex was evaluated and had the following misclassifications:
  - {n_fp} false positives (regex matched but feature did not activate)
  - {n_fn} false negatives (feature activated but regex did not match)

Below we will list activating examples that were misclassified.
The tokens on which the feature activates most strongly will be marked with <<...>>

NOTE: these tags << and >> are not part of the actual sentence, they're just 
used to highlight the places where the neuron is most active; do not match them.

False positives — the regex matched these but the feature did NOT activate ({fp_shown} of {n_fp}):
{fp_examples}
False negatives — the feature activated on these but the regex did NOT match ({fn_shown} of {n_fn}):
{fn_examples}
{reference_section}
Please generate an improved regex that better captures this feature.
Requirements:
  - Return the raw regex pattern wrapped in <pattern> tags, e.g. <pattern>\\$[\\d,]+</pattern>
  - No Python code, no re.compile(), no surrounding quotes or markdown
  - The pattern will be used with re.search() on the full document text (re.IGNORECASE)
  - Aim for high F1: fix the dominant failure mode above without introducing new errors \
on the reference examples
  - Avoid catastrophic backtracking (no nested quantifiers over open-ended classes)

<pattern>YOUR_REGEX_HERE</pattern>"""


def _keywords_line(keywords: str | None) -> str:
  """Return the 'Keywords: ...' prompt line, or '' when keywords are absent."""
  return f"Keywords: {keywords}" if keywords else ""


# ─── Gemini generation configs ────────────────────────────────────────────────
# Screening uses temperature=0 for deterministic amenability judgements.
# Regex generation uses the same creative settings as autointerpret.py so that
# multiple candidates explore meaningfully different patterns.

_SCREENING_CONFIG = genai.types.GenerateContentConfig(
    temperature=0,
    thinking_config=genai.types.ThinkingConfig(thinking_level="low"),
)
_GENERATION_CONFIG = genai.types.GenerateContentConfig(
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    thinking_config=genai.types.ThinkingConfig(thinking_level="low"),
)


# ─── LLM helpers ──────────────────────────────────────────────────────────────


def _parse_screening(response: str) -> tuple[bool, str]:
  """Parse a screening response into (is_amenable, reason).

  Tolerates markdown code fences; if the JSON does not parse, falls back to a
  substring check and returns the first 200 characters as the reason.
  """
  text = response.strip()
  text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.MULTILINE)
  text = re.sub(r"```$", "", text.strip())
  try:
    obj = json.loads(text)
    return bool(obj.get("amenable", False)), str(obj.get("reason", ""))
  except json.JSONDecodeError:
    lower = text.lower()
    amenable = '"amenable": true' in lower or lower.startswith("true")
    return amenable, text[:200]


def _parse_regex(response: str) -> str | None:
  """Extract a single regex pattern from a <pattern>...</pattern> block.

  Falls back to a fenced code block, then to the whole response if it is
  short enough to plausibly be a bare pattern; returns None otherwise.
  """
  m = re.search(r"<pattern>(.*?)</pattern>", response, re.DOTALL)
  if m:
    return m.group(1).strip()
  m = re.search(r"```[a-z]*\n?(.*?)```", response, re.DOTALL)
  if m:
    return m.group(1).strip()
  stripped = response.strip()
  return stripped if len(stripped) < 300 else None


def _parse_all_regexes(response: str) -> list[str]:
  """Extract all <pattern>...</pattern> blocks, deduplicated in order."""
  patterns = [
      m.strip()
      for m in re.findall(r"<pattern>(.*?)</pattern>", response, re.DOTALL)
  ]
  seen, unique = set(), []
  for p in patterns:
    if p and p not in seen:
      seen.add(p)
      unique.append(p)
  return unique


def _llm_call(
    llm: GeminiGenerator,
    prompt: str,
    config: genai.types.GenerateContentConfig | None = None,
) -> str:
  """Make a single synchronous Gemini call and return the response text."""
  return (llm.generate([prompt], config) if config else llm.generate([prompt]))[
      0
  ].text


# ─── Dataset helpers ──────────────────────────────────────────────────────────


def get_text_column(dataset, text_col: str | None = None) -> str:
  """Return the text column name, trying 'text' then 'review' if unspecified.

  Raises:
    ValueError: If no text column can be found.
  """
  if text_col and text_col in dataset.column_names:
    return text_col
  for col in ("text", "review"):
    if col in dataset.column_names:
      return col
  raise ValueError(f"No text column found. Available: {dataset.column_names}")


def load_all_texts(dataset, text_col: str | None = None) -> list[str]:
  """Load every document text into a plain Python list."""
  col = get_text_column(dataset, text_col)
  return dataset[col]


def apply_regex(pattern: str, texts: list[str]) -> set[int]:
  """Return the set of document indices where `pattern` matches.

  Uses re.search with re.IGNORECASE, matching the contract stated in the
  prompts.

  Raises:
    ValueError: If `pattern` does not compile.
  """
  try:
    compiled = re.compile(pattern, re.IGNORECASE)
  except re.error as e:
    raise ValueError(f"Invalid regex '{pattern}': {e}") from e
  return {i for i, text in enumerate(texts) if compiled.search(text)}


# ─── Metrics ──────────────────────────────────────────────────────────────────


def compute_metrics(
    regex_active: set[int],
    sae_active: set[int],
    total_docs: int,
) -> dict:
  """Compute the confusion matrix and derived metrics for one regex.

  Treats the regex as ground truth and SAE activations as the prediction:
    TP: SAE fires on a doc the regex matches (correct detection).
    FP: SAE fires on a doc the regex does not match (spurious activation).
    FN: SAE does not fire on a doc the regex matches (missed activation).
    TN: neither fires.

  Returns:
    Dict with the counts, `regex_active`/`sae_active` set sizes and
    precision/recall/f1 rounded to 4 places; zero when undefined.
  """
  tp = len(sae_active & regex_active)
  fp = len(sae_active - regex_active)
  fn = len(regex_active - sae_active)
  tn = total_docs - tp - fp - fn

  precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
  recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
  f1 = (
      2 * precision * recall / (precision + recall)
      if (precision + recall) > 0
      else 0.0
  )

  return {
      "total_docs": total_docs,
      "regex_active": len(regex_active),
      "sae_active": len(sae_active),
      "tp": tp,
      "fp": fp,
      "fn": fn,
      "tn": tn,
      "precision": round(precision, 4),
      "recall": round(recall, 4),
      "f1": round(f1, 4),
  }


# ─── LOR computation ──────────────────────────────────────────────────────────


def compute_regex_lor(
    regex_active: set[int],
    human_indices: set[int],
    target_indices: set[int],
    frequency_threshold: int = 25,
) -> float:
  """Compute the normalized log-odds ratio for a regex match set.

  Uses the same add-0.5 smoothing and class-size normalization as
  detectorscope/data/process.py lor, so regex LORs are directly
  comparable with the feature LORs in layer<N>_log_odds_ratios.csv.
  Positive LOR is more predictive of human text, negative of target (AI) text.

  Args:
    regex_active: Document indices the regex matches.
    human_indices: Document indices labelled human.
    target_indices: Document indices of the target (AI) class.
    frequency_threshold: Minimum pooled match count; below it the LOR is NaN.
  """
  N_h, N_t = len(human_indices), len(target_indices)
  N = min(N_h, N_t)
  h = len(regex_active & human_indices)
  t = len(regex_active & target_indices)
  if h + t < frequency_threshold:
    return float("nan")
  d_h = h * N / N_h
  d_t = t * N / N_t
  return math.log((d_h + 0.5) / (N - d_h + 0.5)) - math.log(
      (d_t + 0.5) / (N - d_t + 0.5)
  )


def load_sae_lors(data_path: str, layer: int) -> dict[int, float]:
  """Load precomputed feature LORs from layer<N>_log_odds_ratios.csv.

  Returns:
    Mapping feature_id -> lor; empty (with a warning) if the file is absent.
  """
  csv_path = os.path.join(data_path, f"layer{layer}_log_odds_ratios.csv")
  if not os.path.exists(csv_path):
    print(f"Warning: SAE LOR file not found at {csv_path}")
    return {}
  df = pd.read_csv(csv_path)
  return dict(zip(df["feature_id"].astype(int), df["lor"]))


def get_split_indices(
    dataset,
    model_col: str,
    target_model: str | None = None,
) -> tuple[set[int], set[int]]:
  """Return (human_indices, target_indices) from the dataset's model column.

  Without `target_model`, every non-human document counts as target.
  """
  labels = dataset[model_col]
  human_indices = {i for i, m in enumerate(labels) if m == "human"}
  target_indices = (
      {i for i, m in enumerate(labels) if m == target_model}
      if target_model
      else {i for i, m in enumerate(labels) if m != "human"}
  )
  return human_indices, target_indices


# ─── Activation examples ──────────────────────────────────────────────────────

CSR_KEYS = ["crow_indices", "col_indices", "values", "size"]


def _get_hook_name(dataset, layer: int) -> str | None:
  """Return the CSR hook column prefix for the given layer, or None.

  Dataset column names follow the pattern
  `model.layers.{layer+1}_input_sae_acts_post_{csr_key}`, e.g. for layer 7:
  `model.layers.8_input_sae_acts_post_values` (the SAE reads the input of
  the next layer). Falls back to any `*_values` column that mentions the
  incremented layer number if the canonical name is absent.
  """
  hook = sae_hook(layer) + SAE_COLUMN_INFIX.rstrip("_")
  if f"{hook}_values" in dataset.column_names:
    return hook
  # Fallback: any column referencing the incremented layer number.
  for col in dataset.column_names:
    if col.endswith("_values") and f"layers.{layer + 1}" in col:
      return col[: -len("_values")]
  return None


def _token_activations(
    row: dict, hook_name: str, feature_index: int
) -> list[float]:
  """Extract one feature's per-token activations from a dataset row.

  The dataset stores each sparse CSR tensor as four columns (CSR_KEYS):
  `{hook_name}_crow_indices`, `_col_indices`, `_values` and `_size`.
  The reconstructed tensor has shape [1, seq_len, d_sae]; it is densified and
  sliced at [0, :, feature_index], matching pool_csr in evaluate.py.
  """
  crow = row[f"{hook_name}_crow_indices"]
  col = row[f"{hook_name}_col_indices"]
  val = row[f"{hook_name}_values"]
  size = row[f"{hook_name}_size"]
  if isinstance(size, torch.Tensor):
    size = size.tolist()
  # Reconstruct [1, seq_len, d_sae] and densify.
  sparse = torch.sparse_csr_tensor(crow, col, val, size)
  return sparse.to_dense()[0, :, feature_index].tolist()


def get_activation_examples(
    feature_index: int,
    cache: MaxActivationCache,
    dataset,
    text_col: str,
    layer: int,
    n_examples: int = 5,
    extractor=None,
) -> str:
  """Format the top-activating examples with highlights for the regex prompt.

  Reconstructs per-token SAE activations from the dataset's CSR columns (see
  _get_hook_name) and wraps tokens whose activation exceeds
  HIGHLIGHT_THRESHOLD * max_activation in HIGHLIGHTS["angled"] markers,
  cropped to 20 tokens either side of the peak. Falls back to the first 500
  characters of plain text if no extractor or CSR columns are available.

  Returns:
    The "Activation examples" prompt section, or "" if the feature never
    fires in the cache.
  """
  top_docs = sorted(
      cache.get_max_activations(feature_index), key=lambda x: x[1], reverse=True
  )[:n_examples]
  if not top_docs:
    return ""

  max_activation = top_docs[0][1]
  threshold = max_activation * HIGHLIGHT_THRESHOLD
  pre, post = HIGHLIGHTS["angled"]

  hook_name = _get_hook_name(dataset, layer)
  use_highlights = extractor is not None and hook_name is not None

  if use_highlights and hook_name is not None:
    # Torch format so the CSR components come back as tensors, not lists.
    ds_torch = dataset.with_format("torch")
  else:
    ds_torch = None

  examples = []
  for doc_id, _ in top_docs:
    text = dataset[doc_id][text_col]
    if use_highlights:
      try:
        tokens = extractor.get_token_strings(text)
        a_seq = _token_activations(ds_torch[doc_id], hook_name, feature_index)
        # Crop to a window around the peak activation token.
        max_i = a_seq.index(max(a_seq)) if any(a_seq) else 0
        s = 20
        tokens = tokens[max(max_i - s, 0) : max_i + s]
        a_seq = a_seq[max(max_i - s, 0) : max_i + s]
        fmt = "".join(
            f"{pre}{tok}{post}" if a >= threshold else tok
            for tok, a in zip(tokens, a_seq)
        )
        examples.append(fmt.replace(f"{post}{pre}", ""))
        continue
      except Exception:
        pass
    examples.append(text[:500])

  header = (
      f"\nActivation examples "
      f"(strongest activating tokens marked with {pre}...{post}):\n"
  )
  return (
      header
      + "\n".join(f"{i+1}. {repr(ex)}" for i, ex in enumerate(examples))
      + "\n"
  )


# ─── Refinement helpers ───────────────────────────────────────────────────────


def _regex_match_snippet(
    doc_id: int,
    pattern: str,
    texts: list[str],
    context: int = 100,
) -> str:
  """Return a snippet of a document centred on where the regex matched.

  Includes up to `context` characters on each side of the match, with an
  ellipsis where the text was cut. Falls back to the first 300 characters if
  no match is found (should not happen for true FPs, but guards against edge
  cases such as an invalid pattern).
  """
  text = texts[doc_id]
  try:
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
      start = max(0, m.start() - context)
      end = min(len(text), m.end() + context)
      snippet = text[start:end]
      if start > 0:
        snippet = "…" + snippet
      if end < len(text):
        snippet = snippet + "…"
      return snippet
  except re.error:
    pass
  return text[:300]


def _highlight_doc(
    doc_id: int,
    feature_index: int,
    hook_name: str,
    ds_torch,
    extractor,
    max_activation: float,
    text_col: str,
    context_window: int = 20,
    max_len: int = 300,
) -> str:
  """Return a highlighted snippet of a document for the refinement prompt.

  Reconstructs the per-token activation sequence from the CSR columns, crops
  `context_window` tokens either side of the peak activation token, and wraps
  tokens above HIGHLIGHT_THRESHOLD * max_activation in angled-bracket markers
  (<<token>>). Falls back to the first `max_len` characters of plain text on
  any error.
  """
  pre, post = HIGHLIGHTS["angled"]
  threshold = max_activation * HIGHLIGHT_THRESHOLD
  try:
    row = ds_torch[doc_id]
    text = row[text_col]
    if not isinstance(text, str):
      text = str(text)
    tokens = extractor.get_token_strings(text)
    a_seq = _token_activations(row, hook_name, feature_index)
    max_i = a_seq.index(max(a_seq)) if any(a_seq) else 0
    s = context_window
    tok_win = tokens[max(max_i - s, 0) : max_i + s]
    act_win = a_seq[max(max_i - s, 0) : max_i + s]
    fmt = "".join(
        f"{pre}{tok}{post}" if a >= threshold else tok
        for tok, a in zip(tok_win, act_win)
    )
    return fmt.replace(f"{post}{pre}", "")
  except Exception:
    try:
      text = ds_torch[doc_id][text_col]
      if not isinstance(text, str):
        text = str(text)
    except Exception:
      text = ""
    return text[:max_len]


def sample_classified_examples(
    regex_active: set[int],
    sae_active: set[int],
    n_texts: int,
    n_misclassified: int,
    n_reference: int = 3,
    rng: random.Random | None = None,
) -> tuple[list[int], list[int], list[int], list[int], int, int]:
  """Sample misclassified and reference example indices for refinement.

  The `n_misclassified` slots are split between FP and FN in proportion to
  their counts, so the prompt emphasises the dominant failure mode. Reference
  TP/TN examples are included to help the model avoid regressions; TNs are
  drawn lazily from a random sample of the corpus rather than materialising
  the full complement.

  Returns:
    (fp_indices, fn_indices, tp_indices, tn_indices, n_fp, n_fn), where the
    first four are lists of document indices (not text; the caller converts
    them and applies highlights) and n_fp/n_fn are the full error counts.
  """
  rng = rng or random.Random(42)
  fp_indices = list(regex_active - sae_active)
  fn_indices = list(sae_active - regex_active)
  tp_indices = list(regex_active & sae_active)
  n_fp, n_fn = len(fp_indices), len(fn_indices)
  total = n_fp + n_fn
  if total == 0:
    return [], [], [], [], 0, 0

  # Split misclassified slots between FP and FN in proportion.
  n_show_fp = min(round(n_fp / total * n_misclassified), n_fp)
  n_show_fn = min(n_misclassified - n_show_fp, n_fn)

  fp_sample = rng.sample(fp_indices, n_show_fp) if n_show_fp else []
  fn_sample = rng.sample(fn_indices, n_show_fn) if n_show_fn else []

  # Reference TP/TN; TN sampled lazily from the corpus.
  n_ref_tp = min(n_reference, len(tp_indices))
  tp_sample = rng.sample(tp_indices, n_ref_tp) if n_ref_tp else []

  active_any = regex_active | sae_active
  tn_pool = [
      i
      for i in rng.sample(range(n_texts), min(10 * n_reference, n_texts))
      if i not in active_any
  ][:n_reference]

  return (fp_sample, fn_sample, tp_sample, tn_pool, n_fp, n_fn)


def _format_refine_prompt(
    description: str,
    keywords: str | None,
    pattern: str,
    fp_texts: list[str],
    fn_texts: list[str],
    tp_texts: list[str],
    tn_texts: list[str],
    n_fp: int,
    n_fn: int,
) -> str:
  """Build the refinement prompt from the previous pattern and its errors.

  The example lists are already text (snippets or highlighted windows); n_fp
  and n_fn are the full error counts shown alongside how many are listed.
  The reference section is omitted when there are no TP/TN examples.
  """

  def _fmt(items: list[str]) -> str:
    return (
        "\n".join(f"  {i+1}. {repr(t)}" for i, t in enumerate(items))
        or "  (none)"
    )

  reference_section = ""
  if tp_texts or tn_texts:
    parts = []
    if tp_texts:
      parts.append(
          f"True positives — correctly matched by both regex and feature:\n{_fmt(tp_texts)}"
      )
    if tn_texts:
      parts.append(
          f"True negatives — correctly rejected by both regex and feature:\n{_fmt(tn_texts)}"
      )
    reference_section = (
        "For reference, here are some correctly classified examples "
        "(avoid regressions on these):\n" + "\n".join(parts) + "\n"
    )

  return _REFINE_PROMPT.format(
      description=description,
      keywords_line=_keywords_line(keywords),
      pattern=pattern,
      n_fp=n_fp,
      n_fn=n_fn,
      fp_shown=len(fp_texts),
      fn_shown=len(fn_texts),
      fp_examples=_fmt(fp_texts),
      fn_examples=_fmt(fn_texts),
      reference_section=reference_section,
  )


# ─── Candidate evaluation ─────────────────────────────────────────────────────


def _evaluate_pattern(
    pattern: str,
    texts: list[str],
    sae_active: set[int],
    total_docs: int,
    human_indices: set[int],
    target_indices: set[int],
    frequency_threshold: int,
) -> tuple[dict, set[int]]:
  """Evaluate a single regex pattern.

  Returns:
    (candidate, regex_active). An uncompilable pattern gives
    `{"pattern", "valid": False, "error"}` and an empty set; otherwise the
    candidate carries `valid: True`, `lor` (None when NaN) and the
    compute_metrics fields.
  """
  try:
    regex_active = apply_regex(pattern, texts)
  except ValueError as e:
    return {"pattern": pattern, "valid": False, "error": str(e)}, set()

  metrics = compute_metrics(regex_active, sae_active, total_docs)
  lor = compute_regex_lor(
      regex_active, human_indices, target_indices, frequency_threshold
  )
  return {
      "pattern": pattern,
      "valid": True,
      "lor": None if math.isnan(lor) else round(lor, 6),
      **metrics,
  }, regex_active


def _pick_best(candidates: list[dict]) -> dict | None:
  """Select the valid candidate with the highest F1 against the SAE feature."""
  valid = [c for c in candidates if c.get("valid", False)]
  if not valid:
    return None
  return max(valid, key=lambda c: c.get("f1", 0.0))


# ─── Core pipeline ────────────────────────────────────────────────────────────


def process_features(
    features: list[dict],
    dataset,
    cache: MaxActivationCache,
    llm: GeminiGenerator,
    output_path: Path,
    verbose: bool = False,
    texts: list[str] | None = None,
    layer: int | None = None,
    # LOR arguments
    human_indices: set[int] | None = None,
    target_indices: set[int] | None = None,
    sae_lors: dict[int, float] | None = None,
    frequency_threshold: int = 25,
    # Candidate generation
    num_candidates: int = 1,
    # Refinement
    refine: bool = False,
    num_refinement_examples: int = 10,
    num_reference_examples: int = 3,
    # Activation examples
    extractor=None,
    num_activation_examples: int = 5,
    text_col: str | None = None,
    # Screening
    use_screen_prompt: bool = True,
    # Early stopping
    f1_threshold: float | None = None,
    # Refinement source
    refine_use_best: bool = True,
):
  """Run the screen/generate/score pipeline, appending one record per feature.

  Each feature dict is an auto-interp result with `feature_index` and
  `description.terse` (optionally `description.keywords`). A record is
  written to `output_path` immediately after each feature, so an interrupted
  run keeps its partial results. Non-amenable features get a short record
  (`is_amenable: False` plus `screening_reason`); scored features add
  `best_regex` and the full `candidates` list (one _evaluate_pattern dict per
  pattern, or a `pattern: None` entry when no pattern could be parsed), and
  every record carries the precomputed `sae_lor` for comparison.

  Args:
    features: Feature description dicts to process.
    dataset: HF dataset the cache was built from (document order must match).
    cache: MaxActivationCache for the layer; defines doc counts and activity.
    llm: Gemini generator used for screening, generation and refinement.
    output_path: JSONL file to append to.
    verbose: Print per-candidate detail instead of a tqdm bar.
    texts: Pre-loaded document texts; loaded from `dataset` if None.
    layer: SAE layer, needed to locate the CSR hook columns for highlights.
    human_indices: Human document indices for regex LOR; None disables LOR.
    target_indices: Target (AI) document indices for regex LOR.
    sae_lors: Precomputed feature LORs keyed by feature index.
    frequency_threshold: Minimum pooled match count for a non-NaN LOR.
    num_candidates: Patterns per feature; also the number of refinement
      iterations when `refine` is set.
    refine: Condition each new pattern on the previous one's errors (one LLM
      call per candidate) instead of asking for all candidates in one call.
    num_refinement_examples: Misclassified examples shown per iteration.
    num_reference_examples: TP/TN reference examples shown per iteration.
    extractor: HFExtractor (tokenizer only) enabling token-level highlights.
    num_activation_examples: Top-activating examples in the prompt; 0 omits
      the section.
    text_col: Text column name; auto-detected if None.
    use_screen_prompt: Run the amenability screen; if False every feature is
      scored and `screening_reason` is "screening skipped".
    f1_threshold: In the iterative loop, stop a feature early once the best
      candidate reaches this F1.
    refine_use_best: Refine from the best-so-far candidate rather than the
      most recent valid one.
  """
  total_docs = cache.shape[0]
  n_total = len(features)
  n_amenable = 0
  n_scored = 0

  if texts is None:
    print("Loading document texts…")
    texts = load_all_texts(dataset, text_col)
    print(f"  {len(texts):,} documents loaded.")

  _text_col = get_text_column(dataset, text_col)

  # ── One-time setup for refinement highlights ───────────────────────────────
  # ds_torch and hook_name_for_refine give token-level activation highlights
  # to the FN and TP examples shown during iterative refinement.
  hook_name_for_refine: str | None = None
  ds_torch = None
  if refine and extractor is not None and layer is not None:
    hook_name_for_refine = _get_hook_name(dataset, layer)
    if hook_name_for_refine is not None:
      ds_torch = dataset.with_format("torch")
      print(
          f"  Refinement highlights enabled via hook '{hook_name_for_refine}'"
      )
    else:
      print(
          f"  Warning: no hook columns found for layer {layer} — "
          "refinement will use plain text examples."
      )

  print(f"\nProcessing {n_total} feature(s)…\n")
  rng = random.Random(42)

  _log = tqdm.write if not verbose else print
  feature_iter = (
      features
      if verbose
      else tqdm(features, desc="Features", unit="feat", dynamic_ncols=True)
  )

  for i, feat in enumerate(feature_iter, 1):
    fidx = feat["feature_index"]
    prefix = f"[{i}/{n_total}] Feature {fidx:>6}"

    # ── Resolve description ────────────────────────────────────────────────
    terse = feat["description"]["terse"]
    keywords = feat["description"].get("keywords", "")

    # ── Precomputed SAE LOR ────────────────────────────────────────────────
    sae_lor = (sae_lors or {}).get(fidx)

    if verbose:
      print(f"  description : {terse}")
      if keywords is not None:
        print(f"  keywords    : {keywords}")
    # ── Screen ────────────────────────────────────────────────────────────
    if use_screen_prompt:
      print(f"{prefix}  querying amenability…")
      screen_prompt = _SCREEN_PROMPT.format(
          description=terse, keywords_line=_keywords_line(keywords)
      )
      is_amenable, reason = _parse_screening(
          _llm_call(llm, screen_prompt, _SCREENING_CONFIG)
      )

      if verbose:
        tag = "✓ AMENABLE" if is_amenable else "✗ not amenable"
        print(f"  → {tag}: {reason}")
      else:
        tag = "amenable" if is_amenable else "not amenable"
        _log(f"{prefix}  {tag}")

      if not is_amenable:
        record = {
            "feature_index": fidx,
            "terse_description": terse,
            "is_amenable": False,
            "screening_reason": reason,
            "sae_lor": sae_lor,
        }
        if keywords is not None:
          record["keywords"] = keywords
        _append_jsonl(output_path, record)
        if verbose:
          print()
        continue
    else:
      is_amenable = True
      reason = "screening skipped"
      if verbose:
        print(f"{prefix}  screening skipped")

    n_amenable += 1

    # ── Build activation examples section (optional) ───────────────────────
    activ_section = ""
    if num_activation_examples > 0:
      activ_section = get_activation_examples(
          fidx,
          cache,
          dataset,
          _text_col,
          layer=layer,
          n_examples=num_activation_examples,
          extractor=extractor,
      )

    # ── Build base regex prompt ────────────────────────────────────────────
    regex_prompt = _REGEX_PROMPT.format(
        description=terse,
        keywords_line=_keywords_line(keywords),
        activation_section=activ_section,
    )

    sae_active = set(cache.get_active_docs(fidx))

    # ── Per-feature max activation (for refinement highlight threshold) ────
    feat_max_activ: float | None = None
    if ds_torch is not None:
      top_docs_all = sorted(
          cache.get_max_activations(fidx), key=lambda x: x[1], reverse=True
      )
      if top_docs_all:
        feat_max_activ = top_docs_all[0][1]

    # ── Generate candidates ────────────────────────────────────────────────
    if verbose:
      print(f"  generating {num_candidates} candidate regex(es)…")

    candidates = []
    best_cand: dict | None = None
    best_regex_active: set[int] = set()
    last_cand: dict | None = None
    last_regex_active: set[int] = set()

    def _record_candidate(pattern: str | None, k: int) -> tuple[dict, set[int]]:
      """Evaluate one pattern, log it, and update best-so-far and last valid."""
      nonlocal best_cand, best_regex_active, last_cand, last_regex_active
      if pattern is None:
        if verbose:
          print(f"  candidate {k+1}: ✗ no pattern extracted")
        cand = {
            "pattern": None,
            "valid": False,
            "error": "LLM returned no parseable pattern",
        }
        candidates.append(cand)
        return cand, set()

      cand, regex_active = _evaluate_pattern(
          pattern,
          texts,
          sae_active,
          total_docs,
          human_indices or set(),
          target_indices or set(),
          frequency_threshold,
      )
      candidates.append(cand)

      if verbose:
        lor_str = (
            f"LOR={cand['lor']:.3f}"
            if cand.get("lor") is not None
            else "LOR=n/a"
        )
        if cand["valid"]:
          print(
              f"  candidate {k+1}: {lor_str}  "
              f"P={cand['precision']:.3f}  R={cand['recall']:.3f}  "
              f"F1={cand['f1']:.3f}"
              f"(regex={cand['regex_active']:,}  sae={cand['sae_active']:,}  "
              f"TP={cand['tp']:,}  FP={cand['fp']:,}  FN={cand['fn']:,}  TN={cand['tn']:,}) "
              f"pattern={pattern[:60]}"
          )
        else:
          print(f"  candidate {k+1}: ✗ {cand.get('error', 'invalid')}")

      if cand.get("valid") and (
          best_cand is None or cand["f1"] > best_cand["f1"]
      ):
        best_cand = cand
        best_regex_active = regex_active

      if cand.get("valid"):
        last_cand = cand
        last_regex_active = regex_active

      return cand, regex_active

    if not refine and num_candidates > 1:
      # ── Single call: ask for all candidates at once ────────────────────
      # The model sees its own suggestions in one context window and is
      # explicitly asked to vary them, which avoids duplicate patterns.
      multi_prompt = _REGEX_PROMPT_MULTI.format(
          n=num_candidates,
          description=terse,
          keywords_line=_keywords_line(keywords),
          activation_section=activ_section,
      )
      response = _llm_call(llm, multi_prompt, _GENERATION_CONFIG)
      patterns = _parse_all_regexes(response)

      if verbose and len(patterns) < num_candidates:
        print(
            f"  warning: requested {num_candidates} patterns, "
            f"got {len(patterns)} unique"
        )

      for k, pattern in enumerate(patterns[:num_candidates]):
        _record_candidate(pattern, k)

    else:
      # ── Iterative loop: one call per candidate (used for refinement) ───
      current_prompt = regex_prompt
      for k in range(num_candidates):
        pattern = _parse_regex(
            _llm_call(llm, current_prompt, _GENERATION_CONFIG)
        )
        _record_candidate(pattern, k)

        # ── Early stopping: F1 threshold check ────────────────────────
        if (
            f1_threshold is not None
            and best_cand is not None
            and best_cand.get("f1", 0.0) >= f1_threshold
        ):
          if verbose:
            print(
                f"  early stop at iteration {k+1}: "
                f"F1={best_cand['f1']:.4f} ≥ threshold {f1_threshold}"
            )
          else:
            _log(
                f"{prefix}  early stop at iteration {k+1}: "
                f"F1={best_cand['f1']:.4f} ≥ threshold {f1_threshold}"
            )
          break

        if k < num_candidates - 1 and refine and best_cand is not None:
          # Refinement source: best-so-far or most recent valid candidate.
          # Falls back to best when no valid "last" exists yet.
          ref_cand = (
              best_cand if (refine_use_best or last_cand is None) else last_cand
          )
          ref_regex_active = (
              best_regex_active
              if (refine_use_best or last_cand is None)
              else last_regex_active
          )

          fp_idx, fn_idx, tp_idx, tn_idx, n_fp, n_fn = (
              sample_classified_examples(
                  ref_regex_active,
                  sae_active,
                  len(texts),
                  num_refinement_examples,
                  num_reference_examples,
                  rng,
              )
          )

          # Positive examples (FN, TP) get token-level activation highlights
          # when a hook and extractor are available; negative examples (FP,
          # TN) use plain text since the feature did not activate. FPs are
          # centred on the regex match instead.
          _can_highlight = (
              ds_torch is not None
              and hook_name_for_refine is not None
              and feat_max_activ is not None
              and extractor is not None
          )

          def _to_text(
              doc_id: int,
              positive: bool,
              _fidx: int = fidx,
              _max_a: float | None = feat_max_activ,
          ) -> str:
            if positive and _can_highlight:
              return _highlight_doc(
                  doc_id,
                  _fidx,
                  hook_name_for_refine,
                  ds_torch,
                  extractor,
                  _max_a,
                  _text_col,
              )
            return texts[doc_id][:300]

          # Convert indices to text strings.
          fp_texts = [
              _regex_match_snippet(i, ref_cand["pattern"], texts)
              for i in fp_idx
          ]
          fn_texts = [_to_text(i, True) for i in fn_idx]
          tp_texts = [_to_text(i, True) for i in tp_idx]
          tn_texts = [_to_text(i, False) for i in tn_idx]

          current_prompt = _format_refine_prompt(
              terse,
              keywords,
              ref_cand["pattern"],
              fp_texts,
              fn_texts,
              tp_texts,
              tn_texts,
              n_fp,
              n_fn,
          )

    # ── Select best candidate ──────────────────────────────────────────────
    best = _pick_best(candidates)

    record = {
        "feature_index": fidx,
        "terse_description": terse,
        "is_amenable": True,
        "screening_reason": reason,
        "sae_lor": sae_lor,
        "best_regex": best,
        "candidates": candidates,
    }
    if keywords is not None:
      record["keywords"] = keywords
    _append_jsonl(output_path, record)

    n_scored += 1
    if best and best.get("valid"):
      lor_str = (
          f"LOR={best['lor']:.3f}" if best.get("lor") is not None else "LOR=n/a"
      )
      summary = (
          f"{lor_str}  "
          f"P={best['precision']:.3f}  R={best['recall']:.3f}  "
          f"F1={best['f1']:.3f}  "
          f"(regex={best['regex_active']:,}  sae={best['sae_active']:,}  "
          f"TP={best['tp']:,}  FP={best['fp']:,}  FN={best['fn']:,}  TN={best['tn']:,})"
      )
    else:
      summary = "✗ no valid candidate produced"

    if verbose:
      print(f"  ✓ best: {summary}")
    else:
      _log(f"{prefix}  ✓  {summary}")

    if verbose:
      print()

  print(f"\nDone: {n_amenable}/{n_total} amenable, {n_scored} fully scored.")


def _append_jsonl(path: Path, record: dict):
  """Append one JSON record as a line to `path`."""
  with open(path, "a") as f:
    f.write(json.dumps(record) + "\n")


# ─── Data loading ─────────────────────────────────────────────────────────────


def load_descriptions(
    data_path: str,
    layer: int,
    desc_model: str = "gemini-3.1-flash-lite-preview",
) -> dict[int, dict]:
  """Load auto-interp feature descriptions indexed by feature_index.

  Reads <data_path>/layer<N>_<desc_model>_feature_results.jsonl, the output
  of detectorscope/autointerpret.py; exits if it is missing.
  """
  pattern = os.path.join(
      data_path, f"layer{layer}_{desc_model}_feature_results.jsonl"
  )
  matches = glob.glob(pattern)
  if not matches:
    sys.exit(f"Error: no description file found matching {pattern}")
  descriptions = {}
  with open(matches[0]) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      obj = json.loads(line)
      descriptions[obj["feature_index"]] = obj
  print(f"Loaded {len(descriptions):,} feature descriptions from {matches[0]}")
  return descriptions


# ─── Entry point ──────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> None:
  """Load the layer's resources and run process_features."""
  # ── Resolve output path ────────────────────────────────────────────────────
  output_path = Path(
      args.output
      or os.path.join(args.data_path, f"layer{args.layer}_recall_eval.jsonl")
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)
  print(f"Results will be written to: {output_path}")

  # ── Load resources ─────────────────────────────────────────────────────────
  print("\nLoading dataset…")
  dataset = hf_datasets.load_from_disk(args.data_path)
  print(f"  {len(dataset):,} examples")

  print("Loading max activation cache…")
  cache_path = os.path.join(
      args.data_path, f"layer{args.layer}_max_activations.pth"
  )
  cache = MaxActivationCache.from_pth(cache_path)
  print(f"  Cache shape: {cache.shape}")

  descriptions = load_descriptions(
      args.data_path, args.layer, args.desc_gen_model
  )
  sae_lors = load_sae_lors(args.data_path, args.layer)

  # ── LOR split indices ──────────────────────────────────────────────────────
  if args.model_column in dataset.column_names:
    human_indices, target_indices = get_split_indices(
        dataset, args.model_column, args.target_model
    )
    print(
        f"  {len(human_indices):,} human docs  /  "
        f"{len(target_indices):,} target docs"
    )
  else:
    print(
        f"Warning: model column '{args.model_column}' not found — "
        "LOR will not be computed for regexes."
    )
    human_indices = target_indices = None

  # ── Optional: extractor for activation examples ────────────────────────────
  extractor = None
  if args.include_activation_examples:
    if args.extractor_path:
      print(f"Loading extractor from {args.extractor_path}…")
      extractor = HFExtractor(args.extractor_path, tokenizer_only=True)
    else:
      print(
          "No --extractor_path provided — "
          "activation examples will use raw text snippets."
      )

  llm = GeminiGenerator(model_name=args.regex_gen_model)

  # ── Select features to process ─────────────────────────────────────────────
  if args.feature_index is not None:
    missing = [idx for idx in args.feature_index if idx not in descriptions]
    if missing:
      sys.exit(
          f"Error: feature index/indices not found in descriptions file: {missing}"
      )
    features = [descriptions[idx] for idx in args.feature_index]
    label = (
        f"feature {args.feature_index[0]}"
        if len(args.feature_index) == 1
        else f"{len(features)} features: {args.feature_index}"
    )
    print(f"\nProcessing {label}")
  else:
    features = list(descriptions.values())
    if args.num_examples is not None and args.num_examples < len(features):
      rng = random.Random(42)
      features = rng.sample(features, args.num_examples)
      print(
          f"\nSampled {len(features):,} / {len(descriptions):,} features (seed 42)"
      )
    else:
      print(f"\nProcessing all {len(features):,} features")

  # ── Run pipeline ──────────────────────────────────────────────────────────
  try:
    process_features(
        features=features,
        dataset=dataset,
        cache=cache,
        llm=llm,
        output_path=output_path,
        verbose=args.verbose,
        layer=args.layer,
        human_indices=human_indices,
        target_indices=target_indices,
        sae_lors=sae_lors,
        frequency_threshold=args.frequency_threshold,
        num_candidates=args.num_candidates,
        refine=args.refine_with_misclassified_examples,
        num_refinement_examples=args.num_refinement_examples,
        num_reference_examples=args.num_reference_examples,
        extractor=extractor,
        num_activation_examples=(
            args.num_activation_examples
            if args.include_activation_examples
            else 0
        ),
        text_col=args.text_column,
        use_screen_prompt=not args.no_screen_prompt,
        f1_threshold=args.f1_threshold,
        refine_use_best=not args.refine_from_last,
    )
  except KeyboardInterrupt:
    print("\n\nInterrupted — partial results saved to", output_path)
    sys.exit(0)

  # ── Print summary ─────────────────────────────────────────────────────────
  records = []
  with open(output_path) as f:
    for line in f:
      line = line.strip()
      if line:
        records.append(json.loads(line))

  print(f"\nDone. Results written to {output_path}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Evaluate SAE feature recall via LLM-generated regular expressions."
  )
  parser.add_argument(
      "--data_path",
      required=True,
      help="Path to the HuggingFace dataset directory",
  )
  parser.add_argument(
      "--layer",
      required=True,
      type=int,
      help="SAE layer number (used to locate cache and description files)",
  )
  parser.add_argument(
      "--feature_index",
      default=None,
      type=int,
      nargs="+",
      help="If set, evaluate only these feature index/indices (space-separated)",
  )
  parser.add_argument(
      "-n",
      "--num_examples",
      default=None,
      type=int,
      help="Maximum number of features to evaluate, sampled randomly (seed 42)",
  )
  parser.add_argument(
      "--output",
      default=None,
      help="Output JSONL file path (default: <data_path>/layer<N>_recall_eval.jsonl)",
  )
  parser.add_argument(
      "--verbose",
      action="store_true",
      help="Print detailed per-feature progress to the console",
  )
  parser.add_argument(
      "-m",
      "--desc_gen_model",
      type=str,
      default="gemini-3.1-flash-lite-preview",
      help="LLM that generated the descriptions to load in",
  )
  parser.add_argument(
      "-r",
      "--regex_gen_model",
      type=str,
      default="gemini-3.1-flash-lite-preview",
      choices=[
          "gemini-2.5-flash-lite",
          "gemini-2.5-flash",
          "gemini-2.5-pro",
          "gemini-3-pro-preview",
          "gemini-3-flash-preview",
          "gemini-3.1-pro-preview",
          "gemini-3.1-flash-lite-preview",
      ],
      help="Model to use to generate the regex",
  )
  parser.add_argument(
      "--no_screen_prompt",
      action="store_true",
      help=(
          "Skip the amenability screening step and attempt regex generation "
          "for all features regardless of whether they are amenable to a regex."
      ),
  )

  # ── Candidate generation ───────────────────────────────────────────────────
  parser.add_argument(
      "--num_candidates",
      default=1,
      type=int,
      help="Number of candidate regexes to generate per feature (default: 1)",
  )

  # ── LOR computation ────────────────────────────────────────────────────────
  parser.add_argument(
      "--text_column",
      default=None,
      help="Name of text column in dataset (auto-detects 'text'/'review' if omitted)",
  )
  parser.add_argument(
      "--model_column",
      default="model",
      help="Name of model/label column in dataset (default: model)",
  )
  parser.add_argument(
      "--target_model",
      default=None,
      help="Target model to compute LOR against. If omitted, uses all non-human texts.",
  )
  parser.add_argument(
      "--frequency_threshold",
      default=25,
      type=int,
      help="Min pooled frequency for LOR computation; below this, LOR is NaN (default: 25)",
  )

  # ── Refinement ─────────────────────────────────────────────────────────────
  parser.add_argument(
      "--f1_threshold",
      default=None,
      type=float,
      help=(
          "F1 score above which a feature is considered sufficiently captured. "
          "When set, iterative refinement stops early for a feature as soon as "
          "any candidate reaches this F1 (e.g. 0.95). Has no effect when "
          "--refine_with_misclassified_examples is not used."
      ),
  )
  parser.add_argument(
      "--refine_from_last",
      action="store_true",
      help=(
          "When iteratively refining, base each new prompt on the most recent "
          "candidate regex rather than the best-so-far. By default the best "
          "candidate (highest F1) is used as the refinement source."
      ),
  )
  parser.add_argument(
      "--refine_with_misclassified_examples",
      action="store_true",
      help=(
          "Iteratively refine each candidate regex by showing the LLM "
          "examples it misclassified. Iterates --num_candidates times."
      ),
  )
  parser.add_argument(
      "--num_refinement_examples",
      default=10,
      type=int,
      help="Number of misclassified examples to show per refinement iteration (default: 10)",
  )
  parser.add_argument(
      "--num_reference_examples",
      default=3,
      type=int,
      help="Number of TP/TN reference examples to include in refinement prompt (default: 3)",
  )

  # ── Activation examples ────────────────────────────────────────────────────
  parser.add_argument(
      "--include_activation_examples",
      action="store_true",
      help=(
          "Include examples of strongly activating texts in the regex generation "
          "prompt. Uses autointerpret.py style highlighting when --extractor_path "
          "is provided and the activation column is present in the dataset."
      ),
  )
  parser.add_argument(
      "--num_activation_examples",
      default=5,
      type=int,
      help="Number of activation examples to include in the prompt (default: 5)",
  )
  parser.add_argument(
      "--extractor_path",
      default=None,
      help="Path to HF model for tokenization (enables token-level highlights)",
  )

  args = parser.parse_args()
  run(args)
