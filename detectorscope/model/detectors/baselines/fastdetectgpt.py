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
# Adapted from https://github.com/baoguangsheng/fast-detect-gpt
# (scripts/local_infer.py, scripts/fast_detect_gpt.py, scripts/model.py), with
# local modifications: no model cache directory, a use_log_rank option, a guard
# for empty inputs, NaN on floating-point errors in the probability estimate, a
# single pooled calibration file, and this repository's code style.
#
# MIT License
#
# Copyright (c) Guangsheng Bao
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Fast-DetectGPT detector for baseline comparison.

The calibration data it reads lives next to this module in
``resources/fdgpt_calibration.json``: the upstream ``local_infer_ref/``
criterion values (the ``predictions`` key of the six ``local_infer_ref/*.json``
files), pooled into one file with the values unchanged.
"""

import json
import os
import time

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from detectorscope.model.detector import Detector

# Short model names accepted by load_model and load_tokenizer.
_MODEL_FULLNAMES = {
    "gpt2": "gpt2",
    "gpt2-xl": "gpt2-xl",
    "opt-2.7b": "facebook/opt-2.7b",
    "gpt-neo-2.7B": "EleutherAI/gpt-neo-2.7B",
    "gpt-j-6B": "EleutherAI/gpt-j-6B",
    "gpt-neox-20b": "EleutherAI/gpt-neox-20b",
    "mgpt": "sberbank-ai/mGPT",
    "pubmedgpt": "stanford-crfm/pubmedgpt",
    "mt5-xl": "google/mt5-xl",
    "llama-13b": "huggyllama/llama-13b",
    "llama2-13b": "TheBloke/Llama-2-13B-fp16",
    "bloom-7b1": "bigscience/bloom-7b1",
    "opt-13b": "facebook/opt-13b",
}
_FLOAT16_MODELS = (
    "gpt-j-6B",
    "gpt-neox-20b",
    "llama-13b",
    "llama2-13b",
    "bloom-7b1",
    "opt-13b",
)


def get_model_fullname(model_name: str) -> str:
  """Maps a short model name to its Hugging Face id; others pass through."""
  return _MODEL_FULLNAMES.get(model_name, model_name)


def load_model(model_name: str, device: str) -> transformers.PreTrainedModel:
  """Loads a causal LM, in float16 for the larger models, onto ``device``."""
  model_fullname = get_model_fullname(model_name)
  print(f"Loading model {model_fullname}...")
  model_kwargs = {}
  if model_name in _FLOAT16_MODELS:
    model_kwargs["dtype"] = torch.float16
  if "gpt-j" in model_name:
    model_kwargs["revision"] = "float16"
  model = AutoModelForCausalLM.from_pretrained(model_fullname, **model_kwargs)
  print("Moving model to GPU...", end="", flush=True)
  start = time.time()
  model.to(device)
  print(f"DONE ({time.time() - start:.2f}s)")
  return model


def load_tokenizer(
    model_name: str, for_dataset: str
) -> transformers.PreTrainedTokenizerBase:
  """Loads the tokenizer for ``model_name``.

  Pads on the left for pubmed and on the right otherwise. A missing pad token
  falls back to EOS, or to id 0 for the 13B models.
  """
  model_fullname = get_model_fullname(model_name)
  tokenizer_kwargs = {}
  if "facebook/opt-" in model_fullname:
    print("Using non-fast tokenizer for OPT")
    tokenizer_kwargs["fast"] = False
  if for_dataset == "pubmed":
    tokenizer_kwargs["padding_side"] = "left"
  else:
    tokenizer_kwargs["padding_side"] = "right"
  tokenizer = AutoTokenizer.from_pretrained(model_fullname, **tokenizer_kwargs)
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if "13b" in model_fullname:
      tokenizer.pad_token_id = 0
  return tokenizer


def _align(
    logits_ref: torch.Tensor, logits_score: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Trims both logits to their shared vocabulary and gives labels a last dim.

  Returns:
    (logits_ref, logits_score, labels), with labels of shape [1, seq, 1].
  """
  assert logits_ref.shape[0] == 1
  assert logits_score.shape[0] == 1
  assert labels.shape[0] == 1
  if logits_ref.size(-1) != logits_score.size(-1):
    vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
    logits_ref = logits_ref[:, :, :vocab_size]
    logits_score = logits_score[:, :, :vocab_size]
  if labels.ndim == logits_score.ndim - 1:
    labels = labels.unsqueeze(-1)
  return logits_ref, logits_score, labels


def get_sampling_discrepancy_analytic(
    logits_ref: torch.Tensor, logits_score: torch.Tensor, labels: torch.Tensor
) -> float:
  """Analytic sampling discrepancy of ``labels`` under the scoring model.

  The log-likelihood of the observed tokens minus its expectation when tokens
  are sampled from the reference model, divided by the standard deviation of
  that expectation.

  Args:
    logits_ref: Reference-model logits of shape [1, seq, vocab].
    logits_score: Scoring-model logits of shape [1, seq, vocab].
    labels: Observed token ids of shape [1, seq].

  Returns:
    The criterion value.
  """
  logits_ref, logits_score, labels = _align(logits_ref, logits_score, labels)
  lprobs_score = torch.log_softmax(logits_score, dim=-1)
  probs_ref = torch.softmax(logits_ref, dim=-1)
  log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
  mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
  var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(
      mean_ref
  )
  discrepancy = (
      log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)
  ) / var_ref.sum(dim=-1).sqrt()
  return discrepancy.mean().item()


def get_log_sampling_discrepancy_analytic(
    logits_ref: torch.Tensor, logits_score: torch.Tensor, labels: torch.Tensor
) -> float:
  """Log-rank variant of get_sampling_discrepancy_analytic.

  Each token's log-likelihood is replaced by the log of its rank under the
  scoring model; arguments and return value are as there.
  """
  logits_ref, logits_score, labels = _align(logits_ref, logits_score, labels)
  lranks_score = torch.log(
      logits_score.argsort(-1, descending=True).argsort(-1) + 1
  )
  probs_ref = torch.softmax(logits_ref, dim=-1)
  log_rank = lranks_score.gather(dim=-1, index=labels).squeeze(-1)
  mean_ref = (probs_ref * lranks_score).sum(dim=-1)
  var_ref = (probs_ref * torch.square(lranks_score)).sum(dim=-1) - torch.square(
      mean_ref
  )
  rank_discrepancy = (
      log_rank.sum(dim=-1) - mean_ref.sum(dim=-1)
  ) / var_ref.sum(dim=-1).sqrt()
  return rank_discrepancy.mean().item()


class ProbEstimator:
  """Converts a criterion value to P(AI) with upstream's calibration data.

  The calibration criteria come from upstream's ChatGPT and GPT-4 test runs.
  The estimate is the fraction of machine-generated samples among the
  calibration criteria nearest to the input (about the nearest 100).

  Args:
    ref_path: JSON file with "real" (human) and "samples" (machine) criterion
      lists.
    use_log_rank: If True, crit_to_prob returns the criterion unchanged, since
      the calibration data only covers the log-likelihood criterion.
  """

  def __init__(self, ref_path: str, use_log_rank: bool):
    self.use_log_rank = use_log_rank
    with open(ref_path, encoding="utf-8") as f:
      res = json.load(f)
    self.real_crits = list(res["real"])
    self.fake_crits = list(res["samples"])
    print(f"ProbEstimator: total {len(self.real_crits) * 2} samples.")

  def crit_to_prob(self, crit: float) -> float:
    """Returns P(AI) for ``crit``, or ``crit`` itself with use_log_rank."""
    if self.use_log_rank:
      return crit
    real_crits = np.array(self.real_crits)
    fake_crits = np.array(self.fake_crits)
    offset = np.sort(
        np.abs(np.array(self.real_crits + self.fake_crits) - crit)
    )[100]
    cnt_real = np.sum(
        (real_crits > crit - offset) & (real_crits < crit + offset)
    )
    cnt_fake = np.sum(
        (fake_crits > crit - offset) & (fake_crits < crit + offset)
    )
    return cnt_fake / (cnt_real + cnt_fake)


class FastDetectGPTModel:
  """Scoring and reference models plus calibration for one configuration.

  Args:
    scoring_model_name: Short name or Hugging Face id of the scoring model.
    reference_model_name: Short name or id of the reference model. If it equals
      the scoring model, the scoring logits are reused.
    dataset: Upstream dataset name; only selects the tokenizer padding side.
    device: Device both models are moved to.
    ref_path: Calibration JSON passed to ProbEstimator.
    use_log_rank: Use the log-rank criterion and return it uncalibrated.
  """

  def __init__(
      self,
      scoring_model_name: str,
      reference_model_name: str,
      dataset: str,
      device: str,
      ref_path: str,
      use_log_rank: bool,
  ):
    self.device = device
    self.scoring_model_name = scoring_model_name
    self.reference_model_name = reference_model_name

    self.scoring_tokenizer = load_tokenizer(scoring_model_name, dataset)
    self.scoring_model = load_model(scoring_model_name, device)
    self.scoring_model.eval()
    if reference_model_name != scoring_model_name:
      self.reference_tokenizer = load_tokenizer(reference_model_name, dataset)
      self.reference_model = load_model(reference_model_name, device)
      self.reference_model.eval()

    if use_log_rank:
      self.criterion_fn = get_log_sampling_discrepancy_analytic
    else:
      self.criterion_fn = get_sampling_discrepancy_analytic

    self.prob_estimator = ProbEstimator(ref_path, use_log_rank)

  def run(self, text: str) -> float:
    """Scores one text.

    Returns:
      P(AI), or the raw criterion with use_log_rank; 0.5 for texts of fewer
      than two tokens, NaN when the probability estimate hits a floating-point
      error.
    """
    tokenized = self.scoring_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_token_type_ids=False,
    ).to(self.device)
    labels = tokenized.input_ids[:, 1:]
    # Degenerate input: fewer than 2 tokens (empty/whitespace/zero-width text).
    # The discrepancy criterion is undefined (0/0), so return a neutral score.
    if labels.shape[1] == 0:
      return 0.5
    with torch.no_grad():
      logits_score = self.scoring_model(**tokenized).logits[:, :-1]
      if self.reference_model_name == self.scoring_model_name:
        logits_ref = logits_score
      else:
        tokenized = self.reference_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_token_type_ids=False,
        ).to(self.device)
        assert torch.all(
            tokenized.input_ids[:, 1:] == labels
        ), "Tokenizer mismatch."
        logits_ref = self.reference_model(**tokenized).logits[:, :-1]
      crit = self.criterion_fn(logits_ref, logits_score, labels)
    try:
      with np.errstate(invalid="raise", divide="raise"):
        prob = self.prob_estimator.crit_to_prob(crit)
    except FloatingPointError as e:
      print(
          f"FP error in crit_to_prob: {e}; crit={crit}, "
          f"n_tokens={labels.shape[1]}, text={text!r}",
          flush=True,
      )
      prob = float("nan")
    return prob


class FastDetectGPTDetector(Detector):
  """Fast-DetectGPT as a zero-shot baseline detector.

  Scores with gpt-neo-2.7B against gpt-j-6B as the reference model and converts
  the sampling-discrepancy criterion to a probability with the upstream
  calibration data in ``resources/fdgpt_calibration.json``. With
  ``use_log_rank`` the log-rank criterion is used instead and returned raw,
  since the calibration data only covers the default criterion. Both models
  are loaded onto ``cuda``.
  """

  def __init__(self, use_log_rank=False):
    parent_dir = os.path.dirname(__file__)
    ref_path = os.path.join(parent_dir, "resources", "fdgpt_calibration.json")
    self.fast_detect_gpt_instance = FastDetectGPTModel(
        scoring_model_name="gpt-neo-2.7B",
        reference_model_name="gpt-j-6B",
        dataset="xsum",
        device="cuda",
        ref_path=ref_path,
        use_log_rank=use_log_rank,
    )

  def predict(self, texts: list[str]) -> torch.Tensor:
    """Runs a list of texts through Fast-DetectGPT, one text at a time.

    Args:
      texts: Input texts to run through the detector.

    Returns:
      Predictions of shape [n] (P(AI), or the raw criterion with
      ``use_log_rank``; 0.5 for texts of fewer than two tokens, NaN when
      calibration fails).
    """
    probabilities = [self.fast_detect_gpt_instance.run(text) for text in texts]
    return torch.tensor(probabilities)
