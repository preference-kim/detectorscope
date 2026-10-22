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

"""SpecDetect++ detector wrapper.

Replicates the detection logic of SpecDetect++ from the official repo
(https://github.com/luohaitong/SpecDetect, AAAI 2026),
`py_scripts/baselines/specdetect_doubleplus.py`
(`get_sampling_discrepancy_frequency`): the negative total DFT energy of the
mean-centered token log-likelihood sequence, standardized against sequences
sampled from a reference model (Fast-DetectGPT-style). The paper's
"SpecDetect" rows were produced with this class. `_get_freq_features`
reproduces `specdetect_doubleplus.get_freq_features`; the upstream repository
declares no licence.

Known deviation from the released code (which the authors flag as an "initial
version"): the released SpecDetect++ ends with `discrepancy.cpu().item()` on a
numpy array, which crashes (numpy has no `.cpu()`). This wrapper uses
`.item()`.

Score direction: higher = more likely AI-generated (same as Fast-DetectGPT).
The score is a standardized discrepancy, not a probability.
"""

import numpy as np
import torch
from scipy.fft import fft
from transformers import AutoModelForCausalLM, AutoTokenizer

from detectorscope.model.detector import Detector

# Local-path names from their code mapped to HF hub ids (per the URLs in
# their model_fullnames comments).
MODEL_FULLNAMES = {
    "gptj_6b": "EleutherAI/gpt-j-6b",
    "gptneo_2.7b": "EleutherAI/gpt-neo-2.7B",
    "gpt2_xl": "openai-community/gpt2-xl",
    "opt_2.7b": "facebook/opt-2.7b",
    "bloom_7b": "bigscience/bloom-7b1",
    "falcon_7b": "tiiuae/falcon-7b",
    "gemma_7b": "google/gemma-7b",
    "llama1_13b": "huggyllama/llama-13b",
    "llama2_13b": "TheBloke/Llama-2-13B-fp16",
    "llama3_8b": "meta-llama/Meta-Llama-3-8B",
    "opt_13b": "facebook/opt-13b",
    "phi2": "microsoft/phi-2",
    "mgpt": "ai-forever/mGPT",
    "qwen1.5_7b": "Qwen/Qwen1.5-7B",
    "yi1.5_6b": "01-ai/Yi-1.5-6B",
}
FLOAT16_MODELS = [
    "gptj_6b",
    "llama1_13b",
    "llama2_13b",
    "llama3_8b",
    "falcon_7b",
    "bloom_7b",
    "opt_13b",
    "gemma_7b",
    "qwen1.5_7b",
    "yi1.5_6b",
]


def _load_model(model_name):
  """Mirrors their load_model(), with HF hub ids instead of local paths."""
  model_path = MODEL_FULLNAMES.get(model_name, model_name)
  model_kwargs = {}
  if model_name in FLOAT16_MODELS:
    model_kwargs.update(dict(dtype=torch.float16))
  if "gptj" in model_name:
    model_kwargs.update(dict(revision="float16"))
  model = AutoModelForCausalLM.from_pretrained(
      model_path, **model_kwargs, device_map="auto"
  )
  model.eval()
  return model


def _load_tokenizer(model_name):
  """Mirrors their load_tokenizer()."""
  model_fullname = MODEL_FULLNAMES.get(model_name, model_name)
  optional_tok_kwargs = {}
  if "opt-" in model_fullname:
    optional_tok_kwargs["fast"] = False
  optional_tok_kwargs["padding_side"] = "right"
  base_tokenizer = AutoTokenizer.from_pretrained(
      model_fullname, **optional_tok_kwargs
  )
  if base_tokenizer.pad_token_id is None:
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    if "13b" in model_fullname:
      base_tokenizer.pad_token_id = 0
  return base_tokenizer


def _max_length(tokenizer, model):
  """Longest input the model can handle (position-embedding limit)."""
  n = getattr(model.config, "max_position_embeddings", None)
  if n is None:  # e.g. ALiBi models with no hard position limit
    n = tokenizer.model_max_length
  return min(n, tokenizer.model_max_length)


def _get_freq_features(log_likelihood):
  """Copied verbatim from specdetect_doubleplus.get_freq_features.

  Args:
    log_likelihood: Array of shape (1, L, k): L token log-likelihoods for each
      of k sequences.

  Returns:
    Array of shape (k,): the negative total DFT energy of each mean-centered
    sequence.
  """
  log_likelihood = np.swapaxes(log_likelihood, 0, 2)
  log_likelihood = np.squeeze(log_likelihood, axis=-1)

  log_likelihood = log_likelihood - np.mean(
      log_likelihood, axis=1, keepdims=True
  )

  N = log_likelihood.shape[1]

  # fft analysis
  fft_y_orig = fft(log_likelihood)

  power_half_y = (((np.abs(fft_y_orig)) / N) ** 2)[:, : (int(N / 2))]

  energy_total = np.sum(power_half_y, axis=1)

  return -energy_total


class SpecDetectPlusPlusDetector(Detector):
  """SpecDetect++: spectral energy standardized against sampled alternatives.

  White-box setting: reference_model_name == scoring_model_name == source model.
  Black-box setting: both set to the proxy model ("gptj_6b" in their scripts).
  ``n_samples`` alternative sequences are drawn per text from the reference
  model (100 as in their scripts); ``seed`` seeds the global RNGs once at
  construction, so a text's score depends on the seed and on the order texts
  are scored. Models are loaded with ``device_map="auto"`` and inputs are
  truncated to the shorter of the two models' position limits.
  """

  def __init__(
      self,
      scoring_model_name: str = "gptj_6b",
      reference_model_name: str = "gptj_6b",
      n_samples: int = 100,
      device: str = "cuda",
      seed: int = 0,
  ):
    self.device = device
    self.n_samples = n_samples
    self.scoring_model_name = scoring_model_name
    self.reference_model_name = reference_model_name

    self.scoring_tokenizer = _load_tokenizer(scoring_model_name)
    self.scoring_model = _load_model(scoring_model_name)
    self.max_length = _max_length(self.scoring_tokenizer, self.scoring_model)
    if reference_model_name != scoring_model_name:
      self.reference_tokenizer = _load_tokenizer(reference_model_name)
      self.reference_model = _load_model(reference_model_name)
      self.max_length = min(
          self.max_length,
          _max_length(self.reference_tokenizer, self.reference_model),
      )
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

  def _get_samples(self, logits, labels):
    """Their get_samples()."""
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    nsamples = self.n_samples
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples

  @staticmethod
  def _get_likelihood(logits, labels):
    """Their get_likelihood()."""
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood

  def _get_sampling_discrepancy_frequency(
      self, logits_ref, logits_score, labels
  ):
    """Their get_sampling_discrepancy_frequency().

    The final `.cpu().item()` (invalid on numpy) is corrected to `.item()`.
    """
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
      vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
      logits_ref = logits_ref[:, :, :vocab_size]
      logits_score = logits_score[:, :, :vocab_size]

    samples = self._get_samples(logits_ref, labels)
    log_likelihood_x = self._get_likelihood(logits_score, labels)
    log_likelihood_x_tilde = self._get_likelihood(logits_score, samples)

    lastde_x = _get_freq_features(log_likelihood_x.cpu().numpy())
    sampled_lastde = _get_freq_features(log_likelihood_x_tilde.cpu().numpy())

    miu_tilde = sampled_lastde.mean(axis=0)
    sigma_tilde = sampled_lastde.std(axis=0)
    discrepancy = (lastde_x - miu_tilde) / sigma_tilde

    return discrepancy.item()

  def predict(self, texts: list[str]) -> torch.Tensor:
    """Runs a list of texts through SpecDetect++, one text at a time.

    Args:
      texts: Input texts to run through the detector.

    Returns:
      Standardized discrepancies of shape [n] (higher = more likely AI; 0.0
      for texts of fewer than three tokens).
    """
    preds = []
    for text in texts:
      tokenized = self.scoring_tokenizer(
          text,
          return_tensors="pt",
          padding=True,
          return_token_type_ids=False,
          truncation=True,
          max_length=self.max_length,
      ).to(self.device)
      labels = tokenized.input_ids[:, 1:]
      # Degenerate input: <3 tokens (empty/whitespace/zero-width text);
      # the criterion is undefined, so score neutrally.
      if labels.shape[1] <= 1:
        preds.append(0.0)
        continue
      with torch.no_grad():
        logits_score = self.scoring_model(**tokenized).logits[:, :-1]
        if self.reference_model_name == self.scoring_model_name:
          logits_ref = logits_score
        else:
          tokenized = self.reference_tokenizer(
              text,
              return_tensors="pt",
              padding=True,
              return_token_type_ids=False,
              truncation=True,
              max_length=self.max_length,
          ).to(self.device)
          assert torch.all(
              tokenized.input_ids[:, 1:] == labels
          ), "Tokenizer is mismatch."
          logits_ref = self.reference_model(**tokenized).logits[:, :-1]
        crit = self._get_sampling_discrepancy_frequency(
            logits_ref, logits_score, labels
        )
      preds.append(crit)
    return torch.tensor(preds)
