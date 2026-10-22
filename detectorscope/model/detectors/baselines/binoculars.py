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
# The perplexity and entropy functions and the scoring logic are adapted from
# https://github.com/ahans30/Binoculars (binoculars/metrics.py,
# binoculars/detector.py).
#
# BSD 3-Clause License
#
# Copyright (c) 2023, Abhimanyu Hans, Avi Schwarzschild, Tom Goldstein
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Implementation of the Binoculars detector for baseline comparison."""

import numpy as np
import torch
import transformers
import torch.nn.functional as F

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from detectorscope.model.detector import Detector

torch.set_grad_enabled(False)


def perplexity(
    encoding: transformers.BatchEncoding,
    logits: torch.Tensor,
    median: bool = False,
    temperature: float = 1.0,
):
  """Per-sequence mean (or median) next-token cross-entropy of ``logits``.

  Returns:
    Array of shape [batch]
  """
  shifted_logits = logits[..., :-1, :].contiguous() / temperature
  shifted_labels = encoding.input_ids[..., 1:].contiguous()
  shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

  ce = F.cross_entropy(
      shifted_logits.transpose(1, 2), shifted_labels, reduction="none"
  )

  if median:
    ce_nan = ce.masked_fill(~shifted_attention_mask.bool(), float("nan"))
    ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)

  else:
    ppl = (ce * shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
    ppl = ppl.to("cpu").float().numpy()

  return ppl


def entropy(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    encoding: transformers.BatchEncoding,
    pad_token_id: int,
    median: bool = False,
    sample_p: bool = False,
    temperature: float = 1.0,
):
  """Cross-perplexity: cross-entropy of ``q_logits`` against ``p_logits``.

  The softmax of ``p_logits`` (observer) is the soft target for ``q_logits``
  (performer); ``sample_p`` replaces it with one sampled token per position.

  Returns:
    Array of shape [batch], averaged (or median) over non-pad tokens.
  """
  vocab_size = p_logits.shape[-1]
  total_tokens_available = q_logits.shape[-2]
  p_scores, q_scores = p_logits / temperature, q_logits / temperature

  p_proba = F.softmax(p_scores, dim=-1).view(-1, vocab_size)

  if sample_p:
    p_proba = torch.multinomial(
        p_proba.view(-1, vocab_size), replacement=True, num_samples=1
    ).view(-1)

  q_scores = q_scores.view(-1, vocab_size)

  ce = F.cross_entropy(input=q_scores, target=p_proba, reduction="none").view(
      -1, total_tokens_available
  )
  padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

  if median:
    ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
    agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
  else:
    agg_ce = (
        ((ce * padding_mask).sum(1) / padding_mask.sum(1))
        .to("cpu")
        .float()
        .numpy()
    )

  return agg_ce


class BinocularsDetector(Detector):
  """Loads and runs inference on a Binoculars detector."""

  def __init__(
      self,
      observer_name_or_path: str = "tiiuae/falcon-7b",
      performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
      use_bfloat16: bool = True,
      max_token_observed: int = 512,
  ):
    obs_tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
    perf_tokenizer = AutoTokenizer.from_pretrained(performer_name_or_path)

    if obs_tokenizer.vocab != perf_tokenizer.vocab:
      raise ValueError("Tokenizers not same for observer and performer models.")

    self.device_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
    self.device_2 = "cuda:1" if torch.cuda.device_count() > 1 else self.device_1

    self.observer_model = AutoModelForCausalLM.from_pretrained(
        observer_name_or_path,
        device_map={"": self.device_1},
        dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
    )
    self.performer_model = AutoModelForCausalLM.from_pretrained(
        performer_name_or_path,
        device_map={"": self.device_2},
        dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
    )

    self.observer_model.eval()
    self.performer_model.eval()

    self.tokenizer = obs_tokenizer
    if not self.tokenizer.pad_token:
      self.tokenizer.pad_token = self.tokenizer.eos_token

    self.max_token_observed = max_token_observed

  def _tokenize(self, batch: list) -> transformers.BatchEncoding:
    """Tokenizes a batch onto the observer's device, padding only if needed."""
    batch_size = len(batch)
    encodings = self.tokenizer(
        batch,
        return_tensors="pt",
        padding="longest" if batch_size > 1 else False,
        truncation=True,
        max_length=self.max_token_observed,
        return_token_type_ids=False,
    ).to(self.observer_model.device)
    return encodings

  @torch.inference_mode()
  def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
    """Returns (observer, performer) logits, each on its own model's device."""
    observer_logits = self.observer_model(**encodings.to(self.device_1)).logits
    performer_logits = self.performer_model(
        **encodings.to(self.device_2)
    ).logits
    if self.device_1 != "cpu":
      torch.cuda.synchronize()
    return observer_logits, performer_logits

  def _compute_score(self, input_text):
    """Raw Binoculars score ppl / x_ppl: a float for a str, a list otherwise."""
    batch = [input_text] if isinstance(input_text, str) else input_text
    encodings = self._tokenize(batch)
    observer_logits, performer_logits = self._get_logits(encodings)
    ppl = perplexity(encodings, performer_logits)
    x_ppl = entropy(
        observer_logits.to(self.device_1),
        performer_logits.to(self.device_1),
        encodings.to(self.device_1),
        self.tokenizer.pad_token_id,
    )
    binoculars_scores = ppl / x_ppl
    binoculars_scores = binoculars_scores.tolist()
    return (
        binoculars_scores[0]
        if isinstance(input_text, str)
        else binoculars_scores
    )

  def predict(self, texts: list[str]) -> torch.Tensor:
    """Runs a list of texts through Binoculars, one text at a time.

    Args:
      texts: Input texts to run through the detector.

    Returns:
      Predictions of shape [n] (``1 - score``, higher = more likely AI; 0.0 for
      texts of at most one token).
    """
    scores = []
    for text in texts:
      encodings = self._tokenize([text])
      # Perplexity needs >=1 next-token prediction; a <=1-token text
      # yields 0/0 = NaN in perplexity(). Score neutrally instead.
      if encodings.attention_mask[..., 1:].sum() == 0:
        scores.append(0.0)
        continue
      scores.append(1 - self._compute_score(text))
    return torch.Tensor(scores)
