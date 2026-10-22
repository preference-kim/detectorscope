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

"""Classes for loading and using a fine-tuned HuggingFace detector model.

This is the black-box `FT-Model` baseline: a sequence-classification
checkpoint produced by finetune.py, selected with `-dt huggingface`.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from detectorscope.model.detector import Detector


class HFDetector(Detector):
  """Loads and runs a fine-tuned sequence-classification detector.

  Label index 1 is the AI class. The tokenizer defaults to the model path.
  """

  def __init__(self, model_path: str, tokenizer_path: str = None):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.tokenizer_path = tokenizer_path if tokenizer_path else model_path
    self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
    self.model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    ).to(self.device)

  def predict(self, texts: list[str]) -> torch.Tensor:
    """Runs a list of texts through the detector as one padded batch.

    Args:
      texts: Input texts, truncated to 1024 tokens.

    Returns:
      P(AI) of shape [n].
    """
    inputs = self.tokenizer(
        texts,
        truncation=True,
        max_length=1024,
        padding=True,
        return_tensors="pt",
    ).to(self.device)

    logits = self.model(**inputs).logits.detach()
    probabilities = torch.softmax(logits, dim=-1)

    # Return likelihood of the positive class (index 1)
    return probabilities[:, 1]
