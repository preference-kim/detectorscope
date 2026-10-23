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

"""ReMoDetect detector wrapper.

Replicates the ReMoDetect scoring logic from the official repo
(https://github.com/hyunseoklee-ai/ReMoDetect, scripts/supervised.py,
`experiment_with_reward`): the trained reward model is loaded as a
sequence-classification model and its raw logit is the detection score
(higher = more likely AI-generated; not a probability).
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from detectorscope.model.detector import Detector


class ReMoDetectDetector(Detector):
  """ReMoDetect (Lee et al., 2024) reward-model detector.

  Defaults to the released DeBERTa reward model; texts are truncated to
  ``max_length`` tokens.
  """

  def __init__(
      self,
      model_name_or_path: str = "hyunseoki/ReMoDetect-deberta",
      device: str = "cuda",
      max_length: int = 512,
  ):
    self.device = device
    self.max_length = max_length
    self.detector = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path
    ).to(device)
    self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    self.detector.eval()

  def predict(self, texts: list[str]) -> torch.Tensor:
    """Runs a list of texts through ReMoDetect, one text at a time.

    Args:
      texts: Input texts to run through the detector.

    Returns:
      Raw reward logits of shape [n].
    """
    preds = []
    for text in texts:
      tokenized = self.tokenizer(
          text,
          padding=True,
          truncation=True,
          max_length=self.max_length,
          return_tensors="pt",
      ).to(self.device)
      with torch.no_grad():
        crit = self.detector(**tokenized).logits[0].cpu().detach()
      preds.append(crit)
    return torch.cat(preds)
