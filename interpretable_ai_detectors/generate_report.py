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

"""Generates a report explaining an AI detector's prediction for a given text.

This script uses a trained detector model and a Gemini generator to create a
human-readable report detailing why the detector classified a piece of text
as AI-generated or human-written, highlighting the most influential features.
"""

import argparse
import string

from interpretable_ai_detectors import utils
from interpretable_ai_detectors.model.detector import Detector
from interpretable_ai_detectors.model.gemini import GeminiGenerator
from interpretable_ai_detectors.model.sae_extractor import SAEExtractor
import rich
import torch


prompt_template = string.Template(
    "Can you generate a report for an interpretable AI text detector output"
    " for the following piece of text:\n${text}\nThe AI detector detected"
    " this to be from $model with a likelihood of ${likelihood}\nThe top 5"
    " features from its classifier that it used are as follows (in log"
    " space):\n${features}\nPlease write a two paragraph summary of why"
    " the AI detector selected this piece of text which mentions the most"
    " relevant features and how they relate to the text.\nPlease do not"
    " add your own opinion of the result and only report what the detector"
    " claims."
)


def _format_top_k_features(
    features: torch.Tensor, top_k: int = 5, ai_name: str = "GPT4"
) -> list[str]:
  """Formats top K features for use in generated report."""
  lines = []
  for _, weight, desc in utils.get_top_k_explanations(features, top_k):
    polarity = "Human" if weight < 0 else ai_name
    lines.append(f"{abs(weight):.4f} ({polarity}) -- {desc}")
  return lines


def generate_report(
    model: GeminiGenerator,
    example: str,
    importances: torch.Tensor,
    pred: torch.Tensor,
    ai_name: str = "GPT4",
) -> str:
  """Generate an interpretable detector report using the given importances."""
  pred = torch.sigmoid(pred).item()
  model_str = "a human" if pred < 0.5 else ai_name
  likelihood_str = f"{(1.0 - pred):.2%}" if pred < 0.5 else f"{pred:.2%}"
  features_str = "\n".join(_format_top_k_features(importances))

  prompt = prompt_template.substitute(
      text=example,
      model=model_str,
      likelihood=likelihood_str,
      features=features_str,
  )

  response = model.generate([prompt])[0]

  return response.text


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-d",
      "--detector_path",
      type=str,
      help="Path to the detector's .pth file",
  )
  parser.add_argument(
      "--use_sae",
      action="store_true",
      help="Use sae activations as detector input instead of residual stream",
  )
  args = parser.parse_args()

  hook_id = "hook_sae_acts_post" if args.use_sae else "hook_sae_input"

  generator = GeminiGenerator()
  extractor = SAEExtractor()
  detector = Detector(args.detector_path, use_sae_features=args.use_sae)
  detector_kwargs = {}

  if not args.use_sae:
    detector_kwargs["W_dec"] = extractor.sae.W_dec.cpu()

  while True:
    text = input("Input Text: ")

    # Get the last token activations from the extractor
    activations = extractor.get_activations(text)[hook_id][:, -1, :]

    # Get the detector prediction and feature importance scores
    prediction = detector.predict(activations)
    scores = detector.get_importance_scores(activations, **detector_kwargs)

    # If the activations are sae features, print the top k activations
    if args.use_sae:
      utils.print_top_k_sae_activations(activations)

    # Print the top 5 most important SAE features for the input
    utils.print_prediction(prediction)
    utils.print_top_k_importance_scores(scores, top_k=5)

    # Generate the final report
    print("Generating Report... ")
    report = generate_report(generator, text, scores, prediction)

    # Print out the report
    rich.print(rich.panel.Panel(report))
