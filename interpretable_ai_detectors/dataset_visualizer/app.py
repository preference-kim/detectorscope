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

"""Flask app for visualizing dataset activations."""

import argparse
import os

import datasets
import flask
from interpretable_ai_detectors.model.activation_cache import MaxActivationCache
from interpretable_ai_detectors.model.sae_extractor import SAEExtractor
from interpretable_ai_detectors.utils import get_explanations_for_sae_feature
import torch


# TODO(liamdugan): change this to be dynamically calculated from dataset
SAE_WIDTH = 16384


def create_app(
    dataset: datasets.Dataset,
    extractor: SAEExtractor,
    cache: MaxActivationCache,
    args: argparse.Namespace,
):
  """Creates the Flask app using the pre-loaded data."""
  app = flask.Flask(__name__)

  @app.route("/")
  def index():
    return flask.render_template("token_highlighter.html")

  @app.route("/api/info")
  def get_info():
    return flask.jsonify(
        {
            "total_rows": len(dataset),
            "dataset_name": args.dataset,
            "text_column": args.text_col,
            "values_column": args.values_col,
            "featureDimension": SAE_WIDTH,
        }
    )

  @app.route("/api/row/<int:index>/<int:feature_index>")
  def get_row(index, feature_index):
    """Serve the activation values for a feature."""
    if not (0 <= index < len(dataset)):
      return flask.jsonify({"error": "Index out of bounds"}), 404

    try:
      # Tokenize the input text
      tokens = extractor.model.to_str_tokens(
          dataset[index][args.text_col], prepend_bos=extractor.prepend_bos
      )

      # Get the activation for the sequence
      tensor_values = torch.sparse_csr_tensor(
          **dataset[index][args.values_col]
      )

      # Convert to list of floats for frontend
      values = tensor_values.to_dense()[0, :, feature_index].tolist()

      # Error if sequence lengths don't match
      if len(tokens) != len(values):
        return (
            flask.jsonify(
                {
                    "error": f"{len(tokens)} toks != {len(values)} vals"
                }
            ),
            500,
        )

      return flask.jsonify({"index": index, "tokens": tokens, "values": values})
    except TypeError as e:
      return flask.jsonify({"error": f"Failed to get row: {str(e)}"}), 500

  @app.route("/api/feature_explanation/<int:feature_index>")
  def feature_explanation(feature_index):
    """Serve the explanation text for a feature."""
    try:
      return flask.jsonify(
          {"explanation": get_explanations_for_sae_feature(feature_index)}
      )
    except TypeError as e:
      return (
          flask.jsonify({"error": f"Failed to get explanation: {str(e)}"}),
          500,
      )

  @app.route("/api/active_rows/<int:feature_index>")
  def active_rows(feature_index):
    """Serve the list of active rows for a feature."""
    try:
      return flask.jsonify(
          {"active_rows": cache.get_active_docs(feature_index)}
      )
    except TypeError as e:
      return (
          flask.jsonify({"error": f"Failed to get active rows: {str(e)}"}),
          500,
      )

  return app


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Annotation Interface")
  parser.add_argument(
      "-d",
      "--dataset",
      type=str,
      required=True,
      help="Path to the dataset",
  )
  parser.add_argument(
      "-t",
      "--text_col",
      type=str,
      default="response",
      help="The name of the column containing the text/tokens.",
  )
  parser.add_argument(
      "-v",
      "--values_col",
      type=str,
      default="hook_sae_acts_post",
      help="The name of the column containing the sae activations.",
  )
  parsed_args = parser.parse_args()

  ds = datasets.load_from_disk(parsed_args.dataset)

  sae_extractor = SAEExtractor()
  activation_cache = MaxActivationCache(
      os.path.join(parsed_args.dataset, "max_activations.pth")
  )

  web_app = create_app(
      dataset=ds,
      extractor=sae_extractor,
      cache=activation_cache,
      args=parsed_args,
  )

  web_app.run(host="0.0.0.0", port=5003, debug=False)
