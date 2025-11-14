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

"""Interactive demo for Detector Scope (written using Gemini)."""

import argparse
import html
import json

from interpretable_ai_detectors.interactive_demo import utils
from interpretable_ai_detectors.model.hf_detector import HFDetector
from interpretable_ai_detectors.model.hf_extractor import HFExtractor
from interpretable_ai_detectors.sae.models.jumprelu_sae import JumpReLUSAE
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


TITLE = "<div class='main-title'>🔬 Detector Scope 🔬</div>"

GLOBAL_STYLE_CSS = """
<style>
  /* Hide Streamlit Header/Footer */
  header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0%;
  }
  footer {
    visibility: hidden;
    height: 0%;
  }
  .stApp > header {
    display: none;
  }

  /* Global Font and Colors */
  body, .stApp {
    font-family: 'Google Sans', 'Roboto', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    color: #202124;
    background-color: #f8f9fa;
  }

  h1, h2, h3 {
    font-family: 'Google Sans', 'Roboto', sans-serif;
    font-weight: 500;
    color: #202124;
  }

  /* Header Style */
  .main-title {
    text-align: center;
    font-size: 2.5rem;
    font-weight: 400;
    margin-top: -4rem; /* Pull title up since header is gone */
    margin-bottom: 2.5rem;
    color: #3c4043;
    letter-spacing: -0.5px;
  }

  /* Card-like Containers */
  .feature-box {
    padding: 10px; /* Reduced from 16px */
    border-radius: 8px;
    margin-bottom: 8px; /* Reduced from 12px */
    background-color: #ffffff;
    box-shadow: 0 1px 3px rgba(60,64,67, 0.3), 0 4px 8px 3px rgba(60,64,67, 0.15);
    transition: box-shadow 0.2s ease-in-out;
    border-left-width: 6px !important; /* Ensure color border is visible */
  }
  .feature-box:hover {
    box-shadow: 0 2px 6px rgba(60,64,67, 0.3), 0 8px 12px 6px rgba(60,64,67, 0.15);
  }

  .score-box {
    font-size: 48px;
    font-weight: 300;
    text-align: center;
    padding: 24px;
    border-radius: 12px;
    background-color: #ffffff;
    color: #202124;
    box-shadow: 0 1px 3px rgba(60,64,67, 0.3), 0 4px 8px 3px rgba(60,64,67, 0.15);
    margin-bottom: 8px;
  }

  .highlight-text {
    line-height: 1.7;
    font-size: 16px;
    padding: 24px;
    border-radius: 12px;
    background-color: #ffffff;
    min-height: 400px;
    overflow: visible;
    box-shadow: 0 1px 3px rgba(60,64,67, 0.3), 0 4px 8px 3px rgba(60,64,67, 0.15);
    color: #3c4043;
  }

  /* Subheaders */
  .subheader {
    text-transform: uppercase;
    font-size: 0.85rem;
    font-weight: 700;
    color: #5f6368;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
  }

  /* Tooltip Styling */
  .tooltip {
    position: relative;
    display: inline;
    border-bottom: 2px solid transparent;
    cursor: pointer;
  }

  .tooltip.highlighted {
    border-radius: 4px;
    padding: 2px 3px;
    margin: 0 -1px;
    border-bottom: none;
    font-weight: 500;
  }

  .tooltip .tooltiptext {
    visibility: hidden;
    width: 260px;
    background-color: #202124;
    color: #e8eaed;
    text-align: left;
    border-radius: 8px;
    padding: 16px;
    font-size: 13px;
    line-height: 1.5;
    font-family: 'Google Sans', 'Roboto', sans-serif;
    white-space: normal;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);

    position: absolute;
    z-index: 1000;
    bottom: 135%;
    left: 50%;
    margin-left: -130px; /* Center the tooltip */

    opacity: 0;
    transition: opacity 0.2s;
  }

  .tooltip .tooltiptext::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -6px;
    border-width: 6px;
    border-style: solid;
    border-color: #202124 transparent transparent transparent;
  }

  .tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
  }
</style>
"""

# Runs only on initial start
if "initialized" not in st.session_state:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-d",
      "--gpt4o_detector_path",
      type=str,
      help="Path to the detector's .pth file",
  )
  parser.add_argument(
      "-ld",
      "--llama_detector_path",
      type=str,
      help="Path to the detector's .pth file",
  )
  parser.add_argument(
      "-s",
      "--gpt4o_sae_path",
      type=str,
      help="Path to the trained sae to use for W_dec",
  )
  parser.add_argument(
      "-ls",
      "--llama_sae_path",
      type=str,
      help="Path to the trained sae to use for W_dec",
  )
  parser.add_argument(
      "-f",
      "--gpt4o_features_path",
      type=str,
      help="Path to the csv containing feature explanations",
  )
  parser.add_argument(
      "-lf",
      "--llama_features_path",
      type=str,
      help="Path to the csv containing feature explanations",
  )
  args = parser.parse_args()

  # Load the detectors, SAEs, and feature descriptions for each model
  for model in ["gpt4o", "llama"]:
    st.session_state[f"{model}_extractor"] = HFExtractor(
        model_path=getattr(args, f"{model}_detector_path"),
        default_hooks=["norm"]
    )
    st.session_state[f"{model}_detector"] = HFDetector(
        getattr(args, f"{model}_detector_path")
    )
    st.session_state[f"{model}_sae"] = JumpReLUSAE.from_state_dict(
        getattr(args, f"{model}_sae_path")
    )
    st.session_state[f"{model}_feature_df"] = pd.read_csv(
        getattr(args, f"{model}_features_path")
    )

  st.session_state["initialized"] = True

st.set_page_config(layout="wide", page_title="Detector Scope")

st.markdown(GLOBAL_STYLE_CSS, unsafe_allow_html=True)
st.markdown(TITLE, unsafe_allow_html=True)

# --- CONTROLS (in sidebar) ---
st.sidebar.header("Analysis Controls")

AUTOINTERP_THRESHOLD = st.sidebar.slider(
    "Auto-Interpretation Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.05,
    help="Minimum auto-interpretation score for a feature to be visualized.",
)

MAX_PERCENT_HIGHLIGHTED = st.sidebar.slider(
    "Max % Tokens Highlighted",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.05,
    help="Maximum number of active tokens for a feature to be visualized.",
)

SORT_BY_LOR = st.sidebar.checkbox(
    "Sort by Log Odds Ratio",
    value=True,
    help="If checked, sort by log odds ratio.",
)

FILTER_ZERO_LOR = st.sidebar.checkbox(
    "Filter Features with Zero LOR",
    value=True,
    help="If checked, remove features that have a Log-Odds Ratio (LOR) of 0.",
)

NUM_FEATURES_TO_SHOW = st.sidebar.slider(
    "Number of Features",
    min_value=1,
    max_value=15,
    value=10,
    step=1,
    help="The maximum number of top features to display.",
)

FEATURE_COLORS = [
    "#f28b82",  # muted red
    "#fdd663",  # muted yellow
    "#81c995",  # muted green
    "#8ab4f8",  # muted blue
    "#c58af9",  # muted purple
    "#78d9c7",  # muted teal
    "#ffab70",  # muted orange
    "#f4a4c4",  # muted pink
    "#a0cfff",  # muted light blue
    "#b6a69a",  # muted brown-gray
    "#f7cc7f",  # light-orange
    "#c1e1c1",  # pastel-green
    "#f7c6d6",  # light-pink
    "#d1a384",  # light-brown
    "#bccce0",  # light-slate
]


input_text = st.text_area(
    "Input Text",
    height=150,
    placeholder="Paste text here to analyze...",
    label_visibility="collapsed",
)

if input_text:
  with st.spinner("Analyzing text..."):
    predicted_model = ""
    gpt4o_extractor = st.session_state["gpt4o_extractor"]
    gpt4o_detector = st.session_state["gpt4o_detector"]
    gpt4o_sae = st.session_state["gpt4o_sae"]
    gpt4o_df = st.session_state["gpt4o_feature_df"]

    gpt4o_tokens = utils.get_tokens(gpt4o_extractor, input_text)
    gpt4o_activations, gpt4o_score = utils.run_detector(
        input_text, gpt4o_extractor, gpt4o_detector, "norm_input"
    )

    llama_extractor = st.session_state["llama_extractor"]
    llama_detector = st.session_state["llama_detector"]
    llama_sae = st.session_state["llama_sae"]
    llama_df = st.session_state["llama_feature_df"]

    llama_tokens = utils.get_tokens(llama_extractor, input_text)
    llama_activations, llama_score = utils.run_detector(
        input_text, llama_extractor, llama_detector, "norm_input"
    )

    # If we predicted AI for either
    if llama_score > 50 or gpt4o_score > 50:
      # set variables to whichever detector got higher
      if llama_score > gpt4o_score:
        predicted_model = "llama"
        extractor = llama_extractor
        detector = llama_detector
        sae = llama_sae
        df = llama_df
        tokens = llama_tokens
        activations = llama_activations
        score = llama_score
      else:
        predicted_model = "gpt4o"
        extractor = gpt4o_extractor
        detector = gpt4o_detector
        sae = gpt4o_sae
        df = gpt4o_df
        tokens = gpt4o_tokens
        activations = gpt4o_activations
        score = gpt4o_score
    else:
      # set variables to whichever detector got higher
      if llama_score < gpt4o_score:
        predicted_model = "human"
        extractor = llama_extractor
        detector = llama_detector
        sae = llama_sae
        df = llama_df
        tokens = llama_tokens
        activations = llama_activations
        score = llama_score
      else:
        predicted_model = "human"
        extractor = gpt4o_extractor
        detector = gpt4o_detector
        sae = gpt4o_sae
        df = gpt4o_df
        tokens = gpt4o_tokens
        activations = gpt4o_activations
        score = gpt4o_score

    # Run the detector and run the sae on the features
    features, sae_features = utils.get_important_features(
        activations,
        detector,
        sae,
        df,
        FILTER_ZERO_LOR,
        AUTOINTERP_THRESHOLD,
        MAX_PERCENT_HIGHLIGHTED,
    )

    # Replace importance score by density ratio
    # if sorting by LOR
    if SORT_BY_LOR:
      for f in features:
        if f["human_density"] > f["target_density"]:
          f["importance_score"] = (f["human_density"] + 0.5) / (
              f["target_density"] + 0.5
          )
        elif f["human_density"] <= f["target_density"]:
          f["importance_score"] = (f["target_density"] + 0.5) / (
              f["human_density"] + 0.5
          )

      features = sorted(
          features, key=lambda x: abs(x["importance_score"]), reverse=True
      )

    # Filter features to the maximum number of features allowed
    features = features[:NUM_FEATURES_TO_SHOW]

    feature_activations = {}
    feature_max_activations = {}
    for i, feature in enumerate(features):
      features[i]["color"] = FEATURE_COLORS[i % len(FEATURE_COLORS)]
      feature_activations[feature["feature_index"]] = (
          utils.get_feature_highlights(
              sae_features, feature["feature_index"]
          )
      )
      feature_max_activations[feature["feature_index"]] = max(
          feature_activations[feature["feature_index"]]
      )

  col_highlights, col_right = st.columns([3, 1], gap="large")

  # --- RIGHT COLUMN: Score & Features ---
  with col_right:
    st.markdown(
        '<div class="subheader">AI Probability Score</div>',
        unsafe_allow_html=True,
    )

    score_color = (
        "#137333" if score < 50 else "#e37400" if score < 80 else "#c5221f"
    )

    # --- Create a display-friendly name ---
    if predicted_model == "human":
      display_name = "Human"
    elif predicted_model == "gpt4o":
      display_name = "GPT-4o"
    elif predicted_model == "llama":
      display_name = "Llama"
    else:
      display_name = predicted_model.title()  # Fallback

    st.markdown(
        f"""
    <div class="score-box" style="padding-top: 20px; padding-bottom: 16px;">

    <div style="color: {score_color}; line-height: 1.1; margin-bottom: 8px;">
        {score:.2f}%
    </div>

    <div style="font-size: 25px; font-weight: 500; color: #3c4043; margin-top: 3px;">
        <strong>{display_name}</strong>
    </div>

    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subheader">Top Influential Features</div>',
        unsafe_allow_html=True,
    )
    for feature in features:
      f_id = feature["feature_index"]

      importance_score = (
          feature["lor"] if SORT_BY_LOR else feature["importance_score"]
      )
      description = "Log Odds Ratio" if SORT_BY_LOR else "Importance"
      human_density = feature.get("human_density", "N/A")
      target_density = feature.get("target_density", "N/A")
      correlation_val = feature.get("correlation", 0.0)

      # Add label for AI vs. Human features
      raw_lor = feature["lor"]
      if raw_lor < 0:
        class_label = "AI"
        class_color = "#d93025"  # Red
      else:
        class_label = "Human"
        class_color = "#188038"  # Green

      # Create a flex container for the two metrics
      metric_display = f"""
      <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-top: 8px; padding-top: 6px; border-top: 1px solid #f1f3f4;">
          <div style="text-align: left;">
              <div style="font-size: 16px; font-weight: 700; color: #202124; line-height: 1.2;">{importance_score:.2f}</div>
              <div style="font-size: 10px; color: #5f6368; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px;">{description}</div>
          </div>
          <div style="text-align: center;">
              <div style="font-size: 16px; font-weight: 700; color: #202124; line-height: 1.2;">{correlation_val:.2f}</div>
              <div style="font-size: 10px; color: #5f6368; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px;">AutoInterp</div>
          </div>
          <div style="text-align: right;">
              <div style="font-size: 16px; font-weight: 700; color: #202124; line-height: 1.2;">{human_density} : {target_density}</div>
              <div style="font-size: 10px; color: #5f6368; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px;">Human : AI Count</div>
          </div>
      </div>
      """

      st.markdown(
          f"""
      <div class="feature-box interactive-feature-box" style="border-left: 6px solid {feature['color']}; cursor: pointer;" data-feature-id="{f_id}">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
      <div style="font-weight: 500; color: #202124;">{feature['description'].capitalize()}</div>
      <div style="font-weight: 700; font-size: 12px; color: {class_color}; background-color: {class_color}15; padding: 2px 6px; border-radius: 4px;">
          {class_label}
      </div>
      </div>
      <div style="font-size: 12px; color: #5f6368;">{metric_display}</div>
      </div>
      """,
          unsafe_allow_html=True,
      )

  # Token Analysis
  with col_highlights:
    st.markdown(
        '<div class="subheader">Token-Level Analysis</div>',
        unsafe_allow_html=True,
    )

    highlighted_html = []
    for i, token in enumerate(tokens):
      escaped_token = html.escape(token)

      if "\n" in escaped_token:
        highlighted_html.append(escaped_token.replace("\n", "<br>"))
        continue

      if token.isspace():
        highlighted_html.append(escaped_token)
        continue

      # Find the most active feature on the token
      max_activation = 0
      winning_feature = None
      for feature in features:
        f_idx = feature["feature_index"]
        if i < len(feature_activations[f_idx]):
          activation = feature_activations[f_idx][i]
          if activation > max_activation:
            max_activation = activation
            winning_feature = feature

      # --- MODIFIED LOGIC: Build class and data-attribute strings ---
      active_classes_list = []
      data_act_attributes_list = []
      if winning_feature:
        for feature in features:
          f_idx = feature["feature_index"]
          if i < len(feature_activations[f_idx]):
            activation = feature_activations[f_idx][i]
            if activation > 0:
              active_classes_list.append(f"feat-{f_idx}")
              data_act_attributes_list.append(
                  f'data-act-{f_idx}="{activation}"'
              )

      active_classes_str = " ".join(active_classes_list)
      data_act_attributes_str = " ".join(data_act_attributes_list)

      # If a feature is active
      if winning_feature:
        lor_val = winning_feature.get("lor", 0.0)
        human_density = winning_feature.get("human_density", "N/A")
        target_density = winning_feature.get("target_density", "N/A")
        keyword_desc = winning_feature.get("keyword_description", "")
        extended_desc = winning_feature.get("extended_desc", "")
        correlation_val = winning_feature.get(
            "correlation", 0.0
        )

        keyword_html = ""
        if keyword_desc:
          keyword_html = f"""
          <div style='margin-top: 8px; font-size: 13px; line-height: 1.4; opacity: 0.9;'>
          <strong style='color: #e8eaed; opacity: 0.9;'>Keywords:</strong> {html.escape(keyword_desc)}
          </div>
          """

        extended_html = ""
        if extended_desc:
          extended_html = f"""
          <div style='margin-top: 8px; margin-bottom: 10px; font-size: 13px; line-height: 1.4; opacity: 0.9;'>
          {extended_desc}
          </div>
          """

        tooltip_content = f"""
          <strong style='color: {winning_feature['color']}; font-size: 14px;'>{winning_feature['feature_index']}: {winning_feature['description'].capitalize()}</strong><br>
          <div style='margin-top: 8px; margin-bottom: 10px; font-size: 13px; line-height: 1.4; opacity: 0.9;'>
          {winning_feature['extended_desc']}
          </div>
          <div style='margin-top: 8px; font-size: 13px; line-height: 1.4; opacity: 0.9;'>
          <strong style='color: #e8eaed; opacity: 0.9;'>Keywords:</strong> {html.escape(winning_feature['keywords_description'])}
          </div>
          <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-top: 10px; padding-top: 10px; border-top: 1px solid #444;">
          <div style="text-align: left;">
          <div style="font-size: 14px; font-weight: 700; color: #e8eaed; line-height: 1.2;">{lor_val:.2f}</div>
          <div style="font-size: 10px; color: #9aa0a6; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px;">Log Odds Ratio</div>
          </div>
          <div style="text-align: center;">
          <div style="font-size: 14px; font-weight: 700; color: #e8eaed; line-height: 1.2;">{correlation_val:.2f}</div>
          <div style="font-size: 10px; color: #9aa0a6; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px;">AutoInterp</div>
          </div>
          <div style="text-align: right;">
          <div style="font-size: 14px; font-weight: 700; color: #e8eaed; line-height: 1.2;">{human_density} : {target_density}</div>
          <div style="font-size: 10px; color: #9aa0a6; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px;">Human : AI Count</div>
          </div>
          </div>
        """

        # --- Base color is still from the winner ---
        feature_max = feature_max_activations[
            winning_feature["feature_index"]
        ]
        normalized_strength = (
            (max_activation / feature_max) if feature_max > 0 else 0
        )
        opacity_hex = f"{int(normalized_strength * 255):02x}"
        highlight_color = winning_feature["color"] + opacity_hex

        # --- MODIFIED HTML SPAN ---
        # It now includes all classes and all data-act attributes
        highlighted_html.append(
            f"""<span class="tooltip highlighted {active_classes_str}"
                data-color="{highlight_color}"
                {data_act_attributes_str}
                style="background-color: {highlight_color};">
            {escaped_token}<span class="tooltiptext">{tooltip_content}</span></span>"""
        )
      else:
        highlighted_html.append(escaped_token)

    final_html = "".join(highlighted_html)

    st.markdown(
        f"""
    <div class="highlight-text">
        {final_html}
    </div>
    """,
        unsafe_allow_html=True,
    )

  # --- JAVASCRIPT BLOCK (MODIFIED) ---

  # 1. Create a data structure for JavaScript
  # This will map feature IDs to their color and max activation
  js_features_data = {}
  for feature in features:
    f_idx = feature["feature_index"]
    js_features_data[f_idx] = {
        "color": feature["color"],
        "max_act": feature_max_activations[f_idx],
    }

  # Convert to a JSON string, handling quotes safely
  features_json = json.dumps(js_features_data)

  # 2. Create the unique key (unchanged)
  feature_key = "_".join([str(f["feature_index"]) for f in features])

  # 3. Define the new script
  script_html = f"""
  <script>
    // --- Parse the feature data from Python ---
    const featuresData = JSON.parse('{features_json}');

    // --- 1. Helper Functions (MODIFIED) ---
    function highlightFeature(featureId) {{
      // console.log('--- highlightFeature triggered for ID: ' + featureId);
      const allHighlights = window.parent.document.getElementsByClassName('highlighted');

      // Get the data for the specific feature being hovered
      const featureInfo = featuresData[featureId];
      if (!featureInfo) return; // Safety check

      const color = featureInfo.color;
      const maxAct = featureInfo.max_act;

      for (let i = 0; i < allHighlights.length; i++) {{
        const el = allHighlights[i];

        // Check if this token is associated with the hovered feature
        if (el.classList.contains('feat-' + featureId)) {{

            // Get this token's activation for THIS feature
            const activation = parseFloat(el.getAttribute('data-act-' + featureId));

            if (activation > 0 && maxAct > 0) {{
              const normalized_strength = activation / maxAct;
              // Clamp opacity between 0 and 255
              const opacity = Math.max(0, Math.min(255, Math.floor(normalized_strength * 255)));
              const opacity_hex = opacity.toString(16).padStart(2, '0');

              // Set the style to the HOVERED feature's color and strength
              el.style.backgroundColor = color + opacity_hex;
              el.style.opacity = '1';
            }} else {{
              // Active, but no data (shouldn't happen)
              el.style.backgroundColor = 'transparent';
            }}
        }} else {{
          // This token is NOT active for the hovered feature, so hide it
          el.style.backgroundColor = 'transparent';
        }}
      }}
    }}

    // --- resetHighlights is UNCHANGED ---
    function resetHighlights() {{
      // console.log('--- resetHighlights triggered');
      const allHighlights = window.parent.document.getElementsByClassName('highlighted');
      for (let i = 0; i < allHighlights.length; i++) {{
        const el = allHighlights[i];
        // Resets to the original "winner" color
        el.style.backgroundColor = el.getAttribute('data-color');
        el.style.opacity = '1';
      }}
    }}

    // --- 2. Event Listener Attachment (Unchanged) ---
    console.log('Attaching listeners for key: {feature_key}');
    const boxes = window.parent.document.getElementsByClassName('interactive-feature-box');
    // ... (rest of the attachment logic is unchanged) ...
    console.log('Found ' + boxes.length + ' interactive-feature-box elements.');
    for (let i = 0; i < boxes.length; i++) {{
      const box = boxes[i];
      const featureId = box.getAttribute('data-feature-id');
      if (featureId) {{
        // console.log('Attaching to box ' + i + ' (Feature ID: ' + featureId + ')');
        box.addEventListener('mouseover', () => {{
          highlightFeature(featureId);
        }});
        box.addEventListener('mouseout', () => {{
          resetHighlights();
        }});
      }}
    }}

  </script>
  <style>
    .highlighted {{
      transition: background-color 0.2s ease, opacity 0.2s ease;
    }}
  </style>
  """

  # 4. Call the component
  components.html(script_html, height=0)
