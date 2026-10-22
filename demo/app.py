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

"""Interactive demo for Detector Scope: a Gradio app, locally and as a Hugging Face Space.

Sidebar: a detector's Hub coordinates (repo, branch, head type, tau), offered
exactly as the released repos hold them. Main page: a
text panel that is either an editor or the token-highlighted view, and an
evidence panel (gauge + per-side feature cards) built from the per-feature
zero-ablation counterfactuals. AuditableDetector.extract/.explain and
cache.evidence do the work; registry.py loads the detectors; render.py
builds the HTML; this file is the pickers, the state and the layout.

  python demo/app.py [-c demo/config/settings.yml]
"""

try:
  import spaces  # ZeroGPU; must precede torch, a no-op anywhere else
  from spaces.zero.torch import enter_mode
except ImportError:
  spaces = None

import argparse
import functools
import os
import threading
import traceback

import gradio as gr
import pandas as pd
import torch
import yaml

from detectorscope.data import cache
from detectorscope.model.detectors.auditable import Detection
import registry
import render

N_ARC = 40  # per-feature arcs per side; the rest collapse into the "other" arc
REPOS = [  # what the demo offers without a config
    "liamdugan/detector-scope",
    "liamdugan/detector-scope-peer-review-gpt4o",
    "liamdugan/detector-scope-peer-review-llama",
]
MAIN_SETTING = {  # the setting each repo's `main` branch holds
    "liamdugan/detector-scope": "ft-sae",
    "liamdugan/detector-scope-peer-review-gpt4o": "ft-model-sae",
    "liamdugan/detector-scope-peer-review-llama": "ft-model-sae",
}
LABELS = {"logreg": "Logistic regression", "xgboost": "XGBoost"}
# Pseudo features: parts of the logit that are not any single feature.
PSEUDO = {
    "bias": (
        "Model bias (prior)",
        "The head's intercept: the logit of an empty document.",
    ),
    "resid": (
        "Feature interactions",
        "Part of the logit not attributable to any single feature (XGBoost only).",
    ),
}
PLACEHOLDER = "Paste text here, then switch to Run (Ctrl/⌘+Enter)."
THEME = gr.themes.Default(
    font=[
        "Google Sans",
        "Roboto",
        "Helvetica Neue",
        "Helvetica",
        "Arial",
        "sans-serif",
    ]
)


def load_config(path: str | None) -> dict:
  """The config file over the built-in default (the released repos)."""
  cfg = {"repos": REPOS}
  if path:
    with open(path, encoding="utf-8") as f:
      cfg |= yaml.safe_load(f) or {}
  return cfg


parser = argparse.ArgumentParser()
HERE = os.path.dirname(os.path.abspath(__file__))
with open(
    f"{HERE}/examples/prdb_gpt4o_vs_human_pairs.txt", encoding="utf-8"
) as f:
  EXAMPLES = [t.strip() for t in f.read().split("=" * 100)[2::2]][:4]
DEFAULT_CONFIG = (
    os.environ.get("DSCOPE_CONFIG") or f"{HERE}/config/settings.yml"
)
parser.add_argument(
    "-c",
    "--config",
    default=DEFAULT_CONFIG if os.path.exists(DEFAULT_CONFIG) else None,
    help="settings.yml (default: config/settings.yml beside this file if present, else the released repos from the Hub)",
)
args = parser.parse_args()
REG = registry.Registry(
    load_config(args.config), int(os.environ.get("DSCOPE_RESIDENT", "1"))
)


def _extract(repo: str, revision: str, text: str):
  """SAE activations of `text` under the branch's extractor: the GPU step."""
  return REG.detector(repo, revision).extract(text)


if spaces:  # seconds of GPU to reserve per call, by document length
  _extract = spaces.GPU(
      duration=lambda repo, revision, text: 10
      + 20 * min(len(text), 20000) / 20000
  )(_extract)
extract = functools.lru_cache(maxsize=4)(_extract)
EMULATED = threading.local()


def emulate():
  """Enters ZeroGPU's per-thread CUDA emulation in a Gradio handler thread."""
  if spaces and not getattr(EMULATED, "on", False):
    enter_mode()
    EMULATED.on = True


@functools.lru_cache(maxsize=4)
def features(repo: str, revision: str) -> pd.DataFrame:
  """The branch's feature table (descriptions + log-odds ratios)."""
  return cache.feature_frame(REG.detector(repo, revision).feature_table)


def build_panel(
    a: Detection, ev: pd.DataFrame
) -> tuple[str, pd.DataFrame, dict]:
  """Evidence panel HTML, the features that color the text, and side masses.

  Every feature is a card (columns scroll); percentages are shares of the
  whole prediction: |contribution| / (all features + bias + interactions).
  The gauge draws bias at the outer corners, up to N_ARC per-feature arcs per
  side (the rest folded into one lighter arc), and interactions at the seam.
  Interactions exist for XGBoost only; for logreg the arcs sum to the logit.

  Args:
    a: The scored document.
    ev: Its evidence table from cache.evidence (non-zero cf_delta rows).

  Returns:
    (panel, feats, mass): the panel HTML; `ev` with a `side` column, AI rows
    first; and {"ai", "hu"} -> total |evidence| on that side.
  """
  sides = {"ai": ev[ev.cf_delta > 0], "hu": ev[ev.cf_delta < 0]}
  pseudo = {"bias": a.bias, "resid": a.residual}
  side_of = lambda v: "ai" if v > 0 else "hu"
  on_side = lambda s: {
      k: v for k, v in pseudo.items() if abs(v) > 1e-6 and side_of(v) == s
  }
  mass = {
      s: sides[s].cf_delta.abs().sum()
      + sum(abs(v) for v in on_side(s).values())
      for s in sides
  }
  total = mass["ai"] + mass["hu"]

  cards, arcs = {}, {}
  for s, df in sides.items():
    cards[s] = [
        render.card(f, render.terse(f, r), r.cf_delta, total)
        for f, r in df.iterrows()
    ]
    cards[s] += [
        render.card(k, *PSEUDO[k][:1], v, total, "pseudo", PSEUDO[k][1])
        for k, v in on_side(s).items()
    ]
    feature_arcs = [render.segment(r.cf_delta, s, f) for f, r in df.iterrows()]
    rest = sum(seg["v"] for seg in feature_arcs[N_ARC:])
    arcs[s] = (
        (
            [
                render.segment(
                    pseudo["bias"], s, "bias", *PSEUDO["bias"], kind="pseudo"
                )
            ]
            if "bias" in on_side(s)
            else []
        )
        + feature_arcs[:N_ARC]
        + (
            [
                render.segment(
                    rest,
                    s,
                    "other-" + s,
                    f"{len(feature_arcs) - N_ARC} more {render.SIDES[s]} features",
                    "Smaller features, not drawn individually.",
                    kind="other",
                )
            ]
            if rest
            else []
        )
        + (
            [
                render.segment(
                    pseudo["resid"], s, "resid", *PSEUDO["resid"], kind="pseudo"
                )
            ]
            if "resid" in on_side(s)
            else []
        )
    )

  segments = arcs["ai"] + arcs["hu"][::-1]  # corners: bias; seam: interactions
  columns = "".join(
      render.column_html(s, "".join(cards[s])) for s in ("ai", "hu")
  )
  panel = render.evidence_panel(
      render.gauge_svg(segments, 100 * a.p_ai),
      mass,
      {s: len(df) for s, df in sides.items()},
      columns,
  )
  feats = pd.concat(
      [sides["ai"].assign(side="ai"), sides["hu"].assign(side="hu")]
  )
  return panel, feats, mass


def analyze(
    repo: str, revision: str, kind: str, tau: float, text: str
) -> tuple[str, str]:
  """The highlighted text and the evidence panel of `text` under one head."""
  det = REG.detector(repo, revision)
  a = det.explain(extract(repo, revision, text), kind, tau)
  ev = cache.evidence(features(repo, revision), a)
  panel, feats, totals = build_panel(a, ev)
  acts = torch.zeros(len(a.tokens), len(feats))  # [seq, N]
  highlights = {f.id: f.highlights for f in a.features}
  for k, fid in enumerate(feats.index):
    for token, act in highlights[fid]:
      acts[token, k] = act
  info = {
      str(f): render.feature_info(f, r, r.side, acts[:, k].max().item())
      for k, (f, r) in enumerate(feats.iterrows())
  }
  return render.highlight_html(a.tokens, feats, acts, totals), render.with_info(
      panel, info
  )


def setting_label(repo: str, revision: str) -> str:
  """A branch by its setting name; `main` is marked as the default."""
  if revision != "main":
    return revision
  return f"{MAIN_SETTING.get(repo, revision)} (Default)"


def select(repo, revision, kind, tau, mode, text, last):
  """Resolves the pickers against the Hub, top-down, and loads the detector.

  A picker keeps its value while the pickers above it still offer it and
  otherwise falls back to the Hub's default (`main`; the branch's config.json
  head). In Run mode with text the branch's detector is loaded here, on the
  CPU, so that `run`, a separate request, starts its GPU call with a fresh
  ZeroGPU token; in Edit mode the editor shows and the last panel stays, dimmed.
  """
  emulate()
  revisions = REG.branches(repo)
  revision = revision if revision in revisions else revisions[0]
  heads = REG.heads(repo, revision)
  d_kind, d_tau = REG.default(repo, revision)
  kinds = sorted({k for k, _ in heads})
  kind = kind if kind in kinds else d_kind if d_kind in kinds else kinds[0]
  taus = [t for k, t in heads if k == kind]
  tau = tau if tau in taus else d_tau if d_tau in taus else taus[0]
  caption = (
      f"{REG.caption(repo, revision)} · {LABELS.get(kind, kind)} · τ = {tau:g}"
  )

  run = mode == "Run" and bool(text.strip())
  panel = render.stale_panel(last) if last else render.empty_panel()
  if run:
    try:
      REG.detector(repo, revision)
    except Exception as e:  # pylint: disable=broad-exception-caught
      traceback.print_exc()
      run, panel = False, render.error_panel(f"{type(e).__name__}: {e}")
  pick = lambda choices, value: gr.update(
      choices=choices, value=value, visible=len(choices) > 1
  )
  return [
      pick([(setting_label(repo, r), r) for r in revisions], revision),
      pick([(LABELS.get(k, k), k) for k in kinds], kind),
      pick(
          [(f"{t:g} (Default)" if t == d_tau else f"{t:g}", t) for t in taus],
          tau,
      ),
      caption,
      "Run" if run else "Edit",
      gr.update(visible=not run),
      gr.update(visible=run),
      panel,
      last,
  ]


def run(repo, revision, kind, tau, mode, text, last):
  """Scores the document under the resolved head and renders both panels."""
  emulate()
  if mode != "Run" or not text.strip():
    return gr.update(), gr.update(), gr.update(), gr.update(), last
  try:
    view, panel = analyze(repo, revision, kind, tau, text)
    return (
        "Run",
        gr.update(visible=False),
        gr.update(value=view, visible=True),
        panel,
        panel,
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    traceback.print_exc()
    error = render.error_panel(f"{type(e).__name__}: {e}")
    return (
        "Edit",
        gr.update(visible=True),
        gr.update(visible=False),
        error,
        last,
    )


with gr.Blocks(
    title="Detector Scope",
    theme=THEME,
    css=render.CSS,
    js=render.PAGE_JS,
    fill_width=True,
    analytics_enabled=False,
) as demo:
  gr.HTML(render.INTRO, padding=False, elem_id="intro")
  with gr.Sidebar(width=352, open=True, elem_id="sidebar"):
    model = gr.Dropdown(REG.repos, value=REG.repos[0], label="Model")
    setting = gr.Dropdown([], label="Setting")
    with gr.Row():
      head = gr.Dropdown([], label="Head")
      tau = gr.Dropdown([], label="Interp. threshold τ")
    caption = gr.Markdown(elem_id="caption")
    mode = gr.Radio(
        ["Edit", "Run"],
        value="Edit",
        show_label=False,
        container=False,
        elem_id="mode-row",
    )
    gr.HTML(render.INTRO_LINK, padding=False)
  with gr.Row(elem_id="main-row"):
    with gr.Column(scale=0, min_width=96, elem_id="examples"):
      gr.Markdown("Examples")
      examples = [
          gr.Button(str(i + 1), size="sm") for i in range(len(EXAMPLES))
      ]
    with gr.Column(scale=50, min_width=0, elem_id="text-col"):
      text = gr.Textbox(
          show_label=False,
          container=False,
          lines=20,
          max_lines=1000,
          placeholder=PLACEHOLDER,
          elem_id="text",
      )
      view = gr.HTML(visible=False, padding=False, elem_id="panel-slot")
    with gr.Column(scale=23, min_width=305):
      evidence = gr.HTML(
          render.empty_panel(), padding=False, elem_id="evidence"
      )
  last = gr.State(None)

  inputs = [model, setting, head, tau, mode, text, last]
  outputs = [setting, head, tau, caption, mode, text, view, evidence, last]
  results = [mode, text, view, evidence, last]
  demo.load(select, inputs, outputs).then(run, inputs, results)
  for picker in (model, setting, head, tau, mode):
    picker.input(select, inputs, outputs).then(run, inputs, results)
  for i, button in enumerate(examples):
    button.click(lambda i=i: (EXAMPLES[i], "Run"), None, [text, mode]).then(
        select, inputs, outputs
    ).then(run, inputs, results)

if __name__ == "__main__":
  demo.queue(default_concurrency_limit=1, max_size=16)
  demo.launch(
      show_api=False,
      allowed_paths=[],
      enable_monitoring=False,
      ssr_mode=False,
      show_error=True,
  )
