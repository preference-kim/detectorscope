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

"""HTML/CSS/JS builders for the Detector Scope demo (no Gradio dependency).

Layout: the text panel shades every token by the side (AI / human) of the
feature that fires most strongly on it; the evidence panel shows a half gauge
whose arcs are per-feature zero-ablation counterfactuals, plus one scrollable
column of features per side. Hovering a card, an arc or a token isolates that
feature everywhere and shows a tooltip.

The three views are linked by `data-f` (feature id) attributes that PAGE_JS
matches at hover time, so every builder here must emit them consistently;
the tooltip payload travels inside the evidence panel's HTML (with_info).
Panel heights are CSS variables that PAGE_JS writes on <html>, never inline
styles set from JS (see the comment in PAGE_JS.layout for why).
"""

import html
import json
import math

import pandas as pd
import torch

SIDES = {"ai": "AI", "hu": "Human"}
# Single source of the side colors for CSS, token shading and JS.
SIDE_RGB = {"ai": (192, 57, 43), "hu": (31, 138, 112)}
TITLE = "🔬 Detector Scope"

CSS = """
  :root {
    --ink: #1c2027; --ink-2: #4c5563; --ink-3: #7b8493; --line: #dfe3e9; --line-2: #eef0f3;
    --panel: #ffffff; --ai: rgb(__AI__); --ai-ink: #9b2c21; --ai-lite: #e8a59e;
    --hu: rgb(__HU__); --hu-ink: #176b57; --hu-lite: #9fd3c2;
  }
  /* chrome: no footer, no sidebar toggle, no block boxes around the widgets. Geometry: 352px sidebar with
     30px gutters; main padding 72px top (the Hub's floating header takes the top 56px), 80px right, 16px
     bottom, 40px beside the sidebar */
  footer, .sidebar .toggle-button { display: none !important; }
  body, .gradio-container { background: #f4f5f7 !important; }
  .gradio-container { max-width: 100% !important; padding: 0 !important; margin: 0 !important;
    font-family: 'Google Sans', 'Roboto', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    color: var(--ink); font-variant-numeric: tabular-nums; }
  main.app { padding: 72px 80px 16px 40px !important; }
  .wrap.sidebar-parent { padding-left: 352px !important; }
  .sidebar { width: 352px !important; background: #f0f2f6 !important; box-shadow: none !important; }
  .sidebar-content { padding: 78px 30px 24px !important; gap: 0 !important; }
  .sidebar .column { gap: 8px !important; }
  .sidebar .form, .sidebar .block, #text { background: transparent !important; border: none !important;
    border-radius: 0; padding: 0 !important; margin: 0 !important; box-shadow: none !important; }
  .sidebar .form { gap: 8px !important; }
  .sidebar span[data-testid="block-info"] { display: block; font-size: 12px; line-height: 19px; font-weight: 400;
    color: #31333f; margin-bottom: 6px; }
  .sidebar .wrap-inner { min-height: 40px; background: var(--panel); border-radius: 8px; border: none;
    box-shadow: none; font-size: 16px; color: #31333f; }
  .sidebar .row { gap: 16px; }
  #caption { font-size: 14px; color: var(--ink-3); }
  .dscope { font-size: 16px; line-height: 1.6; }
  #mode-row .wrap { display: flex; width: 100%; gap: 0; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
  #mode-row label { flex: 1; min-height: 44px; justify-content: center; font-size: 16px; font-weight: 600;
    background: #f0f2f6; color: #31333f; border: none; border-radius: 0; margin: 0; box-shadow: none; }
  #mode-row label.selected { background: rgba(255, 75, 75, .2); color: #ff4b4b; box-shadow: inset 0 0 0 1px #ff4b4b; }
  #mode-row input { position: absolute; opacity: 0; width: 0; height: 0; }
  #main-row { gap: 8px !important; }
  #examples { gap: 8px !important; }
  #examples .block { background: transparent !important; border: none !important; padding: 0 !important; }
  #examples .prose p { margin: 0; font-size: 12px; line-height: 19px; color: #31333f; text-align: center; }
  #examples button { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; font-size: 14px;
    font-weight: 600; min-width: 0; }
  .card-box { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; }
  .intro-back { position: fixed; inset: 0; z-index: 10000; background: rgba(28, 32, 39, .45); display: none;
    align-items: center; justify-content: center; }
  .intro-back.show { display: flex; }
  .intro { width: 640px; max-width: calc(100% - 48px); padding: 28px 32px 24px; font-size: 15px; line-height: 1.55;
    color: var(--ink); box-shadow: 0 24px 64px rgba(0, 0, 0, .35); }
  .intro .sub { text-align: center; color: var(--ink-3); margin: 0 0 18px; }
  .intro ol { margin: 0; padding-left: 22px; }
  .intro li { margin: 0 0 10px; padding-left: 4px; }
  .intro .ok { display: block; margin: 18px auto 0; padding: 10px 28px; border: none; border-radius: 8px;
    background: var(--ink); color: #fff; font: inherit; font-weight: 600; cursor: pointer; }
  #intro { position: absolute !important; }
  .intro-link { display: block; margin-top: 10px; font-size: 13px; color: var(--ink-3); text-decoration: underline;
    cursor: pointer; }

  /* text panel: editor and highlighted view share font + geometry; their height (and the
     evidence panel's) is measured from the viewport by PAGE_JS.layout, so nothing here is a guess */
  :root { --text-font: 'Google Sans', 'Roboto', 'Helvetica Neue', Helvetica, Arial, sans-serif; }
  .highlight-text, #text textarea {
    font-family: var(--text-font) !important; font-size: 16px !important; line-height: 1.85 !important; font-weight: 400 !important;
    letter-spacing: normal !important; color: #3c4043 !important; padding: 22px 26px !important; box-sizing: border-box;
    height: var(--text-h, 60vh) !important; min-height: 0 !important; max-height: none !important; overflow-y: auto !important; }
  #text textarea { background: #fbfbfc !important; border: 1px solid var(--line) !important; border-radius: 10px !important;
    box-shadow: none !important; resize: none !important; caret-color: var(--ink); }
  #text textarea:focus { border-color: var(--ink-3) !important; }
  #text textarea::placeholder { color: var(--ink-3); }
  .evi .error { margin-top: 16px; font-size: 13px; color: var(--ai-ink); word-break: break-word; }
  .stale .evi { opacity: .5; }
  .tok { border-radius: 3px; padding: 1px 1px; transition: background-color .15s; }
  .focused .tok:not(.on) { background: transparent !important; }
  .focused .tok.on { outline: 1.5px solid currentColor; outline-offset: 1px; }
  .focused .tok.on[data-side="ai"] { color: var(--ai-ink); }
  .focused .tok.on[data-side="hu"] { color: var(--hu-ink); }

  /* evidence panel */
  .evi { position: relative; padding: 14px 18px 18px; display: flex; flex-direction: column; box-sizing: border-box; overflow: hidden;
    height: var(--evi-h, auto); }
  .panel-title { text-align: center; font-size: 1.25rem; font-weight: 500; color: #3c4043; letter-spacing: -0.3px; margin: 0 0 4px; }
  .help { position: absolute; top: 12px; right: 12px; width: 20px; height: 20px; border-radius: 50%; border: 1.5px solid var(--ink-3);
    color: var(--ink-3); font-size: 12px; font-weight: 700; display: inline-flex; align-items: center; justify-content: center; cursor: help; }
  .help .help-tip { visibility: hidden; opacity: 0; position: absolute; right: 0; top: 26px; width: 300px; z-index: 20;
    background: var(--ink); color: #e8ebf0; padding: 12px 14px; border-radius: 8px; font-size: 12.5px; line-height: 1.45;
    font-weight: 400; box-shadow: 0 6px 18px rgba(0,0,0,.3); transition: opacity .12s; }
  .help:hover .help-tip, .help:focus .help-tip { visibility: visible; opacity: 1; }
  .gauge { display: block; width: 100%; max-width: 340px; height: auto; margin: 0 auto; overflow: visible; }
  .gauge .seg { cursor: pointer; stroke: var(--panel); stroke-width: 1.5; transition: opacity .15s, stroke .15s; }
  .gauge .seg.on { stroke: var(--ink); stroke-width: 2.5; filter: drop-shadow(0 0 5px rgba(28,32,39,.45)); }
  .focused .gauge .seg:not(.on) { opacity: .28; }
  .gauge text { font-family: inherit; fill: var(--ink-3); font-size: 11px; }
  .gauge .score { font-size: 58px; font-weight: 700; fill: var(--ink); letter-spacing: -.03em; }
  .gauge .verdict { font-size: 13px; font-weight: 600; }
  .totals { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 4px; font-size: 12px; color: var(--ink-3); }
  .totals b { display: block; font-size: 20px; font-weight: 600; }
  .totals .ai b { color: var(--ai-ink); } .totals .hu b { color: var(--hu-ink); } .totals .hu { text-align: right; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 16px; flex: 1; min-height: 0; }
  .col { display: flex; flex-direction: column; min-height: 0; }
  .col .col-title { font-size: 12px; letter-spacing: .06em; text-transform: uppercase; padding-bottom: 6px; border-bottom: 2px solid;
    margin: 0 0 6px; font-weight: 700; }
  .col.ai .col-title { color: var(--ai-ink); border-color: var(--ai); } .col.hu .col-title { color: var(--hu-ink); border-color: var(--hu); }
  .col .list { flex: 1; min-height: 0; overflow-y: auto; padding-right: 4px; scrollbar-width: thin; }
  .card { padding: 8px 6px; border-radius: 6px; cursor: pointer; transition: background .15s, opacity .15s; }
  .card:hover, .card.on { background: var(--line-2); }
  .focused .card:not(.on) { opacity: .45; }
  .card .d { font-size: 13px; line-height: 1.3; color: var(--ink); display: -webkit-box; -webkit-line-clamp: 2;
    -webkit-box-orient: vertical; overflow: hidden; }
  .card .row { display: flex; align-items: center; gap: 8px; margin-top: 5px; }
  .card .track { flex: 1; height: 6px; background: var(--line-2); border-radius: 3px; overflow: hidden; }
  .card .fill { height: 100%; border-radius: 3px; }
  .col.ai .fill { background: var(--ai); } .col.hu .fill { background: var(--hu); }
  .col.ai .card.other .fill { background: var(--ai-lite); } .col.hu .card.other .fill { background: var(--hu-lite); }
  .col.ai .card.pseudo .fill { background: repeating-linear-gradient(45deg, var(--ai) 0 2px, var(--ai-lite) 2px 5px); }
  .col.hu .card.pseudo .fill { background: repeating-linear-gradient(-45deg, var(--hu) 0 2px, var(--hu-lite) 2px 5px); }
  .card .pct { font-size: 12px; width: 34px; text-align: right; color: var(--ink-2); }
  .card .lg { font-size: 11px; color: var(--ink-3); width: 44px; text-align: right; }
  .card.other .d, .card.pseudo .d { color: var(--ink-3); font-style: italic; }

  /* tooltip (positioned by JS) */
  #dscope-tip { position: fixed; z-index: 1000; width: 300px; padding: 12px 14px; border-radius: 8px; background: var(--ink);
    color: #e8ebf0; font-size: 12.5px; line-height: 1.45; box-shadow: 0 6px 18px rgba(0,0,0,.35); pointer-events: none;
    opacity: 0; transition: opacity .12s; font-family: 'Google Sans', 'Roboto', sans-serif; }
  #dscope-tip.show { opacity: 1; }
  #dscope-tip .t { font-weight: 600; font-size: 13px; margin-bottom: 6px; }
  #dscope-tip .t.ai { color: #f39189; } #dscope-tip .t.hu { color: #6fd3b3; }
  #dscope-tip .long { opacity: .9; margin-bottom: 6px; }
  #dscope-tip .kw { color: #9aa3b0; margin-bottom: 8px; } #dscope-tip .kw b { color: #e8ebf0; }
  #dscope-tip .m { display: flex; justify-content: space-between; border-top: 1px solid rgba(255,255,255,.14); padding-top: 8px; }
  #dscope-tip .m div { text-align: center; } #dscope-tip .m div:first-child { text-align: left; } #dscope-tip .m div:last-child { text-align: right; }
  #dscope-tip .m b { display: block; font-size: 14px; font-weight: 600; }
  #dscope-tip .m span { font-size: 10px; letter-spacing: .05em; text-transform: uppercase; color: #9aa3b0; }
""".replace("__AI__", ",".join(map(str, SIDE_RGB["ai"]))).replace(
    "__HU__", ",".join(map(str, SIDE_RGB["hu"]))
)

INTRO = f"""
<div class="intro-back" id="intro-back"><div class="intro card-box">
<div class="panel-title">{TITLE}</div>
<p class="sub">An AI-text detector you can audit, feature by feature.</p>
<ol>
<li><b>Pick a detector</b> in the sidebar: a model, one of its settings and a head.</li>
<li><b>Paste a text and press Run</b> (Ctrl/⌘+Enter).</li>
<li><b>Read the verdict.</b> The gauge is the probability the text is AI-written; each arc is one feature's share, red toward AI, green toward human.</li>
<li><b>Hover</b> a feature card, an arc or a highlighted word to see what the feature means and where it fires.</li>
</ol>
<button class="ok">Got it</button>
</div></div>
"""
INTRO_LINK = '<a class="intro-link">How it works</a>'
HELP_TEXT = (
    "Each feature's value is its zero-ablation counterfactual on this text: how much "
    "the detector's logit drops when the feature is removed (positive pushes toward AI). "
    "Percentages are shares of the whole prediction: features on both sides plus the "
    "model's bias (the logit of an empty document), drawn hatched at the outer corners of "
    "the gauge. Arc length is that share; the lighter arcs fold in the smallest features. "
    "For XGBoost a hatched arc at the seam is the part of the logit that feature "
    "interactions leave unattributed; for logistic regression the arcs sum exactly to the logit."
)

# Page script: one function, run by Gradio once the page loads (Blocks(js=...)), so its
# observers and listeners live in the page's own JS realm for the life of the app. The
# per-run tooltip payload is read at event time from the JSON block that with_info()
# embeds in the evidence panel; the side colors are substituted in at import.
PAGE_JS = r"""
() => {
  if (window.__dscope) return;
  window.__dscope = true;
  const doc = document, RGB = __RGB__;
  doc.body.classList.remove('dark');  // drawn for the light theme; the Hub passes a viewer's dark theme in
  const px = v => parseFloat(v) || 0;

  // ---- layout: text box and evidence panel end at the same viewport bottom edge
  function layout() {
    const col = doc.querySelector('#text-col');
    const box = col && (col.querySelector('.highlight-text') || col.querySelector('textarea'));
    const main = doc.querySelector('main.app') || doc.querySelector('.gradio-container');
    if (!box || !main) return;
    // a Space's iframe is sized to its content, so the viewport cannot be the target there
    const page = window.self === window.top ? window.innerHeight : 900;
    const bottom = page - px(getComputedStyle(main).paddingBottom);
    // Only CSS variables on <html> are written: the panels are re-rendered from HTML on every
    // run, so inline styles set on their nodes would be lost or land on the wrong element.
    const root = doc.documentElement.style, evi = doc.querySelector('.evi');
    root.setProperty('--text-h', Math.max(240, bottom - box.getBoundingClientRect().top) + 'px');
    if (evi) root.setProperty('--evi-h', Math.max(240, bottom - evi.getBoundingClientRect().top) + 'px');
  }
  let raf = 0;
  const schedule = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(layout); };
  window.addEventListener('resize', schedule);
  new MutationObserver(schedule).observe(doc.body, { childList: true, subtree: true });
  schedule();

  // ---- hover: delegated, so it works on nodes created by later runs
  const tip = doc.createElement('div'); tip.id = 'dscope-tip'; doc.body.appendChild(tip);
  const fmt = v => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2);
  const rgba = (side, a) => `rgba(${RGB[side].join(',')},${a.toFixed(2)})`;
  let infoNode = null, infoCache = {};
  const getInfo = () => {  // the payload of the evidence panel on the page, parsed once per panel
    const node = doc.getElementById('dscope-info');
    if (node !== infoNode) { infoNode = node; infoCache = node ? JSON.parse(node.textContent) : {}; }
    return infoCache;
  };
  function focus(id) {
    const info = getInfo();
    doc.querySelectorAll('.dscope').forEach(p => p.classList.toggle('focused', !!id));
    doc.querySelectorAll('[data-f]').forEach(el => el.classList.toggle('on', !!id && el.dataset.f === id));
    doc.querySelectorAll('.tok[data-a]').forEach(el => {
      const a = id && info[id] && el.dataset.a.split(',').map(s => s.split(':')).find(kv => kv[0] === id);
      if (a) { el.classList.add('on'); el.dataset.side = info[id].side; el.style.background = rgba(info[id].side, 0.15 + 0.7 * a[1] / info[id].max); }
      else { el.dataset.side = el.dataset.side0; el.style.background = el.dataset.bg; }
    });
  }
  function show(el, ev) {
    const f = getInfo()[el.dataset.f];
    if (f) tip.innerHTML = `<div class="t ${f.side}">${f.title}</div><div class="long">${f.long}</div>` +
      (f.kw ? `<div class="kw"><b>Keywords:</b> ${f.kw}</div>` : '') +
      `<div class="m"><div><b>${fmt(f.cf)}</b><span>Counterfactual</span></div><div><b>${f.ratio}</b><span>Human :: AI</span></div><div><b>${f.ai}</b><span>AutoInterp</span></div></div>`;
    else tip.innerHTML = `<div class="t">${el.dataset.title || ''}</div><div class="long">${el.dataset.desc || ''}</div>`;
    tip.classList.add('show'); move(ev);
  }
  function move(ev) {
    const m = 14, w = 300, h = tip.offsetHeight;
    let x = ev.clientX + m, y = ev.clientY + m;
    if (x + w > doc.documentElement.clientWidth - 8) x = ev.clientX - w - m;
    if (y + h > doc.documentElement.clientHeight - 8) y = ev.clientY - h - m;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }
  const target = ev => ev.target && ev.target.closest ? ev.target.closest('.card, .gauge .seg, .tok[data-f]') : null;
  let current = null;
  doc.addEventListener('mouseover', ev => {
    const el = target(ev);
    if (el === current) return;
    current = el;
    try { if (el) { focus(el.dataset.f || null); show(el, ev); } else { focus(null); tip.classList.remove('show'); } }
    catch (e) { console.error('dscope hover', e); }
  });
  doc.addEventListener('mousemove', ev => { if (current) move(ev); });

  // ---- intro: the how-to card, on every load; the sidebar link brings it back
  const intro = doc.getElementById('intro-back');
  if (intro) intro.classList.add('show');
  doc.addEventListener('click', ev => {
    if (!intro) return;
    if (ev.target.closest('.intro-link')) intro.classList.add('show');
    if (ev.target === intro || ev.target.closest('.intro .ok')) {
      intro.classList.remove('show');
    }
  });

  // ---- keyboard: Ctrl/Cmd+Enter in the editor selects the "Run" segment (a radio input)
  doc.addEventListener('keydown', ev => {
    if (ev.key !== 'Enter' || !(ev.metaKey || ev.ctrlKey)) return;
    if (!ev.target.closest || !ev.target.closest('#text textarea')) return;
    const run = doc.querySelector('#mode-row input[type=radio][value="Run"]');
    if (run && !run.checked) run.click();
  }, true);
}
""".replace("__RGB__", json.dumps(SIDE_RGB))


def fmt(v: float) -> str:
  """Signed logit with a typographic minus, e.g. "+1.23" / "−0.40"."""
  return f"{v:+.2f}".replace("-", "−")


def _num(v, spec: str) -> str:
  """format(v, spec), or an en dash for NaN (features missing LOR stats)."""
  return "–" if pd.isna(v) else format(v, spec)


def terse(fid, row: pd.Series) -> str:
  """Short description of a feature, with a fallback for undescribed ones."""
  d = row.get("description.terse")
  return (
      str(d).capitalize()
      if isinstance(d, str)
      else f"Feature {fid} (no description)"
  )


def help_icon() -> str:
  """The "?" badge whose hover / focus tooltip is HELP_TEXT."""
  return f'<span class="help" tabindex="0">?<span class="help-tip">{HELP_TEXT}</span></span>'


def highlight_html(
    tokens: list[str], feats: pd.DataFrame, acts: torch.Tensor, totals: dict
) -> str:
  """Shade each token by the side of its strongest listed feature.

  Shade strength is the token's activation (normalized per feature) times
  the feature's share of its side's evidence, capped so no token is opaque.
  Each shaded token carries its full active-feature list in `data-a` so the
  hover script can re-shade it for a single focused feature.

  Args:
    tokens: The document's tokens, concatenated verbatim (newlines -> <br>).
    feats: Listed features (both sides) with columns cf_delta, side.
    acts: [seq, N] per-token activations for the N rows of feats.
    totals: {"ai": A, "hu": H} evidence totals for share normalization.
  """
  ids = feats.index.tolist()
  if not ids or acts.shape[1] == 0:
    return f'<div class="highlight-text card-box dscope">{html.escape("".join(tokens))}</div>'
  share = torch.tensor(
      [abs(c) / totals[s] for c, s in zip(feats.cf_delta, feats.side)]
  )
  norm = acts / acts.max(0).values.clamp_min(1e-9)
  weight = norm * (0.18 + 2.2 * share).clamp(max=0.85)  # [seq, N]
  w, winner = weight.max(1)
  out = []
  for i, tok in enumerate(tokens):
    esc = html.escape(tok).replace("\n", "<br>")
    if not tok.strip() or w[i] == 0:
      out.append(esc)
      continue
    j = winner[i].item()
    active = ",".join(
        f"{ids[k]}:{acts[i, k]:.3f}"
        for k in (acts[i] > 0).nonzero().flatten().tolist()
    )
    side = feats.side.iloc[j]
    bg = f"rgba({','.join(map(str, SIDE_RGB[side]))},{w[i]:.2f})"
    out.append(
        f'<span class="tok" data-f="{ids[j]}" data-side="{side}" data-side0="{side}" data-bg="{bg}" '
        f'data-a="{active}" style="background:{bg}">{esc}</span>'
    )
  return f'<div class="highlight-text card-box dscope">{"".join(out)}</div>'


def segment(
    value: float,
    cls: str,
    fid="",
    title: str = "",
    desc: str = "",
    kind: str = "",
) -> dict:
  """One gauge arc, as the dict gauge_svg consumes (v is |value|).

  Args:
    kind: "" (feature), "other" (collapsed), "pseudo" (bias / interactions).
    cls: Side ("ai" / "hu"); picks the fill.
  """
  fill = {
      "": f"var(--{cls})",
      "other": f"var(--{cls}-lite)",
      "pseudo": f"url(#hatch-{cls})",
  }[kind]
  return dict(v=abs(value), cls=cls, fill=fill, fid=fid, title=title, desc=desc)


def _arc(
    a0: float, a1: float, cx: float, cy: float, ro: float, ri: float
) -> str:
  """SVG path of an annulus sector from angle a0 to a1 (radians, a0 > a1)."""
  p = lambda a, r: f"{cx + r * math.cos(a):.1f},{cy - r * math.sin(a):.1f}"
  return f"M{p(a0, ro)} A{ro},{ro} 0 0 1 {p(a1, ro)} L{p(a1, ri)} A{ri},{ri} 0 0 0 {p(a0, ri)} Z"


def gauge_svg(segments: list[dict], score: float | None) -> str:
  """Half gauge; AI on the left, human on the right.

  Args:
    segments: Arcs from segment(), left to right; arc length is v / sum(v).
    score: P(AI) in percent, shown with an AI / Human verdict (> 50 is AI);
      None draws the placeholder (no number, no verdict).
  """
  cx, cy, ro, ri = 165, 160, 150, 104
  total = sum(s["v"] for s in segments) or 1.0
  defs = "".join(
      f'<pattern id="hatch-{k}" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate({r})">'
      f'<rect width="6" height="6" fill="var(--{k}-lite)"/><rect width="2.2" height="6" fill="var(--{k})"/></pattern>'
      for k, r in (("ai", 45), ("hu", -45))
  )
  a, paths = math.pi, []
  for s in segments:
    span = s["v"] / total * math.pi
    paths.append(
        f'<path class="seg {s["cls"]}" data-f="{s["fid"]}" data-title="{html.escape(s["title"])}" '
        f'data-desc="{html.escape(s["desc"])}" d="{_arc(a, a - span, cx, cy, ro, ri)}" style="fill:{s["fill"]}"/>'
    )
    a -= span
  side = "ai" if (score or 0) > 50 else "hu"
  verdict = (
      ""
      if score is None
      else (
          f'<text class="score" x="{cx}" y="{cy - 4}" text-anchor="middle">{score:.0f}</text>'
          f'<text class="verdict" x="{cx}" y="{cy + 16}" text-anchor="middle" style="fill:var(--{side}-ink)">{SIDES[side]}</text>'
      )
  )
  return (
      f'<svg class="gauge" viewBox="0 0 330 190"><defs>{defs}</defs>{"".join(paths)}{verdict}'
      f'<text x="{cx - ro + 4}" y="{cy + 14}">AI</text><text x="{cx + ro - 4}" y="{cy + 14}" text-anchor="end">Human</text></svg>'
  )


def card(
    fid, desc: str, value: float, total: float, kind: str = "", tip: str = ""
) -> str:
  """One feature row (a card): description, bar, percentage and logit delta.

  Args:
    kind: "" (feature), "other" (collapsed), "pseudo" (bias / interactions).
    tip: Tooltip body for the pseudo kinds; features use feature_info.
  """
  pct = abs(value) / total * 100 if total else 0
  return (
      f'<div class="card {kind}" data-f="{fid}" data-title="{html.escape(desc)}" data-desc="{html.escape(tip)}">'
      f'<div class="d">{html.escape(desc)}</div><div class="row"><div class="track">'
      f'<div class="fill" style="width:{pct:.0f}%"></div></div><span class="pct">{pct:.0f}%</span>'
      f'<span class="lg">{fmt(value)}</span></div></div>'
  )


def column_html(side: str, cards: str) -> str:
  """One side's titled, scrollable column of cards."""
  # a div, not <h3>: the column title must never become a page heading
  return f'<div class="col {side}"><div class="col-title">{SIDES[side]} features</div><div class="list">{cards}</div></div>'


def evidence_panel(gauge: str, totals: dict, counts: dict, columns: str) -> str:
  """The whole evidence panel: help badge, title, gauge, totals, columns.

  Args:
    totals: Logit totals keyed "ai" / "hu"; the human total is negated so
      both display with the sign of their logit contribution.
    counts: Active-feature counts keyed "ai" / "hu".
  """
  return (
      f'<div class="evi card-box dscope">{help_icon()}<div class="panel-title">{TITLE}</div>{gauge}'
      f'<div class="totals"><div class="ai">Evidence for AI<b>{fmt(totals["ai"])}</b>{counts["ai"]} active features</div>'
      f'<div class="hu">Evidence for human<b>{fmt(-totals["hu"] or 0.0)}</b>{counts["hu"]} active features</div></div>'
      f'<div class="cols">{columns}</div></div>'
  )


def stale_panel(panel: str) -> str:
  """The last run's evidence panel, dimmed, shown while the text is edited."""
  return f'<div class="stale">{panel}</div>'


def empty_panel() -> str:
  """Placeholder before the first run: blank gauge, zero totals, empty columns."""
  gauge = gauge_svg(
      [segment(1, "ai", kind="other"), segment(1, "hu", kind="other")], None
  )
  columns = "".join(column_html(s, "") for s in ("ai", "hu"))
  return evidence_panel(
      gauge, {"ai": 0.0, "hu": 0.0}, {"ai": 0, "hu": 0}, columns
  )


def error_panel(message: str) -> str:
  """The panel frame with an error message where the evidence would be."""
  return (
      f'<div class="evi card-box dscope"><div class="panel-title">{TITLE}</div>'
      f'<div class="error">{html.escape(message)}</div></div>'
  )


def with_info(panel: str, info: dict) -> str:
  """The evidence panel with its tooltip payload embedded for PAGE_JS.

  Args:
    panel: evidence_panel() HTML.
    info: Feature id -> feature_info().
  """
  data = json.dumps(info).replace("</", "<\\/")
  return (
      f'{panel}<script type="application/json" id="dscope-info">{data}</script>'
  )


def feature_info(fid, row: pd.Series, side: str, max_act: float) -> dict:
  """Tooltip payload for one listed feature (read by PAGE_JS.show and focus).

  "max" is the feature's peak activation on this document, used by focus to
  scale the focused-token shading; "ratio" is the human :: target density
  pair from the LOR csv and "ai" the AutoInterp correlation.
  """
  return {
      "side": side,
      "title": f"{fid}: {html.escape(terse(fid, row))}",
      "long": html.escape(str(row.get("description.long", "") or "")),
      "kw": html.escape(str(row.get("description.keywords", "") or "")),
      "cf": float(row.cf_delta),
      "ratio": f"{_num(row.human_density, '.0f')} :: {_num(row.target_density, '.0f')}",
      "ai": _num(row.correlation, ".2f"),
      "max": max_act,
  }
