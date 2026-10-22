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

"""Exports one detector setting to the layout AuditableDetector loads.

One run writes one repo directory: the SAEs (model.safetensors), every head
of the setting under heads/ (the SGD+L1 logreg sweep and the XGBoost sweep,
one file per (type, tau), its decision threshold taken from the head's
evaluation JSON), features.jsonl (the auto-interpretation records of every
described feature plus its log-odds ratios) and the model card. It then
reloads the directory, checks that the demo examples get identical detections
under every head, and pushes the directory to a branch of a private Hub repo
when asked: a repo's branches are a detector's settings (and, for PRDB, its
targets), `main` being the paper's flagship.

  python scripts/huggingface_hub/export_hub.py -c demo/config/settings.yml \
      --setting "PRDB · Gemma-2 fine-tuned (gpt4o)" --target gpt4o \
      --logreg_dir outputs/logreg_sgd_l1_ablation/models/<dataset> \
      --logreg_eval outputs/logreg_sgd_l1_ablation/eval_results/<split>/<dir> \
      --xgboost_eval eval_results/<split>/<dir> \
      --repo_id <ns>/detector-scope-prdb --revision gpt4o-ft-sae \
      -o outputs/hub/detector-scope-prdb/gpt4o-ft-sae [--push] [--alias gpt4o]
"""

import argparse
import csv
import json
import os
import re
import shutil
import typing

import huggingface_hub
import numpy as np
import yaml

from detectorscope.data import cache
from detectorscope.model.detectors.auditable import AuditableDetector
from detectorscope.model.detectors.auditable import head_file
from detectorscope.model.detectors.heads.head import DetectorHead
from detectorscope.model.detectors.heads.logreg import LogRegHead

CARDS = os.path.join(os.path.dirname(__file__), "cards")
LOGREG = r"sgd_lr1e-02_l1-1e-01_threshold([0-9.]+)_{target}_logreg_ep3\.pkl"
XGBOOST = r"threshold([0-9.]+)_{target}_xgbmodel\.pkl"


def load_setting(config: str, name: str) -> dict:
  """The named setting of settings.yml, defaults merged, `sae_paths` filled."""
  with open(config, encoding="utf-8") as f:
    raw = yaml.safe_load(f)
  cfg = raw.get("defaults", {}) | next(
      c for c in raw["settings"] if c["name"] == name
  )
  if "sae_paths" not in cfg:
    cfg["sae_paths"] = [
        f"{cfg['sae_dir']}/{h}_input_finetuned_sae.pth"
        for h in cfg["hook_points"]
    ]
  return cfg


def heads(
    kind: str, pattern: str, directory: str, eval_dir: str, block: str
) -> dict[tuple[str, float], DetectorHead]:
  """One head per tau matching `pattern` in `directory`, thresholds attached.

  The threshold is `optimal_threshold` of `block` in the head's evaluation
  JSON (`<name>_results.json`, with `_ep3` moved before `_logreg` for the
  sweep heads); 0.0 with a warning when the JSON is missing.
  """
  out = {}
  for name in sorted(os.listdir(directory)):
    if not (m := re.fullmatch(pattern, name)):
      continue
    head = DetectorHead.create_detector_head(
        kind, detector_path=f"{directory}/{name}"
    )
    results = re.sub(
        r"_logreg_ep(\d+)\.pkl$", r"_ep\1_logreg_results.json", name
    )
    results = f"{eval_dir}/{results.replace('.pkl', '_results.json')}"
    if os.path.exists(results):
      with open(results, encoding="utf-8") as f:
        head.threshold = json.load(f)[block]["optimal_threshold"]
    else:
      print(f"no evaluation for {name}: threshold 0.0")
    head.metadata = {"source": name}
    out[kind, round(float(m[1]), 2)] = head
  return out


def feature_table(cfg: dict, target: str, d_sae: int) -> dict[int, dict]:
  """features.jsonl records: the description files verbatim, plus id and LOR."""
  lor_files = dict(
      cache.get_lor_files(cfg["data_dir"], target)
      or cache.get_lor_files(cfg["data_dir"])
  )
  table = {}
  desc_files = cache.get_desc_files(cfg["data_dir"], cfg["desc_model"])
  for rank, (layer, path) in enumerate(desc_files):
    lor = {}
    if layer in lor_files:
      with open(lor_files[layer], encoding="utf-8") as f:
        for row in csv.DictReader(f):
          lor[int(row["feature_id"])] = {
              k: float(row[k]) for k in cache.LOR_COLS if row[k]
          }
    with open(path, encoding="utf-8") as f:
      for record in map(json.loads, f):
        index = record.pop("feature_index")
        fid = rank * d_sae + index
        table[fid] = {"id": fid, "layer": layer, "index": index}
        table[fid] |= record | lor.get(index, {})
  return dict(sorted(table.items()))


def model_card(out: str, repo_id: str, revision: str, card: dict) -> str:
  """The card of an exported branch: usage, performance, top features.

  Reads the export directory (config, default head, features) and the card's
  data sources: `stats` (the paper's feature_stats npz, for weight x mean
  activation), `perf` ([label, eval dir, block] rows; the default head's
  evaluation JSON lives at `<eval dir>/<eval_dir>/`) and its prose.
  """
  with open(f"{out}/config.json", encoding="utf-8") as f:
    config = json.load(f)
  kind, tau = config["head"], config["tau"]
  head = typing.cast(
      LogRegHead, DetectorHead.load(f"{out}/{head_file(kind, tau)}")
  )
  n_heads = len(os.listdir(f"{out}/heads"))
  with open(f"{out}/features.jsonl", encoding="utf-8") as f:
    feats = {r["id"]: r for r in map(json.loads, f)}
  st = np.load(card["stats"])
  w = (head.detector.weight[0].detach().cpu() * head.interp_mask.cpu()).numpy()
  mean_cf = w * st["act_mean"]
  rows = {+1: [], -1: []}
  for i in np.argsort(-np.abs(mean_cf)):
    if (
        w[i] == 0
        or st["n_active"][i] < 10
        or i not in feats
        or len(rows[np.sign(mean_cf[i])]) >= 10
    ):
      continue
    r = feats[i]
    ai = st["n_active_ai"][i] / int(st["n_ai_docs"])
    human = st["n_active_human"][i] / int(st["n_human_docs"])
    rows[np.sign(mean_cf[i])].append(
        f"| {r['layer']}_{r['index']} | {r['description']['terse']} |"
        f" {mean_cf[i]:+.3f} | {ai:.0%} | {human:.0%} |"
    )
  stem = f"sgd_lr1e-02_l1-1e-01_threshold{tau:.2f}_{{}}_ep3_logreg_results.json"
  perf = []
  for label, eval_root, block in card["perf"]:
    target = eval_root.rsplit("_", 1)[-1] if "PRDB" in eval_root else "coling"
    with open(
        f"{eval_root}/{card['eval_dir']}/{stem.format(target)}",
        encoding="utf-8",
    ) as f:
      m = json.load(f)[block]
    perf.append(
        f"| {label} | {m['auroc']:.3f} | {m['f1']:.3f} | {m['accuracy']:.3f}"
        f" | {m['n_human'] + m['n_ai']:,} |"
    )
  with open(f"{CARDS}/detector_README.md", encoding="utf-8") as f:
    template = f.read()
  return template.format(
      name=card["name"],
      about=card["about"],
      repo_id=repo_id,
      revision="" if revision == "main" else f', revision="{revision}"',
      threshold=head.threshold,
      tau=f"{tau:.2f}",
      n_heads=n_heads - 1,
      perf_rows="\n".join(perf),
      ai_rows="\n".join(rows[1]),
      human_rows="\n".join(rows[-1]),
      variants=card["variants"],
      extractor=card["extractor"],
  )


def same(a, b) -> bool:
  """Whether two detections are identical."""
  return (a.logit, a.bias, a.label, a.features) == (
      b.logit,
      b.bias,
      b.label,
      b.features,
  )


def push(repo_id: str, revision: str, folder: str, alias: str | None) -> None:
  """Uploads folder to the branch of repo_id (created private if new)."""
  api = huggingface_hub.HfApi()
  api.create_repo(repo_id, private=True, exist_ok=True)
  if revision != "main":
    api.create_branch(repo_id, branch=revision, exist_ok=True)
  print(
      api.upload_folder(folder_path=folder, repo_id=repo_id, revision=revision)
  )
  if alias:
    if alias in {b.name for b in api.list_repo_refs(repo_id).branches}:
      api.delete_branch(repo_id, branch=alias)
    api.create_branch(repo_id, branch=alias, revision=revision)


def write_card(args: argparse.Namespace) -> None:
  """Writes the branch's README.md from scripts/huggingface_hub/cards/branches.json."""
  with open(f"{CARDS}/branches.json", encoding="utf-8") as f:
    card = json.load(f)[f"{args.repo_id.split('/')[1]}/{args.revision}"]
  with open(f"{args.output_dir}/README.md", "w", encoding="utf-8") as f:
    f.write(model_card(args.output_dir, args.repo_id, args.revision, card))


def main(args: argparse.Namespace) -> None:
  """Builds, checks and optionally pushes the repo directory."""
  cfg = load_setting(args.config, args.setting)
  det = AuditableDetector.from_paths(
      cfg["model_path"], "logreg", None, cfg["hook_points"], cfg["sae_paths"]
  )
  block = args.block or args.target
  found = heads(
      "logreg",
      LOGREG.format(target=args.target),
      args.logreg_dir,
      args.logreg_eval,
      block,
  )
  found |= heads(
      "xgboost",
      XGBOOST.format(target=args.target),
      cfg["data_dir"],
      args.xgboost_eval,
      block,
  )
  assert det.default in found, f"no default head {det.default}"
  for (kind, tau), head in found.items():
    if kind == "xgboost":
      used = head.get_global_importance().nonzero().flatten().tolist()
      assert all(head.interp_mask[used] == 1), f"xgboost {tau} splits off-mask"
    else:
      assert head.means is None, f"logreg {tau} has means"
    det.add_head(head, kind, tau)
  det.feature_table = feature_table(cfg, args.target, det.saes[0].d_hidden)

  with open(args.examples, encoding="utf-8") as f:
    texts = [b.strip() for b in f.read().split("=" * 100) if len(b) > 300]
  extractions = [det.extract(t) for t in texts]
  reference = {
      k: [det.explain(ex, *k) for ex in extractions] for k in det.heads()
  }

  os.makedirs(args.output_dir, exist_ok=True)
  shutil.copy(f"{CARDS}/NOTICE", args.output_dir)
  det.save_pretrained(
      args.output_dir,
      config={
          "base_model": None,
          "hook_points": cfg["hook_points"],
          "head": det.default[0],
          "tau": det.default[1],
      },
  )
  del det
  write_card(args)

  det = AuditableDetector.from_pretrained(args.output_dir)
  assert det.heads() == sorted(reference), det.heads()
  extractions = [det.extract(t) for t in texts]
  for k, dets in reference.items():
    new = [det.explain(ex, *k) for ex in extractions]
    assert all(map(same, dets, new)), f"reload differs for {k}"
  print(
      f"{args.output_dir}: {len(reference)} heads, {len(det.feature_table)}"
      f" features, reload identical on {len(texts)} texts"
  )

  if args.push:
    push(args.repo_id, args.revision, args.output_dir, args.alias)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-c", "--config", required=True, help="settings.yml")
  parser.add_argument("--setting", required=True, help="Setting name in it")
  parser.add_argument("--target", required=True, help="gpt4o, llama or coling")
  parser.add_argument(
      "--logreg_dir", required=True, help="SGD+L1 sweep head pickles"
  )
  parser.add_argument(
      "--logreg_eval", required=True, help="Their evaluation JSONs"
  )
  parser.add_argument(
      "--xgboost_eval",
      required=True,
      help="Evaluation JSONs of the XGBoost heads in data_dir",
  )
  parser.add_argument(
      "--block", default=None, help="Evaluation block (default: the target)"
  )
  parser.add_argument(
      "--examples",
      default="demo/examples/prdb_gpt4o_vs_human_pairs.txt",
      help="Texts for the reload check",
  )
  parser.add_argument("--repo_id", required=True, help="Hub repo id")
  parser.add_argument("--revision", default="main", help="Its branch")
  parser.add_argument("-o", "--output_dir", required=True)
  parser.add_argument(
      "--push", action="store_true", help="Upload to the branch, private"
  )
  parser.add_argument("--alias", default=None, help="Branch to alias it as")
  parser.add_argument(
      "--card_only",
      action="store_true",
      help="Only rewrite README.md of an existing export (and push it)",
  )
  parsed = parser.parse_args()
  if not parsed.card_only:
    main(parsed)
  else:
    write_card(parsed)
    if parsed.push:
      huggingface_hub.HfApi().upload_file(
          path_or_fileobj=f"{parsed.output_dir}/README.md",
          path_in_repo="README.md",
          repo_id=parsed.repo_id,
          revision=parsed.revision,
      )
