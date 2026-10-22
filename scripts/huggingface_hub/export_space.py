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

"""Assembles the demo as a Hugging Face Space and publishes it.

The Space runs demo/ unchanged on ZeroGPU: every branch of the released
detector repos is mounted read-only under /models/<repo name>/<revision>, the
config pins those branches so the app never asks the Hub, and the package is
copied in until it is on PyPI. Needs huggingface_hub >= 2.2 (Space volumes).

  python scripts/huggingface_hub/export_space.py                 # outputs/space/
  python scripts/huggingface_hub/export_space.py --create --push # private Space
  python scripts/huggingface_hub/export_space.py --volumes       # after a new branch
"""

import argparse
import pathlib
import shutil
import subprocess
import tomllib

import huggingface_hub
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
MOUNT = "/models"
SPACE_VARIABLES = {
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DSCOPE_RESIDENT": "1",
}
FRONT_MATTER = """---
title: Detector Scope
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12.12"
app_file: app.py
header: mini
startup_duration_timeout: 45m
pinned: false
license: apache-2.0
short_description: Auditable AI-text detection with sparse-autoencoder features
models:
{models}
---
"""
BODY = """# Detector Scope

Paste a document, pick a released detector and see its AI probability, the
sparse-autoencoder features behind that score and the tokens each feature fires
on. The sidebar's four pickers are a detector's Hub coordinates: model (repo),
setting (branch), head type and interpretability threshold τ; choosing them is
`AuditableDetector.from_pretrained(repo, revision=..., head=..., tau=...)`.

Documents are truncated to the extractor's 1024 tokens. Each run reserves a few
seconds of ZeroGPU time per visitor quota; switching to a detector that is not
yet loaded takes a minute while its weights are read from the mounted repo.

Weights: Gemma derivatives under the Gemma Terms of Use (see NOTICE); SAEs and
heads CC-BY-4.0. App code Apache-2.0. Paper: Dugan et al. (2026), *Building
Auditable AI Text Detectors with Sparse Autoencoders*.
"""


def branches(api: huggingface_hub.HfApi, repo: str) -> list[str]:
  names = [b.name for b in api.list_repo_refs(repo).branches]
  return sorted(names, key=lambda n: (n != "main", n))


def volumes(revisions: dict[str, list[str]]) -> list[huggingface_hub.Volume]:
  return [
      huggingface_hub.Volume(
          type="model",
          source=repo,
          revision=branch,
          mount_path=f"{MOUNT}/{repo.split('/')[-1]}/{branch}",
      )
      for repo, names in revisions.items()
      for branch in names
  ]


def requirements() -> str:
  """The package's core, xgboost and demo dependencies, minus Gradio."""
  with open(REPO / "pyproject.toml", "rb") as f:
    project = tomllib.load(f)["project"]
  deps = project["dependencies"] + project["optional-dependencies"]["xgboost"]
  deps += [
      d for d in project["optional-dependencies"]["demo"] if "gradio" not in d
  ]
  deps = ["torch==2.8.0" if d.startswith("torch") else d for d in deps]
  return "\n".join(sorted(deps)) + "\n"


def build(out: pathlib.Path, revisions: dict[str, list[str]]) -> None:
  shutil.rmtree(out, ignore_errors=True)
  out.mkdir(parents=True)
  for name in ("app.py", "registry.py", "render.py"):
    shutil.copy(REPO / "demo" / name, out / name)
  shutil.copytree(REPO / "demo" / "examples", out / "examples")
  shutil.copy(
      pathlib.Path(__file__).with_name("cards") / "NOTICE", out / "NOTICE"
  )
  tracked = subprocess.check_output(
      ["git", "ls-files", "detectorscope"], cwd=REPO, text=True
  ).split()
  for path in tracked:
    (out / path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / path, out / path)
  (out / "config").mkdir()
  config = {
      "repos": list(revisions),
      "revisions": revisions,
      "local_root": MOUNT,
  }
  (out / "config" / "settings.yml").write_text(
      yaml.safe_dump(config, sort_keys=False)
  )
  (out / "requirements.txt").write_text(requirements())
  models = "\n".join(f"  - {repo}" for repo in revisions)
  (out / "README.md").write_text(FRONT_MATTER.format(models=models) + BODY)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--space", default="liamdugan/detector-scope-demo")
  parser.add_argument("--out", default=str(REPO / "outputs" / "space"))
  parser.add_argument(
      "--create",
      action="store_true",
      help="create the private Space with its volumes",
  )
  parser.add_argument(
      "--push", action="store_true", help="upload the assembled directory"
  )
  parser.add_argument(
      "--volumes",
      action="store_true",
      help="re-apply the volume list from the Hub's branches",
  )
  args = parser.parse_args()

  api = huggingface_hub.HfApi()
  with open(
      REPO / "demo" / "config" / "settings.example.yml", encoding="utf-8"
  ) as f:
    repos = yaml.safe_load(f)["repos"]
  revisions = {repo: branches(api, repo) for repo in repos}
  out = pathlib.Path(args.out)
  build(out, revisions)
  print(f"assembled {out}: {sum(len(v) for v in revisions.values())} volumes")
  if args.create:
    api.create_repo(
        args.space,
        repo_type="space",
        space_sdk="gradio",
        space_hardware="zero-a10g",
        space_variables=[
            {"key": k, "value": v} for k, v in SPACE_VARIABLES.items()
        ],
        space_volumes=volumes(revisions),
        private=True,
        exist_ok=True,
    )
    print(f"created https://huggingface.co/spaces/{args.space}")
  if args.volumes:
    api.set_space_volumes(args.space, volumes(revisions))
  if args.push:
    api.upload_folder(
        folder_path=str(out),
        repo_id=args.space,
        repo_type="space",
        ignore_patterns=["__pycache__/*"],
        commit_message="Demo build",
    )
    print(f"pushed; https://huggingface.co/spaces/{args.space}")
