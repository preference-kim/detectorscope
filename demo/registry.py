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

"""The demo's detectors, addressed as on the Hub: repo id and revision.

A Registry answers, for the configured repos, what the Hub holds (branches,
the heads of a branch, its default head, its card title) from a few small
files, and loads a branch's AuditableDetector when asked, keeping the most
recently used ones resident. A branch's files come from the Hub cache unless
the config maps it to a local directory (an export, a Space volume).
"""

import collections
import gc
import json
import os

import huggingface_hub

from detectorscope.model.detectors import auditable

META = ["config.json", "heads/*", "features.jsonl", "README.md"]


class Registry:
  """Branches, heads and loaded detectors of the configured repos.

  Config keys: `repos` (ids, in display order); optional `revisions`
  {repo: [branch]} pinning the branches instead of asking the Hub,
  `local_root` (a directory holding <repo name>/<revision>/ copies, e.g.
  export directories or Space volume mounts), `local` {"repo@revision": directory}, and `heads` (head
  types to offer; default all).
  """

  def __init__(self, cfg: dict, resident: int = 1):
    self.cfg = cfg
    self.resident = resident
    self._branches: dict[str, list[str]] = {}
    self._roots: dict[tuple[str, str], str] = {}
    self._detectors = collections.OrderedDict()

  @property
  def repos(self) -> list[str]:
    return self.cfg["repos"]

  def branches(self, repo: str) -> list[str]:
    """The repo's branches, `main` first, then alphabetical."""
    if repo not in self._branches:
      names = self.cfg.get("revisions", {}).get(repo) or [
          b.name for b in huggingface_hub.list_repo_refs(repo).branches
      ]
      self._branches[repo] = sorted(names, key=lambda n: (n != "main", n))
    return self._branches[repo]

  def local(self, repo: str, revision: str) -> str | None:
    """The directory configured for the branch, if any."""
    path = self.cfg.get("local", {}).get(f"{repo}@{revision}")
    root = self.cfg.get("local_root")
    mount = f"{root}/{repo.split('/')[-1]}/{revision}"
    return path or (mount if root and os.path.isdir(mount) else None)

  def root(self, repo: str, revision: str) -> str:
    """A directory with the branch's config, heads, features and card."""
    if (repo, revision) not in self._roots:
      self._roots[repo, revision] = self.local(
          repo, revision
      ) or huggingface_hub.snapshot_download(
          repo, revision=revision, allow_patterns=META
      )
    return self._roots[repo, revision]

  def heads(self, repo: str, revision: str) -> list[tuple[str, float]]:
    """The branch's (head type, tau) pairs, restricted to `heads` if set."""
    kinds = self.cfg.get("heads")
    heads = auditable.list_heads(self.root(repo, revision))
    return [h for h in heads if not kinds or h[0] in kinds]

  def default(self, repo: str, revision: str) -> tuple[str, float]:
    """The branch's default head, from its config.json."""
    with open(
        f"{self.root(repo, revision)}/config.json", encoding="utf-8"
    ) as f:
      cfg = json.load(f)
    return cfg["head"], round(cfg["tau"], 2)

  def caption(self, repo: str, revision: str) -> str:
    """The branch's card title (the README's first heading)."""
    path = f"{self.root(repo, revision)}/README.md"
    if os.path.exists(path):
      with open(path, encoding="utf-8") as f:
        for line in f:
          if line.startswith("# "):
            return line[2:].strip()
    return f"{repo}@{revision}"

  def detector(self, repo: str, revision: str) -> auditable.AuditableDetector:
    """The branch's detector; the `resident` most recent stay loaded."""
    key = (repo, revision)
    if key not in self._detectors:
      while len(self._detectors) >= self.resident:
        self._detectors.popitem(last=False)
        gc.collect()
      local = self.local(repo, revision)
      dtype = os.environ.get(
          "DSCOPE_DTYPE"
      )  # e.g. bfloat16 when host RAM is short
      self._detectors[key] = auditable.AuditableDetector.from_pretrained(
          local or repo, revision=None if local else revision, torch_dtype=dtype
      )
    self._detectors.move_to_end(key)
    return self._detectors[key]
