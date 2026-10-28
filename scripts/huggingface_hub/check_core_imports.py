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

"""Checks that importing AuditableDetector needs only the core dependencies.

Walks the package imports from detectorscope/model/detectors/auditable.py
and fails if any module on that path imports, at import time, a third-party
package outside CORE (numpy comes with transformers). Run from the repo root.
"""

import ast
import os
import sys

CORE = {"huggingface_hub", "numpy", "safetensors", "torch", "transformers",
        "typing_extensions"}
STDLIB = set(sys.stdlib_module_names)


def imports(path: str) -> tuple[set[str], set[str]]:
  """(package modules, third-party roots) imported at the top level of path."""
  internal, third_party = set(), set()
  for node in ast.parse(open(path, encoding="utf-8").read()).body:
    names = []
    if isinstance(node, ast.Import):
      names = [a.name for a in node.names]
    elif isinstance(node, ast.ImportFrom) and node.module:
      names = [node.module]
    for name in names:
      root = name.split(".")[0]
      if root == "detectorscope":
        internal.add(name)
      elif root not in STDLIB:
        third_party.add(root)
  return internal, third_party


def closure(module: str) -> dict[str, set[str]]:
  """Module -> third-party roots, over the import closure of module."""
  seen = {}
  todo = [module]
  while todo:
    name = todo.pop()
    if name in seen:
      continue
    path = name.replace(".", "/")
    path = (
        f"{path}.py" if os.path.exists(f"{path}.py") else f"{path}/__init__.py"
    )
    internal, seen[name] = imports(path)
    parts = name.split(".")
    todo += internal | {".".join(parts[:i]) for i in range(1, len(parts))}
  return seen


if __name__ == "__main__":
  bad = {
      m: roots - CORE
      for m, roots in closure("detectorscope.model.detectors.auditable").items()
      if roots - CORE
  }
  for mod, roots in sorted(bad.items()):
    print(f"{mod}: {sorted(roots)}")
  print("core imports:", "FAIL" if bad else "PASS")
  sys.exit(bool(bad))
