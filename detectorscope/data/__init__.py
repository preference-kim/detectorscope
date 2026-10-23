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


"""Caches on disk and the stages that build them.

extract.py turns text into features (per-token SAE activations, or gram2vec
features straight into a cache), process.py collapses per-token
activations to per-document caches and derives feature scales and log-odds
ratios, cache.py reads every cache, and loading.py turns a split into a
head's training set.
"""
