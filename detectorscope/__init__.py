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

"""DetectorScope: auditable AI-text detectors built on SAE features.

AuditableDetector.from_pretrained(repo)(text) scores a text with a released
detector; the pipeline stages run as python -m detectorscope.<stage>.
"""

from detectorscope.model.detectors.auditable import AuditableDetector
from detectorscope.model.detectors.auditable import Detection
from detectorscope.model.detectors.auditable import Feature
