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

"""CLI to monitor pending and completed batch jobs from Gemini Batch API."""

import time

from google import genai


client = genai.Client()

COMPLETED_JOB_STATES = [
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
]

MAX_TIME = 60 * 60 * 2  # 2 hours


def main():
  time_start = time.time()
  while time.time() - time_start < MAX_TIME:
    batches = client.batches.list(config={"page_size": 10})
    for b in batches.page:
      if b.state is not None and b.state.name not in COMPLETED_JOB_STATES:
        print("-" * 20)
        print(f"Batch Job: {b.name}")
        print(f"- Display Name: {b.display_name}")
        print(f"- State: {getattr(b.state, 'name', 'N/A')}")
        if b.create_time is not None:
          print(f"- Create Time: {b.create_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 20)
    time.sleep(15)

if __name__ == "__main__":
  main()
