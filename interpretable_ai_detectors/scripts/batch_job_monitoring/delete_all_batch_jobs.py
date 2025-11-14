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

"""CLI to delete pending and completed batch jobs from Gemini Batch API."""

from google import genai

client = genai.Client()


def main():
  batches = client.batches.list(config={"page_size": 10})

  for b in batches.page:
    print(f"Batch Job: {b.name}")
    print(f"  - Display Name: {b.display_name}")
    print(f"  - State: {getattr(b.state, "name", "N/A")}")
    if b.create_time is not None:
      print(f"  - Create Time: {b.create_time.strftime('%Y-%m-%d %H:%M:%S')}")
    selection = input("Delete? (Y/N)")
    if selection == "Y":
      print("Deleting...")
      client.batches.delete(name=b.name)
    else:
      print("Not Deleting")


if __name__ == "__main__":
  main()
