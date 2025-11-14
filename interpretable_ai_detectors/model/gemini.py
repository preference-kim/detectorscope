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

"""Provides a class for interacting with the Gemini API for text generation."""

import asyncio
import json
import random
import time
import uuid

from google import genai
from google.genai import errors
from google.genai import types
import tqdm


class GeminiGenerator:
  """Provides a class for interacting with the Gemini API for text generation.

  Handles synchronous, asynchronous, and batch generation requests, including
  error handling and rate limiting.
  """

  def __init__(self, model_name: str = "gemini-2.5-pro"):
    self.client = genai.Client()
    self.model_name = model_name

    self.random_seed = 42
    self.async_batch_size = 20
    self.model_name = model_name

    self.async_batch_size = 20
    self.async_max_retries = 10
    self.sync_max_retries = 10
    self.backoff_max_retries = 100

    self.default_config = types.GenerateContentConfig(
        temperature=1.0, top_p=0.95, top_k=64
    )

  def generate(
      self, prompts: list[str], config: types.GenerateContentConfig = None
  ) -> list[str]:
    """Generate outputs for prompts and handle errors synchronously."""
    if config is None:
      config = self.default_config

    # TODO(liamdugan): make this interface work with single prompt str
    #                  and output type str if input type is str
    responses = []

    for prompt in tqdm.tqdm(prompts):
      retries = 0
      while retries < self.sync_max_retries:
        try:
          # Generate text from Gemini
          response = self.client.models.generate_content(
              model=self.model_name, contents=prompt, config=config
          )

          # If the above line succeeds, append response & exit loop
          responses.append(response)
          retries = 0
          break

        except errors.ClientError as e:
          # On failure, print error and try again
          print(f"Generation Error: {e}")
          retries = retries + 1

    return responses

  async def _async_generate(
      self, prompts: list[str], config: types.GenerateContentConfig = None
  ) -> list[str]:
    """Async helper function for async_generate."""
    if config is None:
      config = self.default_config

    random.seed(self.random_seed)
    retries = 0
    responses = [""] * len(prompts)
    with tqdm.tqdm(total=len(prompts)) as pbar:
      while not all(responses):
        # Get a random batch of the remaining prompts
        remaining_response_indices = [
            i for i, response in enumerate(responses) if not response
        ]
        sample_size = min(
            len(remaining_response_indices), self.async_batch_size
        )
        batch_indices = random.sample(
            remaining_response_indices, sample_size
        )
        batch = [prompts[i] for i in batch_indices]

        # Async request the prompt completions
        requests = [
            self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            for prompt in batch
        ]

        # Await all responses from the API
        batch_responses = list(
            await asyncio.gather(*requests, return_exceptions=True)
        )

        # Check for any errors and update the progress bar
        rows_added = False
        for i, r in enumerate(batch_responses):
          if isinstance(r, types.GenerateContentResponse):
            responses[batch_indices[i]] = r.text
            rows_added = True
            pbar.update(1)
          else:
            print(f"Error: {r}")

        # Update retry counter
        retries = retries + 1 if not rows_added else 0

        # If we've retried too many times, break out of loop
        if retries >= self.async_max_retries:
          break

    return responses

  def async_generate(
      self, prompts: list[str], config: types.GenerateContentConfig = None
  ) -> list[str]:
    """Generates outputs asynchronously, handling event loop management."""
    if config is None:
      config = self.default_config

    try:
      # Try to get the running event loop
      loop = asyncio.get_running_loop()

      # If it succeeds, schedule the coroutine
      return asyncio.run_coroutine_threadsafe(
          self._async_generate(prompts, config), loop
      ).result()
    except RuntimeError:
      # A RuntimeError means no event loop was running.
      # We fall back to asyncio.run(), which creates a new loop
      return asyncio.run(self._async_generate(prompts, config))

  def format_prompt(
      self,
      prompt: str,
      prompt_id: str,
      config: types.GenerateContentConfig = None,
  ) -> str:
    """Formats a prompt as a GenerateContentRequest JSON string."""
    if config is None:
      config = self.default_config

    # See https://ai.google.dev/gemini-api/docs/batch-api for format details.
    return json.dumps({
        "key": prompt_id,
        "request": {
            "contents": [{"parts": [{"text": prompt}]}],
            "generation_config": config.to_json_dict(),
        },
    })

  def submit_batch_job(
      self, batch: list[str], display_name: str = None
  ) -> tuple[str, str, str]:
    """Writes batch to temp file and submits it as a batch job."""
    tempfile = f".temp_{str(uuid.uuid4())}"
    with open(tempfile, "w+") as f:
      for line in batch:
        f.write(line + "\n")

    if not display_name:
      display_name = str(uuid.uuid4())

    uploaded_file = self.client.files.upload(
        file=tempfile,
        config=types.UploadFileConfig(
            display_name=display_name, mime_type="jsonl"
        ),
    )

    backoff = 0
    job_name = ""
    while backoff < self.backoff_max_retries:
      try:
        batch_job = self.client.batches.create(
            model=self.model_name,
            src=uploaded_file.name,
            config={"display_name": display_name},
        )
        job_name = batch_job.name
        break
      except errors.ClientError as e:
        print(f"ClientError : {e}")
        if "429" in str(e):  # HACK - TODO(liamdugan) should use e.code
          print(f"Rate limit error!.. sleeping for {2 ** backoff}s")
          time.sleep(2**backoff)
          backoff += 1

    return tempfile, job_name
