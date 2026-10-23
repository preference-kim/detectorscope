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

"""Thin wrapper around the Gemini API for text generation.

`GeminiGenerator` adds retries to synchronous calls and batches asynchronous
ones; the client authenticates from the environment (`GEMINI_API_KEY`, or the
Vertex variables) as `genai.Client()` does.
"""

import asyncio
import random

from google import genai
from google.genai import errors
from google.genai import types


def _why_empty(r) -> str:
  """Best-effort reason string for a response with no text."""
  try:
    fb = getattr(r, "prompt_feedback", None)
    if fb is not None and getattr(fb, "block_reason", None):
      return f"prompt blocked: {fb.block_reason}"
    cands = getattr(r, "candidates", None) or []
    if cands:
      return f"finish_reason={getattr(cands[0], 'finish_reason', None)}"
    return "no candidates"
  except Exception as e:  # pragma: no cover - diagnostics only
    return f"unknown ({e})"


class GeminiGenerator:
  """Gemini text generation with retries, synchronous or batched async.

  Note the asymmetry the auto-interp code relies on: `generate` returns the
  raw response objects (read `.text`), `async_generate` returns strings.

  Args:
    model_name: Gemini model name passed on every request.
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

    self.default_config = types.GenerateContentConfig(
        temperature=1.0, top_p=0.95, top_k=64
    )

  def generate(
      self, prompts: list[str], config: types.GenerateContentConfig = None
  ) -> list[str]:
    """Generates one response per prompt, sequentially and with retries.

    Returns:
      Response objects, in prompt order. A prompt that still fails after
      sync_max_retries attempts is skipped, so the list can be shorter than
      prompts.
    """
    if config is None:
      config = self.default_config

    responses = []

    for prompt in prompts:
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

        except (errors.ClientError, errors.ServerError) as e:
          # On failure, print error and try again
          print(f"Generation Error: {e}")
          retries = retries + 1

    return responses

  async def _async_generate(
      self, prompts: list[str], config: types.GenerateContentConfig = None
  ) -> list[str]:
    """Fills responses in random batches until every slot has text or we quit.

    Returns:
      Response texts aligned with prompts; a slot that never succeeded stays
      "".
    """
    if config is None:
      config = self.default_config

    # Seeded so the batch order is reproducible for a given set of prompts
    random.seed(self.random_seed)
    retries = 0
    rounds = 0
    # Every prompt gets at most async_max_retries attempts once it is the only
    # one left; before that, batches of async_batch_size share the rounds.
    max_rounds = self.async_max_retries * (
        -(-len(prompts) // self.async_batch_size) + 1
    )
    responses = [""] * len(prompts)
    while not all(responses):
      # Get a random batch of the remaining prompts
      remaining_response_indices = [
          i for i, response in enumerate(responses) if not response
      ]
      sample_size = min(len(remaining_response_indices), self.async_batch_size)
      batch_indices = random.sample(remaining_response_indices, sample_size)
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

      # Check for any errors and update the progress bar. A response object
      # whose .text is None/empty (e.g. the prompt was blocked by the safety
      # filter, or the candidate has no text part) is a failure, not progress:
      # storing it would leave the slot empty, and counting it as a success
      # would reset the retry counter and spin forever on a blocked prompt.
      rows_added = False
      for i, r in enumerate(batch_responses):
        if isinstance(r, types.GenerateContentResponse):
          text = r.text
          if text:
            responses[batch_indices[i]] = text
            rows_added = True
          else:
            print(f"Empty response (blocked or no text part): {_why_empty(r)}")
        else:
          print(f"Error: {r}")

      # A round with no progress counts as a retry; any progress resets it
      retries = retries + 1 if not rows_added else 0
      rounds += 1

      # If we've retried too many times, break out of loop. The hard cap on
      # total rounds is a backstop in case a prompt keeps failing while others
      # still make progress.
      if retries >= self.async_max_retries or rounds >= max_rounds:
        unfilled = sum(1 for r in responses if not r)
        print(
            f"Giving up after {rounds} rounds with {unfilled} unfilled prompt(s)"
        )
        break

    return responses

  def async_generate(
      self, prompts: list[str], config: types.GenerateContentConfig = None
  ) -> list[str]:
    """Runs _async_generate from synchronous code.

    Schedules it on the running loop when called from another thread, else
    uses asyncio.run.

    Returns:
      Response texts aligned with prompts ("" where generation failed).
    """
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
