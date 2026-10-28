"""Complete-document windows and a bounded cache for a persistent worker."""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import threading


@dataclass(frozen=True)
class TextWindow:
  start: int
  end: int
  text: str
  tokens: int


def split_document(text, tokenizer, max_tokens=1024):
  """Partition source characters without overlap or silent token truncation.

  Each window is independently tokenized, including special tokens. These are
  window-level detector inputs; no whole-document probability is inferred.
  """
  if not 8 <= max_tokens <= 1024:
    raise ValueError("max_tokens must be between 8 and 1024.")
  offsets = tokenizer(text, add_special_tokens=False,
                      return_offsets_mapping=True)["offset_mapping"]
  budget = max_tokens - tokenizer.num_special_tokens_to_add(pair=False)
  windows = []
  start, cursor = 0, 0
  while start < len(text):
    while cursor < len(offsets) and offsets[cursor][1] <= start:
      cursor += 1
    stop = min(cursor + budget, len(offsets))
    end = len(text) if stop == len(offsets) else offsets[stop - 1][1]
    # Retokenization at a character boundary can change token counts.
    while True:
      count = len(tokenizer.encode(text[start:end]))
      if count <= max_tokens:
        break
      stop -= 1
      if stop <= cursor:
        raise ValueError("Could not fit a source character into one window.")
      end = offsets[stop - 1][1]
    if end <= start:
      raise ValueError("Tokenizer offsets did not advance through the document.")
    windows.append(TextWindow(start, end, text[start:end], count))
    start = end
  return windows


class DocumentAnalyzer:
  """Serialize model calls and cache exact windows within one configuration."""

  def __init__(self, detector, max_tokens=1024, cache_size=128, batch_size=1):
    if cache_size < 0 or batch_size < 1:
      raise ValueError("cache_size must be nonnegative; batch_size positive.")
    self.detector = detector
    self.max_tokens = max_tokens
    self.cache_size = cache_size
    self.batch_size = batch_size
    self.cache = OrderedDict()
    self.lock = threading.Lock()

  def analyze(self, text, highlights=False, top_features=10):
    if not isinstance(text, str):
      raise ValueError("text must be a string.")
    if not isinstance(highlights, bool):
      raise ValueError("highlights must be a boolean.")
    if type(top_features) is not int or not 1 <= top_features <= 100:
      raise ValueError("top_features must be an integer from 1 to 100.")
    with self.lock:
      tokenizer = self.detector.extractor.tokenizer
      windows = split_document(text, tokenizer, self.max_tokens)
      keys = [(hashlib.sha256(w.text.encode()).hexdigest(), highlights)
              for w in windows]
      found, missing = {}, {}
      hits = 0
      for key, window in zip(keys, windows):
        if key in self.cache:
          hits += 1
          found[key] = self.cache[key]
          self.cache.move_to_end(key)
        elif key not in missing:
          missing[key] = window.text
      if missing:
        detections = self.detector(list(missing.values()),
                                   return_highlights=highlights,
                                   batch_size=self.batch_size)
        for key, detection in zip(missing, detections):
          found[key] = detection
          if self.cache_size:
            self.cache[key] = detection
            while len(self.cache) > self.cache_size:
              self.cache.popitem(last=False)
      results = []
      for key, window in zip(keys, windows):
        detection = found[key]
        offsets = tokenizer(window.text, return_offsets_mapping=True)[
            "offset_mapping"] if highlights else None
        features = []
        for f in detection.features[:top_features]:
          item = {"id": f.id, "importance": f.importance,
                  "description": f.description}
          if highlights:
            item["spans"] = [
                {"start": window.start + offsets[t][0],
                 "end": window.start + offsets[t][1], "activation": a}
                for t, a in (f.highlights or []) if offsets[t][0] < offsets[t][1]
            ]
          features.append(item)
        results.append({"start": window.start, "end": window.end,
                        "tokens": window.tokens, "label": detection.label,
                        "p_ai": detection.p_ai, "logit": detection.logit,
                        "bias": detection.bias,
                        "active_features": len(detection.features),
                        "features": features})
      return {"model": self.detector.identity, "characters": len(text),
              "max_batch_size": self.batch_size,
              "offset_unit": "Unicode code points",
              "window_count": len(windows), "cache_hits": hits,
              "computed_windows": len(missing), "windows": results,
              "scope": "Independent text windows; no document probability."}
