"""Compare Gemma and fused FP32 normalization in shuffled pairs on one MPS runner.

Pause the document worker first. This complements the fresh-process comparison
in benchmark_prefetch_mac.py; neither benchmark controls macOS file caching.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import resource
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_mac import cases, serialize
from benchmark_prefetch_mac import check_outputs, disk_read_bytes, save
from profile_mac_modules import profiling_cases
from detectorscope.inference import OptimizedDetector
import torch


def summarize(report):
  summary = {}
  for case, lengths in report["sample_tokens"].items():
    rows = [r for r in report["rows"] if r["case"] == case and r["repeat"] >= 0]
    values = {}
    for enabled in (False, True):
      selected = [r for r in rows if r["fused_norm"] == enabled]
      tokens = sum(lengths) * len(selected)
      values[str(enabled)] = {
          "p50_seconds": statistics.median(r["seconds"] for r in selected),
          "input_tps": tokens / sum(r["seconds"] for r in selected),
          "backbone_prefill_tps": tokens / sum(r["profile"]["forward_seconds"] for r in selected),
          "compute_p50_seconds": statistics.median(
              sum(s["compute_seconds"] for s in r["profile"]["stages"]) for r in selected),
          "disk_read_gib_p50": statistics.median(r["process_disk_read_bytes"] for r in selected) / 2**30,
      }
    pairs = {repeat: {r["fused_norm"]: r["seconds"] for r in rows if r["repeat"] == repeat}
             for repeat in range(report["repeats"])}
    savings = [pair[False] - pair[True] for pair in pairs.values()]
    values["paired_seconds_saved"] = savings
    values["paired_median_seconds_saved"] = statistics.median(savings)
    summary[case] = values
  return summary


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True, help="New JSON output file")
  parser.add_argument("--repeats", type=int, default=8)
  args = parser.parse_args()
  if args.repeats < 1:
    parser.error("--repeats must be positive")
  path = args.output.resolve()
  if path.exists():
    raise FileExistsError(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.set_num_threads(8)
  detector = OptimizedDetector.from_pretrained(device="mps", torch_dtype="float32", local_files_only=True)
  fixtures = profiling_cases(detector.extractor.tokenizer)
  short = next(s["text"] for s in cases(detector.extractor.tokenizer)[0] if s["id"] == "gpt-4o-256")
  fixtures = {"single_256": [short], "mixed_1024_420": fixtures["pair_mixed_1024_420"],
              "equal_1024": fixtures["pair_equal_1024"]}
  report = {"started_utc": datetime.now(timezone.utc).isoformat(), "repeats": args.repeats,
            "model": detector.identity, "cpu_threads": 8,
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in ("detectorscope/streaming.py", "detectorscope/inference.py",
                             "scripts/benchmark_norm_mac.py")},
            "sample_tokens": {case: [len(detector.extractor.tokenizer.encode(t)) for t in texts]
                              for case, texts in fixtures.items()},
            "rows": [], "highlights": {}, "status": "running"}
  try:
    for repeat in range(-1, args.repeats):
      order = list(fixtures)
      random.Random(123 + repeat).shuffle(order)
      for case in order:
        modes = [False, True]
        random.Random(500 + repeat * 10 + list(fixtures).index(case)).shuffle(modes)
        for enabled in modes:
          detector.extractor.fused_norm = enabled
          torch.mps.synchronize()
          disk_before = disk_read_bytes()
          start = time.perf_counter()
          result = detector(fixtures[case], batch_size=len(fixtures[case]))
          torch.mps.synchronize()
          seconds = time.perf_counter() - start
          profile = detector.extractor.profile
          assert profile["max_cpu_prefetch_blocks"] == profile["max_gpu_blocks"] == 1
          assert detector.extractor.reader._buffer is None
          report["rows"].append({"case": case, "repeat": repeat, "fused_norm": enabled,
              "seconds": seconds, "process_disk_read_bytes": disk_read_bytes() - disk_before,
              "outputs": [serialize(x) for x in result], "profile": profile})
          save(path, report)
          print(json.dumps({"case": case, "repeat": repeat, "fused_norm": enabled,
                            "seconds": seconds}), flush=True)
    maxima = {}
    for case, texts in fixtures.items():
      reference = next(r["outputs"] for r in report["rows"]
                       if r["case"] == case and not r["fused_norm"])
      for row in report["rows"]:
        if row["case"] == case:
          check_outputs(reference, row["outputs"], maxima, 1e-4, 1e-3)
      highlights = []
      for enabled in (False, True):
        detector.extractor.fused_norm = enabled
        highlights.append([serialize(x, True) for x in
            detector(texts, batch_size=len(texts), return_highlights=True)])
      check_outputs(*highlights, maxima, 1e-4, 1e-3)
      report["highlights"][case] = highlights
    report["summary"] = summarize(report)
    report["max_absolute_differences"] = maxima
    report["status"] = "complete"
  finally:
    report["peak_process_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    save(path, report)
  print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
  main()
