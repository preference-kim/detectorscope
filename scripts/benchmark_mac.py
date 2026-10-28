"""Streamed inference benchmarks with comparison to saved original outputs.

Run from the checkout with .venv/bin/python scripts/benchmark_mac.py suite.
Raw timings and feature contributions are saved after each sample, so an
interrupted run retains its evidence. This is not a detection-quality benchmark.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import re
import resource
import statistics
import subprocess
import sys
import time

import huggingface_hub
import psutil
import torch
from transformers import AutoTokenizer

from detectorscope.inference import DEFAULT_MODEL, DEFAULT_REVISION, OptimizedDetector


ROOT = Path(__file__).resolve().parents[1]
MODES = {
    "cpu-stream-fp32": ("cpu", "float32", True),
    "mps-stream-fp32": ("mps", "float32", True),
    "mps-stream-fp16": ("mps", "float16", True),
}


def system_memory():
  memory, swap = psutil.virtual_memory(), psutil.swap_memory()
  return {"available_gib": memory.available / 2**30,
          "swap_used_gib": swap.used / 2**30}


def write_json(path, data):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  candidate = path.with_suffix(".partial")
  candidate.write_text(json.dumps(data, indent=2, allow_nan=False))
  candidate.replace(path)


def cases(tokenizer):
  corpus = ROOT / "demo/examples/prdb_gpt4o_vs_human_pairs.txt"
  raw = corpus.read_text()
  matches = re.findall(r"={20,}\n(PAIR[^\n]+)\n={20,}\n(.*?)(?=\n={20,}|\Z)",
                       raw, re.DOTALL)
  result = []
  for category in ("GPT-4o", "HUMAN"):
    candidates = [(title, text.strip()) for title, text in matches
                  if f"| {category} |" in title]
    # Some bundled reviews are shorter than 1,024 tokens. Concatenate reviews
    # within each category to create a sufficiently long, fixed public input.
    title = [title for title, _ in candidates]
    text = "\n\n".join(text for _, text in candidates)
    offsets = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    for size in (256, 512, 1024):
      if len(offsets) < size:
        raise RuntimeError(f"Corpus has no {category} document of {size} tokens.")
      end = offsets[size - 1][1]
      while end > 0 and len(tokenizer.encode(text[:end])) > size:
        end -= 1
      sample = text[:end]
      if len(tokenizer.encode(sample)) != size:
        raise RuntimeError("A benchmark prefix did not round-trip to its token count.")
      result.append({"id": f"{category.lower()}-{size}", "tokens": size,
                     "text": sample, "source": title,
                     "sha256": hashlib.sha256(sample.encode()).hexdigest()})
  return result, hashlib.sha256(corpus.read_bytes()).hexdigest()


def serialize(detection, highlights=False):
  result = {"logit": detection.logit, "p_ai": detection.p_ai,
            "label": detection.label, "bias": detection.bias,
            "features": {str(f.id): f.importance for f in detection.features}}
  if highlights:
    result["tokens"] = detection.tokens
    result["highlights"] = {str(f.id): f.highlights for f in detection.features}
  return result


def sync(device):
  if device == "mps":
    torch.mps.synchronize()


def run(args):
  device, dtype, compact = MODES[args.mode]
  torch.set_num_threads(args.cpu_threads)
  if device == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError("MPS unavailable: run with Metal device access.")
  snapshot = huggingface_hub.snapshot_download(DEFAULT_MODEL,
      revision=DEFAULT_REVISION, local_files_only=True)
  tokenizer = AutoTokenizer.from_pretrained(Path(snapshot) / "extractor")
  samples, corpus_hash = cases(tokenizer)
  samples = [s for s in samples if s["tokens"] in args.lengths]
  out = Path(args.output) / f"{args.mode}.json"
  info = {"mode": args.mode, "model": DEFAULT_MODEL, "revision": DEFAULT_REVISION,
          "started_utc": datetime.now(timezone.utc).isoformat(),
          "platform": platform.platform(), "python": sys.version,
          "versions": {p: importlib.metadata.version(p) for p in
                       ("torch", "transformers", "safetensors", "huggingface_hub")},
          "cpu_threads": torch.get_num_threads(),
          "cpu_interop_threads": torch.get_num_interop_threads(),
          "batch_size": 1, "attention_implementation": "sdpa", "dtype": dtype,
          "device": device, "corpus_sha256": corpus_hash,
          "runtime_source_sha256": {
              name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in ("detectorscope/streaming.py", "detectorscope/inference.py",
                           "detectorscope/model/extractor.py", "scripts/benchmark_mac.py")},
          "system_memory_start": system_memory(),
          "samples": samples, "warmups": {}, "timings": [], "outputs": {},
          "status": "loading"}
  write_json(out, info)
  start = time.perf_counter()
  detector = OptimizedDetector.from_pretrained(snapshot, device=device,
      torch_dtype=dtype, local_files_only=True)
  info["retained_per_layer"] = detector.retained_per_layer
  info["weight_loading"] = detector.identity["weight_loading"]
  info["max_mps_driver_bytes"] = 0
  sync(device)
  info["load_seconds"] = time.perf_counter() - start
  info["status"] = "running"
  print(json.dumps({"mode": args.mode, "load_seconds": info["load_seconds"]}), flush=True)

  # Warm every input shape before collecting timings. Tokenization, feature
  # extraction, scoring and explanation construction are all inside the timer.
  for sample in samples:
    sync(device)
    start = time.perf_counter()
    detector(sample["text"])
    sync(device)
    info["max_mps_driver_bytes"] = max(info["max_mps_driver_bytes"],
        detector.extractor.profile["max_mps_driver_bytes"])
    info["warmups"][sample["id"]] = time.perf_counter() - start
    write_json(out, info)
    print(json.dumps({"mode": args.mode, "stage": "warmup", "sample": sample["id"],
                      "seconds": info["warmups"][sample["id"]]}), flush=True)
  for repeat in range(args.repeats):
    order = list(samples)
    random.Random(42 + repeat).shuffle(order)
    for sample in order:
      sync(device)
      start = time.perf_counter()
      detection = detector(sample["text"])
      sync(device)
      elapsed = time.perf_counter() - start
      info["timings"].append({"id": sample["id"], "tokens": sample["tokens"],
                              "repeat": repeat, "seconds": elapsed})
      info["outputs"][sample["id"]] = serialize(detection)
      info["max_mps_driver_bytes"] = max(info["max_mps_driver_bytes"],
          detector.extractor.profile["max_mps_driver_bytes"])
      write_json(out, info)
      print(json.dumps({"mode": args.mode, "sample": sample["id"],
                        "repeat": repeat, "seconds": elapsed}), flush=True)
  # The upstream MPS highlight path calls unsupported sparse ops. Compare
  # original CPU highlights with compact CPU/MPS output, using original IDs.
  if compact or device == "cpu":
    sample = samples[0]
    info["highlight_sample"] = sample["id"]
    info["highlight_output"] = serialize(
        detector(sample["text"], return_highlights=True), highlights=True)
    info["max_mps_driver_bytes"] = max(info["max_mps_driver_bytes"],
        detector.extractor.profile["max_mps_driver_bytes"])
  info["median_seconds_by_tokens"] = {
      str(n): statistics.median(t["seconds"] for t in info["timings"]
                                 if t["tokens"] == n)
      for n in sorted({s["tokens"] for s in samples})
  }
  info["peak_process_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
  info["process_rss_gib_at_end"] = psutil.Process().memory_info().rss / 2**30
  if device == "mps":
    info["mps_driver_gib_at_end"] = torch.mps.driver_allocated_memory() / 2**30
    info["mps_tensors_gib_at_end"] = torch.mps.current_allocated_memory() / 2**30
  info["status"] = "complete"
  info["pipeline_profile_last_call"] = detector.extractor.profile
  info["system_memory_end"] = system_memory()
  write_json(out, info)
  print(json.dumps({k: info[k] for k in
                    ("mode", "median_seconds_by_tokens", "peak_process_rss_gib")}),
        flush=True)


def compare(args):
  root = Path(args.output)
  runs = {p.stem: json.loads(p.read_text()) for p in root.glob("*-fp*.json")}
  available = {k: v for k, v in runs.items() if v.get("outputs")}
  runs = {k: v for k, v in runs.items() if v["status"] == "complete"}
  comparison = {}
  for reference, candidate in [
      ("cpu-original-fp32", "cpu-stream-fp32"),
      ("cpu-stream-fp32", "mps-stream-fp32"),
      ("cpu-original-fp32", "mps-stream-fp32"),
      ("mps-original-fp32", "mps-stream-fp32"),
      ("mps-stream-fp32", "mps-stream-fp16"),
  ]:
    if reference not in available or candidate not in available:
      continue
    ref, cand = available[reference], available[candidate]
    rows = []
    for key, before in ref["outputs"].items():
      if key not in cand["outputs"]:
        continue
      after = cand["outputs"][key]
      ids = set(before["features"]) | set(after["features"])
      top_a = set(sorted(before["features"], key=lambda i: -abs(before["features"][i]))[:10])
      top_b = set(sorted(after["features"], key=lambda i: -abs(after["features"][i]))[:10])
      rows.append({"id": key, "logit_abs_delta": abs(before["logit"] - after["logit"]),
                   "p_ai_abs_delta": abs(before["p_ai"] - after["p_ai"]),
                   "label_equal": before["label"] == after["label"],
                   "active_ids_equal": set(before["features"]) == set(after["features"]),
                   "top10_overlap": len(top_a & top_b) / max(1, len(top_a | top_b)),
                   "max_feature_abs_delta": max((abs(before["features"].get(i, 0)
                       - after["features"].get(i, 0)) for i in ids), default=0)})
    pair = {"reference_status": ref["status"], "candidate_status": cand["status"],
            "samples": rows,
            "labels_equal": all(r["label_equal"] for r in rows),
            "active_ids_equal": all(r["active_ids_equal"] for r in rows),
            "max_logit_abs_delta": max(r["logit_abs_delta"] for r in rows),
            "max_p_ai_abs_delta": max(r["p_ai_abs_delta"] for r in rows),
            "max_feature_abs_delta": max(r["max_feature_abs_delta"] for r in rows)}
    pair["strict_sample_agreement"] = (
        pair["labels_equal"] and pair["active_ids_equal"]
        and pair["max_logit_abs_delta"] <= 1e-4
        and pair["max_feature_abs_delta"] <= 1e-4
    )
    if "highlight_output" in ref and "highlight_output" in cand:
      if ref["highlight_sample"] != cand["highlight_sample"]:
        raise ValueError("Highlight samples differ.")
      a, b = ref["highlight_output"], cand["highlight_output"]
      a_values = {(fid, t): v for fid, values in a["highlights"].items()
                  for t, v in values}
      b_values = {(fid, t): v for fid, values in b["highlights"].items()
                  for t, v in values}
      coords = a_values.keys() | b_values.keys()
      pair["highlights"] = {
          "sample": ref["highlight_sample"], "tokens_equal": a["tokens"] == b["tokens"],
          "positions_equal": a_values.keys() == b_values.keys(),
          "positions_in_reference": len(a_values),
          "positions_changed": len(a_values.keys() ^ b_values.keys()),
          "max_activation_abs_delta": max((abs(a_values.get(i, 0)
              - b_values.get(i, 0)) for i in coords), default=0),
      }
    comparison[f"{reference} -> {candidate}"] = pair
  report = {"latency": {k: v["median_seconds_by_tokens"] for k, v in runs.items()},
            "comparisons": comparison}
  if "cpu-stream-fp32" in runs and "mps-stream-fp32" in runs:
    cpu, mps = runs["cpu-stream-fp32"], runs["mps-stream-fp32"]
    fields = ("model", "revision", "dtype", "cpu_threads", "cpu_interop_threads",
              "batch_size", "attention_implementation", "corpus_sha256",
              "runtime_source_sha256", "samples")
    mismatches = [key for key in fields if key not in cpu or key not in mps
                  or cpu[key] != mps[key]]
    if mismatches:
      raise ValueError(f"CPU/MPS comparison settings differ or are missing: {mismatches}")
    report["matched_fp32"] = {
        "matched_fields": list(fields),
        "cpu_over_mps_latency_ratio": {
            n: seconds / mps["median_seconds_by_tokens"][n]
            for n, seconds in cpu["median_seconds_by_tokens"].items()},
    }
  write_json(root / "comparison.json", report)
  print(json.dumps(report, indent=2))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("command", choices=["run", "suite", "compare"])
  parser.add_argument("--mode", choices=MODES, default="mps-stream-fp32")
  parser.add_argument("--output", default=str(ROOT / "outputs/mac-prototype"))
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--cpu-threads", type=int, default=8)
  parser.add_argument("--suite-modes", nargs="+", choices=MODES,
                      default=["cpu-stream-fp32", "mps-stream-fp32"])
  parser.add_argument("--lengths", type=int, nargs="+", default=[256, 512, 1024])
  args = parser.parse_args()
  if args.cpu_threads < 1 or args.repeats < 1:
    parser.error("--cpu-threads and --repeats must be positive")
  if args.command == "run":
    run(args)
  elif args.command == "compare":
    compare(args)
  else:
    Path(args.output).mkdir(parents=True, exist_ok=True)
    for mode in args.suite_modes:
      command = [sys.executable, str(Path(__file__).resolve()), "run", "--mode", mode,
                 "--output", args.output, "--repeats", str(args.repeats),
                 "--cpu-threads", str(args.cpu_threads),
                 "--lengths", *map(str, args.lengths)]
      with (Path(args.output) / f"{mode}.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                       check=True, cwd=ROOT)
    compare(args)


if __name__ == "__main__":
  main()
