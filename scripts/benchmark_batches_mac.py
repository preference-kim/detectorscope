"""Compare bounded CPU/MPS batches on the same four document chunks."""

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
import torch
from transformers import AutoTokenizer

from benchmark_mac import ROOT, serialize, sync, system_memory, write_json
from detectorscope.documents import split_document
from detectorscope.inference import DEFAULT_MODEL, DEFAULT_REVISION, OptimizedDetector


def chunks(tokenizer):
  raw = (ROOT / "demo/examples/prdb_gpt4o_vs_human_pairs.txt").read_text()
  matches = re.findall(r"={20,}\n(PAIR[^\n]+)\n={20,}\n(.*?)(?=\n={20,}|\Z)",
                       raw, re.DOTALL)
  result = []
  for category in ("GPT-4o", "HUMAN"):
    text = "\n\n".join(text.strip() for title, text in matches
                       if f"| {category} |" in title)
    for i, window in enumerate(split_document(text, tokenizer)[:2]):
      if window.tokens != 1024:
        raise ValueError("Expected four distinct, full 1,024-token chunks.")
      result.append({"id": f"{category.lower()}-{i}", "text": window.text,
                     "tokens": window.tokens,
                     "sha256": hashlib.sha256(window.text.encode()).hexdigest()})
  if len(result) != 4 or len({s["sha256"] for s in result}) != 4:
    raise ValueError("Benchmark requires four distinct chunks.")
  return result


def run(args):
  torch.set_num_threads(8)
  if args.device == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError("MPS is unavailable in this process.")
  snapshot = huggingface_hub.snapshot_download(DEFAULT_MODEL,
      revision=DEFAULT_REVISION, local_files_only=True)
  tokenizer = AutoTokenizer.from_pretrained(Path(snapshot) / "extractor",
                                          local_files_only=True)
  samples = chunks(tokenizer)
  path = Path(args.output) / f"{args.device}-b{args.batch_size}.json"
  info = {"device": args.device, "batch_size": args.batch_size,
          "model": DEFAULT_MODEL, "revision": DEFAULT_REVISION, "dtype": "float32",
          "cpu_threads": torch.get_num_threads(),
          "cpu_interop_threads": torch.get_num_interop_threads(),
          "started_utc": datetime.now(timezone.utc).isoformat(),
          "platform": platform.platform(), "samples": samples,
          "versions": {p: importlib.metadata.version(p) for p in
                       ("torch", "transformers", "safetensors")},
          "runtime_source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in ("detectorscope/streaming.py", "detectorscope/inference.py",
                           "detectorscope/documents.py", "scripts/benchmark_batches_mac.py")},
          "system_memory_start": system_memory(), "status": "loading",
          "timings": [], "outputs": {}, "max_mps_driver_bytes": 0}
  write_json(path, info)
  try:
    start = time.perf_counter()
    detector = OptimizedDetector.from_pretrained(snapshot, device=args.device,
        torch_dtype="float32", local_files_only=True)
    sync(args.device)
    info["load_seconds"] = time.perf_counter() - start
    for repeat in range(-1, args.repeats):
      order = list(samples)
      if repeat >= 0:
        random.Random(42 + repeat).shuffle(order)
      profiles, outputs = [], {}
      sync(args.device)
      start = time.perf_counter()
      for i in range(0, len(order), args.batch_size):
        batch = order[i:i + args.batch_size]
        results = detector([s["text"] for s in batch], batch_size=args.batch_size)
        profiles.append(detector.extractor.profile)
        outputs.update({s["id"]: r for s, r in zip(batch, results)})
      sync(args.device)
      elapsed = time.perf_counter() - start
      info["max_mps_driver_bytes"] = max(info["max_mps_driver_bytes"],
          *(p["max_mps_driver_bytes"] for p in profiles))
      row = {"repeat": repeat, "seconds": elapsed,
             "seconds_per_chunk": elapsed / len(samples),
             "chunks_per_second": len(samples) / elapsed,
             "actual_batch_sizes": [p["batch_size"] for p in profiles],
             "blocks_loaded": sum(len(p["stages"]) for p in profiles)}
      if repeat < 0:
        info["warmup"] = row
      else:
        info["timings"].append(row)
        info["outputs"] = {key: serialize(value) for key, value in outputs.items()}
      info["status"] = "running"
      info["profiles_last_workload"] = profiles
      write_json(path, info)
      print(json.dumps({"device": args.device, "batch_size": args.batch_size, **row}), flush=True)
    median = statistics.median(t["seconds"] for t in info["timings"])
    info.update(status="complete", median_workload_seconds=median,
                median_seconds_per_chunk=median / len(samples),
                chunks_per_second=len(samples) / median)
  except RuntimeError as error:
    if args.device != "mps" or "MPS backend out of memory" not in str(error):
      raise
    info.update(status="allocator_limit", error=str(error))
    if not info["max_mps_driver_bytes"]:
      info["max_mps_driver_bytes"] = None
      info["memory_sampling_note"] = "Forward failed before a completed memory profile was returned."
  finally:
    info["peak_process_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    info["system_memory_end"] = system_memory()
    write_json(path, info)


def compare(args):
  runs = {p.stem: json.loads(p.read_text()) for p in Path(args.output).glob("*-b*.json")}
  complete = {k: v for k, v in runs.items() if v["status"] == "complete"}
  reference = complete["cpu-b1"]
  for run in complete.values():
    for key in ("model", "revision", "dtype", "cpu_threads", "cpu_interop_threads",
                "runtime_source_sha256", "samples"):
      if run[key] != reference[key]:
        raise ValueError(f"Unmatched configuration: {key}")
  summary = {"runs": {}, "numerical_comparisons": {}}
  for name, run in runs.items():
    row = {k: run[k] for k in ("status", "peak_process_rss_gib", "max_mps_driver_bytes")}
    if run["status"] == "complete":
      row.update({k: run[k] for k in ("median_workload_seconds", "median_seconds_per_chunk",
                                     "chunks_per_second")})
      serial = complete[f"{run['device']}-b1"]
      row["speedup_vs_same_device_batch1"] = serial["median_workload_seconds"] / run["median_workload_seconds"]
    summary["runs"][name] = row
  for name, run in complete.items():
    pairs = [(reference["outputs"][key], out) for key, out in run["outputs"].items()]
    summary["numerical_comparisons"][f"cpu-b1 -> {name}"] = {
        "labels_equal": all(a["label"] == b["label"] for a, b in pairs),
        "active_ids_equal": all(a["features"].keys() == b["features"].keys() for a, b in pairs),
        "max_probability_delta": max(abs(a["p_ai"] - b["p_ai"]) for a, b in pairs),
        "max_logit_delta": max(abs(a["logit"] - b["logit"]) for a, b in pairs),
        "max_feature_delta": max(abs(a["features"].get(k, 0) - b["features"].get(k, 0))
            for a, b in pairs for k in a["features"].keys() | b["features"].keys()),
    }
  write_json(Path(args.output) / "comparison.json", summary)
  print(json.dumps(summary, indent=2))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("command", choices=["suite", "run", "compare"])
  parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
  parser.add_argument("--batch-size", type=int, choices=[1, 2, 4], default=4)
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--output", default="outputs/mac-batches")
  args = parser.parse_args()
  if args.command == "run":
    run(args)
  elif args.command == "compare":
    compare(args)
  else:
    Path(args.output).mkdir(parents=True, exist_ok=True)
    for device, size in [("cpu", 1), ("mps", 1), ("mps", 2), ("cpu", 2),
                         ("cpu", 4), ("mps", 4)]:
      with (Path(args.output) / f"{device}-b{size}.log").open("w") as log:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "run",
                        "--device", device, "--batch-size", str(size),
                        "--repeats", str(args.repeats), "--output", args.output],
                       stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
    compare(args)


if __name__ == "__main__":
  main()
