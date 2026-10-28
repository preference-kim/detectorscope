"""Joint batch-size/input-length sweep of the bounded FP32 detector.

Each cell runs in a fresh process. One measured request is exactly one forward
of BS independent ISL-token chunks, so its latency is not a divided document
workload time. No full backbone is loaded; no decode/output tokens are produced.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import resource
import subprocess
import sys
import time

import torch

from benchmark_mac import ROOT, cases, serialize, sync, system_memory, write_json
from benchmark_mlp_padding_mac import compare_outputs
from detectorscope.inference import OptimizedDetector


def source_hashes():
  return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
      "detectorscope/streaming.py", "detectorscope/inference.py",
      "scripts/benchmark_grid_mac.py")}


def cell(args):
  path = Path(args.output) / f"{args.device}-b{args.batch_size}-s{args.chunk_length}.json"
  if path.exists():
    raise FileExistsError(f"Refusing to overwrite {path}")
  torch.set_num_threads(8)
  info = {"status": "loading", "started_utc": datetime.now(timezone.utc).isoformat(),
          "device": args.device, "batch_size": args.batch_size,
          "input_sequence_length": args.chunk_length, "output_sequence_length": 0,
          "precision": "float32", "repeats": args.repeats, "warmup_requests": 1,
          "cpu_threads": torch.get_num_threads(),
          "cpu_interop_threads": torch.get_num_interop_threads(),
          "mps_allocator_budget_bytes": 2 * 2**30 if args.device == "mps" else None,
          "sources": source_hashes(),
          "versions": {name: importlib.metadata.version(name) for name in
                       ("torch", "transformers", "safetensors")},
          "system_memory_start": system_memory(), "requests": [],
          "max_mps_driver_bytes_at_block_completion": 0,
          "timing_scope": "local detector call; tokenization, weight streaming, backbone, SAE, scoring; no HTTP, startup or highlights",
          "backbone_scope": "StreamingExtractor forward wall time, including preparation and weight streaming; 23 blocks; excludes SAE and classifier"}
  write_json(path, info)
  try:
    detector = OptimizedDetector.from_pretrained(device=args.device, torch_dtype="float32",
                                                local_files_only=True)
    info["model"] = detector.identity
    samples = [x for x in cases(detector.extractor.tokenizer)[0]
               if x["tokens"] == args.chunk_length]
    if len(samples) != 2:
      raise ValueError("Expected the same GPT-4o/human prefixes as the historical benchmark.")
    info["samples"] = samples
    info["batch_fill"] = "alternate the two exact public-text prefixes; repeat them for BS=4; no result cache"
    info["status"] = "running"
    write_json(path, info)
    for repeat in range(-1, args.repeats):
      selected = [samples[(repeat + rank) % len(samples)] for rank in range(args.batch_size)]
      sync(args.device)
      started = time.perf_counter()
      values = detector([x["text"] for x in selected], batch_size=args.batch_size)
      sync(args.device)
      elapsed = time.perf_counter() - started
      profile = detector.extractor.profile
      if (profile["batch_size"] != args.batch_size
          or profile["sequence_length"] != args.chunk_length
          or profile["real_tokens"] != args.batch_size * args.chunk_length
          or profile["max_cpu_prefetch_blocks"] != 1
          or profile["max_gpu_blocks"] != int(args.device == "mps")
          or len(profile["stages"]) != 23):
        raise RuntimeError("Requested batch/sequence shape or bounded pipeline was not used.")
      row = {"repeat": repeat, "sample_ids": [x["id"] for x in selected],
             "input_tokens": profile["real_tokens"], "e2e_seconds": elapsed,
             "backbone_forward_seconds": profile["forward_seconds"],
             "stages_seconds": {name: sum(s[name + "_seconds"] for s in profile["stages"])
                                for name in ("load", "wait", "transfer", "compute", "release")},
             "outputs": [serialize(x) for x in values]}
      info["max_mps_driver_bytes_at_block_completion"] = max(
          info["max_mps_driver_bytes_at_block_completion"], profile["max_mps_driver_bytes"])
      if repeat < 0:
        info["warmup"] = row
      else:
        info["requests"].append(row)
      write_json(path, info)
    import statistics
    requests = info["requests"]
    input_tokens = sum(x["input_tokens"] for x in requests)
    info["metrics"] = {
        "e2e_latency_p50_seconds": statistics.median(x["e2e_seconds"] for x in requests),
        "e2e_latency_min_seconds": min(x["e2e_seconds"] for x in requests),
        "e2e_latency_max_seconds": max(x["e2e_seconds"] for x in requests),
        "input_token_throughput": input_tokens / sum(x["e2e_seconds"] for x in requests),
        "request_throughput": len(requests) / sum(x["e2e_seconds"] for x in requests),
        "backbone_prefill_token_throughput": input_tokens / sum(x["backbone_forward_seconds"] for x in requests),
        "interactivity_tokens_per_second_per_user": None,
        "interactivity_note": "Not applicable: the detector produces no decoded token stream.",
    }
    info["status"] = "complete"
  except RuntimeError as error:
    if args.device == "mps" and "out of memory" in str(error).lower():
      info.update(status="oom", error=str(error))
    else:
      info.update(status="error", error=repr(error))
      raise
  finally:
    info["peak_process_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    info["system_memory_end"] = system_memory()
    info["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(path, info)
  print(json.dumps({k: info[k] for k in
                    ("device", "batch_size", "input_sequence_length", "status")}
                   | {"metrics": info.get("metrics")}), flush=True)


def summarize(output):
  runs = [json.loads(path.read_text()) for path in sorted(output.glob("*-b*-s*.json"))]
  expected = {(device, batch, length) for device in ("cpu", "mps")
              for batch in (1, 2, 4) for length in (256, 512, 1024)}
  found = {(x["device"], x["batch_size"], x["input_sequence_length"]) for x in runs}
  if found != expected:
    raise ValueError("Joint sweep is incomplete.")
  first = runs[0]
  for run in runs:
    for field in ("sources", "versions", "precision", "cpu_threads", "repeats"):
      if run[field] != first[field]:
        raise ValueError(f"Unmatched sweep setting: {field}")
  comparisons = []
  for batch in (1, 2, 4):
    for length in (256, 512, 1024):
      cpu = next(x for x in runs if x["device"] == "cpu" and x["batch_size"] == batch
                 and x["input_sequence_length"] == length)
      mps = next(x for x in runs if x["device"] == "mps" and x["batch_size"] == batch
                 and x["input_sequence_length"] == length)
      if mps["status"] != "complete" or cpu["status"] != "complete":
        continue
      if cpu["samples"] != mps["samples"]:
        raise ValueError("CPU and MPS inputs differ.")
      rows = []
      for a, b in zip(cpu["requests"], mps["requests"], strict=True):
        if a["sample_ids"] != b["sample_ids"]:
          raise ValueError("CPU and MPS request order differs.")
        rows.append(compare_outputs(a["outputs"], b["outputs"]))
      comparisons.append({"batch_size": batch, "input_sequence_length": length,
                          "all_repetitions_strict_sample_agreement": all(
                              x["strict_sample_agreement"] for x in rows),
                          "repetitions": rows})
  summary = {"status": "complete", "runs": runs, "cpu_mps_comparisons": comparisons,
             "interactivity": "N/A: no generated output tokens", "sources": first["sources"]}
  write_json(output / "summary.json", summary)
  if not all(x["all_repetitions_strict_sample_agreement"] for x in comparisons):
    raise RuntimeError("CPU/MPS numerical comparison failed; inspect summary.json.")
  return summary


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("command", choices=["suite", "cell", "summarize"])
  parser.add_argument("--output", type=Path, default=ROOT / "outputs/mac-grid-20261029")
  parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
  parser.add_argument("--batch-size", type=int, choices=[1, 2, 4], default=2)
  parser.add_argument("--chunk-length", type=int, choices=[256, 512, 1024], default=1024)
  parser.add_argument("--repeats", type=int, default=6)
  args = parser.parse_args()
  if args.repeats < 2:
    parser.error("At least two repetitions are required.")
  if args.command == "cell":
    cell(args)
    return
  if args.command == "summarize":
    summarize(args.output)
    return
  args.output.mkdir(parents=True, exist_ok=True)
  shapes = [(batch, length) for batch in (1, 2, 4) for length in (256, 512, 1024)]
  random.Random(20261029).shuffle(shapes)
  for index, (batch, length) in enumerate(shapes):
    devices = ("cpu", "mps") if index % 2 == 0 else ("mps", "cpu")
    for device in devices:
      stem = f"{device}-b{batch}-s{length}"
      existing = args.output / f"{stem}.json"
      if existing.exists():
        previous = json.loads(existing.read_text())
        if (previous["status"] in ("complete", "oom")
            and previous["sources"] == source_hashes()
            and previous["repeats"] == args.repeats):
          print(json.dumps({"cell": stem, "status": "reused"}), flush=True)
          continue
        raise RuntimeError(f"Incomplete or incompatible {existing}; preserve it and use a fresh output directory.")
      print(json.dumps({"cell": stem, "status": "starting"}), flush=True)
      with (args.output / f"{stem}.log").open("w") as log:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "cell",
                        "--output", str(args.output), "--device", device,
                        "--batch-size", str(batch), "--chunk-length", str(length),
                        "--repeats", str(args.repeats)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
      done = json.loads(existing.read_text())
      print(json.dumps({"cell": stem, "status": done["status"],
                        "metrics": done.get("metrics")}), flush=True)
  summary = summarize(args.output)
  print(json.dumps({"complete": len(summary["runs"]),
                    "validated_pairs": len(summary["cpu_mps_comparisons"])}), flush=True)


if __name__ == "__main__":
  main()
