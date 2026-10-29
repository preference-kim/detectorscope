"""Compare a recorded streaming implementation with current FP32 inference.

Pause the document worker first. Each variant/round uses a fresh process;
rounds reverse variant order. Only one compute block and one CPU prefetch block
are allowed. No full-model loader is called. Raw outputs stay in outputs/.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import random
import resource
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save(path, data):
  temporary = path.with_suffix(".partial")
  temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
  temporary.replace(path)


def disk_read_bytes():
  """macOS process disk-read accounting, separate from requested tensor bytes."""
  if sys.platform != "darwin":
    return None
  import ctypes
  import os
  class Usage(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in
        "user system pkg intr pageins wired resident footprint start exit childuser childsys "
        "childpkg childintr childpageins childelapsed diskread diskwrite".split()]
  library = ctypes.CDLL("/usr/lib/libproc.dylib")
  library.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
  usage = Usage()
  if library.proc_pid_rusage(os.getpid(), 2, ctypes.byref(usage)):
    raise RuntimeError("proc_pid_rusage failed")
  return usage.diskread


def run(args):
  import time
  import torch
  from benchmark_mac import cases, serialize, sync
  from profile_mac_modules import profiling_cases
  import detectorscope.inference as inference

  torch.set_num_threads(8)
  source = args.output / f"{args.variant}_streaming.py"
  spec = importlib.util.spec_from_file_location("prefetch_variant", source)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  inference.StreamingExtractor = module.StreamingExtractor
  if "fused_norm" not in inspect.signature(module.StreamingExtractor).parameters:
    # Current factory metadata also describes snapshots predating this option.
    # Keep their implementation intact and explicitly mark it unfused.
    class PreviousExtractor(module.StreamingExtractor):
      def __init__(self, *args, fused_norm=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fused_norm = False
    inference.StreamingExtractor = PreviousExtractor
  def forbidden(*a, **kw):
    raise AssertionError("Full-model loading is forbidden")
  inference.AuditableDetector.from_pretrained = forbidden
  detector = inference.OptimizedDetector.from_pretrained(
      device=args.device, torch_dtype="float32", local_files_only=True)
  fixtures = profiling_cases(detector.extractor.tokenizer)
  short = next(s["text"] for s in cases(detector.extractor.tokenizer)[0]
               if s["id"] == "gpt-4o-256")
  fixtures = {"single_256": [short], "single_1024": fixtures["single_1024"],
              "mixed_1024_420": fixtures["pair_mixed_1024_420"],
              "equal_1024": fixtures["pair_equal_1024"]}
  fixtures = {name: fixtures[name] for name in args.cases}
  data = {"variant": args.variant, "round": args.round, "device": args.device,
          "dtype": "float32", "cpu_threads": 8, "model": detector.identity,
          "started_utc": datetime.now(timezone.utc).isoformat(),
          "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
          "inference_sha256": hashlib.sha256((ROOT / "detectorscope/inference.py").read_bytes()).hexdigest(),
          "sample_tokens": {k: [len(detector.extractor.tokenizer.encode(t)) for t in ts]
                            for k, ts in fixtures.items()},
          "rows": [], "highlights": {}, "status": "running"}
  path = args.output / f"round{args.round}-{args.variant}.json"
  if path.exists():
    raise FileExistsError(path)
  try:
    for repeat in range(-1, args.repeats):
      order = list(fixtures)
      random.Random(83 + repeat).shuffle(order)
      for case in order:
        sync(args.device)
        disk_before = disk_read_bytes()
        start = time.perf_counter()
        result = detector(fixtures[case], batch_size=len(fixtures[case]))
        sync(args.device)
        elapsed = time.perf_counter() - start
        disk_after = disk_read_bytes()
        profile = detector.extractor.profile
        assert len(profile["stages"]) == 23
        assert profile["max_cpu_prefetch_blocks"] == 1
        assert profile["max_gpu_blocks"] == int(args.device == "mps")
        assert profile["real_tokens"] == sum(data["sample_tokens"][case])
        io = profile.get("coalesced_weight_io")
        if io:
          assert io["max_staging_bytes"] <= profile["layer_weight_bytes"]
        assert getattr(detector.extractor.reader, "_buffer", None) is None
        data["rows"].append({"case": case, "repeat": repeat, "seconds": elapsed,
                             "process_disk_read_bytes": (None if disk_before is None
                                 else disk_after - disk_before),
                             "outputs": [serialize(x) for x in result], "profile": profile})
        save(path, data)
    for case, texts in fixtures.items():
      data["highlights"][case] = [serialize(x, True) for x in
          detector(texts, batch_size=len(texts), return_highlights=True)]
    data["status"] = "complete"
  finally:
    data["peak_process_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    data["max_driver_gib"] = max((r["profile"]["max_mps_driver_bytes"] / 2**30
                                  for r in data["rows"]), default=0)
    save(path, data)


def check_outputs(expected, actual, maxima, score_atol, highlight_atol):
  """Check categorical outputs exactly; bound and report every numeric change."""
  assert len(expected) == len(actual), "Detection count changed"
  def check(kind, a, b, tolerance):
    delta = abs(a - b)
    maxima[kind] = max(maxima.get(kind, 0), delta)
    assert delta <= tolerance, f"{kind} difference {delta} exceeds {tolerance}"
  for before, after in zip(expected, actual):
    assert before["label"] == after["label"], "Label changed"
    assert before["features"].keys() == after["features"].keys(), "Feature IDs changed"
    for field in ("logit", "p_ai", "bias"):
      check(field, before[field], after[field], score_atol)
    for key in before["features"]:
      check("feature_contribution", before["features"][key], after["features"][key], score_atol)
    if "highlights" in before:
      assert before["tokens"] == after["tokens"], "Highlight tokens changed"
      assert before["highlights"].keys() == after["highlights"].keys(), "Highlight IDs changed"
      for key, values in before["highlights"].items():
        other = after["highlights"][key]
        assert [x[0] for x in values] == [x[0] for x in other], "Highlight positions changed"
        for (_, a), (_, b) in zip(values, other):
          check("highlight_activation", a, b, highlight_atol)


def summarize(output, score_atol=0.0, highlight_atol=0.0):
  runs = [json.loads(p.read_text()) for p in sorted(output.glob("round*-*.json"))]
  if not runs or any(r["status"] != "complete" for r in runs):
    raise ValueError("Incomplete comparison")
  references = {row["case"]: row["outputs"] for row in runs[0]["rows"]}
  maxima = {}
  for run in runs:
    assert run["sample_tokens"] == runs[0]["sample_tokens"]
    for row in run["rows"]:
      check_outputs(references[row["case"]], row["outputs"], maxima, score_atol, highlight_atol)
    assert run["highlights"].keys() == runs[0]["highlights"].keys()
    for case, values in run["highlights"].items():
      check_outputs(runs[0]["highlights"][case], values, maxima, score_atol, highlight_atol)
  summary = {"status": "complete", "exact_output_and_highlight_agreement": not any(maxima.values()),
             "labels_features_tokens_and_highlight_positions_match": True,
             "max_absolute_differences": maxima,
             "tolerances": {"score_atol": score_atol, "highlight_atol": highlight_atol},
             "cases": {}, "memory": {}}
  for case, lengths in runs[0]["sample_tokens"].items():
    row = {"batch_size": len(lengths), "input_tokens_per_chunk": lengths}
    for variant in ("baseline", "production"):
      calls = [r for run in runs if run["variant"] == variant
               for r in run["rows"] if r["case"] == case and r["repeat"] >= 0]
      row[variant] = {"timed_calls": len(calls),
                      "e2e_latency_p50_seconds": statistics.median(r["seconds"] for r in calls),
                      "input_token_throughput": sum(lengths) * len(calls) / sum(r["seconds"] for r in calls),
                      "backbone_prefill_token_throughput": sum(lengths) * len(calls) /
                          sum(r["profile"]["forward_seconds"] for r in calls),
                      "stage_sum_p50_seconds": {key: statistics.median(
                          sum(stage[key] for stage in r["profile"]["stages"]) for r in calls)
                          for key in ("load_seconds", "wait_seconds", "transfer_seconds",
                                      "compute_seconds", "release_seconds")},
                      "process_disk_read_bytes_p50": statistics.median(
                          r["process_disk_read_bytes"] for r in calls
                          if r.get("process_disk_read_bytes") is not None)
                          if any(r.get("process_disk_read_bytes") is not None for r in calls) else None,
                      "round_p50_seconds": [statistics.median(r["seconds"] for r in run["rows"]
                          if r["case"] == case and r["repeat"] >= 0)
                          for run in runs if run["variant"] == variant]}
    row["latency_reduction_percent"] = 100 * (1 - row["production"]["e2e_latency_p50_seconds"]
                                                   / row["baseline"]["e2e_latency_p50_seconds"])
    summary["cases"][case] = row
  for variant in ("baseline", "production"):
    selected = [r for r in runs if r["variant"] == variant]
    summary["memory"][variant] = {
        "max_run_peak_process_rss_gib": max(r["peak_process_rss_gib"] for r in selected),
        "observed_driver_max_gib": max(r["max_driver_gib"] for r in selected)}
  save(output / "summary.json", summary)
  return summary


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("command", choices=["suite", "run", "summarize"])
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--baseline-ref", default="c0106d1")
  parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument("--rounds", type=int, default=2)
  parser.add_argument("--variant", choices=["baseline", "production"])
  parser.add_argument("--round", type=int, default=0)
  parser.add_argument("--score-atol", type=float, default=0.0)
  parser.add_argument("--highlight-atol", type=float, default=0.0)
  parser.add_argument("--cases", nargs="+", default=["single_256", "single_1024", "mixed_1024_420", "equal_1024"])
  args = parser.parse_args()
  if args.score_atol < 0 or args.highlight_atol < 0:
    raise ValueError("Output tolerances must be nonnegative")
  args.output = args.output.resolve()
  if args.command == "run":
    run(args)
    return
  if args.command == "suite":
    if args.repeats < 1 or args.rounds < 1:
      raise ValueError("Positive repeat and round counts required")
    args.output.mkdir(parents=True, exist_ok=False)
    before = subprocess.check_output(["git", "show", f"{args.baseline_ref}:detectorscope/streaming.py"], cwd=ROOT)
    (args.output / "baseline_streaming.py").write_bytes(before)
    (args.output / "production_streaming.py").write_bytes((ROOT / "detectorscope/streaming.py").read_bytes())
    for round_id in range(args.rounds):
      order = ("baseline", "production") if round_id % 2 == 0 else ("production", "baseline")
      for variant in order:
        print(json.dumps({"round": round_id, "variant": variant, "device": args.device}), flush=True)
        command = [sys.executable, str(Path(__file__)), "run", "--output", str(args.output),
                   "--device", args.device, "--variant", variant, "--round", str(round_id),
                   "--repeats", str(args.repeats), "--cases", *args.cases]
        with (args.output / f"round{round_id}-{variant}.log").open("w") as log:
          subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
  print(json.dumps(summarize(args.output, args.score_atol, args.highlight_atol), indent=2))


if __name__ == "__main__":
  main()
