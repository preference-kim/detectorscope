"""Bounded layer/module profiling with explicit instrumentation-overhead checks.

MPS events measure stream-completion intervals, not hardware kernel counters.
CPU traces use record_shapes=False so the profiler does not retain tensor inputs.
No hooks, measurements or trace rows keep references to layer weights.
"""

import argparse
from collections import defaultdict
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import torch
import transformers
from transformers.models.gemma2 import modeling_gemma2

from benchmark_mac import ROOT, serialize, write_json
from benchmark_batches_mac import chunks
from detectorscope.documents import split_document
from detectorscope.inference import OptimizedDetector


def tensor_info(value):
  if isinstance(value, torch.Tensor):
    return {"shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "device": str(value.device)}
  if isinstance(value, (tuple, list)):
    return next((tensor_info(x) for x in value if isinstance(x, torch.Tensor)), None)
  return None


def profiling_cases(tokenizer):
  """Rebuild the measured inputs from the repository's public review corpus."""
  samples = chunks(tokenizer)
  first, second = samples[0]['text'], samples[2]['text']
  short = split_document(first + '\n\n' + second[:1800], tokenizer)[-1].text
  if len(tokenizer.encode(short)) != 420:
    raise RuntimeError('The mixed-length profiling fixture changed token count.')
  return {'single_1024': [first], 'pair_equal_1024': [first, second],
          'pair_mixed_1024_420': [first, short]}


class ModuleRecorder:
  def __init__(self, detector, mode):
    self.detector, self.mode = detector, mode
    self.rows, self.pending, self.active, self.handles = [], [], {}, []
    self.layer = -1

  def begin(self, layer, label, metadata):
    key = (layer, label)
    record = {"layer": layer, "module": label, **metadata}
    if self.mode == "events":
      start, end = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
      start.record()
      record["events"] = (start, end)
    else:
      record["context"] = torch.profiler.record_function(f"layer{layer}:{label}")
      record["context"].__enter__()
    record["host_start"] = time.perf_counter()
    self.active[key] = record

  def end(self, layer, label, output=None):
    record = self.active.pop((layer, label))
    record["host_call_ms"] = (time.perf_counter() - record.pop("host_start")) * 1000
    if self.mode == "events":
      record["events"][1].record()
    else:
      record.pop("context").__exit__(None, None, None)
    record["output"] = tensor_info(output)
    self.pending.append(record)

  def hook(self, module, layer, label):
    metadata = {"class": type(module).__name__}
    if hasattr(module, "weight") and module.weight is not None:
      metadata["weight_shape"] = list(module.weight.shape)

    def before(mod, args, kwargs):
      if label == "block":
        self.layer = layer
      value = args[0] if args else kwargs.get("hidden_states")
      self.begin(layer, label, {**metadata, "input": tensor_info(value)})

    def after(mod, args, kwargs, result):
      self.end(layer, label, result)

    self.handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
    self.handles.append(module.register_forward_hook(after, with_kwargs=True, always_call=True))

  def __enter__(self):
    self.original_load = self.detector.extractor._load_layer
    self.original_rope = modeling_gemma2.apply_rotary_pos_emb
    self.original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def load(index):
      layer, size, seconds = self.original_load(index)
      for name, module in layer.named_modules():
        if name == "" or name in ("self_attn", "mlp") or not list(module.children()):
          self.hook(module, index, name or "block")
      return layer, size, seconds

    def rope(*args, **kwargs):
      layer = self.layer
      self.begin(layer, "self_attn.rope", {"input": tensor_info(args[0])})
      try:
        result = self.original_rope(*args, **kwargs)
      except BaseException:
        self.end(layer, "self_attn.rope")
        raise
      self.end(layer, "self_attn.rope", result)
      return result

    def sdpa(*args, **kwargs):
      layer = self.layer
      mask = kwargs.get("attn_mask", args[3] if len(args) > 3 else None)
      self.begin(layer, "self_attn.sdpa", {
          "query": tensor_info(args[0]), "key": tensor_info(args[1]),
          "value": tensor_info(args[2]), "mask": tensor_info(mask),
          "enable_gqa": kwargs.get("enable_gqa", False),
          "is_causal": kwargs.get("is_causal", False)})
      try:
        result = self.original_sdpa(*args, **kwargs)
      except BaseException:
        self.end(layer, "self_attn.sdpa")
        raise
      self.end(layer, "self_attn.sdpa", result)
      return result

    self.detector.extractor._load_layer = load
    modeling_gemma2.apply_rotary_pos_emb = rope
    torch.nn.functional.scaled_dot_product_attention = sdpa
    self.hook(self.detector.extractor.rotary, -1, "prepare.rotary")
    for i, encoder in enumerate(self.detector.encoders):
      self.hook(encoder, -1, f"sae.{i}")
    return self

  def __exit__(self, *exception):
    self.detector.extractor._load_layer = self.original_load
    modeling_gemma2.apply_rotary_pos_emb = self.original_rope
    torch.nn.functional.scaled_dot_product_attention = self.original_sdpa
    for handle in self.handles:
      handle.remove()
    self.handles.clear()

  def collect(self):
    for record in self.pending:
      if "events" in record:
        start, end = record.pop("events")
        try:
          record["mps_interval_ms"] = start.elapsed_time(end)
        except RuntimeError as error:
          record["mps_interval_ms"] = None
          record["timing_error"] = str(error)
      self.rows.append(record)
    self.pending.clear()
    if self.active:
      raise RuntimeError(f"Unclosed measurements: {list(self.active)}")
    return self.rows


def run_forward(detector, texts, mode):
  recorder = ModuleRecorder(detector, mode) if mode != "baseline" else None
  torch.mps.synchronize()
  with recorder if recorder else nullcontext():
    started = time.perf_counter()
    result = detector(texts, batch_size=len(texts))
    torch.mps.synchronize()
    elapsed = time.perf_counter() - started
  collected = time.perf_counter()
  rows = recorder.collect() if recorder else []
  return {"mode": mode, "wall_seconds": elapsed,
          "event_collection_seconds": time.perf_counter() - collected,
          "pipeline": detector.extractor.profile, "modules": rows,
          "outputs": [serialize(d) for d in result]}


def summarize(report, output):
  summary = {"cases": {}}
  csv_rows = []
  for case in report["cases"]:
    rows = [r for r in report["runs"] if r["case"] == case]
    baseline = [r for r in rows if r["mode"] == "baseline"]
    events = [r for r in rows if r["mode"] == "events"]
    grouped = defaultdict(list)
    for run in events:
      for item in run["modules"]:
        if item.get("mps_interval_ms") is not None:
          grouped[(item["layer"], item["module"])].append(item["mps_interval_ms"])
    modules = []
    for (layer, module), values in sorted(grouped.items()):
      item = {"case": case, "layer": layer, "module": module,
              "median_mps_interval_ms": statistics.median(values),
              "min_ms": min(values), "max_ms": max(values), "samples": len(values)}
      csv_rows.append(item)
      modules.append(item)
    wall = statistics.median(r["wall_seconds"] for r in baseline)
    event_wall = statistics.median(r["wall_seconds"] for r in events)
    equality = all(r["outputs"] == baseline[0]["outputs"] for r in rows)
    summary["cases"][case] = {
        "baseline_median_seconds": wall, "event_median_seconds": event_wall,
        "event_wall_overhead_percent": (event_wall / wall - 1) * 100,
        "outputs_exactly_equal": equality,
        "invalid_event_intervals": sum("timing_error" in v for r in events for v in r["modules"]),
        # Sum within each forward first. Parents include their children: these
        # totals must not be added across overlapping module names.
        "module_totals_median_ms": {
            name: statistics.median(sum(v["mps_interval_ms"] for v in r["modules"]
                if v["module"] == name and v.get("mps_interval_ms") is not None)
                for r in events)
            for name in sorted({v["module"] for r in events for v in r["modules"]})},
        "modules": modules,
    }
  write_json(output / "summary.json", summary)
  with (output / "layer-modules.csv").open("w") as file:
    writer = csv.DictWriter(file, fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)
  return summary


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", default="outputs/mac-module-profile")
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--skip-cpu-traces", action="store_true")
  args = parser.parse_args()
  if args.repeats < 1:
    parser.error('--repeats must be positive')
  output = Path(args.output)
  if (output / 'results.json').exists():
    parser.error('results.json already exists; choose a fresh --output directory to preserve evidence')
  output.mkdir(parents=True, exist_ok=True)
  torch.set_num_threads(8)
  detector = OptimizedDetector.from_pretrained(device="mps", local_files_only=True)
  cases = profiling_cases(detector.extractor.tokenizer)
  report = {"cases": cases, "model": detector.identity, "torch_git_version": torch.version.git_version,
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "cpu_threads": torch.get_num_threads(),
            "status": "running", "warmups": [], "runs": [], "cpu_traces": [],
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in ["detectorscope/streaming.py", "detectorscope/inference.py",
                             "scripts/profile_mac_modules.py"]}}
  for case, texts in cases.items():
    for mode in ("baseline", "events"):
      row = run_forward(detector, texts, mode)
      row.update(case=case, repeat=-1)
      report["warmups"].append(row)
      write_json(output / "results.json", report)
      print(json.dumps({"case": case, "mode": mode, "warmup_seconds": row["wall_seconds"]}), flush=True)
  for repeat in range(args.repeats):
    order = [(case, mode) for case in cases for mode in ("baseline", "events")]
    random.Random(101 + repeat).shuffle(order)
    for case, mode in order:
      row = run_forward(detector, cases[case], mode)
      row.update(case=case, repeat=repeat)
      report["runs"].append(row)
      write_json(output / "results.json", report)
      print(json.dumps({"case": case, "mode": mode, "repeat": repeat,
                        "seconds": row["wall_seconds"], "intervals": len(row["modules"])}), flush=True)
  if not args.skip_cpu_traces:
    for case, texts in cases.items():
      # record_shapes would retain input references, including streamed weights.
      with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
          record_shapes=False, profile_memory=True, with_stack=False) as prof:
        row = run_forward(detector, texts, "cpu")
      prof.export_chrome_trace(str(output / f"{case}-cpu-trace.json"))
      ops = [{"operator": e.key, "count": e.count,
              "self_cpu_ms": e.self_cpu_time_total / 1000,
              "inclusive_cpu_ms": e.cpu_time_total / 1000,
              "self_cpu_memory_bytes": e.self_cpu_memory_usage}
             for e in prof.key_averages()]
      write_json(output / f"{case}-cpu-operators.json", ops)
      report["cpu_traces"].append({"case": case, **row})
      write_json(output / "results.json", report)
      print(json.dumps({"cpu_trace": case, "seconds": row["wall_seconds"],
                        "operator_types": len(ops)}), flush=True)
  report["status"] = "complete"
  write_json(output / "results.json", report)
  summary = summarize(report, output)
  print(json.dumps({k: {a: b for a, b in v.items() if a != "modules"}
                   for k, v in summary["cases"].items()}, indent=2), flush=True)


if __name__ == "__main__":
  main()
