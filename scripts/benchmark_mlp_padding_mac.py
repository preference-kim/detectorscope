"""Compare padded and valid-token-only computation without a full backbone.

Both variants use the same right-padded SDPA path, FP32 weights, 2 GiB MPS
allocator budget, and one-current/one-prefetched-block streaming pipeline.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import resource
import statistics
import time

import psutil
import torch

from benchmark_mac import ROOT, serialize, sync, system_memory, write_json
from profile_mac_modules import profiling_cases
from detectorscope.inference import DEFAULT_MODEL, DEFAULT_REVISION, OptimizedDetector
from detectorscope.streaming import _UnpaddedAttention, _UnpaddedMLP


def compare_outputs(before, after):
  result = {"labels_equal": True, "active_ids_equal": True,
            "max_logit_abs_delta": 0., "max_probability_abs_delta": 0.,
            "max_feature_abs_delta": 0.}
  if len(before) != len(after):
    raise ValueError("Output counts differ.")
  for a, b in zip(before, after):
    result["labels_equal"] &= a["label"] == b["label"]
    result["active_ids_equal"] &= a["features"].keys() == b["features"].keys()
    result["max_logit_abs_delta"] = max(result["max_logit_abs_delta"],
                                         abs(a["logit"] - b["logit"]))
    result["max_probability_abs_delta"] = max(result["max_probability_abs_delta"],
                                               abs(a["p_ai"] - b["p_ai"]))
    result["max_feature_abs_delta"] = max(result["max_feature_abs_delta"],
        max((abs(a["features"].get(k, 0) - b["features"].get(k, 0))
             for k in a["features"].keys() | b["features"].keys()), default=0.))
  result["strict_sample_agreement"] = (result["labels_equal"]
      and result["active_ids_equal"] and result["max_logit_abs_delta"] <= 1e-4
      and result["max_feature_abs_delta"] <= 1e-4)
  return result


def compare_highlights(before, after):
  results = []
  for a, b in zip(before, after):
    x = {(fid, token): value for fid, values in a["highlights"].items()
         for token, value in values}
    y = {(fid, token): value for fid, values in b["highlights"].items()
         for token, value in values}
    results.append({"tokens_equal": a["tokens"] == b["tokens"],
                    "positions_equal": x.keys() == y.keys(),
                    "positions_changed": len(x.keys() ^ y.keys()),
                    "max_activation_abs_delta": max((abs(x.get(k, 0) - y.get(k, 0))
                        for k in x.keys() | y.keys()), default=0.)})
  return results


@torch.inference_mode()
def microbenchmark(detector, cases, repeats, device):
  layer, size, _ = detector.extractor._load_layer(0)
  layer.to(device)
  mlp = layer.mlp
  results = {}
  torch.manual_seed(42)
  for case, texts in cases.items():
    mask = detector.extractor.tokenizer(texts, padding=True, return_tensors="pt")["attention_mask"]
    indices = mask.flatten().nonzero().flatten().to(device)
    hidden = torch.randn(*mask.shape, detector.extractor.config.hidden_size, device=device)
    compact = _UnpaddedMLP(mlp, indices) if len(indices) < mask.numel() else mlp
    functions = {"padded": mlp, "compact": compact}
    expected = mlp(hidden).flatten(0, 1).index_select(0, indices)
    actual = compact(hidden).flatten(0, 1).index_select(0, indices)
    error = (expected - actual).abs().max().item()
    del expected, actual
    for function in functions.values():
      for _ in range(2):
        function(hidden)
    rows = []
    for repeat in range(repeats):
      order = list(functions)
      random.Random(100 + repeat).shuffle(order)
      for variant in order:
        sync(device)
        started = time.perf_counter()
        # Enough work to amortize the one host synchronization per measurement.
        for _ in range(5):
          functions[variant](hidden)
        sync(device)
        rows.append({"repeat": repeat, "variant": variant,
                     "milliseconds": (time.perf_counter() - started) * 1000 / 5})
    results[case] = {"rows": rows, "real_tokens": int(mask.sum()),
                     "padded_tokens": mask.numel(), "valid_max_abs_delta": error,
                     "median_ms": {variant: statistics.median(r["milliseconds"]
                         for r in rows if r["variant"] == variant) for variant in functions}}
    del hidden, indices, compact, functions
  del mlp, layer
  detector.extractor._release_cache()
  return {"layer": 0, "layer_weight_bytes": size, "cases": results}


@torch.inference_mode()
def attention_microbenchmark(detector, cases, repeats, device):
  layer, size, _ = detector.extractor._load_layer(0)
  layer.to(device)
  results = {}
  torch.manual_seed(42)
  for case, texts in cases.items():
    mask = detector.extractor.tokenizer(texts, padding=True, return_tensors="pt")["attention_mask"]
    spans = [(0, int(row.sum())) for row in mask]
    hidden = torch.randn(*mask.shape, detector.extractor.config.hidden_size, device=device)
    positions = torch.arange(mask.shape[1], device=device)[None, :]
    embeddings = detector.extractor.rotary(hidden, positions)
    compact = _UnpaddedAttention(layer.self_attn, spans) if int(mask.sum()) < mask.numel() else layer.self_attn
    def padded():
      return layer.self_attn(hidden_states=hidden, position_embeddings=embeddings, attention_mask=None)[0]
    def unpadded():
      return compact(hidden_states=hidden, position_embeddings=embeddings, attention_mask=None)[0]
    valid = mask.bool().to(device)
    error = (padded()[valid] - unpadded()[valid]).abs().max().item()
    functions = {"padded": padded, "compact": unpadded}
    for function in functions.values():
      for _ in range(2):
        function()
    rows = []
    for repeat in range(repeats):
      order = list(functions)
      random.Random(100 + repeat).shuffle(order)
      for variant in order:
        sync(device)
        started = time.perf_counter()
        for _ in range(5):
          functions[variant]()
        sync(device)
        rows.append({"repeat": repeat, "variant": variant,
                     "milliseconds": (time.perf_counter() - started) * 1000 / 5})
    results[case] = {"rows": rows, "real_tokens": int(mask.sum()),
                     "padded_tokens": mask.numel(), "valid_max_abs_delta": error,
                     "median_ms": {variant: statistics.median(r["milliseconds"]
                         for r in rows if r["variant"] == variant) for variant in functions}}
    del hidden, compact, functions, embeddings, positions, valid
  del layer
  detector.extractor._release_cache()
  return {"layer": 0, "layer_weight_bytes": size, "cases": results}


def configure(detector, target, variant):
  detector.extractor.compact_mlp = target != "mlp" or variant == "compact"
  detector.extractor.compact_attention = target == "attention" and variant == "compact"


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", default="outputs/mac-mlp-padding")
  parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument("--micro-repeats", type=int, default=7)
  parser.add_argument("--target", choices=["mlp", "attention"], default="mlp")
  parser.add_argument("--micro-only", action="store_true")
  parser.add_argument("--cases", nargs="+", default=["single_1024", "pair_equal_1024",
                                                     "pair_mixed_1024_420"])
  args = parser.parse_args()
  if min(args.repeats, args.micro_repeats) < 1:
    parser.error("Repeat counts must be positive.")
  output = Path(args.output) / "results.json"
  if output.exists():
    parser.error("Choose a fresh output directory to preserve previous evidence.")
  torch.set_num_threads(8)
  detector = OptimizedDetector.from_pretrained(device=args.device, local_files_only=True)
  available = profiling_cases(detector.extractor.tokenizer)
  available["pair_mixed_1024_1"] = [available["single_1024"][0], ""]
  for length in (900, 1023):
    tokenizer = detector.extractor.tokenizer
    text = tokenizer.decode(tokenizer.encode(available["single_1024"][0])[1:length])
    if len(tokenizer.encode(text)) != length:
      raise RuntimeError("A near-equal-length fixture did not round-trip.")
    available[f"pair_mixed_1024_{length}"] = [available["single_1024"][0], text]
  cases = {name: available[name] for name in args.cases}
  info = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
          "model": DEFAULT_MODEL, "revision": DEFAULT_REVISION,
          "device": args.device, "dtype": "float32", "attention": "sdpa",
          "platform": platform.platform(), "cpu_threads": torch.get_num_threads(),
          "cpu_interop_threads": torch.get_num_interop_threads(),
          "mps_allocator_budget_bytes": 2 * 2**30 if args.device == "mps" else None,
          "versions": {name: importlib.metadata.version(name) for name in
                       ("torch", "transformers", "huggingface_hub")},
          "sources": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in ("detectorscope/streaming.py", "detectorscope/inference.py",
                           "scripts/benchmark_mlp_padding_mac.py")},
          "cases": cases, "target": args.target, "repeats": args.repeats, "micro_repeats": args.micro_repeats,
          "system_memory_start": system_memory(), "runs": [], "validation": {}}
  write_json(output, info)
  measure = microbenchmark if args.target == "mlp" else attention_microbenchmark
  info["microbenchmark"] = measure(detector, cases, args.micro_repeats, args.device)
  write_json(output, info)
  print(json.dumps({"microbenchmark": info["microbenchmark"]}), flush=True)
  if args.micro_only:
    info["status"] = "complete"
    write_json(output, info)
    return
  tasks = [(case, variant) for case in cases for variant in ("padded", "compact")]
  for repeat in range(-1, args.repeats):
    order = list(tasks)
    random.Random(42 + repeat).shuffle(order)
    for case, variant in order:
      configure(detector, args.target, variant)
      sync(args.device)
      started = time.perf_counter()
      values = detector(cases[case], batch_size=len(cases[case]))
      sync(args.device)
      elapsed = time.perf_counter() - started
      row = {"repeat": repeat, "case": case, "variant": variant,
             "seconds": elapsed, "outputs": [serialize(x) for x in values],
             "pipeline": detector.extractor.profile,
             "process_rss_gib_after": psutil.Process().memory_info().rss / 2**30}
      info["runs"].append(row)
      write_json(output, info)
      print(json.dumps({key: row[key] for key in
                        ("repeat", "case", "variant", "seconds")}), flush=True)
  for case, texts in cases.items():
    outputs = {}
    for variant in ("padded", "compact"):
      configure(detector, args.target, variant)
      outputs[variant] = [serialize(x, highlights=True)
          for x in detector(texts, batch_size=len(texts), return_highlights=True)]
    baseline = next(r["outputs"] for r in info["runs"]
                    if r["case"] == case and r["variant"] == "padded")
    comparisons = [compare_outputs(baseline, r["outputs"]) for r in info["runs"]
                   if r["case"] == case and r["variant"] == "compact"]
    info["validation"][case] = {
        "scores_all_repetitions": comparisons,
        "highlight_scores": compare_outputs(outputs["padded"], outputs["compact"]),
        "highlights": compare_highlights(outputs["padded"], outputs["compact"]),
        "outputs": outputs}
    write_json(output, info)
  info["summary"] = {}
  for case in cases:
    summary = {}
    for variant in ("padded", "compact"):
      rows = [r for r in info["runs"] if r["repeat"] >= 0
              and r["case"] == case and r["variant"] == variant]
      summary[variant] = {
          "median_seconds": statistics.median(r["seconds"] for r in rows),
          "min_seconds": min(r["seconds"] for r in rows),
          "max_seconds": max(r["seconds"] for r in rows),
          "median_block_compute_seconds": statistics.median(sum(s["compute_seconds"]
              for s in r["pipeline"]["stages"]) for r in rows),
          "max_mps_driver_bytes": max(r["pipeline"]["max_mps_driver_bytes"] for r in rows),
          "mlp_tokens": rows[0]["pipeline"]["mlp_tokens"],
          "attention_tokens": rows[0]["pipeline"]["attention_tokens"],
          "real_tokens": rows[0]["pipeline"]["real_tokens"],
          "padded_tokens": rows[0]["pipeline"]["padded_tokens"]}
    summary["latency_reduction_percent"] = (
        1 - summary["compact"]["median_seconds"] / summary["padded"]["median_seconds"]) * 100
    info["summary"][case] = summary
  info["peak_process_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
  info["system_memory_end"] = system_memory()
  info["status"] = "complete"
  write_json(output, info)
  write_json(output.with_name("summary.json"), info["summary"])
  print(json.dumps(info["summary"], indent=2), flush=True)
  if not all(v["highlight_scores"]["strict_sample_agreement"]
      and all(c["strict_sample_agreement"] for c in v["scores_all_repetitions"])
      and all(h["tokens_equal"] and h["positions_equal"] for h in v["highlights"])
      for v in info["validation"].values()):
    raise RuntimeError("Numerical or highlight-position validation failed; inspect results.json.")


if __name__ == "__main__":
  main()
