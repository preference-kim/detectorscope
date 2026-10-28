"""Evaluate a locally downloaded Metal Flash SDPA build on streamed inference.

Download with the official kernels loader in an isolated dependency directory;
pass the resulting build directory explicitly. No full backbone is loaded.
The sdpa-equivalent comparison disables softcapping deliberately. A separate
softcap comparison checks Gemma's eager semantics; it is not the same baseline.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import resource
import statistics
import sys
import time

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from benchmark_mac import ROOT, serialize, sync, system_memory, write_json
from benchmark_mlp_padding_mac import compare_outputs, compare_highlights
from profile_mac_modules import profiling_cases
from detectorscope.inference import OptimizedDetector


def load_local_kernel(path):
  path = Path(path).absolute()
  name = "_detectorscope_metal_flash_sdpa"
  spec = importlib.util.spec_from_file_location(name, path / "__init__.py",
                                               submodule_search_locations=[str(path)])
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


class FlashAttention:
  """Experimental adapter preserving independent sequences and GQA layout."""

  def __init__(self, kernel, mask, softcap=False):
    self.kernel, self.softcap = kernel, softcap
    lengths = mask.sum(dim=1, dtype=torch.int32)
    self.indices = mask.flatten().nonzero().flatten().to("mps")
    self.has_padding = len(self.indices) != mask.numel()
    self.cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32), lengths.cumsum(
        0, dtype=torch.int32)]).to("mps")
    self.maximum = int(lengths.max())

  def __call__(self, module, query, key, value, attention_mask, dropout=0.,
               scaling=None, **kwargs):
    if dropout or module.training:
      raise ValueError("The experiment supports inference without dropout only.")
    batch, heads, length, width = query.shape
    def pack(tensor):
      flat = tensor.transpose(1, 2).reshape(-1, tensor.shape[1], width)
      return flat.index_select(0, self.indices) if self.has_padding else flat
    q, k, v = (pack(tensor) for tensor in (query, key, value))
    window = getattr(module, "sliding_window", None)
    output = self.kernel.flash_attn_varlen_func(
        q, k, v, self.cu_seqlens, self.cu_seqlens, self.maximum, self.maximum,
        softmax_scale=scaling, causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        softcap=(module.attn_logit_softcapping or 0.) if self.softcap else 0.)
    if self.has_padding:
      output = output.new_zeros(batch * length, heads, width).index_copy_(
          0, self.indices, output)
    return output.reshape(batch, length, heads, width), None


@contextmanager
def attention_adapter(adapter):
  original = ALL_ATTENTION_FUNCTIONS["sdpa"]
  ALL_ATTENTION_FUNCTIONS.register("sdpa", adapter)
  try:
    yield
  finally:
    ALL_ATTENTION_FUNCTIONS.register("sdpa", original)


@torch.inference_mode()
def microbenchmark(kernel, repeats):
  from types import SimpleNamespace
  results = {}
  for name, lengths, window, softcap in [
      ("b1_1024", [1024], None, False),
      ("b2_1024", [1024, 1024], None, False),
      ("b2_1024_420", [1024, 420], None, False),
      ("sliding_128_73", [128, 73], 64, False),
      ("softcap_sliding_128_73", [128, 73], 64, True)]:
    maximum, batch = max(lengths), len(lengths)
    cpu_mask = torch.arange(maximum)[None, :] < torch.tensor(lengths)[:, None]
    adapter = FlashAttention(kernel, cpu_mask, softcap)
    module = SimpleNamespace(training=False, sliding_window=window, attn_logit_softcapping=50.)
    torch.manual_seed(42)
    # Match the model's transposed projection layout and head dimension.
    q = torch.randn(batch, maximum, 8, 256, device="mps").transpose(1, 2)
    k = torch.randn(batch, maximum, 4, 256, device="mps").transpose(1, 2)
    v = torch.randn(batch, maximum, 4, 256, device="mps").transpose(1, 2)
    scale = 256**-.5
    if window or softcap:
      positions = torch.arange(maximum, device="mps")
      allowed = positions[:, None] >= positions[None, :]
      if window:
        allowed &= positions[:, None] - positions[None, :] < window
      def baseline():
        scores = q @ k.repeat_interleave(2, dim=1).transpose(-1, -2) * scale
        if softcap:
          scores = (scores / 50.).tanh() * 50.
        scores = scores.masked_fill(~allowed, float("-inf"))
        return (scores.softmax(-1) @ v.repeat_interleave(2, dim=1)).transpose(1, 2)
    else:
      def baseline():
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=scale, enable_gqa=True).transpose(1, 2)
    def flash():
      return adapter(module, q, k, v, None, scaling=scale)[0]
    valid = cpu_mask.to("mps")
    before, after = baseline()[valid].cpu(), flash()[valid].cpu()
    delta = (before - after).abs()
    results[name] = {"lengths": lengths, "window": window, "softcap": softcap,
        "input_query_stride": list(q.stride()), "max_abs_delta": delta.max().item(),
        "rms_delta": delta.square().mean().sqrt().item(), "finite": bool(after.isfinite().all()),
        "rows": []}
    functions = {"baseline": baseline, "flash": flash}
    for function in functions.values():
      for _ in range(3):
        function()
    for repeat in range(repeats):
      order = list(functions)
      random.Random(repeat + 42).shuffle(order)
      for variant in order:
        torch.mps.synchronize()
        started = time.perf_counter()
        for _ in range(10):
          functions[variant]()
        torch.mps.synchronize()
        results[name]["rows"].append({"variant": variant, "repeat": repeat,
            "ms": (time.perf_counter() - started) * 1000 / 10})
    results[name]["median_ms"] = {variant: statistics.median(r["ms"]
        for r in results[name]["rows"] if r["variant"] == variant) for variant in functions}
    del q, k, v, before, after, delta, valid, adapter, functions
    torch.mps.empty_cache()
  return results


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--kernel-build", required=True)
  parser.add_argument("--output", default="outputs/mac-flash-sdpa")
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument("--micro-repeats", type=int, default=7)
  parser.add_argument("--micro-only", action="store_true")
  args = parser.parse_args()
  if min(args.repeats, args.micro_repeats) < 1:
    parser.error("Repeat counts must be positive.")
  output = Path(args.output) / "results.json"
  if output.exists():
    parser.error("Choose a fresh output directory to preserve evidence.")
  torch.set_num_threads(8)
  torch.mps.set_per_process_memory_fraction(2 * 2**30 / torch.mps.recommended_max_memory())
  kernel = load_local_kernel(args.kernel_build)
  info = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
          "torch": torch.__version__, "kernel_build": str(Path(args.kernel_build).absolute()),
          "kernel_metadata": json.loads((Path(args.kernel_build) / "metadata.json").read_text()),
          "sources": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in ("detectorscope/streaming.py", "detectorscope/inference.py",
                           "scripts/benchmark_flash_sdpa_mac.py")},
          "microbenchmark": {}, "runs": [], "validation": {},
          "system_memory_start": system_memory()}
  write_json(output, info)
  info["microbenchmark"] = microbenchmark(kernel, args.micro_repeats)
  write_json(output, info)
  print(json.dumps({"microbenchmark": info["microbenchmark"]}), flush=True)
  if args.micro_only:
    info["status"] = "complete"
    write_json(output, info)
    return
  detector = OptimizedDetector.from_pretrained(device="mps", local_files_only=True, compact_mlp=True)
  cases = profiling_cases(detector.extractor.tokenizer)
  info.update(cases=cases, model=detector.identity, repeats=args.repeats,
              cpu_threads=torch.get_num_threads(),
              cpu_interop_threads=torch.get_num_interop_threads(),
              mps_allocator_budget_bytes=2 * 2**30)
  adapters = {case: FlashAttention(kernel, detector.extractor.tokenizer(
      texts, padding=True, return_tensors="pt")["attention_mask"]) for case, texts in cases.items()}
  original = ALL_ATTENTION_FUNCTIONS["sdpa"]
  tasks = [(case, variant) for case in cases for variant in ("baseline", "flash")]
  for repeat in range(-1, args.repeats):
    order = list(tasks)
    random.Random(42 + repeat).shuffle(order)
    for case, variant in order:
      with attention_adapter(adapters[case] if variant == "flash" else original):
        sync("mps")
        started = time.perf_counter()
        values = detector(cases[case], batch_size=len(cases[case]))
        sync("mps")
        elapsed = time.perf_counter() - started
      row = {"repeat": repeat, "case": case, "variant": variant,
             "seconds": elapsed, "outputs": [serialize(x) for x in values],
             "pipeline": detector.extractor.profile}
      info["runs"].append(row)
      write_json(output, info)
      print(json.dumps({key: row[key] for key in
                        ("repeat", "case", "variant", "seconds")}), flush=True)
  for case, texts in cases.items():
    outputs = {}
    for variant in ("baseline", "flash"):
      with attention_adapter(adapters[case] if variant == "flash" else original):
        outputs[variant] = [serialize(x, highlights=True) for x in detector(
            texts, batch_size=len(texts), return_highlights=True)]
    baseline = next(r["outputs"] for r in info["runs"]
                    if r["case"] == case and r["variant"] == "baseline")
    info["validation"][case] = {
        "scores_all_repetitions": [compare_outputs(baseline, r["outputs"])
            for r in info["runs"] if r["case"] == case and r["variant"] == "flash"],
        "highlight_scores": compare_outputs(outputs["baseline"], outputs["flash"]),
        "highlights": compare_highlights(outputs["baseline"], outputs["flash"]),
        "outputs": outputs}
    write_json(output, info)
  # Separately measure the behavioral effect of the softcap currently ignored
  # by Transformers' SDPA adapter. Compare the candidate with eager on a 256-token
  # real prefix, then compare both with the current uncapped SDPA detector.
  from benchmark_mac import cases as review_cases
  sample = review_cases(detector.extractor.tokenizer)[0][0]
  text = sample["text"]
  mask = detector.extractor.tokenizer([text], return_tensors="pt")["attention_mask"]
  capped = FlashAttention(kernel, mask, softcap=True)
  semantic = {"text": text, "sample": sample["id"], "outputs": {}}
  for variant in ("sdpa", "eager_softcap", "flash_softcap"):
    detector.extractor.config._attn_implementation = "eager" if variant == "eager_softcap" else "sdpa"
    with attention_adapter(capped if variant == "flash_softcap" else original):
      semantic["outputs"][variant] = [serialize(x, highlights=True)
          for x in detector([text], return_highlights=True)]
  detector.extractor.config._attn_implementation = "sdpa"
  semantic["sdpa_vs_eager_softcap"] = compare_outputs(semantic["outputs"]["sdpa"],
                                                     semantic["outputs"]["eager_softcap"])
  semantic["eager_vs_flash_softcap"] = compare_outputs(semantic["outputs"]["eager_softcap"],
                                                     semantic["outputs"]["flash_softcap"])
  semantic["eager_vs_flash_highlights"] = compare_highlights(semantic["outputs"]["eager_softcap"],
                                                            semantic["outputs"]["flash_softcap"])
  info["softcap_semantics"] = semantic
  info["summary"] = {}
  for case in cases:
    summary = {}
    for variant in ("baseline", "flash"):
      rows = [r for r in info["runs"] if r["repeat"] >= 0
              and r["case"] == case and r["variant"] == variant]
      summary[variant] = {"median_seconds": statistics.median(r["seconds"] for r in rows),
          "min_seconds": min(r["seconds"] for r in rows),
          "max_seconds": max(r["seconds"] for r in rows),
          "median_block_compute_seconds": statistics.median(sum(s["compute_seconds"]
              for s in r["pipeline"]["stages"]) for r in rows),
          "max_mps_driver_bytes": max(r["pipeline"]["max_mps_driver_bytes"] for r in rows)}
    summary["latency_reduction_percent"] = (
        1 - summary["flash"]["median_seconds"] / summary["baseline"]["median_seconds"]) * 100
    info["summary"][case] = summary
  info["peak_process_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
  info["system_memory_end"] = system_memory()
  info["status"] = "complete"
  write_json(output, info)
  write_json(output.with_name("summary.json"), info["summary"])
  print(json.dumps({"summary": info["summary"],
                    "softcap_sdpa_vs_eager": semantic["sdpa_vs_eager_softcap"],
                    "softcap_eager_vs_flash": semantic["eager_vs_flash_softcap"]}, indent=2))


if __name__ == "__main__":
  main()
