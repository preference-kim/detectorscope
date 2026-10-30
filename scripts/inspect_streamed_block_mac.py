"""Capture actual FP32 block dispatches, not instrumented performance numbers.

Compile metal_dispatch_probe.mm separately and pass its dylib with --probe.
Run while the document worker is stopped. Only one model block is loaded.
"""

import argparse
from collections import Counter
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def capture(args, memory):
  import torch
  from detectorscope.inference import OptimizedDetector

  torch.set_num_threads(8)
  torch.manual_seed(42)
  detector = OptimizedDetector.from_pretrained(device="mps", local_files_only=True,
                                               shared_prefetch=args.shared_prefetch)
  extractor = detector.extractor
  with extractor._weight_staging():
    layer, _, _ = extractor._load_layer(12)
    extractor._transfer_layer(layer)
    torch.mps.synchronize()
  probe = ctypes.CDLL(str(args.probe.resolve()))
  probe.probe_start.argtypes = [ctypes.c_char_p]
  probe.probe_write.argtypes = [ctypes.c_char_p]
  if probe.probe_install() != 0:
    raise RuntimeError("Cannot install the Metal measurement probe.")
  hidden = torch.randn(2, 1024, extractor.config.hidden_size, device="mps")
  positions = torch.arange(1024, device="mps").unsqueeze(0)
  embeddings = extractor.rotary(hidden, positions)
  with torch.inference_mode():
    for label, module in (("block", layer), ("mlp", layer.mlp),
                          ("attention", layer.self_attn), ("norm", layer.input_layernorm)):
      def run():
        if label in ("block", "attention"):
          return module(hidden, position_embeddings=embeddings, attention_mask=None,
                        position_ids=positions, cache_position=positions[0], use_cache=False)
        return module(hidden)
      for _ in range(2):
        run()
      torch.mps.synchronize()
      probe.probe_start(label.encode())
      try:
        run()
        torch.mps.synchronize()
      finally:
        probe.probe_stop()
  if probe.probe_write(str(args.output.resolve()).encode()) != 0:
    raise RuntimeError("Cannot write capture.")
  report = json.loads(args.output.read_text())
  report.update(model=detector.identity, torch=torch.__version__,
                torch_git_version=torch.version.git_version, block=12,
                input_shape=list(hidden.shape), input_dtype=str(hidden.dtype),
                memory=memory.status(),
                probe_sha256=hashlib.sha256(args.probe.read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
  args.output.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report["device"]))
  for label in ("block", "mlp", "attention", "norm"):
    print(label, dict(Counter(r["pipeline_label"] for r in report["dispatches"]
                              if r["case"] == label)))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--probe", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--shared-prefetch", action="store_true")
  args = parser.parse_args()
  if args.output.exists():
    parser.error("Choose a new output file to preserve existing evidence.")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  if args.shared_prefetch:
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.05")
  from detectorscope.memory import MemoryWatchdog
  with MemoryWatchdog(3) as memory:
    capture(args, memory)


if __name__ == "__main__":
  main()
