"""Capture SDPA launch geometry in a separate, untimed instrumentation process."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmark_flash_sdpa_mac import FlashAttention, load_local_kernel


@torch.inference_mode()
def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--probe", default="outputs/mac-gemm-tuning/metal_dispatch_probe.dylib")
  parser.add_argument("--kernel-build", required=True)
  parser.add_argument("--output", required=True)
  args = parser.parse_args()
  output = Path(args.output)
  if output.exists():
    parser.error("Refusing to overwrite an existing capture.")
  output.parent.mkdir(parents=True, exist_ok=True)
  torch.set_num_threads(8)
  torch.manual_seed(42)
  torch.mps.set_per_process_memory_fraction(2 * 2**30 / torch.mps.recommended_max_memory())
  probe = ctypes.CDLL(str(Path(args.probe).resolve()))
  probe.probe_start.argtypes = [ctypes.c_char_p]
  probe.probe_write.argtypes = [ctypes.c_char_p]
  if probe.probe_install() != 0:
    raise RuntimeError("Cannot install the measurement probe.")
  kernel = load_local_kernel(args.kernel_build)
  q = torch.randn(2, 1024, 8, 256, device="mps").transpose(1, 2)
  k = torch.randn(2, 1024, 4, 256, device="mps").transpose(1, 2)
  v = torch.randn(2, 1024, 4, 256, device="mps").transpose(1, 2)
  mask = torch.arange(1024)[None, :] < torch.tensor([1024, 420])[:, None]
  flash = FlashAttention(kernel, mask)
  module = SimpleNamespace(training=False, sliding_window=None, attn_logit_softcapping=50.)
  functions = {
      "pytorch_implicit_causal_b2": lambda: torch.nn.functional.scaled_dot_product_attention(
          q, k, v, is_causal=True, scale=256**-.5, enable_gqa=True),
      "metal_flash_varlen_1024_420": lambda: flash(module, q, k, v, None, scaling=256**-.5)[0]}
  for label, function in functions.items():
    function()
    torch.mps.synchronize()
    probe.probe_start(label.encode())
    function()
    torch.mps.synchronize()
    probe.probe_stop()
  if probe.probe_write(str(output.resolve()).encode()) != 0:
    raise RuntimeError("Cannot write capture.")
  report = json.loads(output.read_text())
  report.update(torch=torch.__version__, torch_git_version=torch.version.git_version,
                query_shape=list(q.shape), query_stride=list(q.stride()),
                kernel_build=args.kernel_build,
                probe_sha256=hashlib.sha256(Path(args.probe).read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
  output.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
