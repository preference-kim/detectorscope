# FP32 shared prefetch and Metal kernel investigation

Measured October 30–31, 2026 KST on the M4 Pro, with the pinned PyTorch 2.14.1
and Transformers 4.57.6 runtime. An optional shared-memory prefetch path reduced
request P50 by **13–22%** against `44302d8` on four measured workloads. Scores,
features and highlights matched exactly. The original FP32 weights, **2 GiB MPS
limit** and **3 GiB worker footprint watchdog** stay in place.

The main change reads checkpoint bytes directly into CPU-visible MPS storage,
avoiding a separate weight copy. Freed MPS allocations are also reused between
blocks. The bigger remaining costs are FP32 FFN matrix multiplication for long
batches and exposed weight reads for short inputs.

## Current bottlenecks

Before changing cache lifetime, three uninstrumented forwards per workload gave
these median stage sums. Loading overlaps computation; the load duration must
not be added to the exposed wait or total request time.

| Input | Request, s | Exposed weight wait, s | Transfer + fence, s | Block compute + fence, s |
|---|---:|---:|---:|---:|
| 1 × 1024 | 2.122 | 0.763 | 0.236 | 1.018 |
| 2 × 1024 | 2.482 | 0.147 | 0.229 | 2.013 |
| 1024 + 420 | 2.308 | 0.305 | 0.247 | 1.632 |

In separate instrumented forwards, the full batch's FFN took about **1.43 s**
out of **2.00 s** of block intervals. Its gate, up and down projections accounted
for roughly 442, 445 and 479 ms. Attention was about 518 ms, including 167 ms
of SDPA and 44 ms of RoPE. Parent intervals include their children.

These are stream-completion event intervals, not hardware-only kernel timings.
Instrumentation changed request medians by −2.8% to +4.7%. Eleven short norm
intervals failed event-order validation and were excluded; block, FFN and
attention intervals did not fail. [Module totals](assets/mac-current-modules.csv).

## What actually runs on Metal

A separate, untimed capture of block 12 at `[2,1024,2304]` observed **33 compute
dispatches**. Seven were labeled `mmul_kernel_a16_float_float_float`. The FFN
had three matrix products and two other dispatches, consistent with separate
GELU and multiply operations. Attention had four matrix products and fourteen
other dispatches. A fused Gemma norm still uses two dispatches: forming its
`1 + weight` scale and normalization. [Observed dispatches](assets/mac-block-dispatch.csv).
Repeating the capture with shared prefetch produced identical dispatch records,
including grids, thread groups and recorded threadgroup-memory sizes.

The FFN gate/up grids were `[144,32,1]`, with 128 threads per group; down used
`[36,32,1]`. These match 64×64 output tiles, inferred from dimensions and grids.
The matching [PyTorch linear implementation](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/operations/Linear.mm)
uses `MPSNDArrayMatrixMultiplication` for this contiguous FP32 path.
`PYTORCH_MPS_PREFER_METAL` selects a different `mm` path; it does not tune the
tiles of these `linear` calls.

Most other pipeline labels were empty, so the capture alone cannot identify
their shader names. The native SDPA prefill path was already checked in the
[previous attention investigation](mac-compute-optimization.md). It should not
be described as an unfused fallback just because its operator contains `math`.

This Mac exposes timestamp counters at stage boundaries, but not dispatch
boundaries. `xctrace` is unavailable. No per-core occupancy, register pressure,
cache bandwidth or instruction-level utilization was measured.

## How shared prefetch works

The released checkpoint executes 23 blocks, each containing about 297 MiB of
FP32 weights. A forward still reads those blocks from disk and retains only
one current compute block and one CPU prefetch slot. There is no full-model
load, persistent shard mapping, quantization or change to the attention formula.

With `shared_prefetch=True`, **both slots are backed by MPS shared storage**.
PyTorch's private [`torch.mps._host_alias_storage` API](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/mps/__init__.py)
exposes a CPU view of each buffer. Coalesced file reads fill that view, including
when a block crosses checkpoint shards. GPU parameters then become views of
the same storage, with their original FP32 values, shapes and strides.

The slots alternate: CPU prefetch writes the next slot while the GPU computes
from the current one. A slot is reused only after its previous GPU computation
has finished. Another fence separates host writes from GPU consumption. On
exit, including errors, the loader thread joins and GPU work drains before
the slots are released. There are two MPS-backed weight buffers, but only one
block executes at a time; the second buffer has the existing CPU prefetch role.

The first prototype hit the unchanged 2 GiB MPS limit on a full batch. The
matching [allocator](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/mps/MPSAllocator.h)
can reserve 1 GiB heaps for requests between 10 and 512 MiB when below its
memory-pressure threshold. Shared slots changed that allocation pattern.
The CLI option sets `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.05` before importing Torch,
unless the environment already defines it. This lowers the reclamation and
heap-selection threshold; the 2 GiB allocation ceiling and 3 GiB watchdog do
not increase. This setting completed all measured workloads on this 24 GB Mac.

Shared prefetch is opt-in and requires host-alias support in PyTorch. It is
enabled only for MPS FP32; CPU and other precisions keep their existing path.
For direct Python use, set the environment before initializing MPS and pass
`shared_prefetch=True` to `OptimizedDetector.from_pretrained`. Direct model calls
still need an explicit `MemoryWatchdog(3)` context for the whole-process budget.

## Cache reuse

The old loop calls `torch.mps.empty_cache()` after every block. In the matching
[allocator implementation](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/mps/MPSAllocator.mm),
this commits and waits for the stream, then releases entirely unused heaps.
Later blocks may need those allocations again. The new MPS FP32 path keeps
that allocator cache during the forward and empties it after the prefetch
thread and queued GPU copies have drained, including on exceptions.

Layer objects are still released every iteration; the reusable cache holds free
allocations. CPU and other precision paths keep their previous cache behavior.
Cache reuse is on by default for MPS FP32, independently of shared prefetch.
`reuse_mps_cache=False` selects the old cache behavior for comparisons.

An earlier cache-only comparison found modest P50 reductions: 0.4% for 256
tokens, 4.0% for 1024, 2.5% for a mixed batch and 2.0% for two full windows.
Single-1024 aggregate throughput was slightly worse. That experiment does not
establish a general throughput gain. Its maximum lifetime footprint was 2.068
GiB old / 2.002 GiB new. [Cache-only results](assets/mac-cache-comparison.csv).

## Combined comparison

The final comparison uses `44302d8`, two rounds in reversed process order,
one warmup per workload/process and **eight timed calls per variant/workload**.
It includes tokenization, weight reads, backbone, SAE and scoring, but excludes
startup, HTTP and highlights. Each process runs under the 3 GiB watchdog.
Baseline uses its default allocator settings; the new path uses shared prefetch,
cache reuse and the lower reclamation threshold together.

| Input | P50 seconds, old → new | P50 reduction | Input tok/s, old → new |
|---|---:|---:|---:|
| 1 × 256 | 2.105 → 1.695 | 19.4% | 118.2 → 146.2 |
| 1 × 1024 | 2.228 → 1.749 | 21.5% | 453.6 → 583.4 |
| 1024 + 420 | 2.311 → 1.915 | 17.1% | 614.6 → 756.8 |
| 2 × 1024 | 2.554 → 2.213 | 13.3% | 791.7 → 888.5 |

Both rounds favored the new path for all four workloads. Across workloads,
median transfer-plus-fence time fell from 249–256 ms to 4.5–5.8 ms per forward.
Weight loading and exposed waits also changed, so the entire request saving
cannot be attributed to removing copies. Full-batch compute did not improve:
its stage median was 1.955 s old / 2.025 s new.

An earlier same-runner control gave both variants the lower reclamation setting.
Four shuffled pairs per workload favored the shared-slot prototype by 5–16%
at P50, with exact score/feature agreement. This supports the shared-storage
change beyond an allocator-only explanation, but is a separate experiment.

Process disk-read accounting was about 6.69 GiB per forward in both variants.
This counter does not isolate SSD service time. The new 256-token case still
spent a median 1.191 s waiting for weights; two full windows spent 0.070 s.
Clocks, filesystem cache and other applications were not controlled. Eight
timed calls per workload are evidence for these fixtures, not a universal
latency guarantee. [Combined comparison CSV](assets/mac-shared-prefetch-comparison.csv).

Maximum lifetime process footprint was **2.000 GiB old / 1.715 GiB new**, including
startup and highlight checks. Sampled MPS driver maxima were 1.079 / 1.188 GiB;
peak RSS was 0.858 / 1.336 GiB. These measures are not additive on unified memory.
The footprint watchdog is a stop-on-breach guard and allows temporary overshoot;
see the [worker memory contract](mac-inference.md#worker-memory-budget).

## Other experiments

Custom FP32 kernels used the same formulas, with no weight conversion. A scalar
GELU×multiply kernel was slower than stock. After vectorizing four elements per
thread, the 2×1024 operator improved from 1.732 to 1.039 ms. A fused RoPE kernel
improved from 1.467 to 0.645 ms on the same batch shape.

Neither established a useful request-level improvement in five shuffled pairs
per workload, on top of the cache change:

| Input | RoPE original → fused, s | GELU×multiply original → fused, s |
|---|---:|---:|
| 1 × 1024 | 1.939 → 1.986 | 2.034 → 2.025 |
| 2 × 1024 | 2.293 → 2.317 | 2.486 → 2.466 |
| 1024 + 420 | 2.058 → 2.063 | 2.168 → 2.178 |

RoPE outputs and highlights matched exactly. GELU's largest differences were
2.27e-6 logit, 5.37e-7 probability, 2.33e-6 feature contribution and 1.61e-4
highlight activation; labels, IDs, tokens and highlight positions matched.
These prototype kernels are not in the production path. Operator speedups
alone would have overstated their value.

## Validation and reproduction

The combined change matched baseline scores, features, tokens and highlights
**exactly** on all comparison fixtures. All **46 tests passed with real MPS**,
including unchanged FP32 weights, shared reads across shards, alternating slots,
layer release, transfer/prefetch/queued-compute failures, recovery, cache cleanup
and an embedding-only request. A separate full-checkpoint CPU comparison of
the cache change also matched exactly; shared prefetch does not activate on CPU.
This is implementation agreement, not a detector-quality evaluation.

The local worker was restored with shared prefetch on October 31 KST. An HTTP
check covered a document split into 1024, 1024 and 284 tokens, highlight bounds,
exact cached results and a rejected invalid request followed by healthy service.
The worker retained FP32, maximum batch 2, token limit 1024 and the 3 GiB guard.

Pause the document worker and run:

```sh
.venv/bin/python scripts/benchmark_prefetch_mac.py suite \
  --baseline-ref 44302d8 --output outputs/new-shared-comparison \
  --rounds 2 --repeats 4 --shared-prefetch
xcrun clang++ -std=c++17 -fobjc-arc -shared -framework Foundation \
  -framework Metal scripts/metal_dispatch_probe.mm -o /tmp/detectorscope-probe.dylib
.venv/bin/python scripts/inspect_streamed_block_mac.py \
  --probe /tmp/detectorscope-probe.dylib --output outputs/new-block-dispatch.json \
  --shared-prefetch
```

Both Python commands enforce the 3 GiB process watchdog. The model keeps its
2 GiB MPS budget. Restore the worker afterward:

```sh
.venv/bin/detectorscope serve --device mps --dtype float32 --batch-size 2 \
  --local-files-only --memory-limit-gib 3 --shared-prefetch
```

Omit `--shared-prefetch` to use the ordinary copy path. The dispatch probe records
launch arguments without changing them and must not be used for timing claims.
Raw timings, source snapshots, traces and experimental scripts are retained
locally in `outputs/mac-kernel-optimization-20261030/`; that ignored directory
does not accompany a clone.
