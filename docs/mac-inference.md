# About this fork

**Status: 2026-10-29 KST.**

## Why this fork exists

I'm a Mac user. I wanted DetectorScope to be practical for everyday document checks without CUDA. This fork focuses on usable **CPU and Apple Silicon/MPS performance**, with manageable memory use.

Against our [historical full-model CPU run](#historical-cpu-reference), the fork's CPU path was **at least 8× faster at P50** and used **under one-sixth the peak process RSS** in FP32 batch-1 tests with 256–1,024 input tokens.

## Model specification

The backbone is [Gemma 2, a decoder-only causal LLM](https://ai.google.dev/gemma/docs/core/model_card_2), used here to extract features from the supplied text. **Decoder-only describes its architecture.** Detection runs forward passes over the input; it does not run the autoregressive loop that generates new tokens one at a time.

The inference path is **input text → Gemma 2 hidden states → sparse autoencoder (SAE) features → logistic regression → detection result**.

| Property | Specification |
|---|---|
| Released model | `liamdugan/detector-scope`, based on Gemma-2-2B |
| Parameters | **2.614B backbone + approximately 0.302B SAE parameters** |
| Parameter precision | **All parameters are FP32**: backbone, SAEs and classifier |
| Weight size | Approximately **10.86 GiB** of backbone + SAE tensor data |
| Transformer blocks | **26 total; 23 executed (0–22)**. Features use inputs to blocks 8, 14, 18 and 23 |
| Block dimensions | Hidden 2304; FFN 9216; 8 query / 4 KV heads; head dimension 256 |
| Native maximum context | **8,192 tokens**; alternating full and sliding attention, with a **4,096-token** local window |
| Detector chunk limit | **1,024 tokens including special tokens**. Longer documents are split into independent contexts |
| Detection head | Four SAEs with 16,384 features each, followed by logistic regression; this runner retains the **449 encoder features** used by the selected head |

Each independent input chunk returns a [detection result](../detectorscope/model/detectors/auditable.py): **`label`** (`ai` or `human`), **`p_ai`** (the classifier's AI probability score, 0–1), and **`features`** with their contributions to the decision. Optional **highlights** locate those features in the input. Feature descriptions come from saved metadata; inference retrieves them. Long documents receive separate results for their chunks.

Model revision: `9baaff9e2d1cdb79cafa90cdb27811b847e69dae`. Parameter counts, dtypes and context limits were checked against this checkpoint.

**Current development and everyday use remain based on the original published weights at FP32.** Low-precision execution is future work, with the original FP32 model as the reference.

## Test environment

| Component | Configuration |
|---|---|
| Chip / CPU | **Apple M4 Pro, 12 CPU cores: 8 performance + 4 efficiency** |
| GPU | **16 Apple GPU cores**, using PyTorch MPS |
| RAM | **24 GB unified memory** |
| OS | macOS 26.6.2 |
| Runtime | Python 3.13; PyTorch 2.14.1; Transformers 4.57.6 |
| Benchmark CPU threads | 8 compute threads; one weight-prefetch thread in streamed runs |
| Current worker | **MPS FP32, maximum batch 2**, chunk limit 1,024 tokens |
| GPU allocation budget | **2 GiB MPS allocator limit**; not a process-wide RAM cap |

## Benchmark configuration: batch size × input length

**BS** (batch size) and **ISL** (input sequence length / tokens per chunk) are independent settings. A forward processes BS independent contexts of ISL tokens each. **OSL=0**: no generated tokens. This is not chunked-prefill scheduling within one shared context.

Example: `detectorscope serve --device mps --dtype float32 --batch-size 2 --max-tokens 512`. Batch size is distinct from HTTP concurrency; the worker serializes HTTP requests.

## Standard inference metrics

The terminology follows [InferenceX](https://inferencex.semianalysis.com/inference/kimi-k3), including its definitions of [E2E latency](https://inferencex.semianalysis.com/glossary/end-to-end-latency), [throughput](https://inferencex.semianalysis.com/glossary/throughput) and [interactivity](https://inferencex.semianalysis.com/glossary/interactivity).

The timed task is to **read input text and return its detection result**. E2E latency measures time to that completed result; input throughput measures how much text is processed per second. **Prefill** below refers to the backbone's input-processing stage. It still executes 23 Transformer blocks, but no token-generation phase follows it.

| Metric | Definition and scope here |
|---|---|
| **E2E Request Latency, P50 (s)** | Median local-call time until all **BS** chunk results are complete; timing scope below |
| **Input Token Throughput (tok/s)** | Total valid input tokens across timed calls divided by their total elapsed time. Padding is excluded; this is whole-machine throughput |
| **Prefill Throughput (tok/s), detector backbone** | Input tokens / streamed extractor wall time: **23 blocks**, preparation and weight I/O; excludes SAE/classifier. Not a full 26-block generative Gemma benchmark |
| **Interactivity (tok/s/user)** | Generated-token delivery rate, the reciprocal of time per output token (TPOT) in compatible units. **P50 Interactivity is N/A** for this detector: there is no decode stream. Time to first token (TTFT), TPOT and inter-token latency (ITL) are likewise N/A |
| **Peak memory (GiB)** | Peak process RSS, with MPS driver allocation reported separately. Driver samples are observed maxima, not guaranteed transient peaks. OOM is a failed operating point |

Plots use **Input Token Throughput vs P50 E2E Request Latency**, with explicit **BS labels and ISL panels**. Input TPS and inverse request latency are not called Interactivity. P95 needs more samples; FP32 output agreement remains an acceptance check, separate from accuracy evaluation.

## Performance — joint sweep, 2026-10-29

Reference commit: **`c0106d1`**, before the prefetch update below. Sweep: **CPU/MPS × BS {1, 2, 4} × ISL {256, 512, 1024}**, all FP32. **17 configurations completed; one OOM.** Each uses a fresh process, one warmup and six timed complete batches. P50 is directly measured, not a multi-batch time divided by batch count.

Both backends use the same public GPT-4o/human review prefixes: BS=1 alternates them, BS=2 includes both, and BS=4 repeats them. Timers include tokenization, weight I/O, backbone, SAE and scoring; exclude startup, HTTP and highlights; bypass result caching. The worker was paused. Clocks, filesystem cache and other apps were not controlled.

![Input token throughput versus P50 E2E latency, faceted by input length and labeled by batch size](assets/mac-throughput-latency.svg)

Panels hold ISL fixed while BS changes. Whiskers are observed min–max, not confidence intervals. Compare within an ISL: changing chunk length also changes the detector's available context.

- At **ISL=256**, CPU had lower latency and higher input throughput than MPS at each tested BS.
- At **ISL=512, BS=4**, MPS reached **864 input tok/s at 2.37 s P50**, versus CPU's **546 tok/s at 3.68 s**.
- At **ISL=1024, BS=2**, MPS reached **772 input tok/s at 2.63 s P50**, versus CPU's **530 tok/s at 3.81 s**.
- **MPS BS=4 works at ISL=256 and 512; it OOMs at ISL=1024** under the unchanged 2 GiB allocator limit.

### Memory across the same BS × ISL grid

![CPU and MPS process peak RSS and separate observed MPS driver memory over batch size and input length](assets/mac-memory-grid.svg)

Each cell has a separate memory measurement. RSS includes initialization; driver memory is sampled at block completion. Do not add them on unified memory. OOM cells are not zeros or interpolated estimates.

[Measured grid, including backbone prefill throughput (CSV)](assets/mac-benchmark-grid.csv) · [Throughput/latency PNG](assets/mac-throughput-latency.png) · [Memory PNG](assets/mac-memory-grid.png)

### Prefetch update — 2026-10-29

The MPS FP32 path now coalesces adjacent weight reads, reuses one CPU staging buffer, and queues nonblocking parameter copies before waiting at each block boundary. GPU parameter layout and kernels stay the same. The CPU path retains its existing loader.

Compared with `c0106d1`, two fresh-process rounds reversed variant order, with ten timed calls per variant/workload after warmup:

| BS | Tokens per chunk | Before P50 | After P50 | Input tok/s, before → after |
|---:|---|---:|---:|---:|
| 1 | 256 | 1.970 s | 1.932 s | 126.6 → 128.2 |
| 1 | 1024 | 2.062 s | 1.966 s | 490.1 → 482.6 |
| 2 | 1024 + 420 | 2.169 s | 2.157 s | 657.4 → 668.3 |
| 2 | 1024 + 1024 | 2.506 s | 2.394 s | 811.4 → 802.5 |

Block-transfer time fell **20–27%**; E2E P50 fell **0.5–4.7%**. Input throughput changed **−1.5% to +1.7%**, so a stable E2E throughput gain is not established. Weight loading remains the main limit for shorter inputs. Clocks, filesystem cache and other applications were not controlled.

Maximum run peak RSS was **0.94 → 0.90 GiB**; observed MPS driver maximum stayed **1.08 GiB**. The **2 GiB allocator limit**, one GPU compute block and one CPU prefetch slot remain unchanged. CPU staging is bounded to **one block (about 297 MiB)** and released after each forward; shard boundaries use separate bounded reads. Scores, labels, feature contributions and highlights matched exactly in both MPS rounds and the CPU regression check. **24 tests passed**, including real MPS buffer reuse and recovery after a queued-copy failure.

[Prefetch comparison, including backbone prefill throughput (CSV)](assets/mac-prefetch-comparison.csv). To repeat with the worker paused and test extras installed: `python scripts/benchmark_prefetch_mac.py suite --baseline-ref c0106d1 --output outputs/new-prefetch-run`.

### Historical CPU reference

The **2026-10-28** full-model CPU run used FP32, BS=1 and the same review prefixes, each warmed and timed three times. It is a historical reference, **not a verified unmodified-upstream baseline**.

| BS | ISL (tokens/chunk) | Historical CPU P50 | Fork CPU P50, Oct 29 | Fork MPS P50, Oct 29 |
|---:|---:|---:|---:|---:|
| 1 | 256 | 14.07 s | 1.67 s | 3.27 s |
| 1 | 512 | 17.81 s | 1.68 s | 2.06 s |
| 1 | 1024 | 19.40 s | 2.09 s | 2.35 s |

Peak process RSS was **6.55 GiB** in the historical CPU run and **at most 1.03 GiB** across the fork's BS=1 CPU runs. A separate 1,024+420-token BS=2 test found **6.9%** lower P50 after FFN padding removal; this uniform-length grid contains no padding. Across all eight completed CPU/MPS pairs in the new sweep, labels and feature IDs matched; maximum logit difference was **3.58e-6**.

## Current constraints and optimizations — 2026-10-29

- **Bounded weight streaming:** storage → CPU → GPU, with one current GPU block and one CPU-prefetched block. CPU execution also retains only current/next blocks.
- **MPS prefetch:** reuse CPU staging only after the GPU copy fence; drain queued copies before releasing storage on errors. GPU parameter layout is unchanged.
- **FP32 baseline:** preserve the original parameter precision and the 2 GiB MPS allocator budget. CPU staging and runtime overhead are outside that allocator budget.
- **Skip unused work:** stop after the required feature hooks, omit vocabulary projection and SAE decoding, and retain only classifier-relevant encoder features.
- **Batch independent chunks:** share each block's weights across chunks. Right padding enables implicit causal SDPA while preserving sliding-window and feature-pooling masks.
- **Remove FFN padding work:** process valid tokens when at least 64 padding rows can be removed; keep the original path for smaller amounts of padding.
- **Existing inference kernels:** no model kernel has been written or modified. Compare existing PyTorch/MPS and Metal Flash SDPA implementations first. External Flash and per-context attention execution remain disabled by default.

### Apple compute and memory references

- [Apple M4 Pro specifications](https://support.apple.com/en-us/121553): this configuration has 12 CPU and 16 GPU cores, with **273 GB/s** specified memory bandwidth. The specification does not list FP32 TFLOPS.
- Local synthetic reference measurements: approximately **4.84 FP32 TFLOP/s** and **200–239 GB/s effective large-buffer bandwidth**. These are neither architectural peaks nor hardware-utilization measurements.
- [Apple MPP Programming Guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf): GEMM tiling, threadgroup layout and cache reuse. Its M5-specific recommendations are not assumed to apply directly to M4. The current MPS GEMM API did not expose tile controls in this investigation.

## TODO

- [ ] Expand representative workloads and sample counts for **P95 and HTTP-level E2E latency**, including mixed chunk lengths.
- [ ] Further tune the **storage/CPU/GPU pipeline and prefetch scheduling**, including how load/transfer overlaps computation within the existing memory budget.
- [ ] Improve **FFN/SDPA** using existing implementations and supported tuning options; then decide whether custom kernels or a framework change are justified.
- [ ] Evaluate **low precision** against the original FP32 reference. **Keep original FP32 weights as the basis for current development and everyday use.**
- [ ] Resolve **attention softcapping and reference-output semantics** before changing attention behavior.

To repeat the joint sweep, pause the local worker, run `python scripts/benchmark_grid_mac.py suite --output outputs/new-grid`, then restore the worker. `python scripts/plot_mac_grid.py --input outputs/new-grid/summary.json --assets docs/assets` regenerates the figures and compact CSV (requires Matplotlib). Raw per-call records remain in ignored outputs; the overview needs only these plots and the CSV.
