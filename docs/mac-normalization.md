# FP32 normalization on Mac

Measured on October 29–30, 2026 KST, on the M4 Pro and pinned runtime described
in [Mac inference](mac-inference.md). Fused normalization reduced long-batch
block compute by about 5% in a shuffled comparison. End-to-end latency generally
improved for those batches, but storage/cache variation prevents a general
throughput claim.

The MPS FP32 runner now uses `torch.nn.functional.rms_norm` for Gemma's four
normalizations per block. The installed PyTorch has an MPS kernel for
`aten::_fused_rms_norm`. It replaces the separate square, reduction, rescaling
and multiplication operations while preserving Gemma's `1 + weight` scale and
epsilon. Original checkpoint parameters remain FP32 and are never rewritten.
The CPU path and other precisions retain their existing normalization.
`OptimizedDetector.from_pretrained(..., fused_norm=False)` selects the original
implementation for comparisons.

The 2 GiB MPS allocator limit, one GPU compute block, one CPU prefetch slot,
copy-completion fence and per-forward staging-buffer release remain in place.
Attention, including the existing SDPA softcapping behavior, is unchanged.

## End-to-end measurements

The fresh-process comparison uses `bf7d995` as its baseline, two rounds in
reversed variant order, one warmup per workload per process and **12 timed calls
per variant/workload**. Each request includes tokenization, streamed weights,
backbone computation, SAE features and scoring. Startup, HTTP and highlights
are excluded; highlights are checked separately. There are eight CPU compute
threads and one prefetch thread. The document worker was stopped during these
measurements.

| Batch × tokens | P50 seconds, original → fused | Input tok/s, original → fused |
|---|---:|---:|
| 1 × 256 | 2.293 → 2.068 | 110.7 → 117.1 |
| 1 × 1024 | 2.308 → 2.145 | 441.3 → 465.0 |
| 1024 + 420 | 2.405 → 2.355 | 595.5 → 610.5 |
| 2 × 1024 | 2.946 → 2.699 | 681.7 → 766.4 |

Loading and preparation also varied between processes, so the entire measured
gain cannot be attributed to normalization. A second comparison toggles only
normalization on one runner, shuffling variant order within **eight timed pairs
per workload**, after warming both modes:

| Batch × tokens | P50 seconds, original → fused | Input tok/s, original → fused | Median paired seconds saved |
|---|---:|---:|---:|
| 1 × 256 | 1.776 → 1.726 | 159.1 → 158.0 | −0.033 |
| 1024 + 420 | 2.073 → 2.030 | 693.6 → 722.6 | 0.100 |
| 2 × 1024 | 2.412 → 2.250 | 850.2 → 804.6 | 0.154 |

The paired statistic subtracts each fused call from its neighboring original
call before taking the median; it is not the difference between the two P50s.
One fused equal-batch request took 4.803 seconds, including 1.607 seconds outside
the extractor. It remains in every aggregate. This outlier reverses that
workload's throughput result even though seven of its eight pairs favor fused
normalization. The outlier's cause was not isolated. Short inputs have no
consistent end-to-end gain in this control.

Long-batch compute-stage medians fell from **1.540 to 1.457 seconds** for the
mixed batch and **1.901 to 1.798 seconds** for the equal batch. These intervals
include host dispatch and synchronization. Isolated normalization on synthetic
FP32 tensors with 256, 1024 and 2048 rows was 7–10× faster; that is an operator
measurement, not a request speedup.

## Loading and memory

The fresh-process runs recorded about **6.69 GiB of process disk reads per
forward**, despite warmup. During the shuffled run, timed requests ranged from
almost no disk reads to about 6.7 GiB. This is consistent with changing macOS
file-cache residency. Its contribution to the timing variation was not isolated.
The counter does not isolate SSD service time, and this experiment does not
fully explain the CPU/MPS gap. Clocks, filesystem cache and other applications
were not controlled.

A separate bounded read experiment tried 1, 4, 16 and 64 MiB read sizes and
temporary block mappings. None established a useful improvement over the
existing coalesced reads; mappings were much slower. Those experiments were
not adopted. No model mapping is retained by the runtime.

Fresh-process peak RSS was **0.945 GiB original / 0.843 GiB fused**; sampled MPS
driver maxima were **1.079 GiB for both**. The shuffled process peaked at
1.101 GiB RSS. RSS and driver allocation are separate measures and should not
be added on unified memory. Driver samples can miss transient peaks.

## Output checks and reproduction

Labels, selected feature IDs, token strings and highlight positions matched
on all measured fixtures. Maximum absolute changes were **7.75e-7 probability**,
**2.97e-6 logit**, **3.04e-6 feature contribution** and **1.68e-4 highlight
activation**. Numeric outputs are not bitwise identical. This is implementation
agreement on fixtures, not a detection-quality evaluation. The CPU comparison
matched exactly, including highlights, for both measured workloads.

All 26 tests passed with real MPS, including original-weight equality,
epsilon-sensitive and noncontiguous normalization inputs, staging ownership,
queued-copy failure and recovery. The original and fused paths both retain the
same weight-residency checks.

After pausing the document worker, run from the checkout:

```sh
.venv/bin/python scripts/benchmark_prefetch_mac.py suite \
  --baseline-ref bf7d995 --output outputs/new-norm-paired \
  --rounds 2 --repeats 6 --score-atol 0.0001 --highlight-atol 0.001
.venv/bin/python scripts/benchmark_norm_mac.py \
  --output outputs/new-norm-crossover.json --repeats 8
```

Restore the worker afterward. Both harnesses reject existing output paths and
retain timings, output values and per-block profiles. Output tolerances are
explicit; the fresh-process harness still requires exact equality by default.
The saved crossover harness was also exercised with one timed pair per workload.

[CSV results](assets/mac-norm-comparison.csv) include input throughput and
backbone prefill throughput, whose scope includes weight loading but excludes
SAE/scoring. Local raw evidence is in `outputs/mac-fused-norm/paired-mps/`,
`outputs/mac-fused-norm/crossover.json` and `outputs/mac-fused-norm/cpu-regression/`.
The executed exploratory scripts and operator/read measurements are in
`outputs/mac-loader-investigation/`. These ignored files do not travel with Git.
