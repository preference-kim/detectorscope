# DetectorScope on my MacBook

A small patch to run [DeepMind's DetectorScope](https://github.com/google-deepmind/detectorscope) on my MacBook. I wanted to check documents locally, so this fork focuses on CPU and Apple Silicon (MPS) inference with manageable memory use.

- Tested on an M4 Pro MacBook with 24 GB of unified memory.
- Keeps the original published weights in FP32.
- Streams one model block at a time, with one CPU prefetch slot and a 2 GiB MPS allocator limit. That limit doesn't cap total process RAM.
- Skips unused model work, reduces computation on padding, and uses PyTorch's fused normalization on MPS.
- Adds a local CLI for checking text and Markdown, with optional feature highlights.

Still a work in progress. Speed depends on input length, batch size and file caching. See the [Mac inference notes](docs/mac-inference.md) for benchmarks, environment details and remaining questions. The upstream link has the original project documentation.

Original work © 2026 Google LLC. See [LICENSE](LICENSE).
