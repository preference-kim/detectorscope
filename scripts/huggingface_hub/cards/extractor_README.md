---
library_name: transformers
license: gemma
base_model: google/gemma-2-2b
pipeline_tag: text-classification
tags:
- ai-text-detection
---

# {name}

`google/gemma-2-2b` fine-tuned as a sequence classifier (human vs. {target}
text on {dataset}) for *Building Auditable AI Text Detectors with Sparse
Autoencoders*. It is the paper's FT-Model baseline and the extractor of its
FT-Model+SAE detectors, which hook its residual stream.

The weights are a modification of Gemma 2 2B (full fine-tuning with
`detectorscope.finetune`; all parameters changed). Gemma is provided under
and subject to the Gemma Terms of Use found at
https://ai.google.dev/gemma/terms, including its use restrictions; see
`NOTICE`.
