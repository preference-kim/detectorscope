---
library_name: detectorscope
license: gemma
base_model: google/gemma-2-2b
pipeline_tag: text-classification
tags:
- ai-text-detection
- sparse-autoencoder
- interpretability
---

# {name}

{about} Every prediction is a sum of contributions from sparse-autoencoder
features, each with a validated natural-language description, so the detector
can say which properties of the text drove its decision and where they occur.
Paper: [Building Auditable AI Text Detectors with Sparse Autoencoders](https://arxiv.org/abs/2610.30006);
code: [google-deepmind/detectorscope](https://github.com/google-deepmind/detectorscope).

## Use

```bash
pip install detectorscope
```

```python
from detectorscope import AuditableDetector

det = AuditableDetector.from_pretrained("{repo_id}"{revision})
out = det("The authors propose a novel framework for ...")

out.p_ai       # probability that the text is AI-written
out.label      # "ai" or "human" (logit threshold {threshold:+.2f})
out.features   # [Feature(id, importance, description, highlights), ...]
```

`out.features` lists every feature active in the text with a non-zero effect,
largest effect first. `importance` is how much the logit falls if that feature
is silenced: positive pushes toward AI, negative toward human, and
`out.logit == out.bias + sum(f.importance for f in out.features)`. A list of
texts returns a list of detections.

To see where a feature fires:

```python
out = det(text, return_highlights=True)
out.tokens                  # the text as a list of tokens
out.features[0].highlights  # [(token index, activation), ...] of the top feature
```

The default head is logistic regression over the features whose
auto-interpretation score is at least {tau}. `head="xgboost"` or another
`tau` (0.00 to 0.95 in steps of 0.05), on `from_pretrained` or on the call,
selects one of the other {n_heads} heads.

## Performance

Default head; the decision threshold is fixed on the development data.

| split | AUROC | F1 | accuracy | documents |
|---|---|---|---|---|
{perf_rows}

## What it looks at

The features with the largest average contribution on the training data
(weight × mean activation when active; density = share of the documents of
each class in which the feature fires).

**Toward AI**

| feature | description | mean contribution | AI density | human density |
|---|---|---|---|---|
{ai_rows}

**Toward human**

| feature | description | mean contribution | AI density | human density |
|---|---|---|---|---|
{human_rows}

## Variants

{variants}

## Licence

The language model under `extractor/` is Gemma-2-2b{extractor}, provided under
and subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
(see `NOTICE`). The sparse autoencoders derive from
[Gemma Scope](https://huggingface.co/google/gemma-scope-2b-pt-res) (CC-BY-4.0);
the heads and feature descriptions are CC-BY-4.0.

## Citation

```bibtex
@misc{{dugan2026detectorscope,
  title={{Building Auditable AI Text Detectors with Sparse Autoencoders}},
  author={{Liam Dugan and Callum McDougall and Matthew Denton and Phillip Pham and Christine Kaeser-Chen and Chris Callison-Burch}},
  year={{2026}},
  eprint={{2610.30006}},
  archivePrefix={{arXiv}},
  url={{https://arxiv.org/abs/2610.30006}},
}}
```
