# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""The counterfactual experiment: zero-ablation vs. targeted paraphrase.

Does removing a feature by rewriting the text move the detector the way
zero-ablating the feature predicts? For each document of a sampled
subpopulation, this script computes the per-feature zero-ablation logit deltas
(the head's get_importance_scores), samples a fixed number of the document's
active interpretable features, asks Gemini to paraphrase the document so that
each sampled feature disappears while everything else stays (the template is
scripts/prompts/counterfactual/paraphrase.txt), and re-scores every
paraphrase. One run is one arm (a dataset x SAE setting x head x tau).

  python scripts/counterfactual_ablation.py --detector_head_path HEAD \
      --sae_model_paths ... --hook_points ... --desc_data_path DIR \
      --desc_model NAME [--output_path OUT] [--resume]

The output (<root>/<arm dir>/threshold<tau>_<head>_counterfactual.jsonl in the
stored runs) holds one record per (document, feature) pair, all deltas in
logit space with the (original - intervention) sign convention: positive means
removing the feature lowered the AI logit. Resumable (--resume). The parquet
run cache, the agreement statistics, permutation nulls, auto-interp bins and
most-frequent tables live in scripts/visualization/stats.py; the figure and
LaTeX tables in scripts/visualization/analysis.ipynb.

The paper arms are SGD+L1 logreg heads, which store no corpus means, so the
script takes --means_path pointing at the sibling XGBoost head.
"""

import argparse
import collections
import concurrent.futures
import json
import math
import os
from pathlib import Path
import pickle
import random
import threading
import time

import datasets
import torch
import tqdm

from detectorscope.data.cache import get_desc_files
from detectorscope.model.detector import Detector
from detectorscope.model.gemini import GeminiGenerator

REPO = Path(__file__).resolve().parents[1]
PROMPT = REPO / "scripts" / "prompts" / "counterfactual" / "paraphrase.txt"

# Sentinel for --target_model meaning "every non-human generator, pooled".
ALL_AI = "all"
# Every arm's feature space: four GemmaScope layers of width 16384, feature
# id = layer_rank * SAE_DIM + index.
SAE_DIM = 16384
LAYERS = [7, 13, 17, 22]


def _format_description(desc: dict) -> dict[str, str]:
  """Normalizes a feature description dict into prompt-ready strings.

  Args:
    desc: a description dict with optional 'terse', 'keywords', and 'long'
      fields (keywords may be a list or a string).

  Returns:
    Dict with string 'terse', 'keywords', and 'long' entries, using a
    placeholder where a field is missing.
  """
  placeholder = "[[ No Description ]]"
  terse = desc.get("terse") or placeholder
  long = desc.get("long") or placeholder
  keywords = desc.get("keywords")
  if isinstance(keywords, (list, tuple)):
    keywords = ", ".join(str(k) for k in keywords)
  keywords = keywords or placeholder
  return {"terse": terse, "keywords": keywords, "long": long}


class CounterfactualAblator:
  """Runs counterfactual ablation + targeted paraphrasing for a detector.

  Args:
    detector: an interpretable detector whose head exposes interp_mask, means
      and get_importance_scores.
    generator: the Gemini client used for the paraphrases (one clone per
      worker thread is created on demand).
    descriptions: concatenated-space feature id -> description dict, as built
      by load_descriptions.
    num_features: active features to sample and paraphrase per document.
    seed: per-document sampling is seeded as (seed + doc_id).
    epsilon: added to the corpus mean in the deviation denominator.
    sample_pool: 'described' restricts sampling to features with a usable
      description; 'all' disables the filter.
    extract_batch_size: paraphrases per extractor forward pass; a GPU memory
      knob that does not change results.
  """

  def __init__(
      self,
      detector: Detector,
      generator: GeminiGenerator,
      descriptions: dict[int, dict],
      num_features: int = 12,
      seed: int = 42,
      epsilon: float = 1e-8,
      sample_pool: str = "described",
      extract_batch_size: int = 8,
  ):
    self.detector = detector
    self.generator = generator
    self.descriptions = descriptions
    self.num_features = num_features
    self.seed = seed
    self.epsilon = epsilon
    self.extract_batch_size = extract_batch_size
    self.sample_pool = sample_pool
    self.template = PROMPT.read_text().rstrip("\n")

    # Features eligible to be targeted by a paraphrase. Two kinds of feature
    # are excluded, and both must be:
    #   1. features whose auto-interp score is NaN. These fire near-identically
    #      on every document (typically BOS / document-start artifacts), so the
    #      correlation between description and activation is undefined. The
    #      description exists but is useless as an intervention target.
    #   2. features with no description record at all, which never fired often
    #      enough in training for auto-interp to run. These may well be
    #      causally relevant; we simply have no description to paraphrase away.
    # NOTE: the NaN check is load-bearing. load_descriptions stores
    # correlation[0] verbatim, so a case-1 score is the float NaN rather than
    # None, and "is not None" alone would let every such feature through.
    self.describable = {
        fid
        for fid, desc in descriptions.items()
        if (score := desc.get("autointerp_score")) is not None
        and not math.isnan(score)
    }

    # Audit trail: which active features were never sampled, so the exclusion
    # never becomes a silent filter (warn_on_frequent_drops).
    self.dropped_counts: collections.Counter = collections.Counter()
    self.n_docs_seen = 0

    # Boolean mask [d] over the concatenated SAE feature space marking the
    # interpretable features that the trained head actually uses.
    self.interp_mask = torch.as_tensor(
        detector.head.interp_mask, dtype=torch.bool
    ).flatten()

    # Corpus mean activation per feature, loaded from the head artifact
    # ("means" key). Required to normalize the paraphrase deviation metrics.
    if detector.head.means is None:
      raise ValueError(
          "The detector head has no 'means' (corpus mean activations), which "
          "are required for the normalized paraphrase deviation metrics. "
          "LogReg heads never store them; pass --means_path pointing at the "
          "sibling XGBoost head for the same (train dir, target). Means are "
          "threshold-independent on retained columns, so any threshold works."
      )
    self.means = torch.as_tensor(
        detector.head.means, dtype=torch.float32
    ).flatten()

    # --workers > 1 runs ablate() from a thread pool so that one document's
    # Gemini round-trips overlap another document's forward passes. The
    # extractor keeps its hook captures on self and every head/SAE forward
    # shares one GPU, so all GPU work (and the sampling counters) runs under
    # this lock; only the Gemini wait runs unlocked.
    self.gpu_lock = threading.Lock()
    # One Gemini client per thread: async_generate() spins up a fresh event
    # loop per call, and one async HTTP client shared across loops running
    # concurrently in different threads is not safe. The main thread keeps the
    # generator it was given.
    self._thread_local = threading.local()
    self._thread_local.generator = generator
    # Per-phase wall-clock totals (seconds) over every document ablated.
    self.timings: collections.Counter = collections.Counter()
    self.n_timed_docs = 0

  def _generator(self) -> GeminiGenerator:
    """The calling thread's Gemini client (created on first use)."""
    gen = getattr(self._thread_local, "generator", None)
    if gen is None:
      gen = GeminiGenerator(model_name=self.generator.model_name)
      gen.async_batch_size = self.generator.async_batch_size
      self._thread_local.generator = gen
    return gen

  def warn_on_frequent_drops(self, doc_frac_threshold: float = 0.10) -> None:
    """Prints a warning for excluded features that fire in many documents.

    A dropped feature with a NaN auto-interp score is expected (those are
    document-start artifacts that fire everywhere). A dropped feature with no
    description at all that is also highly active is worth knowing about: it is
    a feature the detector may rely on that this experiment cannot test.
    """
    if not self.n_docs_seen:
      return
    for fid, n_docs in self.dropped_counts.most_common():
      frac = n_docs / self.n_docs_seen
      if frac < doc_frac_threshold:
        break
      if fid not in self.descriptions:
        print(
            f"WARNING: feature {fid} has no description but is active in "
            f"{frac:.1%} of documents; excluded from paraphrase sampling."
        )

  def describe(self, fid: int) -> dict[str, str]:
    """Returns the prompt-ready description strings for a feature id."""
    return _format_description(self.descriptions.get(fid, {}))

  def parse_response(self, response: str) -> str:
    """Extracts the paraphrased text from a Gemini response."""
    if response is None:
      return ""
    start_tag, end_tag = "<rewrite>", "</rewrite>"
    if start_tag in response and end_tag in response:
      start = response.index(start_tag) + len(start_tag)
      end = response.index(end_tag, start)
      return response[start:end].strip()
    return response.strip()

  def sample_features(self, active_ids: list[int], doc_id: int) -> list[int]:
    """Uniformly samples up to num_features active feature ids.

    Sampling is without replacement and seeded per-document (seed + doc_id) so
    runs are reproducible and resumable. If the document has fewer active
    features than num_features, all of them are returned.

    Args:
      active_ids: the active interpretable feature ids for the document.
      doc_id: index of the document, used to seed the per-document RNG.

    Returns:
      List of sampled concatenated-space feature ids.
    """
    if self.sample_pool == "described":
      pool = [f for f in active_ids if f in self.describable]
    else:
      pool = list(active_ids)

    # Audit what was excluded so warn_on_frequent_drops can report it.
    self.n_docs_seen += 1
    self.dropped_counts.update(set(active_ids) - set(pool))

    k = min(self.num_features, len(pool))
    rng = random.Random(self.seed + doc_id)
    return rng.sample(pool, k)

  def calc_deviations(
      self, delta: torch.Tensor, target_id: int, active_ids: list[int]
  ) -> tuple[float, float]:
    """Normalized activation deviations induced by a paraphrase.

    Given delta = h_paraphrase - h_original in the SAE feature basis, each
    feature's absolute activation change is normalized by that feature's corpus
    mean activation, i.e. deviation_j = |delta_j| / (mean_j + eps).

    Args:
      delta: tensor of shape [d] of per-feature activation changes.
      target_id: the concatenated-space id of the targeted feature.
      active_ids: all active interpretable feature ids in the original
        document.

    Returns:
      (target_feature_deviation, avg_non_target_deviation), where the first is
      the normalized deviation of the targeted feature and the second is the
      mean normalized deviation over all *other* active features (0.0 if none).
    """
    deviation = delta.abs() / (self.means + self.epsilon)
    target_feature_deviation = deviation[target_id].item()
    other_ids = [j for j in active_ids if j != target_id]
    if other_ids:
      avg_non_target_deviation = deviation[other_ids].mean().item()
    else:
      avg_non_target_deviation = 0.0
    return target_feature_deviation, avg_non_target_deviation

  @torch.no_grad()
  def ablate(self, text: str, doc_id: int, label: str) -> list[dict]:
    """Runs the counterfactual ablation experiment for a single document.

    Runs under torch.no_grad(): nothing here needs gradients, and without it
    the extractor's forward pass retains the full autograd graph for every
    layer's activations, which is the main driver of GPU OOM.

    Args:
      text: the document to probe.
      doc_id: index of the document in the source dataset.
      label: the ground-truth model label for the document.

    Returns:
      A list of per-(document, feature) metric records.
    """
    t_start = time.perf_counter()
    with self.gpu_lock:
      # Extract SAE features and score the original document.
      orig_feats = self.detector.get_feature_vectors([text])
      head = self.detector.head
      orig_score = head.predict(orig_feats, logit=False)[0].item()
      orig_logit = head.predict(orig_feats)[0].item()

      # Per-feature zero-ablation counterfactual logit deltas [1, d] -> [d]:
      # (original - ablated), so positive means the feature pushed toward AI.
      cf_deltas = head.get_importance_scores(orig_feats)[0].cpu()
      orig_feats_cpu = orig_feats[0].cpu()

      # All active interpretable features (a feature is active where its
      # max-pooled activation, hence its zero-ablation effect, is non-zero),
      # and a uniform sample of them to target (sample_features updates shared
      # counters, hence under the lock).
      active_ids = cf_deltas.nonzero().flatten().tolist()
      target_ids = (
          self.sample_features(active_ids, doc_id) if active_ids else []
      )
    t_gpu_orig = time.perf_counter() - t_start
    if not active_ids:
      print(f"[doc {doc_id}] no active interpretable features, skipping")
      return []
    if not target_ids:
      print(
          f"[doc {doc_id}] {len(active_ids)} active features but none are "
          "describable, skipping"
      )
      return []
    num_describable_active = sum(1 for f in active_ids if f in self.describable)

    # Log the features (and their descriptions) being ablated for this doc so
    # the description lookup can be sanity-checked at run time.
    for fid in target_ids:
      tqdm.tqdm.write(
          f"[doc {doc_id}] ablating feature {fid} "
          f"(counterfactual_logit_delta={cf_deltas[fid].item():+.4f}): "
          f"{self.describe(fid)['terse']}"
      )

    # Build one paraphrase prompt per targeted feature.
    prompts = [
        self.template.format(text=text, **self.describe(fid))
        for fid in target_ids
    ]

    # Query Gemini for all paraphrases of this document at once. This is the
    # only phase that runs outside the GPU lock.
    t_start = time.perf_counter()
    responses = self._generator().async_generate(prompts)
    t_gemini = time.perf_counter() - t_start
    paraphrases = [self.parse_response(r) for r in responses]

    # A paraphrase can come back empty when the generator gives up on a prompt
    # (e.g. the document trips the safety filter). Scoring an empty string
    # would produce a meaningless record, so drop those features instead.
    dropped = [fid for fid, p in zip(target_ids, paraphrases) if not p]
    if dropped:
      tqdm.tqdm.write(
          f"[doc {doc_id}] {len(dropped)}/{len(target_ids)} paraphrase(s) "
          f"empty (blocked or failed); skipping features {dropped}"
      )
      keep = [i for i, p in enumerate(paraphrases) if p]
      target_ids = [target_ids[i] for i in keep]
      paraphrases = [paraphrases[i] for i in keep]
      if not target_ids:
        return []

    # Re-extract features and re-score every paraphrase, a batch at a time.
    # There is one paraphrase per sampled feature, so this list is -nf long
    # (50 in the sweep), and the extractor runs whatever it is handed in a
    # single forward pass. The SAE widens d_model 2304 -> d_sae 16384, so 50
    # documents x 1024 tokens x 4 layers is ~13 GB of activations, doubled by
    # the masked copy -- more than a 45 GB GPU has. Chunking here does not
    # change the result: pooling is per-document.
    t_start = time.perf_counter()
    with self.gpu_lock:
      para_feats = torch.concat(
          [
              self.detector.get_feature_vectors(
                  paraphrases[i : i + self.extract_batch_size]
              )
              for i in range(0, len(paraphrases), self.extract_batch_size)
          ],
          dim=0,
      )
      para_scores = self.detector.head.predict(para_feats, logit=False)
      para_logits = self.detector.head.predict(para_feats)
      para_feats = para_feats.cpu()
    t_gpu_para = time.perf_counter() - t_start

    with self.gpu_lock:
      self.timings.update(
          {
              "gpu_orig": t_gpu_orig,
              "gemini": t_gemini,
              "gpu_para": t_gpu_para,
          }
      )
      self.n_timed_docs += 1
    tqdm.tqdm.write(
        f"[doc {doc_id}] timing: gpu_orig={t_gpu_orig:.1f}s "
        f"gemini={t_gemini:.1f}s gpu_para={t_gpu_para:.1f}s "
        f"({len(paraphrases)} paraphrases)"
    )

    records = []
    for i, fid in enumerate(target_ids):
      desc = self.describe(fid)
      para_score = para_scores[i].item()
      para_logit = para_logits[i].item()
      # Detector logit with this feature zero-ablated, i.e. f(x | x_fid = 0).
      counterfactual_logit = orig_logit - cf_deltas[fid].item()
      para_feats_cpu = para_feats[i].cpu()
      delta = para_feats_cpu - orig_feats_cpu
      target_feature_deviation, avg_non_target_deviation = self.calc_deviations(
          delta, fid, active_ids
      )
      # Components of target_feature_deviation, separated out for analysis:
      #   target_feature_deviation = |para_activation - orig_activation|
      #                              / (mean_activation + eps)
      orig_activation = orig_feats_cpu[fid].item()
      para_activation = para_feats_cpu[fid].item()
      mean_activation = self.means[fid].item()
      # Fields are ordered so the JSONL is scannable at a glance: identifiers,
      # then the numbers, then the feature descriptions, and finally the long
      # free-text fields (document, paraphrase).
      records.append(
          {
              "doc_id": doc_id,
              "label": label,
              "feature_id": fid,
              # Counterfactual logit delta (original - zero_ablated), surfaced
              # up front as an at-a-glance feature direction: positive means
              # the feature pushes the detector toward AI, negative toward
              # human.
              "feature_weight": cf_deltas[fid].item(),
              # The feature's auto-interp score (correlation[0] from its
              # description record); a measure of how well the description
              # explains the feature. None if not available.
              "autointerp_score": self.descriptions.get(fid, {}).get(
                  "autointerp_score"
              ),
              # P(AI) of the original and paraphrased document.
              "orig_score": orig_score,
              "para_score": para_score,
              "orig_logit": orig_logit,
              "para_logit": para_logit,
              "counterfactual_logit": counterfactual_logit,
              # Both deltas are in logit space and share the same sign
              # convention: (original) - (intervention). Counterfactual
              # zero-ablates the feature; paraphrase removes it via a targeted
              # rewrite. A positive delta means removing the feature lowered
              # the detector's logit.
              "counterfactual_logit_delta": cf_deltas[fid].item(),
              "paraphrase_logit_delta": orig_logit - para_logit,
              "orig_activation": orig_activation,
              "para_activation": para_activation,
              "mean_activation": mean_activation,
              "target_feature_deviation": target_feature_deviation,
              "avg_non_target_deviation": avg_non_target_deviation,
              # All ablatable features in the document, and the subset
              # eligible to be paraphrased. These differ a lot at tau=0.0 and
              # for linear heads.
              "num_active_features": len(active_ids),
              "num_describable_active_features": num_describable_active,
              # Feature descriptions and long free-text fields last.
              "description": desc["terse"],
              "keywords": desc["keywords"],
              "long_description": desc["long"],
              "text": text,
              "paraphrase": paraphrases[i],
          }
      )
    return records


def read_records(path: str) -> list[dict]:
  """Reads per-(doc, feature) records back from a JSONL written by `run`.

  Tolerates a truncated final line, which is what a killed job leaves behind.
  """
  records = []
  with open(path, "r") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        records.append(json.loads(line))
      except json.JSONDecodeError:
        print(f"Skipping malformed trailing line in {path}")
  return records


def load_descriptions(
    desc_data_path: str, desc_model: str, sae_dim: int
) -> dict[int, dict]:
  """Builds a {concatenated_feature_id: description_dict} mapping.

  Mirrors the per-layer indexing used in cache.get_interp_mask, where the
  concatenated feature id is (layer_position * sae_dim) + feature_index.

  Args:
    desc_data_path: directory containing per-layer feature results files.
    desc_model: name of the auto-interp model used in the filenames.
    sae_dim: width of a single SAE (features per layer).

  Returns:
    Mapping from concatenated-space feature id to its description dict, which
    contains the 'terse', 'keywords', and 'long' fields used to build the
    paraphrase prompt, plus an 'autointerp_score' field (the first element of
    the record's 'correlation' list, or None if unavailable).
  """
  descriptions = {}
  for i, (_, path) in enumerate(get_desc_files(desc_data_path, desc_model)):
    with open(path, "r") as f:
      for line in f:
        if not line.strip():
          continue
        record = json.loads(line)
        feature_id = (i * sae_dim) + int(record["feature_index"])
        # Each record's description is a dict {"terse", "keywords", "long"};
        # we keep all fields so the paraphrase prompt can use them, and also
        # capture the auto-interp score (correlation[0]).
        correlation = record.get("correlation") or []
        descriptions[feature_id] = {
            **record.get("description", {}),
            "autointerp_score": correlation[0] if correlation else None,
        }
  return descriptions


def sample_subpopulation(
    ds: datasets.Dataset,
    model_col: str,
    human_label: str,
    target_model: str,
    num_examples: int,
    seed: int,
    stratified: bool = True,
) -> datasets.Dataset:
  """Samples documents of the human and target-model classes.

  With stratified=True (the default), the sample is class-balanced: an equal
  number of human and target-model documents is drawn. Each class is capped at
  the smaller class's available count so the counts stay equal, and, when
  num_examples is given, at num_examples // 2 per class (an odd num_examples
  therefore yields num_examples - 1 documents so the two classes match).

  With stratified=False, a single unstratified random sample of num_examples is
  drawn from the union of the two classes, so the class balance reflects the
  underlying dataset proportions.

  Args:
    ds: the source dataset.
    model_col: column holding each document's class.
    human_label: label of the human class.
    target_model: label of the target AI-model class, or "all" to pool every
      non-human generator.
    num_examples: total sample size (defaults to the full filtered dataset).
    seed: random seed for shuffling/sampling.
    stratified: whether to draw an equal number from each class.

  Returns:
    The sampled (and shuffled) dataset.
  """
  models = ds[model_col]
  human_idx = [i for i, m in enumerate(models) if m == human_label]
  if target_model == ALL_AI:
    # Every non-human generator, pooled. This matches how the COLING heads were
    # trained (no single target model), and sampling is proportional to each
    # generator's share of the corpus, i.e. the distribution the detector
    # actually sees. The true generator name is preserved in each record's
    # label, so per-generator slices remain available afterwards.
    ai_idx = [i for i, m in enumerate(models) if m != human_label]
  else:
    ai_idx = [i for i, m in enumerate(models) if m == target_model]

  if not stratified:
    keep = human_idx + ai_idx
    ds = ds.select(keep)
    if num_examples and num_examples < len(ds):
      ds = ds.shuffle(seed=seed).select(range(num_examples))
    return ds

  # Stratified: equal counts per class.
  per_class = min(len(human_idx), len(ai_idx))
  if num_examples:
    per_class = min(per_class, num_examples // 2)

  rng = random.Random(seed)
  rng.shuffle(human_idx)
  rng.shuffle(ai_idx)
  keep = human_idx[:per_class] + ai_idx[:per_class]
  rng.shuffle(keep)  # interleave the two classes in the final ordering
  return ds.select(keep)


def run(args: argparse.Namespace) -> None:
  """One arm: writes the per-(document, feature) records incrementally.

  A killed job can be continued with --resume: feature sampling is seeded per
  document as (seed + doc_id), so a resumed run reproduces exactly what an
  uninterrupted one would have written.
  """
  if args.exit_if_output_path_exists and os.path.exists(args.output_path):
    print(f"Found results file at {args.output_path}, exiting")
    return

  detector = Detector.create_detector(
      "auditable",
      model_path=args.extractor_model_path,
      head_type=args.head_type,
      head_path=args.detector_head_path,
      hook_points=args.hook_points,
      sae_model_paths=args.sae_model_paths,
      gemma_scope_repo=args.gemma_scope_repo,
  )

  # LogReg artifacts never store corpus mean activations, which the paraphrase
  # deviation metrics normalize by. Borrow them from a sibling head when asked:
  # means depend only on the training cache and the interp mask, and masking a
  # column to zero cannot change the non-zero mean of a column it retains, so
  # any head trained on the same (train dir, target) has identical means on
  # every column this run will touch.
  if args.means_path:
    if detector.head.means is not None:
      print(f"Head already has means; ignoring --means_path {args.means_path}")
    else:
      with open(args.means_path, "rb") as f:
        donor = pickle.load(f)
      if donor.get("means") is None:
        raise ValueError(f"--means_path {args.means_path} has no 'means' key")
      detector.head.means = donor["means"]
      print(f"Loaded corpus means from {args.means_path}")

  # Infer SAE width from the interp mask and the number of layers.
  interp_mask = torch.as_tensor(detector.head.interp_mask).flatten()
  sae_dim = interp_mask.shape[0] // len(args.hook_points)

  print("Loading feature descriptions...")
  descriptions = load_descriptions(
      args.desc_data_path, args.desc_model, sae_dim
  )

  print("Loading dataset...")
  ds = datasets.load_from_disk(args.data_path).with_format("torch")
  ds = sample_subpopulation(
      ds,
      args.model_col,
      args.human_label,
      args.target_model,
      args.num_examples,
      args.seed,
      stratified=args.stratified,
  )

  generator = GeminiGenerator(model_name=args.generator_model_name)
  generator.async_batch_size = args.gemini_batch_size
  ablator = CounterfactualAblator(
      detector,
      generator,
      descriptions,
      num_features=args.num_features,
      seed=args.seed,
      sample_pool=args.sample_pool,
      extract_batch_size=args.extract_batch_size,
  )

  done_doc_ids, n_written = set(), 0
  mode = "w"
  if args.resume and os.path.exists(args.output_path):
    done_doc_ids = {r["doc_id"] for r in read_records(args.output_path)}
    mode = "a"
    print(
        f"Resuming: {len(done_doc_ids)} documents already in {args.output_path}"
    )

  # Rows are pulled lazily (the cache datasets are large), and with --workers
  # > 1 at most that many documents are in flight; the bar advances as
  # documents are handed out.
  def pending_docs():
    for doc_id, row in enumerate(
        tqdm.tqdm(ds, desc="Ablating documents", total=len(ds))
    ):
      if doc_id in done_doc_ids:
        continue
      yield doc_id, row[args.text_col], row[args.model_col]

  with open(args.output_path, mode, encoding="utf-8") as f:

    def commit(records: list[dict]) -> None:
      nonlocal n_written
      for record in records:
        f.write(json.dumps(record) + "\n")
      f.flush()
      n_written += len(records)

    if args.workers <= 1:
      for doc_id, text, label in pending_docs():
        commit(ablator.ablate(text, doc_id, label))
    else:
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=args.workers
      ) as pool:
        in_flight: set[concurrent.futures.Future] = set()
        for doc_id, text, label in pending_docs():
          in_flight.add(pool.submit(ablator.ablate, text, doc_id, label))
          if len(in_flight) >= args.workers:
            finished, in_flight = concurrent.futures.wait(
                in_flight, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for fut in finished:
              commit(fut.result())
        for fut in concurrent.futures.as_completed(in_flight):
          commit(fut.result())

  if ablator.n_timed_docs:
    n = ablator.n_timed_docs
    print(
        f"Mean per-document timing over {n} docs: "
        + ", ".join(f"{k}={v / n:.1f}s" for k, v in ablator.timings.items())
    )
  print(f"Wrote {n_written} records to {args.output_path}")
  ablator.warn_on_frequent_drops()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Counterfactual experiment: one arm -> jsonl."
  )
  parser.add_argument("-d", "--data_path", type=str, help="Path to the dataset")
  parser.add_argument(
      "-m",
      "--extractor_model_path",
      type=str,
      help="Path to the extractor model",
  )
  parser.add_argument(
      "-ht",
      "--head_type",
      type=str,
      default="xgboost",
      choices=["xgboost", "logreg"],
      help="Type of detector head to load",
  )
  parser.add_argument(
      "-e",
      "--detector_head_path",
      type=str,
      required=True,
      help="Path to the detector head to probe",
  )
  parser.add_argument(
      "-sm",
      "--sae_model_paths",
      type=str,
      nargs="+",
      required=True,
      help="Model paths for SAEs (in ascending layer order)",
  )
  parser.add_argument(
      "-hp",
      "--hook_points",
      type=str,
      nargs="+",
      required=True,
      help="Hook points for SAEs to connect to (in ascending layer order)",
  )
  parser.add_argument(
      "--means_path",
      type=str,
      default=None,
      help=(
          "(Optional) path to another head artifact to borrow the 'means' "
          "(corpus mean activations) from, used when the head itself has none. "
          "LogReg heads never store means; point this at the sibling XGBoost "
          "head for the same (train dir, target). Means are threshold-"
          "independent on retained columns, so any threshold's head works."
      ),
  )
  parser.add_argument(
      "--gemma_scope_repo",
      type=str,
      default="google/gemma-scope-2b-pt-res",
      help="Hub repo of the Gemma Scope ids among --sae_model_paths",
  )
  parser.add_argument(
      "--sample_pool",
      type=str,
      default="described",
      choices=["described", "all"],
      help=(
          "Which active features may be sampled for paraphrasing. 'described' "
          "(the default) excludes features with no description and features "
          "whose auto-interp score is NaN, neither of which can be targeted by "
          "a paraphrase prompt. 'all' disables the filter."
      ),
  )
  parser.add_argument(
      "--extract_batch_size",
      type=int,
      default=8,
      help=(
          "How many paraphrases to run through the extractor per forward pass. "
          "Lower this if the GPU runs out of memory; it does not change results."
      ),
  )
  parser.add_argument(
      "--workers",
      type=int,
      default=1,
      help=(
          "Documents ablated concurrently (thread pool). GPU work is "
          "serialized under a lock; >1 only overlaps Gemini round-trips with "
          "forward passes. Results are identical, records land in completion "
          "order."
      ),
  )
  parser.add_argument(
      "--gemini_batch_size",
      type=int,
      default=20,
      help=(
          "Paraphrase requests kept in flight per document (GeminiGenerator."
          "async_batch_size). 50 sends all -nf prompts in one round."
      ),
  )
  parser.add_argument(
      "--exit_if_output_path_exists",
      action="store_true",
      help="If true, exit if the output path already exists",
  )
  parser.add_argument(
      "--resume",
      action="store_true",
      help="Resume from an existing output file, skipping documents already written",
  )
  parser.add_argument(
      "-g",
      "--generator_model_name",
      type=str,
      default="gemini-2.5-pro",
      help="LLM to use for the targeted paraphrase",
  )
  parser.add_argument(
      "-dd",
      "--desc_data_path",
      type=str,
      required=True,
      help="Directory containing per-layer feature description files",
  )
  parser.add_argument(
      "-dm",
      "--desc_model",
      type=str,
      required=True,
      help="Auto-interp model name used in the description filenames",
  )
  parser.add_argument(
      "-nf",
      "--num_features",
      type=int,
      default=12,
      help=(
          "Number of active features to uniformly sample, ablate, and "
          "paraphrase per document (fewer are used if a document has fewer "
          "active features)"
      ),
  )
  parser.add_argument(
      "-c",
      "--model_col",
      type=str,
      default="model",
      help="Column in the dataset that contains the class",
  )
  parser.add_argument(
      "-tc",
      "--text_col",
      type=str,
      default="review",
      help="Column in the dataset that contains the text",
  )
  parser.add_argument(
      "-hl",
      "--human_label",
      type=str,
      default="human",
      help="Label for the human class in the dataset",
  )
  parser.add_argument(
      "-tm",
      "--target_model",
      type=str,
      default="gpt4o",
      help=(
          "Target AI model class to sample alongside human documents, or "
          "'all' to pool every non-human generator (use this for COLING, "
          "whose heads were trained on all generators together)"
      ),
  )
  parser.add_argument(
      "-n",
      "--num_examples",
      type=int,
      default=None,
      help="Size of the sampled subpopulation (defaults to full dataset)",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed for subpopulation sampling",
  )
  parser.add_argument(
      "--stratified",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "Draw an equal number of human and target-model documents "
          "(use --no-stratified for a plain unstratified random sample)"
      ),
  )
  parser.add_argument(
      "-o",
      "--output_path",
      type=str,
      default="counterfactual_ablation.jsonl",
      help="Path for the per-(doc, feature) results to be written to",
  )
  args = parser.parse_args()
  run(args)
