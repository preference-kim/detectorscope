from types import SimpleNamespace
import json
import weakref

import pytest
import safetensors.torch
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import Gemma2Config, Gemma2ForCausalLM, PreTrainedTokenizerFast

from detectorscope.documents import DocumentAnalyzer, split_document
from detectorscope.inference import OptimizedDetector, _SelectedEncoder
from detectorscope.model.detectors.auditable import AuditableDetector, Detection
from detectorscope.model.detectors.heads.logreg import LogRegHead
from detectorscope.model.extractor import HFExtractor
from detectorscope.model.sae import JumpReLUSAE
from detectorscope.streaming import StreamingExtractor, TensorReader, _FusedRMSNorm
from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm


@pytest.fixture
def tokenizer():
  vocab = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2}
  vocab.update({f"word{i}": i + 3 for i in range(60)})
  tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
  tok.pre_tokenizer = pre_tokenizers.Whitespace()
  tok.post_processor = processors.TemplateProcessing(
      single="[BOS] $A", special_tokens=[("[BOS]", 2)]
  )
  return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]",
                                 unk_token="[UNK]", bos_token="[BOS]")


@pytest.fixture
def source(tmp_path, tokenizer):
  torch.manual_seed(12)
  config = Gemma2Config(vocab_size=64, hidden_size=32, intermediate_size=48,
                        num_hidden_layers=4, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=8,
                        max_position_embeddings=128, sliding_window=64,
                        pad_token_id=0, bos_token_id=2, eos_token_id=1,
                        attn_implementation="eager")
  extractor = HFExtractor.__new__(HFExtractor)
  extractor.config = config
  extractor.model = Gemma2ForCausalLM(config).eval()
  extractor.tokenizer = tokenizer
  extractor.device = "cpu"
  extractor.tokenizer_only = False
  extractor.default_hooks = ["model.layers.1", "model.layers.3"]
  extractor.captured_activation = {}
  det = AuditableDetector(None, extractor.default_hooks, d_model=32, d_sae=12)
  det.extractor = extractor
  det.saes[1].use_pre_enc_bias = True
  det.saes[1].b_dec.data.fill_(0.1)
  weight = torch.zeros(1, 24)
  weight[0, [1, 3, 8, 14, 20]] = torch.tensor([0.2, -0.3, 0.7, 0.5, -0.1])
  path = str(tmp_path / "head.safetensors")
  safetensors.torch.save_file({"weight": weight, "bias": torch.tensor([0.25])},
                             path, metadata={"threshold": "0.4"})
  head = LogRegHead(path)
  head.interp_mask[3] = 0
  det.add_head(head, *det.default)
  det.feature_table = {i: {"description": {"terse": f"feature {i}"}}
                       for i in range(24)}
  return det


def test_compact_scores_features_and_highlights(source):
  texts = ["word1 word4 word9 word2", "word3", ""]
  expected = source(texts, return_highlights=True)
  runner = OptimizedDetector(source, "cpu")
  actual = runner(texts, return_highlights=True, batch_size=2)
  assert len(runner.feature_ids) == 4
  assert sum(e.weight.numel() for e in runner.encoders) == 32 * 4
  for before, after in zip(expected, actual):
    assert after.logit == pytest.approx(before.logit, abs=2e-6)
    assert after.label == before.label
    assert after.tokens == before.tokens
    a, b = {f.id: f for f in before.features}, {f.id: f for f in after.features}
    assert a.keys() == b.keys()
    for fid in a:
      assert a[fid].description == b[fid].description
      assert a[fid].importance == pytest.approx(b[fid].importance, abs=2e-6)
      assert [t for t, _ in a[fid].highlights] == [t for t, _ in b[fid].highlights]
      assert [v for _, v in a[fid].highlights] == pytest.approx(
          [v for _, v in b[fid].highlights], abs=2e-6
      )


def test_tail_is_never_executed_and_hooks_removed(source):
  def fail(*args):
    raise AssertionError("Unused block ran")
  source.extractor.model.model.layers[3].forward = fail
  runner = OptimizedDetector(source, "cpu")
  runner("word1 word2")
  assert all(not m._forward_pre_hooks and not m._forward_hooks
             for m in source.extractor.model.modules())


def test_real_model_errors_propagate_and_hooks_are_removed(source):
  def fail(*args, **kwargs):
    raise RuntimeError("fixture model failure")
  source.extractor.model.model.layers[0].forward = fail
  runner = OptimizedDetector(source, "cpu")
  with pytest.raises(RuntimeError, match="fixture model failure"):
    runner("word1")
  assert all(not m._forward_pre_hooks and not m._forward_hooks
             for m in source.extractor.model.modules())


def test_all_zero_head_and_empty_batch(source):
  source.load_head().detector.weight.data.zero_()
  runner = OptimizedDetector(source, "cpu")
  assert runner([]) == []
  output = runner("word1", return_highlights=True)
  assert output.logit == pytest.approx(0.25)
  assert output.features == []


def test_selected_encoder_threshold_boundary_and_pre_bias():
  sae = JumpReLUSAE(2, 3)
  sae.W_enc.data.copy_(torch.tensor([[1., 0., 1.], [0., 1., 1.]]))
  sae.b_enc.data.zero_()
  sae.log_threshold.data.zero_()
  sae.use_pre_enc_bias = True
  sae.b_dec.data.fill_(0.1)
  x = torch.tensor([[1.1, 0.1], [1.1001, 0.1], [1.0999, 0.1]])
  selected = torch.tensor([0, 2])
  torch.testing.assert_close(_SelectedEncoder(sae, selected)(x),
                             sae.encode(x)[:, selected], rtol=0, atol=0)


@pytest.mark.parametrize("text", ["", " ", "word1 " * 2100,
                                  "한글과 emoji 👩🏽‍💻\nword1 " * 80])
def test_windows_cover_every_character(text, tokenizer):
  windows = split_document(text, tokenizer, max_tokens=32)
  assert "".join(w.text for w in windows) == text
  assert all(w.tokens <= 32 and text[w.start:w.end] == w.text for w in windows)


def test_cache_reuses_exact_windows_but_rechecks_edits(tokenizer):
  class FakeDetector:
    extractor = SimpleNamespace(tokenizer=tokenizer)
    identity = {"revision": "fixture"}
    calls = 0

    def __call__(self, texts, **kwargs):
      self.calls += len(texts)
      return [Detection(0.2, 0.2, "human", []) for _ in texts]

  detector = FakeDetector()
  analyzer = DocumentAnalyzer(detector, max_tokens=32, cache_size=1)
  first = analyzer.analyze("word1 word2")
  again = analyzer.analyze("word1 word2")
  changed = analyzer.analyze("word1 word3")
  assert first["computed_windows"] == 1
  assert again["cache_hits"] == 1 and again["computed_windows"] == 0
  assert changed["computed_windows"] == 1 and detector.calls == 2
  assert len(analyzer.cache) == 1


def test_streamed_layers_match_original_and_skip_unused_weights(source, tmp_path):
  root = tmp_path / "extractor"
  source.extractor.model.save_pretrained(root, max_shard_size="10KB")
  source.extractor.tokenizer.save_pretrained(root)
  extractor = StreamingExtractor(root, source.hooks, attention_implementation="eager")
  loaded = []
  original_load = extractor._load_layer
  def load(index):
    loaded.append(index)
    return original_load(index)
  extractor._load_layer = load
  texts = ["word1 word4 word2", "word3"]
  with torch.no_grad():
    expected = source.extractor.get_activations(texts)
  actual, mask = extractor.get_activations(texts, return_mask=True)
  for key in expected:
    # Unused padding activations are unspecified once its MLP work is skipped.
    torch.testing.assert_close(actual[key][mask.bool()], expected[key][mask.bool()],
                               rtol=1e-5, atol=1e-6)
  assert loaded == [0, 1, 2]
  assert extractor.profile["embedding_rows_read"] <= 6
  assert extractor.profile["max_cpu_prefetch_blocks"] == 1
  assert not hasattr(extractor, "model")


def test_streamed_factory_never_calls_full_loader(source, tmp_path, monkeypatch):
  source.saes[1].use_pre_enc_bias = False
  root = tmp_path / "checkpoint"
  (root / "heads").mkdir(parents=True)
  source.extractor.model.save_pretrained(root / "extractor")
  source.extractor.tokenizer.save_pretrained(root / "extractor")
  safetensors.torch.save_file({k: v.contiguous() for k, v in source.state_dict().items()},
                             str(root / "model.safetensors"))
  source.load_head().save(str(root / "heads/logreg_tau0.85.safetensors"))
  (root / "config.json").write_text(json.dumps({"hook_points": source.hooks,
                                               "head": "logreg", "tau": .85}))
  (root / "features.jsonl").write_text("\n".join(json.dumps({"id": i, **r})
                                                 for i, r in source.feature_table.items()))
  def forbidden(*args, **kwargs):
    raise AssertionError("Full-model loader must not run")
  monkeypatch.setattr(AuditableDetector, "from_pretrained", forbidden)
  texts = ["word1 word4 word2", "word3"]
  expected = source(texts, return_highlights=True)
  detector = OptimizedDetector.from_pretrained(str(root), device="cpu",
      local_files_only=True, attention_implementation="eager", compact_mlp=True)
  actual = detector(texts, return_highlights=True, batch_size=2)
  for before, after in zip(expected, actual):
    assert after.logit == pytest.approx(before.logit, abs=2e-6)
    assert [f.id for f in after.features] == [f.id for f in before.features]
    assert after.tokens == before.tokens
  assert len(detector.explainer.feature_table) == 4
  assert detector.identity["compact_mlp"] is True
  assert detector.identity["fused_norm"] is False  # CPU remains on Gemma's path.
  assert detector.identity["reuse_mps_cache"] is False
  assert detector.identity["shared_prefetch"] is False


@pytest.mark.parametrize("padding_side", ["right", "left"])
@pytest.mark.parametrize("compact_mlp", [False, True])
@pytest.mark.parametrize("compact_attention", [False, True])
def test_streamed_batch_padding_preserves_independent_chunks(
    source, tmp_path, padding_side, compact_mlp, compact_attention):
  root = tmp_path / "batch-extractor"
  source.extractor.model.save_pretrained(root)
  source.extractor.tokenizer.save_pretrained(root)
  extractor = StreamingExtractor(root, source.hooks, attention_implementation="sdpa",
                                 compact_mlp=compact_mlp, compact_attention=compact_attention)
  extractor.tokenizer.padding_side = padding_side
  # Exceed the fixture's 64-token sliding window: only the full-attention mask
  # may be omitted. Test empty and padded chunks against independent forwards.
  texts = ["word1 word4 word2", "word3", "", "word9 word8 " * 36]
  expected = [extractor.get_activations(text) for text in texts]
  # Observe rows reaching the original MLP, independently of profile metadata.
  shapes = []
  original_load = extractor._load_layer
  def load(index):
    layer, size, seconds = original_load(index)
    layer.mlp.register_forward_pre_hook(lambda module, args: shapes.append(args[0].shape))
    return layer, size, seconds
  extractor._load_layer = load
  actual, mask = extractor.get_activations(texts, return_mask=True)
  assert extractor.profile["batch_size"] == 4
  assert len(extractor.profile["stages"]) == 3  # One block load shared by all chunks.
  assert extractor.profile["mlp_tokens"] == (
      extractor.profile["real_tokens"] if compact_mlp else mask.numel())
  assert extractor.profile["attention_tokens"] == (
      extractor.profile["real_tokens"] if compact_attention else mask.numel())
  assert all(tuple(shape[:-1]) == ((int(mask.sum()),) if compact_mlp else tuple(mask.shape))
             for shape in shapes)
  assert len(shapes) == 3
  masks = extractor.profile["attention_masks"]
  assert (masks["full_attention"] is None) == (padding_side == "right")
  assert masks["sliding_attention"] is not None
  for i, single in enumerate(expected):
    for hook in single:
      torch.testing.assert_close(actual[hook][i][mask[i].bool()], single[hook][0],
                                 rtol=1e-5, atol=1e-6)


def test_tiny_padding_keeps_the_original_mlp_path(source, tmp_path):
  root = tmp_path / "tiny-padding-extractor"
  source.extractor.model.save_pretrained(root)
  source.extractor.tokenizer.save_pretrained(root)
  extractor = StreamingExtractor(root, source.hooks, attention_implementation="eager")
  shapes = []
  original_load = extractor._load_layer
  def load(index):
    layer, size, seconds = original_load(index)
    layer.mlp.register_forward_pre_hook(lambda module, args: shapes.append(args[0].shape))
    return layer, size, seconds
  extractor._load_layer = load
  texts = ["word1 word2", "word3"]
  with torch.no_grad():
    expected = source.extractor.get_activations(texts)
  actual = extractor.get_activations(texts)
  assert extractor.profile["padded_tokens"] - extractor.profile["real_tokens"] == 1
  assert extractor.profile["mlp_tokens"] == extractor.profile["padded_tokens"]
  assert shapes == [torch.Size((2, 3, 32))] * 3
  for hook in actual:
    torch.testing.assert_close(actual[hook], expected[hook], rtol=1e-5, atol=1e-6)


def test_coalesced_reads_reuse_one_slot_and_do_not_read_gaps(tmp_path):
  tensors = {name: torch.arange(size, dtype=torch.float32)
             for name, size in (("a", 6), ("b", 5), ("c", 4), ("d", 3))}
  safetensors.torch.save_file(tensors, str(tmp_path / "model.safetensors"))
  reader = TensorReader(tmp_path)
  with reader.staging(True):
    first = reader.read_many(["b", "a"])
    pointer = first["a"].untyped_storage().data_ptr()
    assert first["b"].untyped_storage().data_ptr() == pointer
    for key in first:
      torch.testing.assert_close(first[key], tensors[key], rtol=0, atol=0)
    del first
    again = reader.read_many(["a", "b"])
    assert again["a"].untyped_storage().data_ptr() == pointer
    assert reader.read_stats["allocations"] == 1
    assert reader.read_stats["reuses"] == 1
    del again
    separate = reader.read_many(["a", "c"])
    assert reader._buffer is None
    assert sum(x.untyped_storage().nbytes() for x in separate.values()) == (
        tensors["a"].nbytes + tensors["c"].nbytes)
    for key in separate:
      torch.testing.assert_close(separate[key], tensors[key], rtol=0, atol=0)
  assert reader._buffer is None and not reader._reuse_buffer


@pytest.mark.parametrize("shared_buffer", [False, True])
def test_coalesced_reads_respect_shards_and_release_buffer_on_error(tmp_path, shared_buffer):
  weights = {"a": torch.arange(6, dtype=torch.float32),
             "b": torch.arange(4, dtype=torch.float32)}
  for key, value in weights.items():
    safetensors.torch.save_file({key: value}, str(tmp_path / f"{key}.safetensors"))
  (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
      "weight_map": {key: f"{key}.safetensors" for key in weights}}))
  reader = TensorReader(tmp_path)
  buffer = torch.empty(10, dtype=torch.float32).numpy() if shared_buffer else None
  with reader.staging(True):
    actual = reader.read_many(weights, buffer=buffer)
    assert reader._buffer is None
    for key in weights:
      torch.testing.assert_close(actual[key], weights[key], rtol=0, atol=0)
    if shared_buffer:
      assert reader.read_stats["allocations"] == 0
      assert actual["b"].data_ptr() == actual["a"].data_ptr() + weights["a"].nbytes
      with pytest.raises(ValueError, match="does not fit"):
        reader.read_many(weights, buffer=buffer[:5])
  with (tmp_path / "a.safetensors").open("r+b") as file:
    file.truncate(reader.tensors["a"][1] + weights["a"].nbytes - 1)
  with pytest.raises(IOError, match="Incomplete tensor group"):
    with reader.staging(True):
      reader.read_many(["a"], buffer=buffer)
  assert reader._buffer is None and not reader._reuse_buffer


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires MPS")
@pytest.mark.parametrize("fused_norm", [False, True])
@pytest.mark.parametrize("reuse_mps_cache", [False, True])
@pytest.mark.parametrize("shared_prefetch", [False, True])
def test_mps_coalesced_pipeline_matches_reference_and_releases_weights(
    source, tmp_path, fused_norm, reuse_mps_cache, shared_prefetch):
  root = tmp_path / "mps-extractor"
  source.extractor.model.save_pretrained(root, max_shard_size="10KB" if shared_prefetch else "5GB")
  source.extractor.tokenizer.save_pretrained(root)
  source.extractor.config._attn_implementation = "sdpa"
  texts = ["word1 word4 word2", "word9 word8 " * 36]
  with torch.no_grad():
    expected, mask = source.extractor.get_activations(texts, return_mask=True)
  extractor = StreamingExtractor(root, source.hooks, device="mps", fused_norm=fused_norm,
                                 reuse_mps_cache=reuse_mps_cache, shared_prefetch=shared_prefetch)
  releases = []
  release_cache = extractor._release_cache
  def observe_release():
    releases.append(len(references))
    release_cache()
  extractor._release_cache = observe_release
  references = []
  cpu_storages = []
  gpu_weight_bytes = []
  load = extractor._load_layer
  transfer = extractor._transfer_layer
  def record(index):
    layer, size, seconds = load(index)
    assert isinstance(layer.input_layernorm, _FusedRMSNorm) == fused_norm
    # Wrapping normalization must neither transform nor duplicate weights.
    expected_state = source.extractor.model.model.layers[index].state_dict()
    for name, value in layer.state_dict().items():
      assert value.dtype == torch.float32
      torch.testing.assert_close(value, expected_state[name], rtol=0, atol=0)
    layer.fixture_index = index
    references.append(weakref.ref(layer))
    return layer, size, seconds
  extractor._load_layer = record
  def observe_transfer(layer):
    pointer = next(layer.parameters()).untyped_storage().data_ptr()
    if shared_prefetch:
      slot = extractor._shared_cpu_slots[layer.fixture_index % 2]
      start = slot.__array_interface__["data"][0]
      assert start <= pointer < start + slot.nbytes
      cpu_storages.append(start)
    else:
      cpu_storages.append(pointer)
    transfer(layer)
    if shared_prefetch:
      expected_state = source.extractor.model.model.layers[layer.fixture_index].state_dict()
      for name, value in layer.state_dict().items():
        torch.testing.assert_close(value.cpu(), expected_state[name], rtol=0, atol=0)
    storages = {p.untyped_storage().data_ptr(): p.untyped_storage().nbytes()
                for p in layer.parameters()}
    gpu_weight_bytes.append(sum(storages.values()))
  extractor._transfer_layer = observe_transfer
  actual = extractor.get_activations(texts)
  for hook in actual:
    torch.testing.assert_close(actual[hook][mask.bool()], expected[hook][mask.bool()],
                               rtol=1e-4, atol=1e-5)
  io = extractor.profile["coalesced_weight_io"]
  if shared_prefetch:
    assert io["allocations"] == 0 and io["reuses"] > 3  # Multiple shards per block.
    assert io["read_calls"] == io["reuses"]
  else:
    assert io["allocations"] == 1 and io["reuses"] == 2
    assert io["read_calls"] == 3
  assert io["bytes"] == 3 * extractor.profile["layer_weight_bytes"]
  assert io["max_staging_bytes"] == extractor.profile["layer_weight_bytes"]
  assert extractor.reader._buffer is None
  assert extractor._shared_cpu_slots is None and extractor._shared_gpu_slots is None
  assert len(cpu_storages) == 3 and len(set(cpu_storages)) == (2 if shared_prefetch else 1)
  assert gpu_weight_bytes == [extractor.profile["layer_weight_bytes"]] * 3
  assert all(ref() is None for ref in references)
  assert extractor.profile["fused_norm"] == fused_norm
  assert extractor.profile["reuse_mps_cache"] == reuse_mps_cache
  assert extractor.profile["shared_prefetch"] == shared_prefetch
  assert len(releases) == (1 if reuse_mps_cache else 3 + int(shared_prefetch))
  def fail(layer):
    transfer(layer)  # Fail after async copies were queued, before the caller's fence.
    raise RuntimeError("fixture transfer failure")
  extractor._transfer_layer = fail
  releases.clear()
  with pytest.raises(RuntimeError, match="fixture transfer failure"):
    extractor.get_activations(texts)
  assert extractor.reader._buffer is None and not extractor.reader._reuse_buffer
  assert extractor._shared_cpu_slots is None and extractor._shared_gpu_slots is None
  assert len(releases) == int(reuse_mps_cache or shared_prefetch)
  extractor._transfer_layer = observe_transfer
  recovered = extractor.get_activations(texts)
  for hook in recovered:
    torch.testing.assert_close(recovered[hook][mask.bool()], expected[hook][mask.bool()],
                               rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires MPS")
@pytest.mark.parametrize("failure_stage", ["prefetch", "compute"])
def test_shared_prefetch_drains_work_after_failure(source, tmp_path, failure_stage):
  root = tmp_path / "shared-extractor"
  source.extractor.model.save_pretrained(root, max_shard_size="10KB")
  source.extractor.tokenizer.save_pretrained(root)
  extractor = StreamingExtractor(root, source.hooks, device="mps", shared_prefetch=True)
  load = extractor._load_layer
  references = []
  def fail(index):
    layer, size, seconds = load(index)
    references.append(weakref.ref(layer))
    if failure_stage == "prefetch" and index == 1:
      raise IOError("fixture prefetch failure")
    if failure_stage == "compute" and index == 0:
      forward = layer.forward
      def queued_failure(*args, **kwargs):
        forward(*args, **kwargs)  # GPU work is queued before the exception.
        raise IOError("fixture compute failure")
      layer.forward = queued_failure
    return layer, size, seconds
  extractor._load_layer = fail
  with pytest.raises(IOError, match=f"fixture {failure_stage} failure"):
    extractor.get_activations("word1 word4")
  assert extractor._shared_cpu_slots is None and extractor._shared_gpu_slots is None
  assert extractor.reader._buffer is None and not extractor.reader._reuse_buffer
  # The injected forward closes over a bound method, so collect its fixture cycle.
  import gc
  gc.collect()
  assert all(ref() is None for ref in references)
  extractor._load_layer = load
  actual = extractor.get_activations("word1 word4")
  extractor.shared_prefetch = False
  expected = extractor.get_activations("word1 word4")
  for hook in actual:
    torch.testing.assert_close(actual[hook], expected[hook], rtol=0, atol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires MPS")
def test_shared_prefetch_without_executed_blocks(source, tmp_path):
  root = tmp_path / "embedding-only"
  source.extractor.model.save_pretrained(root)
  source.extractor.tokenizer.save_pretrained(root)
  extractor = StreamingExtractor(root, ["model.layers.0"], device="mps", shared_prefetch=True)
  actual = extractor.get_activations("word1 word2")
  assert list(actual) == ["model.layers.0_input"]
  assert extractor.profile["stages"] == []
  assert extractor._shared_cpu_slots is None and extractor._shared_gpu_slots is None


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires MPS")
def test_fused_gemma_norm_preserves_weight_and_epsilon():
  torch.manual_seed(83)
  original = Gemma2RMSNorm(2304, eps=1e-6).to("mps").eval()
  with torch.no_grad():
    original.weight.normal_(mean=0.1, std=0.3)
  before = original.weight.detach().cpu().clone()
  fused = _FusedRMSNorm(original)
  assert fused.weight is original.weight
  with torch.inference_mode():
    # Include inputs where epsilon dominates, and noncontiguous token rows.
    for scale in (0.0, 1e-4, 1.0, 100.0):
      hidden = (torch.randn(2, 9, 2304, device="mps") * scale)[:, ::2]
      expected, actual = original(hidden), fused(hidden)
      assert actual.dtype == torch.float32
      torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
  torch.testing.assert_close(original.weight.cpu(), before, rtol=0, atol=0)
