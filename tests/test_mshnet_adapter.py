from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import subprocess
import sys
import textwrap

from PIL import Image
import pytest
import torch
from torch import nn

from cure_lite.adapters import MSHNetAdapter
from cure_lite.cache import file_sha256
from cure_lite.data import PreprocessConfig
from cure_lite.decoder import CURELiteDecoder
from cure_lite.model import CURELiteModel


_MINIMAL_MSHNET_SOURCE = textwrap.dedent(
    """
    import torch
    from torch import nn


    class MSHNet(nn.Module):
        def __init__(self, input_channels):
            super().__init__()
            self.decoder_0 = nn.Sequential(
                nn.Conv2d(input_channels, 16, kernel_size=1),
                nn.BatchNorm2d(16),
                nn.SiLU(),
            )
            self.output = nn.Conv2d(16, 1, kernel_size=1)

        def forward(self, images, warm_flag):
            if warm_flag is not True:
                raise ValueError("warm_flag must be true")
            feature = self.decoder_0(images)
            logits = self.output(feature)
            return [logits, logits, logits, logits], logits
    """
).lstrip()


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_model_type(source: Path) -> type[nn.Module]:
    module_name = f"_test_mshnet_{file_sha256(source)[:16]}_{id(source)}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import fixture MSHNet source")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    model_type = getattr(module, "MSHNet", None)
    if not isinstance(model_type, type) or not issubclass(model_type, nn.Module):
        raise RuntimeError("fixture source does not expose MSHNet")
    return model_type


@dataclass(frozen=True)
class SyntheticUpstream:
    repository: Path
    source: Path
    checkpoint: Path
    adapter_type: type[MSHNetAdapter]
    preprocessing: PreprocessConfig


@pytest.fixture
def synthetic_upstream(tmp_path: Path) -> SyntheticUpstream:
    repository = tmp_path / "mshnet-upstream"
    source = repository / "model" / "MSHNet.py"
    source.parent.mkdir(parents=True)
    source.write_text(_MINIMAL_MSHNET_SOURCE, encoding="utf-8")

    subprocess.run(("git", "init", "-q", str(repository)), check=True)
    _git(repository, "config", "user.name", "CURE-Lite Test")
    _git(repository, "config", "user.email", "cure-lite@example.invalid")
    _git(repository, "add", "model/MSHNet.py")
    _git(repository, "commit", "-q", "-m", "pinned synthetic MSHNet")

    commit = _git(repository, "rev-parse", "HEAD")
    tree = _git(repository, "rev-parse", "HEAD^{tree}")
    source_sha256 = file_sha256(source)
    adapter_type = type(
        "SyntheticPinnedMSHNetAdapter",
        (MSHNetAdapter,),
        {
            "PINNED_UPSTREAM_COMMIT": commit,
            "PINNED_UPSTREAM_TREE": tree,
            "PINNED_MODEL_SOURCE_SHA256": source_sha256,
        },
    )

    model_type = _load_model_type(source)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(173)
        native_model = model_type(3)
    checkpoint = tmp_path / "raw-state-dict.pt"
    torch.save(native_model.state_dict(), checkpoint)
    return SyntheticUpstream(
        repository=repository,
        source=source,
        checkpoint=checkpoint,
        adapter_type=adapter_type,
        preprocessing=PreprocessConfig(height=16, width=16),
    )


def _adapter(
    upstream: SyntheticUpstream,
    *,
    checkpoint: Path | None = None,
    expected_checkpoint_sha256: str | None = None,
    preprocessing: PreprocessConfig | None = None,
    device: torch.device | str = "cpu",
) -> MSHNetAdapter:
    checkpoint = checkpoint or upstream.checkpoint
    return upstream.adapter_type(
        upstream.repository,
        checkpoint,
        expected_checkpoint_sha256=(
            expected_checkpoint_sha256 or file_sha256(checkpoint)
        ),
        base_training_provenance_fingerprint="d" * 64,
        base_training_final_receipt_sha256="e" * 64,
        preprocessing=preprocessing or upstream.preprocessing,
        device=device,
    )


def _native_output_and_feature(
    source: Path,
    checkpoint: Path,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    model_type = _load_model_type(source)
    model = model_type(3)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval().requires_grad_(False)
    captured: list[torch.Tensor] = []

    def capture_feature(_module, _inputs, output) -> None:
        captured.append(output.detach().clone())

    handle = model.decoder_0.register_forward_hook(capture_feature)
    try:
        with torch.no_grad():
            _, native_logits = model(images, warm_flag=True)
    finally:
        handle.remove()
    assert len(captured) == 1
    return native_logits, captured[0]


def test_checkpoint_load_matches_native_probability_and_decoder_feature(
    synthetic_upstream: SyntheticUpstream,
) -> None:
    images = torch.randn(2, 3, 16, 16, dtype=torch.float32)
    native_logits, native_feature = _native_output_and_feature(
        synthetic_upstream.source,
        synthetic_upstream.checkpoint,
        images,
    )

    with _adapter(synthetic_upstream) as adapter:
        output = adapter.extract(images)
        assert adapter.checkpoint_sha256 == file_sha256(
            synthetic_upstream.checkpoint
        )
        assert adapter.upstream_commit == adapter.PINNED_UPSTREAM_COMMIT
        assert adapter.upstream_tree == adapter.PINNED_UPSTREAM_TREE

    torch.testing.assert_close(
        output.probability,
        torch.sigmoid(native_logits.float()),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(output.feature, native_feature, rtol=0, atol=0)
    assert output.probability.dtype == torch.float32
    assert output.feature.dtype == torch.float32
    assert not output.probability.requires_grad
    assert not output.feature.requires_grad


def test_checkpoint_digest_mismatch_is_rejected(
    synthetic_upstream: SyntheticUpstream,
) -> None:
    with pytest.raises(RuntimeError, match="differs from the expected digest"):
        _adapter(
            synthetic_upstream,
            expected_checkpoint_sha256="0" * 64,
        )


@pytest.mark.parametrize("corruption", ["missing", "unexpected"])
def test_checkpoint_with_bad_keys_is_rejected(
    synthetic_upstream: SyntheticUpstream,
    tmp_path: Path,
    corruption: str,
) -> None:
    state_dict = dict(
        torch.load(
            synthetic_upstream.checkpoint,
            map_location="cpu",
            weights_only=True,
        )
    )
    if corruption == "missing":
        state_dict.pop(next(iter(state_dict)))
    else:
        state_dict["unexpected.weight"] = torch.zeros(1)
    checkpoint = tmp_path / f"bad-{corruption}.pt"
    torch.save(state_dict, checkpoint)

    with pytest.raises(RuntimeError, match="not strictly compatible"):
        _adapter(synthetic_upstream, checkpoint=checkpoint)


def test_preprocess_enforces_rgb_and_fingerprinted_grid(
    synthetic_upstream: SyntheticUpstream,
) -> None:
    gray = Image.new("L", (4, 5), color=128)
    with _adapter(synthetic_upstream) as adapter:
        images = adapter.preprocess(gray)
        assert images.shape == (1, 3, 16, 16)
        assert images.dtype == torch.float32
        assert images.device.type == "cpu"
        pixel = torch.tensor(128.0, dtype=torch.float32) / 255.0
        expected_channels = (
            pixel
            - torch.tensor(
                synthetic_upstream.preprocessing.mean,
                dtype=torch.float32,
            )
        ) / torch.tensor(
            synthetic_upstream.preprocessing.std,
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            images[0, :, 0, 0],
            expected_channels,
            rtol=0,
            atol=0,
        )
        adapter.extract(images)
        with pytest.raises(ValueError, match="evaluation grid"):
            adapter.extract(torch.randn(1, 3, 15, 16))

    monochrome = PreprocessConfig(
        height=16,
        width=16,
        color_mode="L",
        mean=(0.5,),
        std=(0.25,),
    )
    with pytest.raises(ValueError, match="requires RGB"):
        _adapter(synthetic_upstream, preprocessing=monochrome)

    wrong_normalization = PreprocessConfig(
        height=16,
        width=16,
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
    )
    with pytest.raises(ValueError, match="native ImageNet normalization"):
        _adapter(synthetic_upstream, preprocessing=wrong_normalization)

    with pytest.raises(ValueError, match="must be square"):
        _adapter(
            synthetic_upstream,
            preprocessing=PreprocessConfig(height=16, width=32),
        )
    with pytest.raises(ValueError, match="divisible by 16"):
        _adapter(
            synthetic_upstream,
            preprocessing=PreprocessConfig(height=24, width=24),
        )


def test_requested_device_is_canonicalized_to_the_model_device(
    synthetic_upstream: SyntheticUpstream,
) -> None:
    with _adapter(synthetic_upstream, device="cpu:0") as adapter:
        assert adapter.device == torch.device("cpu")
        images = adapter.preprocess(Image.new("RGB", (3, 4), color=64))
        assert images.device == adapter.device
        adapter.extract(images)


def test_cure_update_leaves_all_mshnet_parameters_and_buffers_unchanged(
    synthetic_upstream: SyntheticUpstream,
) -> None:
    adapter = _adapter(synthetic_upstream)
    base = adapter.base
    before = {
        name: value.detach().clone() for name, value in base.state_dict().items()
    }
    model = CURELiteModel(
        adapter,
        CURELiteDecoder(feature_channels=adapter.feature_channels),
    )
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=1e-3)
    try:
        model.train()
        output = model(
            torch.randn(2, 3, 16, 16, dtype=torch.float32),
            residual_threshold=None,
        )
        optimizer.zero_grad(set_to_none=True)
        output.residual_logits.square().mean().backward()
        optimizer.step()
    finally:
        adapter.close()

    assert set(base.state_dict()) == set(before)
    assert all(
        torch.equal(value, before[name])
        for name, value in base.state_dict().items()
    )
    assert all(not parameter.requires_grad for parameter in base.parameters())
    assert all(parameter.grad is None for parameter in base.parameters())
    assert not base.training


def test_close_is_idempotent_removes_hook_and_disables_extract(
    synthetic_upstream: SyntheticUpstream,
) -> None:
    adapter = _adapter(synthetic_upstream)
    feature_module = adapter.base.decoder_0
    assert len(feature_module._forward_hooks) == 1

    adapter.close()
    adapter.close()
    assert len(feature_module._forward_hooks) == 0
    with pytest.raises(RuntimeError, match="is closed"):
        adapter.extract(torch.randn(1, 3, 16, 16))
    with pytest.raises(RuntimeError, match="cannot re-enter"):
        adapter.__enter__()


def test_close_deep_audit_detects_data_write_and_still_removes_hook(
    synthetic_upstream: SyntheticUpstream,
) -> None:
    adapter = _adapter(synthetic_upstream)
    feature_module = adapter.base.decoder_0
    first_parameter = next(adapter.base.parameters())
    first_parameter.data.add_(1.0)

    with pytest.raises(RuntimeError, match="values changed"):
        adapter.close()
    assert len(feature_module._forward_hooks) == 0
    with pytest.raises(RuntimeError, match="is closed"):
        adapter.extract(torch.randn(1, 3, 16, 16))


def test_real_sibling_mshnet_random_weight_interface_parity(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2] / "MSHNet"
    source = repository / "model" / "MSHNet.py"
    if not source.is_file() or not (repository / ".git").exists():
        pytest.skip("pinned sibling MSHNet repository is unavailable")
    if (
        _git(repository, "rev-parse", "HEAD")
        != MSHNetAdapter.PINNED_UPSTREAM_COMMIT
        or _git(repository, "rev-parse", "HEAD^{tree}")
        != MSHNetAdapter.PINNED_UPSTREAM_TREE
        or file_sha256(source) != MSHNetAdapter.PINNED_MODEL_SOURCE_SHA256
    ):
        pytest.skip("sibling MSHNet does not match the adapter's pinned identity")

    model_type = _load_model_type(source)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(947)
        native_model = model_type(3)
    checkpoint = tmp_path / "sibling-random-raw-state-dict.pt"
    torch.save(native_model.state_dict(), checkpoint)
    images = torch.randn(1, 3, 32, 32, dtype=torch.float32)
    native_logits, native_feature = _native_output_and_feature(
        source,
        checkpoint,
        images,
    )

    adapter = MSHNetAdapter(
        repository,
        checkpoint,
        expected_checkpoint_sha256=file_sha256(checkpoint),
        base_training_provenance_fingerprint="d" * 64,
        base_training_final_receipt_sha256="e" * 64,
        preprocessing=PreprocessConfig(height=32, width=32),
        device="cpu",
    )
    try:
        output = adapter.extract(images)
    finally:
        adapter.close()

    torch.testing.assert_close(
        output.probability,
        torch.sigmoid(native_logits.float()),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(output.feature, native_feature, rtol=0, atol=0)
