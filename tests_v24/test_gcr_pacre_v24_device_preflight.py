from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import run_cure_lite_v24_gcr_pacre_bounded_400 as bounded_cli
from tools import run_cure_lite_v24_gcr_pacre_formal_800 as formal_cli
from tools import prepare_cure_lite_v24_gcr_pacre_training_chain as chain_cli


@pytest.mark.parametrize(
    "preflight",
    (
        bounded_cli._preflight_execution_device,
        formal_cli._preflight_execution_device,
        chain_cli._preflight_execution_device,
    ),
)
def test_device_preflight_accepts_cpu_and_forces_context_initialization(
    preflight,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, object]] = []

    def fake_empty(size: object, *, device: object):
        calls.append((size, device))
        return object()

    monkeypatch.setattr(torch, "empty", fake_empty)
    assert preflight("cpu") == "cpu"
    assert calls == [(1, torch.device("cpu"))]


@pytest.mark.parametrize(
    "preflight",
    (
        bounded_cli._preflight_execution_device,
        formal_cli._preflight_execution_device,
        chain_cli._preflight_execution_device,
    ),
)
@pytest.mark.parametrize("device", ("meta", "xpu:0"))
def test_device_preflight_rejects_non_cpu_cuda_types(
    preflight,
    device: str,
) -> None:
    with pytest.raises(ValueError, match="cpu or cuda"):
        preflight(device)


@pytest.mark.parametrize(
    "preflight",
    (
        bounded_cli._preflight_execution_device,
        formal_cli._preflight_execution_device,
        chain_cli._preflight_execution_device,
    ),
)
def test_device_preflight_requires_explicit_cuda_index(preflight) -> None:
    with pytest.raises(ValueError, match="index must be explicit"):
        preflight("cuda")


@pytest.mark.parametrize(
    "preflight",
    (
        bounded_cli._preflight_execution_device,
        formal_cli._preflight_execution_device,
        chain_cli._preflight_execution_device,
    ),
)
def test_device_preflight_checks_cuda_availability(
    preflight,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="not available"):
        preflight("cuda:0")


@pytest.mark.parametrize(
    "preflight",
    (
        bounded_cli._preflight_execution_device,
        formal_cli._preflight_execution_device,
        chain_cli._preflight_execution_device,
    ),
)
def test_device_preflight_checks_cuda_index_range(
    preflight,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="out of range"):
        preflight("cuda:1")


def test_bounded_invalid_device_fails_before_source_audit_marker_or_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    output = tmp_path.resolve()
    authorization_path = output / "authorization.json"
    chain = SimpleNamespace(
        payload={
            "authorization_artifact_path": str(authorization_path),
        }
    )
    authorization = SimpleNamespace(
        output_directory=str(output),
        requested_device="cuda:999",
    )
    monkeypatch.setattr(
        bounded_cli,
        "load_and_verify_gcr_pacre_bounded_chain_config",
        lambda path: chain,
    )
    monkeypatch.setattr(
        bounded_cli,
        "_load_authorization",
        lambda specification, *, chain_config: authorization,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def forbidden(name: str):
        def invoke(*args, **kwargs):
            del args, kwargs
            events.append(name)
            raise AssertionError(f"{name} must not be invoked")

        return invoke

    monkeypatch.setattr(
        bounded_cli,
        "_final_source_closure_audit",
        forbidden("source_audit"),
    )
    monkeypatch.setattr(
        bounded_cli,
        "create_gcr_pacre_bounded_run_start_marker",
        forbidden("marker"),
    )
    monkeypatch.setattr(
        bounded_cli,
        "run_gcr_pacre_paired_bounded_400",
        forbidden("training"),
    )
    with pytest.raises(ValueError, match="out of range"):
        bounded_cli.run_once(
            SimpleNamespace(
                chain_config="/fixed/chain.json",
                input_factory=bounded_cli.FROZEN_INPUT_FACTORY,
                output=str(output),
                authorization_out=str(authorization_path),
                device="cuda:999",
            )
        )
    assert events == []


def test_formal_invalid_device_fails_before_source_audit_marker_or_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    output = tmp_path.resolve()
    authorization = SimpleNamespace(
        output_directory=str(output),
        requested_device="cuda:999",
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def forbidden(name: str):
        def invoke(*args, **kwargs):
            del args, kwargs
            events.append(name)
            raise AssertionError(f"{name} must not be invoked")

        return invoke

    monkeypatch.setattr(
        formal_cli,
        "_final_source_closure_audit",
        forbidden("source_audit"),
    )
    monkeypatch.setattr(
        formal_cli,
        "create_gcr_pacre_formal_run_start_marker",
        forbidden("marker"),
    )
    monkeypatch.setattr(
        formal_cli,
        "run_gcr_pacre_formal_800",
        forbidden("training"),
    )
    with pytest.raises(ValueError, match="out of range"):
        formal_cli._run_one(
            authorization,
            output=output,
            device="cuda:999",
        )
    assert events == []


def test_bounded_verify_only_rejects_non_bound_receipt_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    expected = tmp_path / "expected-result.json"
    supplied = tmp_path / "different-result.json"
    chain = SimpleNamespace(
        payload={"result_artifact_path": str(expected)}
    )
    monkeypatch.setattr(
        bounded_cli,
        "load_and_verify_gcr_pacre_bounded_chain_config",
        lambda path: chain,
    )

    def forbidden_authorization(*args, **kwargs):
        del args, kwargs
        events.append("authorization")
        raise AssertionError("authorization must not be loaded")

    def forbidden_read(*args, **kwargs):
        del args, kwargs
        events.append("receipt_read")
        raise AssertionError("receipt must not be read")

    monkeypatch.setattr(
        bounded_cli,
        "_load_authorization",
        forbidden_authorization,
    )
    monkeypatch.setattr(
        bounded_cli,
        "read_canonical_json",
        forbidden_read,
    )
    with pytest.raises(PermissionError, match="frozen chain"):
        bounded_cli.verify_only(
            SimpleNamespace(
                chain_config="/fixed/chain.json",
                input_factory=bounded_cli.FROZEN_INPUT_FACTORY,
                receipt=str(supplied),
            )
        )
    assert events == []


def test_seal_bounded_invalid_device_invokes_no_chain_or_cache_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def forbidden(name: str):
        def invoke(*args, **kwargs):
            del args, kwargs
            events.append(name)
            raise AssertionError(f"{name} must not be invoked")

        return invoke

    monkeypatch.setattr(
        chain_cli,
        "required_gcr_pacre_bounded_chain_config_path",
        forbidden("chain_path"),
    )
    monkeypatch.setattr(
        chain_cli,
        "_create_parent_once",
        forbidden("cache_parent"),
    )
    monkeypatch.setattr(
        chain_cli,
        "save_formal_cache_neutral_artifact_new",
        forbidden("cache_writer"),
    )
    monkeypatch.setattr(
        chain_cli,
        "seal_gcr_pacre_bounded_chain_config_new",
        forbidden("chain_writer"),
    )
    with pytest.raises(ValueError, match="out of range"):
        chain_cli.seal_bounded(device="cuda:999")
    assert events == []


def test_seal_formal_invalid_device_invokes_no_chain_or_cache_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def forbidden(name: str):
        def invoke(*args, **kwargs):
            del args, kwargs
            events.append(name)
            raise AssertionError(f"{name} must not be invoked")

        return invoke

    monkeypatch.setattr(
        chain_cli,
        "required_gcr_pacre_formal_chain_config_path",
        forbidden("chain_path"),
    )
    monkeypatch.setattr(
        chain_cli,
        "load_and_verify_gcr_pacre_bounded_chain_config",
        forbidden("predecessor_loader"),
    )
    monkeypatch.setattr(
        chain_cli,
        "save_formal_cache_neutral_artifact_new",
        forbidden("cache_writer"),
    )
    monkeypatch.setattr(
        chain_cli,
        "seal_gcr_pacre_formal_chain_config_new",
        forbidden("chain_writer"),
    )
    with pytest.raises(ValueError, match="out of range"):
        chain_cli.seal_formal(
            seed42_device="cpu",
            seed43_device="cuda:999",
        )
    assert events == []
