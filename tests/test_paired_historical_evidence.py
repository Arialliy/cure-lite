from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from cure_lite.experiment.paired_formal_decision import HISTORICAL_COMPARATORS
from cure_lite.experiment.paired_formal_evaluation import (
    load_frozen_comparison_protocol,
)
from cure_lite.experiment.paired_historical_evidence import (
    load_frozen_historical_fx_v3_source,
    load_frozen_historical_fx_v3_sources,
)


_ROOT = Path(__file__).resolve().parents[1]
_COMPARISON = (
    _ROOT / "protocols/IRSTD-1K/paired_formal_evaluation_v1/config.json"
)
_RUN_RELATIVE = {
    42: Path("runs/irstd1k_stage_a_seed42/cure_lite_stage_a_fx_v3"),
    43: Path("runs/irstd1k_stage_a_seed42/cure_lite_stage_a_fx_v3_s43"),
}
_PROTOCOL_RELATIVE = {
    42: Path("protocols/IRSTD-1K/stage_a_seed42_fx_v3"),
    43: Path("protocols/IRSTD-1K/stage_a_seed42_fx_v3_s43"),
}
_RECEIPTS = (
    "anchor.json",
    "calibration.json",
    "config.json",
    "efficiency.json",
    "finalization.json",
    "results.json",
    "support.json",
)
_PROTOCOL_FILES = (
    "protocol_freeze.json",
    "stage_a_config.json",
    "stage_a_decision_rule.json",
)


def _protocol():
    return load_frozen_comparison_protocol(_COMPARISON)


def _copy_receipt_fixture(destination: Path, seed: int = 42) -> tuple[Path, Path]:
    source_run = _ROOT / _RUN_RELATIVE[seed]
    source_protocol = _ROOT / _PROTOCOL_RELATIVE[seed]
    run = destination / _RUN_RELATIVE[seed]
    protocol = destination / _PROTOCOL_RELATIVE[seed]
    (run / "receipts").mkdir(parents=True)
    for directory in ("d_r", "d_v", "decoders"):
        (run / directory).mkdir()
    shutil.copyfile(source_run / "COMPLETE.json", run / "COMPLETE.json")
    for filename in _RECEIPTS:
        shutil.copyfile(
            source_run / "receipts" / filename,
            run / "receipts" / filename,
        )
    protocol.mkdir(parents=True)
    for filename in _PROTOCOL_FILES:
        shutil.copyfile(source_protocol / filename, protocol / filename)
    return run, protocol


def _load_fixture(destination: Path, seed: int = 42):
    run, protocol = _copy_receipt_fixture(destination, seed)
    return load_frozen_historical_fx_v3_source(
        run,
        protocol,
        comparison_protocol=_protocol(),
        seed=seed,
        repository_root=destination,
    )


def _tree_small_file_hashes(root: Path) -> dict[str, str]:
    paths = [
        root / _RUN_RELATIVE[42] / "COMPLETE.json",
        *(
            root / _RUN_RELATIVE[42] / "receipts" / name
            for name in _RECEIPTS
        ),
        *(
            root / _PROTOCOL_RELATIVE[42] / name
            for name in _PROTOCOL_FILES
        ),
    ]
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
    }


def test_authoritative_pair_loads_as_immutable_adapter_inputs() -> None:
    protocol = _protocol()
    sources = load_frozen_historical_fx_v3_sources(
        seed42_run_root=_ROOT / _RUN_RELATIVE[42],
        seed42_protocol_root=_ROOT / _PROTOCOL_RELATIVE[42],
        seed43_run_root=_ROOT / _RUN_RELATIVE[43],
        seed43_protocol_root=_ROOT / _PROTOCOL_RELATIVE[43],
        comparison_protocol=protocol,
    )
    assert tuple(source.seed for source in sources.sources) == (42, 43)
    assert tuple(
        item.method for item in sources.source_for_seed(42).methods
    ) == HISTORICAL_COMPARATORS
    evidence = sources.adapted_evidence(protocol)
    assert len(evidence) == 8
    assert [
        (item.seed, item.method, item.true_targets)
        for item in evidence
    ] == [
        (42, "Base@B", 150),
        (42, "F", 154),
        (42, "F×", 149),
        (42, "U", 151),
        (43, "Base@B", 150),
        (43, "F", 152),
        (43, "F×", 147),
        (43, "U", 152),
    ]
    with pytest.raises(FrozenInstanceError):
        sources.sources = ()  # type: ignore[misc]


def test_loader_is_read_only_and_does_not_open_d_v(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, protocol_root = _copy_receipt_fixture(tmp_path)
    before = _tree_small_file_hashes(tmp_path)
    comparison_protocol = _protocol()
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path.absolute().is_relative_to(tmp_path):
            relative = path.absolute().relative_to(tmp_path)
            if "d_v" in relative.parts:
                pytest.fail("receipt loader opened historical D_V")
        mode = args[0] if args else kwargs.get("mode", "r")
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            pytest.fail("receipt loader attempted to write")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    source = load_frozen_historical_fx_v3_source(
        run,
        protocol_root,
        comparison_protocol=comparison_protocol,
        seed=42,
        repository_root=tmp_path,
    )
    assert source.seed == 42
    assert _tree_small_file_hashes(tmp_path) == before


@pytest.mark.parametrize(
    ("relative", "message"),
    (
        (Path("receipts/results.json"), "receipt hash mismatch"),
        (Path("receipts/calibration.json"), "receipt hash mismatch"),
        (Path("receipts/config.json"), "receipt hash mismatch"),
        (Path("COMPLETE.json"), "COMPLETE fingerprint mismatch"),
    ),
)
def test_loader_rejects_tampered_bound_receipts(
    tmp_path: Path,
    relative: Path,
    message: str,
) -> None:
    run, protocol = _copy_receipt_fixture(tmp_path)
    path = run / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    if relative.name == "results.json":
        payload["methods"]["U"]["pd"] = 0.0
    elif relative.name == "calibration.json":
        payload["methods"]["U"]["protocol"]["selected_threshold"] = 0.98
    elif relative.name == "config.json":
        payload["run_config"]["training"]["global_seed"] = 99
    else:
        payload["status"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        load_frozen_historical_fx_v3_source(
            run,
            protocol,
            comparison_protocol=_protocol(),
            seed=42,
            repository_root=tmp_path,
        )


@pytest.mark.parametrize(
    "filename",
    ("stage_a_config.json", "stage_a_decision_rule.json"),
)
def test_loader_rejects_tampered_frozen_config_hash(
    tmp_path: Path,
    filename: str,
) -> None:
    run, protocol = _copy_receipt_fixture(tmp_path)
    path = protocol / filename
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_frozen_historical_fx_v3_source(
            run,
            protocol,
            comparison_protocol=_protocol(),
            seed=42,
            repository_root=tmp_path,
        )


def test_loader_rejects_frozen_output_path_tamper(tmp_path: Path) -> None:
    run, protocol = _copy_receipt_fixture(tmp_path)
    freeze_path = protocol / "protocol_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["stage_a_output"] = "runs/not-the-frozen-run"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    with pytest.raises(RuntimeError, match="output path changed"):
        load_frozen_historical_fx_v3_source(
            run,
            protocol,
            comparison_protocol=_protocol(),
            seed=42,
            repository_root=tmp_path,
        )


@pytest.mark.parametrize("location", ("run", "receipt", "protocol"))
def test_loader_rejects_noncanonical_inventory(
    tmp_path: Path,
    location: str,
) -> None:
    run, protocol = _copy_receipt_fixture(tmp_path)
    directory = {
        "run": run,
        "receipt": run / "receipts",
        "protocol": protocol,
    }[location]
    (directory / "unexpected").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        load_frozen_historical_fx_v3_source(
            run,
            protocol,
            comparison_protocol=_protocol(),
            seed=42,
            repository_root=tmp_path,
        )


def test_loader_rejects_root_and_member_symlinks(tmp_path: Path) -> None:
    run, protocol = _copy_receipt_fixture(tmp_path)
    linked_root = tmp_path / "linked-run"
    linked_root.symlink_to(run, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        load_frozen_historical_fx_v3_source(
            linked_root,
            protocol,
            comparison_protocol=_protocol(),
            seed=42,
            repository_root=tmp_path,
        )

    results = run / "receipts/results.json"
    real_results = tmp_path / "real-results.json"
    results.rename(real_results)
    results.symlink_to(real_results)
    with pytest.raises(ValueError, match="symlink"):
        load_frozen_historical_fx_v3_source(
            run,
            protocol,
            comparison_protocol=_protocol(),
            seed=42,
            repository_root=tmp_path,
        )


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    run, protocol = _copy_receipt_fixture(tmp_path)
    freeze = protocol / "protocol_freeze.json"
    freeze.write_text('{"schema_version":"a","schema_version":"b"}')
    with pytest.raises(ValueError, match="duplicate"):
        load_frozen_historical_fx_v3_source(
            run,
            protocol,
            comparison_protocol=_protocol(),
            seed=42,
            repository_root=tmp_path,
        )
