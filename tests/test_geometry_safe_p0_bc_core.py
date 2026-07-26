from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.experiment.p0_protocol import load_p0_config
from cure_lite.experiment.p0_support import _TargetRecord, _oof_auc
from tools.run_geometry_safe_p0_bc import (
    _p0_b_screening,
    _p0_c_screening,
)


_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "protocols"
    / "IRSTD-1K"
    / "p0_v1"
    / "p0_config.json"
)


def _records() -> tuple[_TargetRecord, ...]:
    rows: list[_TargetRecord] = []
    for index in range(6):
        group_id = f"group-{index}"
        base = float(index + 1)
        for role_index, role in enumerate(("factual", "legal")):
            offset = 0.15 * role_index
            rows.append(
                _TargetRecord(
                    identity=(
                        f"{group_id}-{role}",
                        role_index + 1,
                        None if role == "factual" else index + 10,
                    ),
                    sample_id=f"{group_id}-{role}",
                    group_id=group_id,
                    role=role,
                    hand=torch.tensor(
                        [base + offset, base * base + offset],
                        dtype=torch.float64,
                    ),
                    joint_feature_raw=torch.tensor(
                        [
                            base + offset,
                            base * base + 2.0 * offset,
                            base * base * base - offset,
                        ],
                        dtype=torch.float64,
                    ),
                    joint_occupancy_raw=torch.tensor(
                        [base / 10.0, role_index / 4.0],
                        dtype=torch.float64,
                    ),
                )
            )
    return tuple(rows)


def _configs():
    config = load_p0_config(_CONFIG)
    overlap = replace(config.overlap, joint_feature_components=1)
    separability = replace(
        config.separability,
        folds=3,
        bootstrap_replicates=25,
    )
    return overlap, separability


def _without_fingerprint(
    payload: dict[str, object],
    field: str = "parameter_fingerprint",
) -> dict[str, object]:
    return {key: value for key, value in payload.items() if key != field}


def _population_fingerprint(
    records: tuple[_TargetRecord, ...],
    groups: set[str],
    *,
    role: str | None = None,
) -> str:
    return stable_fingerprint(
        [
            {
                "identity": list(item.identity),
                "group_id": item.group_id,
                "role": item.role,
            }
            for item in records
            if item.group_id in groups and (role is None or item.role == role)
        ]
    )


def _assert_fold_group_separation(
    folds: list[dict[str, object]],
) -> None:
    observed_test_groups: list[str] = []
    for fold in folds:
        train_groups = set(fold["train_groups"])
        test_groups = set(fold["test_groups"])
        assert train_groups.isdisjoint(test_groups)
        assert train_groups | test_groups == {
            f"group-{index}" for index in range(6)
        }
        observed_test_groups.extend(fold["test_groups"])
    assert sorted(observed_test_groups) == [
        f"group-{index}" for index in range(6)
    ]


def test_handcrafted_oof_fold_receipts_bind_all_training_roles() -> None:
    records = _records()
    overlap, separability = _configs()
    receipt = _oof_auc(
        records,
        space="handcrafted",
        overlap=overlap,
        config=separability,
    )
    folds = receipt["folds"]

    _assert_fold_group_separation(folds)
    for fold in folds:
        train_groups = set(fold["train_groups"])
        assert fold["projection_fit"] is None

        scale_fit = fold["scale_fit"]
        assert scale_fit["fit_role"] == "training-fold-all-roles"
        assert scale_fit["fit_targets"] == 2 * len(train_groups)
        assert scale_fit["fit_groups"] == len(train_groups)
        assert scale_fit["fit_population_fingerprint"] == (
            _population_fingerprint(records, train_groups)
        )
        assert len(scale_fit["median"]) == fold["dimensions"]
        assert len(scale_fit["scale"]) == fold["dimensions"]
        assert scale_fit["parameter_fingerprint"] == stable_fingerprint(
            _without_fingerprint(scale_fit)
        )

        classifier = fold["classifier_parameters"]
        assert len(classifier["coefficient_with_intercept"]) == (
            fold["dimensions"] + 1
        )
        assert classifier["parameter_fingerprint"] == stable_fingerprint(
            _without_fingerprint(classifier)
        )


def test_joint_oof_projection_is_fit_only_on_training_legal_targets() -> None:
    records = _records()
    overlap, separability = _configs()
    receipt = _oof_auc(
        records,
        space="joint",
        overlap=overlap,
        config=separability,
    )
    folds = receipt["folds"]

    _assert_fold_group_separation(folds)
    for fold in folds:
        train_groups = set(fold["train_groups"])
        projection = fold["projection_fit"]
        assert projection["fit_role"] == "training-fold-legal-targets-only"
        assert projection["fit_targets"] == len(train_groups)
        assert projection["fit_groups"] == len(train_groups)
        assert projection["fit_population_fingerprint"] == (
            _population_fingerprint(records, train_groups, role="legal")
        )
        assert projection["fit_population_fingerprint"] != (
            _population_fingerprint(records, train_groups)
        )
        assert len(projection["raw_median"]) == 3
        assert len(projection["raw_scale"]) == 3
        assert len(projection["pca_mean"]) == 3
        assert len(projection["basis"]) == overlap.joint_feature_components
        assert all(len(row) == 3 for row in projection["basis"])
        assert len(projection["singular_values"]) == (
            overlap.joint_feature_components
        )
        assert projection["parameter_fingerprint"] == stable_fingerprint(
            _without_fingerprint(projection)
        )

        scale_fit = fold["scale_fit"]
        assert scale_fit["fit_role"] == "training-fold-all-roles"
        assert scale_fit["fit_population_fingerprint"] == (
            _population_fingerprint(records, train_groups)
        )
        assert scale_fit["parameter_fingerprint"] == stable_fingerprint(
            _without_fingerprint(scale_fit)
        )

        classifier = fold["classifier_parameters"]
        assert classifier["parameter_fingerprint"] == stable_fingerprint(
            _without_fingerprint(classifier)
        )


@pytest.mark.parametrize(
    ("covered", "expected"),
    ((29, "pass"), (28, "fail")),
)
def test_p0_b_screening_uses_the_frozen_29_of_32_gate(
    covered: int,
    expected: str,
) -> None:
    coverage = {
        "factual_total": 32,
        "covered_factual_targets": covered,
        "required_covered_factual_targets": 29,
        "pass": covered >= 29,
    }
    state, spaces = _p0_b_screening(
        {
            "coverage": {
                "handcrafted": dict(coverage),
                "decoder_joint": dict(coverage),
            }
        }
    )
    assert state == expected
    assert spaces == {
        "handcrafted_knn_coverage": expected,
        "decoder_joint_knn_coverage": expected,
    }


@pytest.mark.parametrize(
    ("lower", "upper", "mmd_pass", "expected_auc", "expected_mmd", "expected"),
    (
        (0.40, 0.70, True, "pass", "pass", "pass"),
        (0.71, 0.90, True, "fail", "pass", "fail"),
        (0.60, 0.80, True, "inconclusive", "pass", "inconclusive"),
        (0.40, 0.70, False, "pass", "fail", "fail"),
        (0.40, 0.70, None, "pass", "inconclusive", "inconclusive"),
    ),
)
def test_p0_c_screening_combines_auc_interval_and_mmd_three_values(
    lower: float,
    upper: float,
    mmd_pass: bool | None,
    expected_auc: str,
    expected_mmd: str,
    expected: str,
) -> None:
    classifier = {"bootstrap": {"lower": lower, "upper": upper}}
    mmd = {} if mmd_pass is None else {"pass": mmd_pass}
    state, spaces = _p0_c_screening(
        {
            "grouped_classifier": {
                "handcrafted": dict(classifier),
                "decoder_joint": dict(classifier),
            },
            "mmd": {
                "handcrafted": dict(mmd),
                "decoder_joint": dict(mmd),
            },
        },
        auc_maximum=0.70,
    )
    assert state == expected
    for space in ("handcrafted", "decoder_joint"):
        assert spaces[space] == {
            "auc_bootstrap_interval": expected_auc,
            "mmd_against_legal_reference": expected_mmd,
            "space_status": expected,
        }
