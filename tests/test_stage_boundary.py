from __future__ import annotations

from pathlib import Path


def test_active_package_contains_only_cure_lite_stage() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    package_root = repository_root / "cure_lite"
    assert package_root.is_dir()
    assert (repository_root / "pyproject.toml").is_file()
    assert not (package_root / "cure").exists()
    assert not (package_root / "counterfactual").exists()


def test_root_package_does_not_export_future_full_cure_api() -> None:
    forbidden = {
        "CUREModel",
        "CUREProtocol",
        "CURETrainingPolicy",
        "PropensityConfig",
        "build_atomic_intervention_supervision",
        "build_eligible_sample_catalog",
        "cross_fit_miss_propensity",
    }
    package_root = Path(__file__).resolve().parents[1] / "cure_lite"
    root_source = (package_root / "__init__.py").read_text(encoding="utf-8")
    assert "from .cure import" not in root_source
    assert "from .counterfactual import" not in root_source
    assert all(name not in root_source for name in forbidden)
