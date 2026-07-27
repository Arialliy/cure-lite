"""Read-only real-``D_R`` gate for the v18 PMOPE objective.

The gate binds the frozen negative v17 run, its source closure, the generated
PMOPE receipt, and the existing seed-42 bounded population.  It performs
geometry/mass audits and one gradient probe per clean pair from one common
initial CMIF state.  It never constructs an optimizer, updates a parameter,
accesses ``D_V``/``D_T``, or authorizes Formal800.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import fsum, isfinite
from pathlib import Path

import torch

from ..cache.schema import stable_fingerprint
from ..coverage_state_centered_mixed_interaction import (
    CURELiteCenteredMixedInteractionLevelSet,
    CoverageStateCenteredMixedInteractionConfig,
)
from ..coverage_state_sobolev import (
    coverage_state_pmope_pair_loss_from_targets,
)
from ..paired_types import tensor_content_fingerprint
from .coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_ROLE_COUNT,
    COVERAGE_STATE_BOUNDED_SEED,
    CoverageStateBoundedPopulation,
    build_coverage_state_bounded_population,
)
from .coverage_state_pmope_dataset_free import (
    COVERAGE_STATE_PMOPE_FORMAL_FEATURE_CHANNELS,
    COVERAGE_STATE_PMOPE_FORMAL_FEATURE_STRIDE,
    COVERAGE_STATE_PMOPE_FORMAL_PARAMETER_COUNT,
    COVERAGE_STATE_PMOPE_FORMAL_WIDTH,
    COVERAGE_STATE_PMOPE_MARGIN,
    COVERAGE_STATE_PMOPE_POLICY,
    CoverageStatePMOPEDatasetFreeReceipt,
)
from .coverage_state_pmope_sealed_v17 import (
    COVERAGE_STATE_PMOPE_V17_RUN_REPO_PATH,
    COVERAGE_STATE_PMOPE_V17_SOURCE_ARCHIVE_REPO_PATH,
    COVERAGE_STATE_PMOPE_V17_SOURCE_MANIFEST_REPO_PATH,
    CoverageStatePMOPESealedV17Receipt,
    verify_coverage_state_pmope_sealed_v17_controls,
)
from .coverage_state_real_dr_inputs import CoverageStateRealDRInputs


COVERAGE_STATE_PMOPE_DR_GATE_SCHEMA = (
    "cure-lite-pmope-v18-real-dr-gate-receipt-v1"
)
COVERAGE_STATE_PMOPE_DR_EXECUTION_SEED = 42
COVERAGE_STATE_PMOPE_DR_DATASET = "IRSTD-1K"
COVERAGE_STATE_PMOPE_DR_SPLIT = "D_R"
COVERAGE_STATE_PMOPE_DR_EXPECTED_DATASET_FREE_FINGERPRINT = (
    "9081659faff9dfa69284c7c69fa8efec3ba774a562088ab05211e6262e07c272"
)
COVERAGE_STATE_PMOPE_DR_EXPECTED_POPULATION_FINGERPRINT = (
    "1a53467d57bea595afcc1edd3330708d1dda39e0e2d606325e552e8993e7841c"
)
COVERAGE_STATE_PMOPE_DR_EXPECTED_SOURCE_BINDING_FINGERPRINT = (
    "9689ac7dc4cd95bd0e9bcf79e12e83bc1c8606a96e99ca27945dc07baf4fc74d"
)

def load_coverage_state_pmope_v17_binding(
) -> CoverageStatePMOPESealedV17Receipt:
    root = Path(__file__).resolve().parents[2]
    return verify_coverage_state_pmope_sealed_v17_controls(
        root / COVERAGE_STATE_PMOPE_V17_RUN_REPO_PATH,
        source_manifest_path=(
            root / COVERAGE_STATE_PMOPE_V17_SOURCE_MANIFEST_REPO_PATH
        ),
        source_archive_path=(
            root / COVERAGE_STATE_PMOPE_V17_SOURCE_ARCHIVE_REPO_PATH
        ),
    )


def _float_hex(value: float, *, name: str, positive: bool = False) -> str:
    number = float(value)
    if not isfinite(number) or (positive and not number > 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return number.hex()


def _model_state_fingerprint(
    model: CURELiteCenteredMixedInteractionLevelSet,
) -> str:
    return stable_fingerprint(
        {
            name: tensor_content_fingerprint(value)
            for name, value in sorted(model.state_dict().items())
        }
    )


def _geometry_audit(
    population: CoverageStateBoundedPopulation,
) -> dict[str, object]:
    cache = population.cache
    cache.verify_unchanged()
    clean = tuple(
        sorted(
            cache.clean_positive_records,
            key=lambda value: value.record.pair_id,
        )
    )
    component = tuple(
        sorted(
            cache.component_null_records,
            key=lambda value: value.record.pair_id,
        )
    )
    identity = tuple(
        sorted(
            (
                value
                for value in cache.pair_records
                if value.optimizer_role == "identity_diagnostic"
            ),
            key=lambda value: value.record.pair_id,
        )
    )
    diagnostic = tuple(
        sorted(
            (
                value
                for value in cache.pair_records
                if value.optimizer_role == "diagnostic_only"
            ),
            key=lambda value: value.record.pair_id,
        )
    )
    mass_rows: list[dict[str, object]] = []
    strict_nonzero = True
    valid_pixel_count = 0
    for cached in cache.pair_records:
        targets = cached.joint_targets
        for field in (
            targets.target_field_plus,
            targets.target_field_minus,
        ):
            valid_pixel_count += int(torch.count_nonzero(targets.valid_mask))
            strict_nonzero = strict_nonzero and not bool(
                torch.any(field[targets.valid_mask] == 0.0)
            )
    for cached in clean:
        targets = cached.joint_targets
        added = (
            targets.valid_mask
            & (targets.target_field_minus < 0.0)
            & (targets.target_field_plus > 0.0)
        )
        mass = float(
            targets.integration_measure[added].sum().detach().cpu()
        )
        focus_mass = float(
            targets.integration_measure[
                targets.focus_support
            ].sum().detach().cpu()
        )
        mass_rows.append(
            {
                "pair_id": cached.record.pair_id,
                "sample_id": cached.record.sample_id,
                "added_pixel_count": int(torch.count_nonzero(added)),
                "focus_pixel_count": int(
                    torch.count_nonzero(targets.focus_support)
                ),
                "integration_mass_hex": _float_hex(
                    mass,
                    name="added integration mass",
                    positive=True,
                ),
                "focus_mass_hex": _float_hex(
                    focus_mass,
                    name="focus integration mass",
                    positive=True,
                ),
                "focus_share_hex": _float_hex(
                    mass / focus_mass,
                    name="added focus share",
                    positive=True,
                ),
            }
        )
    masses = sorted(
        float.fromhex(str(row["integration_mass_hex"]))
        for row in mass_rows
    )
    median = 0.5 * (
        masses[len(masses) // 2 - 1] + masses[len(masses) // 2]
    )
    mass_statistics = {
        "count": len(masses),
        "minimum_hex": _float_hex(min(masses), name="minimum mass"),
        "median_hex": _float_hex(median, name="median mass"),
        "mean_hex": _float_hex(
            fsum(masses) / len(masses),
            name="mean mass",
        ),
        "maximum_hex": _float_hex(max(masses), name="maximum mass"),
        "threshold_derived_from_observed_values": False,
        "required_contract": "strictly_positive_per_clean_pair",
    }
    component_contract = all(
        value.record.pair_kind == "component_null"
        and torch.equal(
            value.record.target_plus,
            value.record.target_minus,
        )
        and torch.equal(
            value.joint_targets.target_field_plus,
            value.joint_targets.target_field_minus,
        )
        and not torch.equal(
            value.record.occupancy_plus,
            value.record.occupancy_minus,
        )
        for value in component
    )
    identity_contract = all(
        value.record.pair_kind == "identity_null"
        and torch.equal(
            value.record.target_plus,
            value.record.target_minus,
        )
        and torch.equal(
            value.record.occupancy_plus,
            value.record.occupancy_minus,
        )
        and torch.equal(
            value.joint_targets.target_field_plus,
            value.joint_targets.target_field_minus,
        )
        and value.actual_input_plus_fingerprint
        == value.actual_input_minus_fingerprint
        for value in identity
    )
    diagnostic_contract = all(
        value.record.pair_kind == "component_null"
        and value.optimizer_role == "diagnostic_only"
        for value in diagnostic
    )
    return {
        "mass_rows": mass_rows,
        "mass_statistics": mass_statistics,
        "pair_counts": {
            "clean_positive": len(clean),
            "component_null": len(component),
            "identity_null": len(identity),
            "diagnostic_only_component_null": len(diagnostic),
        },
        "valid_target_field_pixel_count": valid_pixel_count,
        "target_fields_strictly_nonzero_on_valid": strict_nonzero,
        "component_geometry_contract": component_contract,
        "identity_geometry_contract": identity_contract,
        "diagnostic_component_geometry_contract": diagnostic_contract,
    }


def _gradient_probe(
    population: CoverageStateBoundedPopulation,
) -> dict[str, object]:
    before_rng = torch.random.get_rng_state().clone()
    input_before = stable_fingerprint(population.canonical_payload())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(COVERAGE_STATE_PMOPE_DR_EXECUTION_SEED)
        config = CoverageStateCenteredMixedInteractionConfig(
            feature_channels=COVERAGE_STATE_PMOPE_FORMAL_FEATURE_CHANNELS,
            feature_stride=COVERAGE_STATE_PMOPE_FORMAL_FEATURE_STRIDE,
            width=COVERAGE_STATE_PMOPE_FORMAL_WIDTH,
        )
        model = CURELiteCenteredMixedInteractionLevelSet(config)
        initial = _model_state_fingerprint(model)
        parameter_contract = [
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "requires_grad": parameter.requires_grad,
            }
            for name, parameter in model.named_parameters()
        ]
        rows: list[dict[str, object]] = []
        clean = sorted(
            population.cache.clean_positive_records,
            key=lambda value: value.record.pair_id,
        )
        for cached in clean:
            record = cached.record
            feature = torch.cat((record.feature, record.feature), dim=0)
            occupancy = torch.cat(
                (record.occupancy_plus, record.occupancy_minus),
                dim=0,
            )
            field = model(feature, occupancy)
            field_plus, field_minus = field.split(1, dim=0)
            result = coverage_state_pmope_pair_loss_from_targets(
                field_plus,
                field_minus,
                cached.joint_targets,
                config=population.cache.sobolev_config,
                validate=True,
            )
            gradient = torch.autograd.grad(
                result.loss,
                model.scalar_energy_weight,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
            loss = float(result.loss.detach().cpu())
            norm = float(gradient.detach().norm().cpu())
            rows.append(
                {
                    "pair_id": record.pair_id,
                    "sample_id": record.sample_id,
                    "loss_hex": _float_hex(
                        loss,
                        name="PMOPE clean loss",
                        positive=True,
                    ),
                    "scalar_energy_gradient_l2_hex": _float_hex(
                        norm,
                        name="scalar-energy gradient norm",
                        positive=True,
                    ),
                    "scalar_energy_gradient_finite": bool(
                        torch.isfinite(gradient).all()
                    ),
                    "scalar_energy_gradient_nonzero_count": int(
                        torch.count_nonzero(gradient)
                    ),
                    "field_plus_fingerprint": (
                        tensor_content_fingerprint(field_plus)
                    ),
                    "field_minus_fingerprint": (
                        tensor_content_fingerprint(field_minus)
                    ),
                }
            )
        final = _model_state_fingerprint(model)
        gradients_unretained = all(
            parameter.grad is None for parameter in model.parameters()
        )
        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        model_config_payload = {
            "feature_channels": config.feature_channels,
            "feature_stride": config.feature_stride,
            "width": config.width,
            "parameter_count": parameter_count,
            "field_amplitude_hex": config.field_amplitude.hex(),
            "fixed_margin_hex": COVERAGE_STATE_PMOPE_MARGIN.hex(),
            "objective_policy": COVERAGE_STATE_PMOPE_POLICY,
        }
    rng_preserved = torch.equal(before_rng, torch.random.get_rng_state())
    population.verify_unchanged()
    input_after = stable_fingerprint(population.canonical_payload())
    return {
        "execution_seed": COVERAGE_STATE_PMOPE_DR_EXECUTION_SEED,
        "model_config_payload": model_config_payload,
        "parameter_contract": parameter_contract,
        "initial_model_fingerprint": initial,
        "final_model_fingerprint": final,
        "input_before_fingerprint": input_before,
        "input_after_fingerprint": input_after,
        "gradient_rows": rows,
        "parameter_grad_buffers_unretained": gradients_unretained,
        "global_cpu_rng_preserved": rng_preserved,
    }


def _receipt_evidence_payload(
    *,
    dataset_free_receipt_fingerprint: str,
    real_inputs_build_fingerprint: str,
    source_binding_fingerprint: str,
    bounded_population_fingerprint: str,
    bounded_cache_fingerprint: str,
    v17_binding_fingerprint: str,
    execution_seed: int,
    geometry: dict[str, object],
    model_config_payload: dict[str, object],
    parameter_contract: tuple[dict[str, object], ...],
    initial_model_fingerprint: str,
    final_model_fingerprint: str,
    input_before_fingerprint: str,
    input_after_fingerprint: str,
    gradient_rows: tuple[dict[str, object], ...],
    parameter_grad_buffers_unretained: bool,
    global_cpu_rng_preserved: bool,
) -> dict[str, object]:
    return {
        "dataset_free_receipt_fingerprint": (
            dataset_free_receipt_fingerprint
        ),
        "real_inputs_build_fingerprint": real_inputs_build_fingerprint,
        "source_binding_fingerprint": source_binding_fingerprint,
        "bounded_population_fingerprint": bounded_population_fingerprint,
        "bounded_cache_fingerprint": bounded_cache_fingerprint,
        "v17_binding_fingerprint": v17_binding_fingerprint,
        "execution_seed": execution_seed,
        "geometry": deepcopy(geometry),
        "model_config_payload": deepcopy(model_config_payload),
        "parameter_contract": deepcopy(list(parameter_contract)),
        "initial_model_fingerprint": initial_model_fingerprint,
        "final_model_fingerprint": final_model_fingerprint,
        "input_before_fingerprint": input_before_fingerprint,
        "input_after_fingerprint": input_after_fingerprint,
        "gradient_rows": deepcopy(list(gradient_rows)),
        "parameter_grad_buffers_unretained": (
            parameter_grad_buffers_unretained
        ),
        "global_cpu_rng_preserved": global_cpu_rng_preserved,
    }


@dataclass(frozen=True, eq=False)
class CoverageStatePMOPEDRGateReceipt:
    dataset_free_receipt: CoverageStatePMOPEDatasetFreeReceipt
    dataset_free_receipt_fingerprint: str
    real_inputs: CoverageStateRealDRInputs
    real_inputs_build_fingerprint: str
    source_binding_fingerprint: str
    bounded_population: CoverageStateBoundedPopulation
    bounded_population_fingerprint: str
    bounded_cache_fingerprint: str
    v17_binding: CoverageStatePMOPESealedV17Receipt
    v17_binding_fingerprint: str
    execution_seed: int
    geometry: dict[str, object]
    model_config_payload: dict[str, object]
    parameter_contract: tuple[dict[str, object], ...]
    initial_model_fingerprint: str
    final_model_fingerprint: str
    input_before_fingerprint: str
    input_after_fingerprint: str
    gradient_rows: tuple[dict[str, object], ...]
    parameter_grad_buffers_unretained: bool
    global_cpu_rng_preserved: bool
    checks: tuple[tuple[str, bool], ...]
    evidence_fingerprint: str

    def __post_init__(self) -> None:
        self.verify_unchanged()

    def _evidence_payload(self) -> dict[str, object]:
        return _receipt_evidence_payload(
            dataset_free_receipt_fingerprint=(
                self.dataset_free_receipt_fingerprint
            ),
            real_inputs_build_fingerprint=(
                self.real_inputs_build_fingerprint
            ),
            source_binding_fingerprint=self.source_binding_fingerprint,
            bounded_population_fingerprint=(
                self.bounded_population_fingerprint
            ),
            bounded_cache_fingerprint=self.bounded_cache_fingerprint,
            v17_binding_fingerprint=self.v17_binding_fingerprint,
            execution_seed=self.execution_seed,
            geometry=self.geometry,
            model_config_payload=self.model_config_payload,
            parameter_contract=self.parameter_contract,
            initial_model_fingerprint=self.initial_model_fingerprint,
            final_model_fingerprint=self.final_model_fingerprint,
            input_before_fingerprint=self.input_before_fingerprint,
            input_after_fingerprint=self.input_after_fingerprint,
            gradient_rows=self.gradient_rows,
            parameter_grad_buffers_unretained=(
                self.parameter_grad_buffers_unretained
            ),
            global_cpu_rng_preserved=self.global_cpu_rng_preserved,
        )

    def verify_unchanged(self) -> None:
        self.dataset_free_receipt.verify_unchanged()
        self.real_inputs.verify_unchanged()
        self.bounded_population.verify_unchanged()
        current_v17 = load_coverage_state_pmope_v17_binding()
        expected_geometry = _geometry_audit(self.bounded_population)
        expected_checks = recompute_coverage_state_pmope_dr_gate_checks(
            dataset_free_receipt=self.dataset_free_receipt,
            real_inputs=self.real_inputs,
            bounded_population=self.bounded_population,
            v17_binding=self.v17_binding,
            execution_seed=self.execution_seed,
            geometry=self.geometry,
            expected_geometry=expected_geometry,
            model_config_payload=self.model_config_payload,
            parameter_contract=self.parameter_contract,
            initial_model_fingerprint=self.initial_model_fingerprint,
            final_model_fingerprint=self.final_model_fingerprint,
            input_before_fingerprint=self.input_before_fingerprint,
            input_after_fingerprint=self.input_after_fingerprint,
            gradient_rows=self.gradient_rows,
            parameter_grad_buffers_unretained=(
                self.parameter_grad_buffers_unretained
            ),
            global_cpu_rng_preserved=self.global_cpu_rng_preserved,
        )
        if (
            self.dataset_free_receipt.receipt_fingerprint
            != self.dataset_free_receipt_fingerprint
            or self.real_inputs.build_fingerprint
            != self.real_inputs_build_fingerprint
            or self.real_inputs.source_binding.binding_fingerprint
            != self.source_binding_fingerprint
            or self.bounded_population.population_fingerprint
            != self.bounded_population_fingerprint
            or self.bounded_population.cache.cache_fingerprint
            != self.bounded_cache_fingerprint
            or self.v17_binding.receipt_fingerprint
            != self.v17_binding_fingerprint
            or self.v17_binding.canonical_payload()
            != current_v17.canonical_payload()
            or self.geometry != expected_geometry
            or self.checks != expected_checks
            or stable_fingerprint(self._evidence_payload())
            != self.evidence_fingerprint
        ):
            raise RuntimeError("PMOPE real-D_R gate evidence changed")

    @property
    def all_pass(self) -> bool:
        self.verify_unchanged()
        return bool(self.checks) and all(value for _, value in self.checks)

    @property
    def bounded_400_authorized(self) -> bool:
        self.verify_unchanged()
        return False

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        return {
            "schema_version": COVERAGE_STATE_PMOPE_DR_GATE_SCHEMA,
            **self._evidence_payload(),
            "v17_binding": self.v17_binding.canonical_payload(),
            "checks": dict(self.checks),
            "all_pass": bool(self.checks)
            and all(value for _, value in self.checks),
            "runtime_splits": [COVERAGE_STATE_PMOPE_DR_SPLIT],
            "execution_accounting": {
                "execution_seed": self.execution_seed,
                "clean_pair_gradient_probes": len(self.gradient_rows),
                "optimizer_construction_count": 0,
                "optimizer_steps": 0,
                "parameter_updates": 0,
                "training_performed": False,
                "calibration_performed": False,
                "inference_performed": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
            },
            "bounded_400_authorized": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "claim_boundary": {
                "same_seed_deterministic_replay_only": True,
                "multi_seed_claim_supported": False,
                "single_gate_receipt_authorizes_training": False,
                "gradient_probe_is_not_training": True,
                "learnability_proven": False,
                "performance_supported": False,
            },
            "evidence_fingerprint": self.evidence_fingerprint,
        }

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def recompute_coverage_state_pmope_dr_gate_checks(
    *,
    dataset_free_receipt: CoverageStatePMOPEDatasetFreeReceipt,
    real_inputs: CoverageStateRealDRInputs,
    bounded_population: CoverageStateBoundedPopulation,
    v17_binding: CoverageStatePMOPESealedV17Receipt,
    execution_seed: int,
    geometry: dict[str, object],
    expected_geometry: dict[str, object],
    model_config_payload: dict[str, object],
    parameter_contract: tuple[dict[str, object], ...],
    initial_model_fingerprint: str,
    final_model_fingerprint: str,
    input_before_fingerprint: str,
    input_after_fingerprint: str,
    gradient_rows: tuple[dict[str, object], ...],
    parameter_grad_buffers_unretained: bool,
    global_cpu_rng_preserved: bool,
) -> tuple[tuple[str, bool], ...]:
    pair_counts = geometry.get("pair_counts", {})
    mass_rows = geometry.get("mass_rows", [])
    gradients_finite_positive = (
        len(gradient_rows) == COVERAGE_STATE_BOUNDED_ROLE_COUNT
        and all(
            isinstance(row, dict)
            and float.fromhex(str(row.get("loss_hex"))) > 0.0
            and float.fromhex(
                str(row.get("scalar_energy_gradient_l2_hex"))
            )
            > 0.0
            and row.get("scalar_energy_gradient_finite") is True
            and row.get("scalar_energy_gradient_nonzero_count") == 32
            for row in gradient_rows
        )
    )
    checks = {
        "v17_complete_decision_source_closure_bound": (
            v17_binding.receipt_fingerprint
            == stable_fingerprint(v17_binding.canonical_payload())
            and all(value for _, value in v17_binding.checks)
        ),
        "dataset_free_gate_passed": dataset_free_receipt.all_pass,
        "dataset_free_fingerprint_exact": (
            dataset_free_receipt.receipt_fingerprint
            == COVERAGE_STATE_PMOPE_DR_EXPECTED_DATASET_FREE_FINGERPRINT
        ),
        "formal_real_D_R_source_bound": (
            real_inputs.source_binding.dataset
            == COVERAGE_STATE_PMOPE_DR_DATASET
            and real_inputs.source_binding.split
            == COVERAGE_STATE_PMOPE_DR_SPLIT
            and real_inputs.bundle.split == COVERAGE_STATE_PMOPE_DR_SPLIT
            and real_inputs.source_binding.binding_fingerprint
            == COVERAGE_STATE_PMOPE_DR_EXPECTED_SOURCE_BINDING_FINGERPRINT
            and bounded_population.source_cache is real_inputs.scalar_cache
        ),
        "bounded_population_seed42_exact": (
            execution_seed == COVERAGE_STATE_PMOPE_DR_EXECUTION_SEED
            == COVERAGE_STATE_BOUNDED_SEED
            and bounded_population.seed == COVERAGE_STATE_BOUNDED_SEED
            and bounded_population.population_fingerprint
            == COVERAGE_STATE_PMOPE_DR_EXPECTED_POPULATION_FINGERPRINT
        ),
        "bounded_role_counts_exact": (
            isinstance(pair_counts, dict)
            and pair_counts.get("clean_positive")
            == COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and pair_counts.get("component_null")
            == COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and pair_counts.get("identity_null")
            == COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and pair_counts.get("diagnostic_only_component_null") == 1
        ),
        "geometry_recomputes_exactly": geometry == expected_geometry,
        "target_fields_strictly_nonzero_on_valid": (
            geometry.get("target_fields_strictly_nonzero_on_valid")
            is True
            and int(geometry.get("valid_target_field_pixel_count", 0)) > 0
        ),
        "every_clean_added_support_has_positive_measure": (
            isinstance(mass_rows, list)
            and len(mass_rows) == COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and all(
                isinstance(row, dict)
                and int(row.get("added_pixel_count", 0)) > 0
                and float.fromhex(
                    str(row.get("integration_mass_hex"))
                )
                > 0.0
                for row in mass_rows
            )
        ),
        "mass_policy_has_no_observed_value_threshold": (
            isinstance(geometry.get("mass_statistics"), dict)
            and geometry["mass_statistics"].get(
                "threshold_derived_from_observed_values"
            )
            is False
            and geometry["mass_statistics"].get("required_contract")
            == "strictly_positive_per_clean_pair"
        ),
        "component_geometry_contract": (
            geometry.get("component_geometry_contract") is True
        ),
        "identity_geometry_contract": (
            geometry.get("identity_geometry_contract") is True
        ),
        "diagnostic_component_geometry_contract": (
            geometry.get("diagnostic_component_geometry_contract") is True
        ),
        "common_initial_cmif_contract": (
            model_config_payload.get("feature_channels")
            == COVERAGE_STATE_PMOPE_FORMAL_FEATURE_CHANNELS
            and model_config_payload.get("feature_stride")
            == COVERAGE_STATE_PMOPE_FORMAL_FEATURE_STRIDE
            and model_config_payload.get("width")
            == COVERAGE_STATE_PMOPE_FORMAL_WIDTH
            and model_config_payload.get("parameter_count")
            == COVERAGE_STATE_PMOPE_FORMAL_PARAMETER_COUNT
            and model_config_payload.get("fixed_margin_hex")
            == COVERAGE_STATE_PMOPE_MARGIN.hex()
            and model_config_payload.get("objective_policy")
            == COVERAGE_STATE_PMOPE_POLICY
            and len(parameter_contract) == 3
        ),
        "every_clean_loss_and_scalar_gradient_positive": (
            gradients_finite_positive
        ),
        "clean_mass_and_gradient_pair_ids_identical": (
            tuple(
                row.get("pair_id")
                for row in mass_rows
                if isinstance(row, dict)
            )
            == tuple(row.get("pair_id") for row in gradient_rows)
        ),
        "model_parameters_unchanged": (
            initial_model_fingerprint == final_model_fingerprint
        ),
        "bounded_inputs_unchanged": (
            input_before_fingerprint == input_after_fingerprint
        ),
        "gradient_buffers_unretained": parameter_grad_buffers_unretained,
        "global_cpu_rng_preserved": global_cpu_rng_preserved,
        "no_optimizer_training_or_dataset_evaluation": True,
    }
    return tuple(sorted(checks.items()))


def run_coverage_state_pmope_dr_gate(
    *,
    dataset_free_receipt: CoverageStatePMOPEDatasetFreeReceipt,
    real_inputs: CoverageStateRealDRInputs,
    bounded_population: CoverageStateBoundedPopulation,
) -> CoverageStatePMOPEDRGateReceipt:
    """Run the fixed seed-42 PMOPE gate on real ``D_R`` without training."""

    if not isinstance(
        dataset_free_receipt,
        CoverageStatePMOPEDatasetFreeReceipt,
    ):
        raise TypeError(
            "dataset_free_receipt must be "
            "CoverageStatePMOPEDatasetFreeReceipt"
        )
    if not isinstance(real_inputs, CoverageStateRealDRInputs):
        raise TypeError("real_inputs must be CoverageStateRealDRInputs")
    if not isinstance(
        bounded_population,
        CoverageStateBoundedPopulation,
    ):
        raise TypeError(
            "bounded_population must be CoverageStateBoundedPopulation"
        )
    dataset_free_receipt.verify_unchanged()
    if (
        not dataset_free_receipt.all_pass
        or dataset_free_receipt.receipt_fingerprint
        != COVERAGE_STATE_PMOPE_DR_EXPECTED_DATASET_FREE_FINGERPRINT
    ):
        raise PermissionError(
            "PMOPE dataset-free receipt is not the frozen passing receipt"
        )
    real_inputs.verify_unchanged()
    bounded_population.verify_unchanged()
    if (
        bounded_population.source_cache is not real_inputs.scalar_cache
        or bounded_population.seed != COVERAGE_STATE_PMOPE_DR_EXECUTION_SEED
    ):
        raise PermissionError(
            "PMOPE D_R gate requires the exact real bounded population"
        )
    rebuilt = build_coverage_state_bounded_population(
        real_inputs.scalar_cache,
        seed=COVERAGE_STATE_PMOPE_DR_EXECUTION_SEED,
    )
    if (
        rebuilt.population_fingerprint
        != bounded_population.population_fingerprint
        or rebuilt.canonical_payload()
        != bounded_population.canonical_payload()
    ):
        raise PermissionError("bounded population selector changed")
    v17 = load_coverage_state_pmope_v17_binding()
    geometry = _geometry_audit(bounded_population)
    gradient = _gradient_probe(bounded_population)
    parameter_contract = tuple(gradient["parameter_contract"])
    gradient_rows = tuple(gradient["gradient_rows"])
    checks = recompute_coverage_state_pmope_dr_gate_checks(
        dataset_free_receipt=dataset_free_receipt,
        real_inputs=real_inputs,
        bounded_population=bounded_population,
        v17_binding=v17,
        execution_seed=COVERAGE_STATE_PMOPE_DR_EXECUTION_SEED,
        geometry=geometry,
        expected_geometry=_geometry_audit(bounded_population),
        model_config_payload=gradient["model_config_payload"],
        parameter_contract=parameter_contract,
        initial_model_fingerprint=gradient["initial_model_fingerprint"],
        final_model_fingerprint=gradient["final_model_fingerprint"],
        input_before_fingerprint=gradient["input_before_fingerprint"],
        input_after_fingerprint=gradient["input_after_fingerprint"],
        gradient_rows=gradient_rows,
        parameter_grad_buffers_unretained=gradient[
            "parameter_grad_buffers_unretained"
        ],
        global_cpu_rng_preserved=gradient["global_cpu_rng_preserved"],
    )
    kwargs = {
        "dataset_free_receipt": dataset_free_receipt,
        "dataset_free_receipt_fingerprint": (
            dataset_free_receipt.receipt_fingerprint
        ),
        "real_inputs": real_inputs,
        "real_inputs_build_fingerprint": real_inputs.build_fingerprint,
        "source_binding_fingerprint": (
            real_inputs.source_binding.binding_fingerprint
        ),
        "bounded_population": bounded_population,
        "bounded_population_fingerprint": (
            bounded_population.population_fingerprint
        ),
        "bounded_cache_fingerprint": (
            bounded_population.cache.cache_fingerprint
        ),
        "v17_binding": v17,
        "v17_binding_fingerprint": v17.receipt_fingerprint,
        "execution_seed": COVERAGE_STATE_PMOPE_DR_EXECUTION_SEED,
        "geometry": geometry,
        "model_config_payload": gradient["model_config_payload"],
        "parameter_contract": parameter_contract,
        "initial_model_fingerprint": gradient[
            "initial_model_fingerprint"
        ],
        "final_model_fingerprint": gradient["final_model_fingerprint"],
        "input_before_fingerprint": gradient[
            "input_before_fingerprint"
        ],
        "input_after_fingerprint": gradient["input_after_fingerprint"],
        "gradient_rows": gradient_rows,
        "parameter_grad_buffers_unretained": gradient[
            "parameter_grad_buffers_unretained"
        ],
        "global_cpu_rng_preserved": gradient[
            "global_cpu_rng_preserved"
        ],
        "checks": checks,
    }
    evidence = _receipt_evidence_payload(
        dataset_free_receipt_fingerprint=kwargs[
            "dataset_free_receipt_fingerprint"
        ],
        real_inputs_build_fingerprint=kwargs[
            "real_inputs_build_fingerprint"
        ],
        source_binding_fingerprint=kwargs[
            "source_binding_fingerprint"
        ],
        bounded_population_fingerprint=kwargs[
            "bounded_population_fingerprint"
        ],
        bounded_cache_fingerprint=kwargs["bounded_cache_fingerprint"],
        v17_binding_fingerprint=kwargs["v17_binding_fingerprint"],
        execution_seed=kwargs["execution_seed"],
        geometry=kwargs["geometry"],
        model_config_payload=kwargs["model_config_payload"],
        parameter_contract=kwargs["parameter_contract"],
        initial_model_fingerprint=kwargs["initial_model_fingerprint"],
        final_model_fingerprint=kwargs["final_model_fingerprint"],
        input_before_fingerprint=kwargs["input_before_fingerprint"],
        input_after_fingerprint=kwargs["input_after_fingerprint"],
        gradient_rows=kwargs["gradient_rows"],
        parameter_grad_buffers_unretained=kwargs[
            "parameter_grad_buffers_unretained"
        ],
        global_cpu_rng_preserved=kwargs["global_cpu_rng_preserved"],
    )
    return CoverageStatePMOPEDRGateReceipt(
        **kwargs,
        evidence_fingerprint=stable_fingerprint(evidence),
    )


__all__ = [
    "COVERAGE_STATE_PMOPE_DR_DATASET",
    "COVERAGE_STATE_PMOPE_DR_EXECUTION_SEED",
    "COVERAGE_STATE_PMOPE_DR_EXPECTED_DATASET_FREE_FINGERPRINT",
    "COVERAGE_STATE_PMOPE_DR_EXPECTED_POPULATION_FINGERPRINT",
    "COVERAGE_STATE_PMOPE_DR_EXPECTED_SOURCE_BINDING_FINGERPRINT",
    "COVERAGE_STATE_PMOPE_DR_GATE_SCHEMA",
    "COVERAGE_STATE_PMOPE_DR_SPLIT",
    "CoverageStatePMOPEDRGateReceipt",
    "CoverageStatePMOPESealedV17Receipt",
    "load_coverage_state_pmope_v17_binding",
    "recompute_coverage_state_pmope_dr_gate_checks",
    "run_coverage_state_pmope_dr_gate",
]
