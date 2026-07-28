"""Create-only final-model artifacts for PACRE-VC v23 Formal800.

Only the final float32 model state is serialised.  Optimizer state,
intermediate checkpoints, pickle payloads, and evaluation data are outside
this module's contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Final, Mapping

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from cure_lite.cache.schema import (
    canonical_json,
    file_sha256,
    stable_fingerprint,
)
from cure_lite.experiment.coverage_state_training import (
    CoverageStateTrainingResult,
    coverage_state_model_fingerprint,
    coverage_state_optimizer_config_fingerprint,
)
from cure_lite.coverage_state_sobolev import CSLF_PMOPE_POLICY
from cure_lite.frozen_base import module_state_fingerprint

from .factory import (
    PACRE_VC_PARAMETER_NAMES,
    build_pacre_vc_training_model,
)
from .formal_training import (
    PACRE_VC_FORMAL_DEVICE,
    PACRE_VC_FORMAL_FULL_CACHE_FINGERPRINT,
    PACRE_VC_FORMAL_INITIAL_MODEL_FINGERPRINT,
    PACRE_VC_FORMAL_SCHEDULE_FINGERPRINT,
    CoverageStatePACREVCFormal800RunResult,
)
from .pacre_vc import (
    CURELitePACREVerifierCorrectedLevelSet,
    CoverageStatePACREVerifierCorrectedConfig,
)
from .protocol import (
    fingerprinted,
    read_strict_json,
    verify_fingerprinted,
    write_new_json,
)
from .training import (
    PACRE_PMOPE_OBJECTIVE,
    PACRE_PMOPE_TRAINING_CONFIG,
)


PACRE_VC_FORMAL_MODEL_ARTIFACT_SCHEMA: Final = (
    "cure-lite-v23-pacre-vc-formal800-final-model-v3"
)
PACRE_VC_FORMAL_MODEL_FILE: Final = "model.safetensors"
PACRE_VC_FORMAL_MODEL_RECEIPT_FILE: Final = "artifact.json"
PACRE_VC_FORMAL_TRAINING_LEDGER_SCHEMA: Final = (
    "cure-lite-v23-pacre-vc-formal800-artifact-training-ledger-v1"
)
PACRE_VC_FORMAL_TERMINAL_VERIFICATION_SCHEMA: Final = (
    "cure-lite-v23-pacre-vc-formal800-terminal-verification-v1"
)
PACRE_VC_FORMAL_TERMINAL_STATUS: Final = (
    "FORMAL800_TRAINING_COMPLETE_D_V_PREREGISTRATION_ELIGIBLE"
)
PACRE_VC_FORMAL_MODEL_FQCN: Final = (
    "cure_lite_v23.pacre_vc."
    "CURELitePACREVerifierCorrectedLevelSet"
)
PACRE_VC_FORMAL_CONFIG_FQCN: Final = (
    "cure_lite_v23.pacre_vc."
    "CoverageStatePACREVerifierCorrectedConfig"
)
_FORMAL_ARTIFACT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "candidate",
        "serialization",
        "model_file",
        "model_file_sha256",
        "model_config",
        "model_config_fingerprint",
        "state_keys",
        "state_shapes",
        "state_dtypes",
        "parameter_count",
        "coverage_state_model_fingerprint",
        "module_state_fingerprint",
        "formal_result_fingerprint",
        "training_result_fingerprint",
        "formal_training_ledger",
        "authorization_fingerprint",
        "source_closure_fingerprint",
        "final_checkpoint_only",
        "optimizer_state_saved",
        "intermediate_checkpoint_saved",
        "D_V_payload_accessed",
        "D_T_payload_accessed",
        "performance_evaluation_performed",
        "artifact_fingerprint",
    }
)


class _LoadedPACREVCFormalArtifactSeal:
    pass


@dataclass(frozen=True, eq=False)
class LoadedPACREVCFormalArtifact:
    """Exact, revalidatable identity for one strict final-only artifact."""

    directory: Path
    model: CURELitePACREVerifierCorrectedLevelSet
    artifact_json: str
    _seal: _LoadedPACREVCFormalArtifactSeal = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self._seal) is not _LoadedPACREVCFormalArtifactSeal:
            raise TypeError("formal artifact identity seal is invalid")
        if type(self.model) is not CURELitePACREVerifierCorrectedLevelSet:
            raise TypeError("loaded formal artifact has the wrong model")
        payload = json.loads(self.artifact_json)
        if (
            not isinstance(payload, dict)
            or canonical_json(payload) != self.artifact_json
        ):
            raise ValueError("loaded formal artifact receipt is not canonical")

    @property
    def receipt(self) -> dict[str, object]:
        payload = json.loads(self.artifact_json)
        if not isinstance(payload, dict):
            raise AssertionError("validated artifact receipt changed")
        return payload

    @property
    def artifact_fingerprint(self) -> str:
        value = self.receipt["artifact_fingerprint"]
        if not isinstance(value, str):
            raise AssertionError("validated artifact fingerprint changed")
        return value

    def verify_unchanged(self) -> None:
        reloaded = load_pacre_vc_formal_final_model(
            self.directory,
            self.receipt,
        )
        if (
            reloaded.artifact_json != self.artifact_json
            or coverage_state_model_fingerprint(reloaded.model)
            != coverage_state_model_fingerprint(self.model)
            or module_state_fingerprint(reloaded.model)
            != module_state_fingerprint(self.model)
        ):
            raise RuntimeError("loaded formal artifact changed")


class _VerifiedPACREVCFormalTerminalSeal:
    __slots__ = ("issuer", "artifact", "verification_json")

    def __init__(
        self,
        *,
        issuer: object,
        artifact: LoadedPACREVCFormalArtifact,
        verification_json: str,
    ) -> None:
        if issuer is not _PACRE_VC_FORMAL_TERMINAL_ISSUER:
            raise PermissionError(
                "Formal800 terminal seals are verifier-issued only"
            )
        self.issuer = issuer
        self.artifact = artifact
        self.verification_json = verification_json


_PACRE_VC_FORMAL_TERMINAL_ISSUER = object()
_FORMAL_TERMINAL_VERIFICATION_FIELDS: Final = frozenset(
    {
        "schema_version",
        "run_id",
        "output",
        "status",
        "complete_fingerprint",
        "attempt_fingerprint",
        "authorization_fingerprint",
        "training_result_fingerprint",
        "formal_result_fingerprint",
        "source_closure_fingerprint",
        "D_R_gate_receipt_fingerprint",
        "artifact_fingerprint",
        "model_file_sha256",
        "final_model_fingerprint",
        "seed",
        "epochs",
        "steps_per_epoch",
        "updates",
        "from_scratch",
        "training_invocations",
        "artifact_count",
        "D_V_preregistration_eligible",
        "D_V_execution_authorized",
        "D_T_execution_authorized",
        "D_V_tensor_payload_accessed",
        "D_T_tensor_payload_accessed",
        "performance_evaluation_performed",
        "performance_claim_supported",
        "bounded_400_required",
        "bounded_400_authorization_effect",
    }
)


@dataclass(frozen=True, eq=False)
class VerifiedPACREVCFormalTerminal:
    """Verifier-issued identity required before an artifact may enter D_V."""

    artifact: LoadedPACREVCFormalArtifact
    verification_json: str
    _seal: _VerifiedPACREVCFormalTerminalSeal = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self._verify_seal()
        self._verify_binding()

    def _verify_seal(self) -> None:
        if (
            type(self._seal)
            is not _VerifiedPACREVCFormalTerminalSeal
            or self._seal.issuer
            is not _PACRE_VC_FORMAL_TERMINAL_ISSUER
            or self._seal.artifact is not self.artifact
            or self._seal.verification_json != self.verification_json
        ):
            raise PermissionError(
                "Formal800 terminal lacks its verifier-issued seal"
            )

    @property
    def verification(self) -> dict[str, object]:
        payload = json.loads(self.verification_json)
        if not isinstance(payload, dict):
            raise AssertionError("validated terminal verification changed")
        return payload

    def _verify_binding(self) -> None:
        self._verify_seal()
        if type(self.artifact) is not LoadedPACREVCFormalArtifact:
            raise TypeError(
                "verified terminal requires the exact loaded artifact"
            )
        payload = self.verification
        receipt = self.artifact.receipt
        if (
            set(payload) != _FORMAL_TERMINAL_VERIFICATION_FIELDS
            or canonical_json(payload) != self.verification_json
            or payload.get("schema_version")
            != PACRE_VC_FORMAL_TERMINAL_VERIFICATION_SCHEMA
            or payload.get("run_id")
            != "cure_lite_pacre_v23_vc_pmope_formal_800_seed42_r1"
            or payload.get("status") != PACRE_VC_FORMAL_TERMINAL_STATUS
            or Path(str(payload.get("output")))
            != self.artifact.directory.parent
            or payload.get("artifact_fingerprint")
            != self.artifact.artifact_fingerprint
            or payload.get("model_file_sha256")
            != receipt.get("model_file_sha256")
            or payload.get("formal_result_fingerprint")
            != receipt.get("formal_result_fingerprint")
            or payload.get("training_result_fingerprint")
            != receipt.get("training_result_fingerprint")
            or payload.get("authorization_fingerprint")
            != receipt.get("authorization_fingerprint")
            or payload.get("source_closure_fingerprint")
            != receipt.get("source_closure_fingerprint")
            or payload.get("final_model_fingerprint")
            != receipt.get("coverage_state_model_fingerprint")
            or payload.get("seed") != 42
            or payload.get("epochs") != 800
            or payload.get("steps_per_epoch") != 40
            or payload.get("updates") != 32000
            or payload.get("from_scratch") is not True
            or payload.get("training_invocations") != 1
            or payload.get("artifact_count") != 10
            or payload.get("D_V_preregistration_eligible") is not True
            or payload.get("D_V_execution_authorized") is not False
            or payload.get("D_T_execution_authorized") is not False
            or payload.get("D_V_tensor_payload_accessed") is not False
            or payload.get("D_T_tensor_payload_accessed") is not False
            or payload.get("performance_evaluation_performed") is not False
            or payload.get("performance_claim_supported") is not False
            or payload.get("bounded_400_required") is not False
            or payload.get("bounded_400_authorization_effect") is not False
        ):
            raise PermissionError(
                "Formal800 terminal verification/artifact binding changed"
            )
        for name in (
            "complete_fingerprint",
            "attempt_fingerprint",
            "authorization_fingerprint",
            "training_result_fingerprint",
            "formal_result_fingerprint",
            "source_closure_fingerprint",
            "D_R_gate_receipt_fingerprint",
            "artifact_fingerprint",
            "model_file_sha256",
            "final_model_fingerprint",
        ):
            if not _is_digest(payload.get(name)):
                raise ValueError(
                    f"Formal800 terminal {name} is not a digest"
                )

    def verify_unchanged(self) -> None:
        self._verify_binding()
        self.artifact.verify_unchanged()


def _issue_verified_pacre_vc_formal_terminal(
    artifact: LoadedPACREVCFormalArtifact,
    verification: Mapping[str, object],
) -> VerifiedPACREVCFormalTerminal:
    """Issue the process-local terminal seal after independent verification."""

    if type(artifact) is not LoadedPACREVCFormalArtifact:
        raise TypeError("terminal issuer requires the exact loaded artifact")
    if not isinstance(verification, Mapping):
        raise TypeError("terminal verification must be a mapping")
    artifact.verify_unchanged()
    verification_json = canonical_json(dict(verification))
    seal = _VerifiedPACREVCFormalTerminalSeal(
        issuer=_PACRE_VC_FORMAL_TERMINAL_ISSUER,
        artifact=artifact,
        verification_json=verification_json,
    )
    return VerifiedPACREVCFormalTerminal(
        artifact=artifact,
        verification_json=verification_json,
        _seal=seal,
    )


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_model(
    model: object,
) -> CURELitePACREVerifierCorrectedLevelSet:
    if type(model) is not CURELitePACREVerifierCorrectedLevelSet:
        raise TypeError("formal artifact requires the exact v23 model")
    if type(model.config) is not CoverageStatePACREVerifierCorrectedConfig:
        raise TypeError("formal artifact requires the exact v23 config")
    names = tuple(name for name, _ in model.named_parameters())
    if names != PACRE_VC_PARAMETER_NAMES:
        raise ValueError("formal model parameter inventory changed")
    if (
        model.config.feature_channels != 64
        or model.config.feature_stride != 4
        or model.config.width != 32
        or model.config.expected_parameter_count != 64064
    ):
        raise ValueError("formal model coordinates changed")
    if any(
        parameter.dtype != torch.float32
        or not bool(torch.isfinite(parameter).all())
        for parameter in model.parameters()
    ):
        raise ValueError("formal model state must be finite float32")
    return model


def _cpu_state(
    model: CURELitePACREVerifierCorrectedLevelSet,
) -> dict[str, torch.Tensor]:
    state = {
        name: tensor.detach().to(device="cpu").contiguous()
        for name, tensor in model.state_dict().items()
    }
    if tuple(state) != PACRE_VC_PARAMETER_NAMES:
        raise ValueError("formal model state inventory changed")
    if any(
        tensor.dtype != torch.float32
        or tensor.requires_grad
        or not tensor.is_contiguous()
        or not bool(torch.isfinite(tensor).all())
        for tensor in state.values()
    ):
        raise ValueError("formal model state cannot be serialized")
    return state


def formal_model_config_payload(
    config: CoverageStatePACREVerifierCorrectedConfig,
) -> dict[str, object]:
    if type(config) is not CoverageStatePACREVerifierCorrectedConfig:
        raise TypeError("formal config must have the exact v23 type")
    return {
        "model_fqcn": PACRE_VC_FORMAL_MODEL_FQCN,
        "config_fqcn": PACRE_VC_FORMAL_CONFIG_FQCN,
        "config": asdict(config),
        "parameter_count": config.expected_parameter_count,
    }


def _formal_optimizer_config_fingerprint(
    model: CURELitePACREVerifierCorrectedLevelSet,
) -> str:
    config = PACRE_PMOPE_TRAINING_CONFIG
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )
    if type(optimizer) is not torch.optim.Adam or optimizer.state:
        raise RuntimeError("formal artifact fresh Adam policy changed")
    return coverage_state_optimizer_config_fingerprint(model, optimizer)


def formal_training_ledger_payload(
    model: CURELitePACREVerifierCorrectedLevelSet,
    training_result: CoverageStateTrainingResult,
) -> dict[str, object]:
    expected_optimizer = _formal_optimizer_config_fingerprint(model)
    first_nonzero = dict(training_result.first_nonzero_gradient_update)
    if (
        training_result.objective != PACRE_PMOPE_OBJECTIVE
        or training_result.objective_policy != CSLF_PMOPE_POLICY
        or training_result.seed != 42
        or training_result.epochs != 800
        or training_result.steps_per_epoch != 40
        or training_result.completed_updates != 32000
        or training_result.schedule_fingerprint
        != PACRE_VC_FORMAL_SCHEDULE_FINGERPRINT
        or training_result.cache_fingerprint
        != PACRE_VC_FORMAL_FULL_CACHE_FINGERPRINT
        or training_result.execution_device != PACRE_VC_FORMAL_DEVICE
        or training_result.optimizer_config_fingerprint
        != expected_optimizer
        or training_result.initial_model_fingerprint
        != PACRE_VC_FORMAL_INITIAL_MODEL_FINGERPRINT
        or training_result.final_model_fingerprint
        != coverage_state_model_fingerprint(model)
        or training_result.final_model_fingerprint
        == training_result.initial_model_fingerprint
        or training_result.forward_calls != 32000
        or training_result.backward_calls != 32000
        or training_result.optimizer_steps != 32000
        or training_result.logical_state_evaluations != 384000
        or training_result.finite_state_audits != 32001
        or len(training_result.epoch_logs) != 800
        or set(first_nonzero) != set(PACRE_VC_PARAMETER_NAMES)
        or any(
            type(update) is not int or update < 0 or update >= 32000
            for update in first_nonzero.values()
        )
    ):
        raise ValueError(
            "formal artifact requires the exact completed Formal800 ledger"
        )
    body: dict[str, object] = {
        "schema_version": PACRE_VC_FORMAL_TRAINING_LEDGER_SCHEMA,
        "objective": training_result.objective,
        "objective_policy": training_result.objective_policy,
        "seed": training_result.seed,
        "epochs": training_result.epochs,
        "steps_per_epoch": training_result.steps_per_epoch,
        "completed_updates": training_result.completed_updates,
        "schedule_fingerprint": training_result.schedule_fingerprint,
        "cache_fingerprint": training_result.cache_fingerprint,
        "execution_device": training_result.execution_device,
        "device_cache_fingerprint": (
            training_result.device_cache_fingerprint
        ),
        "device_cache_resident_bytes": (
            training_result.device_cache_resident_bytes
        ),
        "optimizer_config_fingerprint": (
            training_result.optimizer_config_fingerprint
        ),
        "initial_model_fingerprint": (
            training_result.initial_model_fingerprint
        ),
        "final_model_fingerprint": (
            training_result.final_model_fingerprint
        ),
        "epoch_count": len(training_result.epoch_logs),
        "epoch_logs_fingerprint": stable_fingerprint(
            list(training_result.epoch_logs)
        ),
        "first_nonzero_gradient_update": first_nonzero,
        "forward_calls": training_result.forward_calls,
        "backward_calls": training_result.backward_calls,
        "optimizer_steps": training_result.optimizer_steps,
        "logical_state_evaluations": (
            training_result.logical_state_evaluations
        ),
        "finite_state_audits": training_result.finite_state_audits,
        "training_result_fingerprint": training_result.result_fingerprint,
        "trained_from_scratch": True,
        "resumed": False,
        "runtime_splits": ["D_R"],
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
        "performance_evaluation_performed": False,
    }
    return {
        **body,
        "ledger_fingerprint": stable_fingerprint(body),
    }


def verify_formal_training_ledger(
    ledger: Mapping[str, object],
    *,
    model: CURELitePACREVerifierCorrectedLevelSet,
    training_result_fingerprint: str,
) -> str:
    """Strictly validate the self-contained Formal800 artifact ledger."""

    if not isinstance(ledger, Mapping):
        raise TypeError("formal artifact training ledger must be a mapping")
    payload = dict(ledger)
    expected_fields = {
        "schema_version",
        "objective",
        "objective_policy",
        "seed",
        "epochs",
        "steps_per_epoch",
        "completed_updates",
        "schedule_fingerprint",
        "cache_fingerprint",
        "execution_device",
        "device_cache_fingerprint",
        "device_cache_resident_bytes",
        "optimizer_config_fingerprint",
        "initial_model_fingerprint",
        "final_model_fingerprint",
        "epoch_count",
        "epoch_logs_fingerprint",
        "first_nonzero_gradient_update",
        "forward_calls",
        "backward_calls",
        "optimizer_steps",
        "logical_state_evaluations",
        "finite_state_audits",
        "training_result_fingerprint",
        "trained_from_scratch",
        "resumed",
        "runtime_splits",
        "D_V_payload_accessed",
        "D_T_payload_accessed",
        "performance_evaluation_performed",
        "ledger_fingerprint",
    }
    if set(payload) != expected_fields:
        raise ValueError("formal artifact training ledger fields changed")
    fingerprint = verify_fingerprinted(
        payload,
        field="ledger_fingerprint",
    )
    digests = (
        payload.get("schedule_fingerprint"),
        payload.get("cache_fingerprint"),
        payload.get("device_cache_fingerprint"),
        payload.get("optimizer_config_fingerprint"),
        payload.get("initial_model_fingerprint"),
        payload.get("final_model_fingerprint"),
        payload.get("epoch_logs_fingerprint"),
        payload.get("training_result_fingerprint"),
    )
    first_nonzero = payload.get("first_nonzero_gradient_update")
    if (
        not all(_is_digest(value) for value in digests)
        or not isinstance(first_nonzero, Mapping)
        or set(first_nonzero) != set(PACRE_VC_PARAMETER_NAMES)
        or any(
            type(update) is not int or update < 0 or update >= 32000
            for update in first_nonzero.values()
        )
        or payload.get("schema_version")
        != PACRE_VC_FORMAL_TRAINING_LEDGER_SCHEMA
        or payload.get("objective") != PACRE_PMOPE_OBJECTIVE
        or payload.get("objective_policy") != CSLF_PMOPE_POLICY
        or payload.get("seed") != 42
        or payload.get("epochs") != 800
        or payload.get("steps_per_epoch") != 40
        or payload.get("completed_updates") != 32000
        or payload.get("schedule_fingerprint")
        != PACRE_VC_FORMAL_SCHEDULE_FINGERPRINT
        or payload.get("cache_fingerprint")
        != PACRE_VC_FORMAL_FULL_CACHE_FINGERPRINT
        or payload.get("execution_device") != PACRE_VC_FORMAL_DEVICE
        or payload.get("device_cache_resident_bytes", 0) < 1
        or payload.get("optimizer_config_fingerprint")
        != _formal_optimizer_config_fingerprint(model)
        or payload.get("initial_model_fingerprint")
        != PACRE_VC_FORMAL_INITIAL_MODEL_FINGERPRINT
        or payload.get("final_model_fingerprint")
        != coverage_state_model_fingerprint(model)
        or payload.get("epoch_count") != 800
        or payload.get("forward_calls") != 32000
        or payload.get("backward_calls") != 32000
        or payload.get("optimizer_steps") != 32000
        or payload.get("logical_state_evaluations") != 384000
        or payload.get("finite_state_audits") != 32001
        or payload.get("training_result_fingerprint")
        != training_result_fingerprint
        or payload.get("trained_from_scratch") is not True
        or payload.get("resumed") is not False
        or payload.get("runtime_splits") != ["D_R"]
        or payload.get("D_V_payload_accessed") is not False
        or payload.get("D_T_payload_accessed") is not False
        or payload.get("performance_evaluation_performed") is not False
    ):
        raise ValueError("formal artifact training ledger changed")
    return fingerprint


def save_pacre_vc_formal_final_model(
    directory: Path,
    *,
    formal_result: CoverageStatePACREVCFormal800RunResult,
) -> dict[str, object]:
    """Persist one engine-issued Formal800 terminal result, exactly once.

    A generic ``CoverageStateTrainingResult`` is intentionally insufficient:
    callers cannot upgrade a hand-assembled ledger into a formal artifact.
    The exact sealed run result is revalidated before any directory is made.
    """

    if (
        type(formal_result)
        is not CoverageStatePACREVCFormal800RunResult
    ):
        raise TypeError(
            "formal_result must be the exact engine-issued "
            "CoverageStatePACREVCFormal800RunResult"
        )
    formal_result.verify_unchanged()
    if not formal_result.training_complete:
        raise PermissionError("Formal800 terminal result is incomplete")
    authorization = formal_result.authorization
    validated = _validate_model(formal_result.final_model)
    training_result = formal_result.training_result
    if (
        validated is not formal_result.model
        or training_result.final_model_fingerprint
        != coverage_state_model_fingerprint(validated)
    ):
        raise RuntimeError("Formal800 terminal result model binding changed")
    authorization_fingerprint = authorization.authorization_fingerprint
    source_closure_fingerprint = (
        authorization.source_closure_fingerprint
    )
    if (
        not _is_digest(authorization_fingerprint)
        or not _is_digest(source_closure_fingerprint)
        or formal_result.source_closure_fingerprint_after
        != source_closure_fingerprint
    ):
        raise ValueError("formal artifact binding digest is invalid")
    training_ledger = formal_training_ledger_payload(
        validated,
        training_result,
    )

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model_path = directory / PACRE_VC_FORMAL_MODEL_FILE
    state = _cpu_state(validated)
    encoded = save_safetensors(
        state,
        metadata={
            "candidate": "PACRE-VC-v23",
            "format": "pt",
            "seed": "42",
            "epochs": "800",
            "updates": "32000",
        },
    )
    with model_path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        import os

        os.fsync(handle.fileno())

    body = {
        "schema_version": PACRE_VC_FORMAL_MODEL_ARTIFACT_SCHEMA,
        "candidate": "PACRE-VC-v23",
        "serialization": "safetensors",
        "model_file": PACRE_VC_FORMAL_MODEL_FILE,
        "model_file_sha256": file_sha256(model_path),
        "model_config": formal_model_config_payload(validated.config),
        "model_config_fingerprint": stable_fingerprint(
            formal_model_config_payload(validated.config)
        ),
        "state_keys": list(state),
        "state_shapes": {
            name: list(tensor.shape) for name, tensor in state.items()
        },
        "state_dtypes": {
            name: str(tensor.dtype) for name, tensor in state.items()
        },
        "parameter_count": sum(tensor.numel() for tensor in state.values()),
        "coverage_state_model_fingerprint": (
            coverage_state_model_fingerprint(validated)
        ),
        "module_state_fingerprint": module_state_fingerprint(validated),
        "formal_result_fingerprint": formal_result.result_fingerprint,
        "training_result_fingerprint": training_result.result_fingerprint,
        "formal_training_ledger": training_ledger,
        "authorization_fingerprint": authorization_fingerprint,
        "source_closure_fingerprint": source_closure_fingerprint,
        "final_checkpoint_only": True,
        "optimizer_state_saved": False,
        "intermediate_checkpoint_saved": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
        "performance_evaluation_performed": False,
    }
    payload = fingerprinted(body, field="artifact_fingerprint")
    write_new_json(
        directory / PACRE_VC_FORMAL_MODEL_RECEIPT_FILE,
        payload,
    )
    loaded = load_pacre_vc_formal_final_model(
        directory,
        payload,
    )
    if (
        coverage_state_model_fingerprint(loaded.model)
        != coverage_state_model_fingerprint(validated)
        or loaded.receipt != payload
    ):
        raise RuntimeError("formal model strict roundtrip changed state")
    return payload


def load_pacre_vc_formal_final_model(
    directory: Path,
    artifact: Mapping[str, object],
) -> LoadedPACREVCFormalArtifact:
    """Strictly load an exact v23 final model from a supplied receipt."""

    if not isinstance(artifact, Mapping):
        raise TypeError("formal artifact receipt must be a mapping")
    payload = dict(artifact)
    if set(payload) != _FORMAL_ARTIFACT_FIELDS:
        raise ValueError("formal model artifact fields changed")
    verify_fingerprinted(payload, field="artifact_fingerprint")
    directory = Path(directory)
    if (
        not directory.is_dir()
        or directory.is_symlink()
        or set(path.name for path in directory.iterdir())
        != {
            PACRE_VC_FORMAL_MODEL_FILE,
            PACRE_VC_FORMAL_MODEL_RECEIPT_FILE,
        }
    ):
        raise RuntimeError("formal model directory inventory changed")
    path = directory / PACRE_VC_FORMAL_MODEL_FILE
    receipt_path = directory / PACRE_VC_FORMAL_MODEL_RECEIPT_FILE
    if read_strict_json(receipt_path) != payload:
        raise RuntimeError("formal model receipt differs from supplied binding")
    if (
        not path.is_file()
        or path.is_symlink()
        or payload.get("schema_version")
        != PACRE_VC_FORMAL_MODEL_ARTIFACT_SCHEMA
        or payload.get("candidate") != "PACRE-VC-v23"
        or payload.get("serialization") != "safetensors"
        or payload.get("model_file") != PACRE_VC_FORMAL_MODEL_FILE
        or payload.get("model_file_sha256") != file_sha256(path)
        or payload.get("state_keys") != list(PACRE_VC_PARAMETER_NAMES)
        or payload.get("parameter_count") != 64064
        or payload.get("final_checkpoint_only") is not True
        or payload.get("optimizer_state_saved") is not False
        or payload.get("intermediate_checkpoint_saved") is not False
        or payload.get("D_V_payload_accessed") is not False
        or payload.get("D_T_payload_accessed") is not False
        or payload.get("performance_evaluation_performed") is not False
        or not _is_digest(payload.get("formal_result_fingerprint"))
    ):
        raise ValueError("formal model artifact contract changed")
    config_payload = payload.get("model_config")
    training_ledger = payload.get("formal_training_ledger")
    if not isinstance(config_payload, Mapping):
        raise ValueError("formal model config receipt is absent")
    expected_config = CoverageStatePACREVerifierCorrectedConfig(
        feature_channels=64,
        feature_stride=4,
        width=32,
    )
    expected_payload = formal_model_config_payload(expected_config)
    if (
        dict(config_payload) != expected_payload
        or payload.get("model_config_fingerprint")
        != stable_fingerprint(expected_payload)
    ):
        raise ValueError("formal model config changed")
    try:
        state = load_safetensors(path.read_bytes())
    except Exception as error:
        raise RuntimeError("formal safetensors payload is invalid") from error
    if set(state) != set(PACRE_VC_PARAMETER_NAMES):
        raise ValueError("formal safetensors keys changed")
    state = {
        name: state[name] for name in PACRE_VC_PARAMETER_NAMES
    }
    expected_shapes = payload.get("state_shapes")
    expected_dtypes = payload.get("state_dtypes")
    if (
        not isinstance(expected_shapes, Mapping)
        or not isinstance(expected_dtypes, Mapping)
        or {
            name: list(tensor.shape) for name, tensor in state.items()
        }
        != dict(expected_shapes)
        or {
            name: str(tensor.dtype) for name, tensor in state.items()
        }
        != dict(expected_dtypes)
        or any(
            tensor.dtype != torch.float32
            or not bool(torch.isfinite(tensor).all())
            for tensor in state.values()
        )
    ):
        raise ValueError("formal safetensors topology changed")
    model = build_pacre_vc_training_model(expected_config)
    model.load_state_dict(state, strict=True)
    model.eval()
    training_result_fingerprint = payload.get(
        "training_result_fingerprint"
    )
    if not _is_digest(training_result_fingerprint):
        raise ValueError("formal training-result fingerprint changed")
    verify_formal_training_ledger(
        training_ledger,
        model=model,
        training_result_fingerprint=training_result_fingerprint,
    )
    if (
        coverage_state_model_fingerprint(model)
        != payload.get("coverage_state_model_fingerprint")
        or module_state_fingerprint(model)
        != payload.get("module_state_fingerprint")
    ):
        raise RuntimeError("formal model fingerprint differs after load")
    return LoadedPACREVCFormalArtifact(
        directory=directory,
        model=model,
        artifact_json=canonical_json(payload),
        _seal=_LoadedPACREVCFormalArtifactSeal(),
    )


__all__ = [
    "PACRE_VC_FORMAL_CONFIG_FQCN",
    "PACRE_VC_FORMAL_MODEL_ARTIFACT_SCHEMA",
    "PACRE_VC_FORMAL_MODEL_FILE",
    "PACRE_VC_FORMAL_MODEL_FQCN",
    "PACRE_VC_FORMAL_MODEL_RECEIPT_FILE",
    "PACRE_VC_FORMAL_TERMINAL_STATUS",
    "PACRE_VC_FORMAL_TERMINAL_VERIFICATION_SCHEMA",
    "PACRE_VC_FORMAL_TRAINING_LEDGER_SCHEMA",
    "LoadedPACREVCFormalArtifact",
    "VerifiedPACREVCFormalTerminal",
    "formal_model_config_payload",
    "formal_training_ledger_payload",
    "load_pacre_vc_formal_final_model",
    "save_pacre_vc_formal_final_model",
    "verify_formal_training_ledger",
]
