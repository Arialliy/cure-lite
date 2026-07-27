"""Frozen-``D_R`` necessary-condition gates for CURE-Lite CMIF-v17.

The audit is deliberately narrower than training.  It asks whether the
literal local state consumed by CMIF already contains a deterministic
contradiction:

* every non-zero clean finite response is inside the radius-two occupancy
  change domain;
* equal endpoint inputs never require unequal endpoint fields;
* equal transition inputs never require unequal finite responses;
* a structurally active endpoint or response never has an exactly-zero
  normalized feature patch.

The mathematical keys are tuples of exact tensor identities and the
row-major output phase.  SHA256 is only a bucket index: repeated digests are
confirmed against dtype, shape, and bytes before sharing an identity.
Neither sample identity nor spatial coordinates participate in a key.

There are three authority levels:

``audit_coverage_state_cmif_population``
    Generic/toy audit.  It is never formally source-bound and cannot
    authorize a run.

``run_coverage_state_cmif_p0_single``
    One real-``D_R`` audit bound to :class:`CoverageStateRealDRInputs`.
    A passing result is only eligible for independent replay and always has
    ``training_authorized == False``.

``replay_coverage_state_cmif_p0_in_memory``
    Runs the complete formal audit twice as a deterministic consistency
    probe.  It never authorizes training: only two separately persisted
    create-only r1/r2 runs may satisfy P0-E.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from ..cache.schema import file_sha256, stable_fingerprint
from ..coverage_state_centered_mixed_interaction import (
    CMIF_COARSE_RADIUS,
    CMIF_INPUT_REPRESENTATION,
)
from ..coverage_state_level_set import (
    CSLF_FIELD_AMPLITUDE,
    normalize_cslf_feature,
)
from ..coverage_state_phase_preserving import (
    pixel_unshuffle_bool_occupancy,
)
from ..coverage_state_precomputed_cache import CoverageStateScalarCache
from ..paired_types import tensor_content_fingerprint
from .coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_ROLE_COUNT,
    CoverageStateBoundedPopulation,
    build_coverage_state_bounded_population,
)
from .coverage_state_cmif_dataset_free import (
    CoverageStateCMIFDatasetFreeReceipt,
)
from .coverage_state_real_dr_inputs import CoverageStateRealDRInputs


COVERAGE_STATE_CMIF_P0_SCHEMA = (
    "cure-lite-cmif-v17-dr-p0-single-receipt-v2"
)
COVERAGE_STATE_CMIF_P0_REPLAY_SCHEMA = (
    "cure-lite-cmif-v17-dr-p0-in-memory-replay-candidate-v2"
)
COVERAGE_STATE_CMIF_ENDPOINT_KEY_POLICY = (
    "exact_normalized_fp32_B_patch5_exact_bool_U_patch5_"
    "row_major_phase_tuple_v2"
)
COVERAGE_STATE_CMIF_TRANSITION_KEY_POLICY = (
    "exact_normalized_fp32_B_patch5_exact_bool_Uplus_Uminus_"
    "patch5_row_major_phase_tuple_v2"
)
COVERAGE_STATE_CMIF_PATCH_PADDING_POLICY = (
    "conv2d_equivalent_constant_zero_padding_radius2_v1"
)
COVERAGE_STATE_CMIF_ENDPOINT_STRATA = (
    "target",
    "response_ring",
    "background",
)
COVERAGE_STATE_CMIF_TRANSITION_ROLES = (
    "clean_nonzero",
    "clean_zero",
    "component_zero",
    "component_nonzero",
)
COVERAGE_STATE_CMIF_EXPECTED_FULL_FACTUAL_MISS = 32
COVERAGE_STATE_CMIF_EXPECTED_FULL_FACTUAL_NO_MISS = 135
COVERAGE_STATE_CMIF_EXPECTED_FULL_CLEAN_PAIRS = 206
COVERAGE_STATE_CMIF_EXPECTED_FULL_COMPONENT_PAIRS = 16
COVERAGE_STATE_CMIF_EXPECTED_FULL_IDENTITY_PAIRS = 160
COVERAGE_STATE_CMIF_EXPECTED_FULL_DIAGNOSTIC_COMPONENT_PAIRS = 1
COVERAGE_STATE_CMIF_EXPECTED_FULL_RESPONSE_PIXELS = 19722
COVERAGE_STATE_CMIF_FORMAL_DATASET = "IRSTD-1K"
COVERAGE_STATE_CMIF_FORMAL_SPLIT = "D_R"
COVERAGE_STATE_CMIF_FROZEN_SOURCE_FILE_SHA256 = (
    (
        "manifest_file_sha256",
        "aa8e33529bd86f564ce6e163e0f9a7b1b3053e9c15054a59c6702a1523f35c02",
    ),
    (
        "state_index_file_sha256",
        "075fc1ad217f365df85b1d29568ad215f06ce6e0b691ef78a5dd85f0affe6298",
    ),
    (
        "geometry_config_file_sha256",
        "719e956b7c51b2b2c8294699fe26c2d36d5c8190b0d8bb5c1d5665a0f4344558",
    ),
    (
        "geometry_receipt_file_sha256",
        "e2a9a986f8819433f3f5efd5c4f627504d10fb32d20f62769b2235b803209283",
    ),
    (
        "observability_config_file_sha256",
        "60d42e657f1daed3cb01c7ee93c8f3fe17417542931d853756ccbbeda1f95713",
    ),
)
COVERAGE_STATE_CMIF_P0_IMPLEMENTATION_PATHS = (
    "cure_lite/coverage_state_level_set.py",
    "cure_lite/coverage_state_phase_preserving.py",
    "cure_lite/coverage_state_centered_mixed_interaction.py",
    "cure_lite/coverage_state_sobolev.py",
    "cure_lite/coverage_state_precomputed_cache.py",
    "cure_lite/experiment/coverage_state_bounded_protocol.py",
    "cure_lite/experiment/coverage_state_cmif_dataset_free.py",
    "cure_lite/experiment/coverage_state_real_dr_inputs.py",
    "cure_lite/experiment/coverage_state_cmif_p0.py",
    "tools/audit_coverage_state_cmif_v17.py",
)

_HEX_DIGITS = frozenset("0123456789abcdef")
_FIELD_AMPLITUDE_FP32_HEX = float(
    torch.tensor(CSLF_FIELD_AMPLITUDE, dtype=torch.float32).item()
).hex()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _current_implementation_binding() -> tuple[tuple[str, str], ...]:
    root = Path(__file__).resolve().parents[2]
    result: list[tuple[str, str]] = []
    for relative in COVERAGE_STATE_CMIF_P0_IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"CMIF P0 implementation path is invalid: {relative}"
            )
        result.append((relative, file_sha256(path)))
    return tuple(result)


def _canonical_json_bytes(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


class _ExactTensorRegistry:
    """Assign exact identities with digest-bucket byte confirmation."""

    def __init__(self) -> None:
        self._buckets: dict[
            str,
            list[tuple[int, str, tuple[int, ...], bytes]],
        ] = {}
        self._next_token = 0
        self.digest_collision_count = 0

    @staticmethod
    def _identity(
        value: Tensor,
    ) -> tuple[str, tuple[int, ...], bytes]:
        if not isinstance(value, Tensor) or value.device.type != "cpu":
            raise TypeError("exact CMIF keys require CPU tensors")
        frozen = value.detach().contiguous()
        return (
            str(frozen.dtype),
            tuple(int(item) for item in frozen.shape),
            frozen.numpy().tobytes(order="C"),
        )

    def intern(self, value: Tensor) -> int:
        if not isinstance(value, Tensor):
            raise TypeError("exact CMIF keys require tensors")
        frozen = value.detach().contiguous()
        digest = tensor_content_fingerprint(frozen)
        dtype, shape, raw = self._identity(frozen)
        bucket = self._buckets.setdefault(digest, [])
        for token, old_dtype, old_shape, old_raw in bucket:
            if (
                dtype == old_dtype
                and shape == old_shape
                and raw == old_raw
            ):
                return token
        if bucket:
            self.digest_collision_count += 1
        token = self._next_token
        self._next_token += 1
        bucket.append((token, dtype, shape, raw))
        return token

    def lookup(self, value: Tensor) -> int | None:
        if not isinstance(value, Tensor):
            raise TypeError("exact CMIF keys require tensors")
        frozen = value.detach().contiguous()
        digest = tensor_content_fingerprint(frozen)
        bucket = self._buckets.get(digest)
        if not bucket:
            return None
        dtype, shape, raw = self._identity(frozen)
        for token, old_dtype, old_shape, old_raw in bucket:
            if (
                dtype == old_dtype
                and shape == old_shape
                and raw == old_raw
            ):
                return token
        return None


class _ExactPatchRegistry:
    """Intern small patches directly by dtype, shape, and raw bytes."""

    def __init__(self) -> None:
        self._values: dict[
            tuple[str, tuple[int, ...], bytes],
            int,
        ] = {}

    @staticmethod
    def _identity(
        value: Tensor,
    ) -> tuple[str, tuple[int, ...], bytes]:
        if not isinstance(value, Tensor) or value.device.type != "cpu":
            raise TypeError("exact CMIF patch keys require CPU tensors")
        frozen = value.detach().contiguous()
        return (
            str(frozen.dtype),
            tuple(int(item) for item in frozen.shape),
            frozen.numpy().tobytes(order="C"),
        )

    def intern(self, value: Tensor) -> int:
        identity = self._identity(value)
        token = self._values.get(identity)
        if token is None:
            token = len(self._values)
            self._values[identity] = token
        return token

    def lookup(self, value: Tensor) -> int | None:
        return self._values.get(self._identity(value))


class _PatchKeyBuilder:
    """Cache exact padded CMIF source tensors and local patch identities."""

    def __init__(self, *, stride: int) -> None:
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise ValueError("stride must be a positive integer")
        self.stride = stride
        self.radius = CMIF_COARSE_RADIUS
        self.kernel_size = 2 * self.radius + 1
        self.feature_sources = _ExactTensorRegistry()
        self.state_sources = _ExactTensorRegistry()
        self.feature_patches = _ExactPatchRegistry()
        self.phase_patches = _ExactPatchRegistry()
        self._feature_padded: dict[int, Tensor] = {}
        self._phase_padded: dict[int, Tensor] = {}
        self._feature_patch_cache: dict[
            tuple[int, int, int],
            tuple[int, bool],
        ] = {}
        self._feature_patch_lookup_cache: dict[
            tuple[int, int, int],
            tuple[int, bool] | None,
        ] = {}
        self._phase_patch_cache: dict[
            tuple[int, int, int],
            int,
        ] = {}
        self._phase_patch_lookup_cache: dict[
            tuple[int, int, int],
            int | None,
        ] = {}

    @property
    def digest_collision_count(self) -> int:
        return sum(
            registry.digest_collision_count
            for registry in (
                self.feature_sources,
                self.state_sources,
            )
        )

    def register_feature(self, feature: Tensor) -> int:
        token = self.feature_sources.intern(feature)
        if token not in self._feature_padded:
            encoded = normalize_cslf_feature(feature)
            self._feature_padded[token] = F.pad(
                encoded,
                (
                    self.radius,
                    self.radius,
                    self.radius,
                    self.radius,
                ),
                value=0.0,
            ).contiguous()
        return token

    def register_state(self, occupancy: Tensor) -> int:
        token = self.state_sources.intern(occupancy)
        if token not in self._phase_padded:
            phase = pixel_unshuffle_bool_occupancy(
                occupancy,
                stride=self.stride,
            )
            self._phase_padded[token] = F.pad(
                phase,
                (
                    self.radius,
                    self.radius,
                    self.radius,
                    self.radius,
                ),
                value=False,
            ).contiguous()
        return token

    def _feature_patch_tensor(
        self,
        source: int,
        row: int,
        column: int,
    ) -> Tensor:
        return self._feature_padded[source][
            :,
            :,
            row : row + self.kernel_size,
            column : column + self.kernel_size,
        ].contiguous()

    def _phase_patch_tensor(
        self,
        state: int,
        row: int,
        column: int,
    ) -> Tensor:
        return self._phase_padded[state][
            :,
            :,
            row : row + self.kernel_size,
            column : column + self.kernel_size,
        ].contiguous()

    def intern_feature_patch(
        self,
        source: int,
        row: int,
        column: int,
    ) -> tuple[int, bool]:
        cache_key = (source, row, column)
        cached = self._feature_patch_cache.get(cache_key)
        if cached is None:
            patch = self._feature_patch_tensor(source, row, column)
            cached = (
                self.feature_patches.intern(patch),
                bool(torch.any(patch != 0.0)),
            )
            self._feature_patch_cache[cache_key] = cached
        return cached

    def lookup_feature_patch(
        self,
        source: int,
        row: int,
        column: int,
    ) -> tuple[int, bool] | None:
        cache_key = (source, row, column)
        if cache_key in self._feature_patch_cache:
            return self._feature_patch_cache[cache_key]
        if cache_key not in self._feature_patch_lookup_cache:
            patch = self._feature_patch_tensor(source, row, column)
            token = self.feature_patches.lookup(patch)
            self._feature_patch_lookup_cache[cache_key] = (
                None
                if token is None
                else (token, bool(torch.any(patch != 0.0)))
            )
        return self._feature_patch_lookup_cache[cache_key]

    def intern_phase_patch(
        self,
        state: int,
        row: int,
        column: int,
    ) -> int:
        cache_key = (state, row, column)
        cached = self._phase_patch_cache.get(cache_key)
        if cached is None:
            cached = self.phase_patches.intern(
                self._phase_patch_tensor(state, row, column)
            )
            self._phase_patch_cache[cache_key] = cached
        return cached

    def lookup_phase_patch(
        self,
        state: int,
        row: int,
        column: int,
        *,
        cache: bool = True,
    ) -> int | None:
        cache_key = (state, row, column)
        if cache_key in self._phase_patch_cache:
            return self._phase_patch_cache[cache_key]
        if not cache:
            return self.phase_patches.lookup(
                self._phase_patch_tensor(state, row, column)
            )
        if cache_key not in self._phase_patch_lookup_cache:
            self._phase_patch_lookup_cache[cache_key] = (
                self.phase_patches.lookup(
                    self._phase_patch_tensor(state, row, column)
                )
            )
        return self._phase_patch_lookup_cache[cache_key]

    def endpoint_key(
        self,
        *,
        source: int,
        state: int,
        row: int,
        column: int,
        phase: int,
        create: bool,
    ) -> tuple[tuple[int, int, int], bool] | None:
        if create:
            feature, nonzero = self.intern_feature_patch(
                source,
                row,
                column,
            )
            occupancy = self.intern_phase_patch(
                state,
                row,
                column,
            )
        else:
            occupancy = self.lookup_phase_patch(
                state,
                row,
                column,
            )
            if occupancy is None:
                return None
            feature_value = self.lookup_feature_patch(
                source,
                row,
                column,
            )
            if feature_value is None:
                return None
            feature, nonzero = feature_value
        return (feature, occupancy, phase), nonzero

    def transition_key(
        self,
        *,
        source: int,
        plus: int,
        minus: int,
        row: int,
        column: int,
        phase: int,
    ) -> tuple[tuple[int, int, int, int], bool]:
        feature, nonzero = self.intern_feature_patch(
            source,
            row,
            column,
        )
        plus_patch = self.intern_phase_patch(
            plus,
            row,
            column,
        )
        minus_patch = self.intern_phase_patch(
            minus,
            row,
            column,
        )
        return (
            feature,
            plus_patch,
            minus_patch,
            phase,
        ), nonzero


@dataclass
class _ValueAccumulator:
    count: int
    values: dict[str, int]
    roles: set[str]
    strata: set[str]

    @classmethod
    def create(
        cls,
        *,
        value: str,
        role: str,
        stratum: str,
    ) -> "_ValueAccumulator":
        return cls(
            count=1,
            values={value: 1},
            roles={role},
            strata={stratum},
        )

    def add(self, *, value: str, role: str, stratum: str) -> None:
        self.count += 1
        self.values[value] = self.values.get(value, 0) + 1
        self.roles.add(role)
        self.strata.add(stratum)


@dataclass(frozen=True)
class _EndpointState:
    state_id: str
    role: str
    source: int
    state: int
    target: Tensor
    target_field: Tensor
    domain: Tensor


def _float_hex(value: Tensor) -> str:
    if (
        not isinstance(value, Tensor)
        or value.numel() != 1
        or value.dtype != torch.float32
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("CMIF P0 target values must be finite FP32 scalars")
    return float(value.detach().reshape(())).hex()


def _phase_index(
    output_row: int,
    output_column: int,
    *,
    stride: int,
) -> int:
    return (
        (output_row % stride) * stride
        + output_column % stride
    )


def _coarse_coordinate(
    output_row: int,
    output_column: int,
    *,
    stride: int,
) -> tuple[int, int]:
    return output_row // stride, output_column // stride


def _minimum_changed_cell_distance(
    changed_cells: Tensor,
    *,
    row: int,
    column: int,
) -> int:
    if (
        not isinstance(changed_cells, Tensor)
        or changed_cells.ndim != 2
        or changed_cells.shape[1] != 2
        or changed_cells.numel() == 0
    ):
        raise ValueError("deletion must contain a changed coarse cell")
    location = torch.tensor(
        [row, column],
        dtype=changed_cells.dtype,
        device=changed_cells.device,
    )
    return int(
        (changed_cells - location)
        .abs()
        .amax(dim=1)
        .amin()
        .item()
    )


def _key_stream_digest(key: tuple[int, ...]) -> str:
    return stable_fingerprint({"exact_token_tuple": list(key)})


def _update_stream(
    hasher: object,
    *,
    key: tuple[int, ...],
    value: str,
    role: str,
    stratum: str,
) -> None:
    if not hasattr(hasher, "update"):
        raise TypeError("stream hasher is invalid")
    digest = stable_fingerprint(
        {
            "key": _key_stream_digest(key),
            "value": value,
            "role": role,
            "stratum": stratum,
        }
    )
    hasher.update(bytes.fromhex(digest))


def _add_group_value(
    groups: dict[tuple[int, ...], _ValueAccumulator],
    *,
    key: tuple[int, ...],
    value: str,
    role: str,
    stratum: str,
) -> None:
    group = groups.get(key)
    if group is None:
        groups[key] = _ValueAccumulator.create(
            value=value,
            role=role,
            stratum=stratum,
        )
    else:
        group.add(value=value, role=role, stratum=stratum)


def _group_statistics(
    groups: dict[tuple[int, ...], _ValueAccumulator],
) -> tuple[int, int, int, int, int, int, str]:
    singleton = sum(group.count == 1 for group in groups.values())
    repeated = sum(group.count > 1 for group in groups.values())
    maximum = max(
        (group.count for group in groups.values()),
        default=0,
    )
    conflicts = tuple(
        group for group in groups.values() if len(group.values) > 1
    )
    conflict_observations = sum(group.count for group in conflicts)
    maximum_span = 0.0
    for group in conflicts:
        numeric = tuple(float.fromhex(value) for value in group.values)
        maximum_span = max(
            maximum_span,
            max(numeric) - min(numeric),
        )
    return (
        len(groups),
        singleton,
        repeated,
        maximum,
        len(conflicts),
        conflict_observations,
        (0.5 * maximum_span).hex(),
    )


def _counter_rows(
    values: dict[tuple[str, str], int],
) -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (role, stratum, count)
        for (role, stratum), count in sorted(values.items())
    )


def _zero_rows(
    totals: dict[tuple[str, str], int],
    zeros: dict[tuple[str, str], int],
) -> tuple[tuple[str, str, int, int], ...]:
    return tuple(
        (
            role,
            stratum,
            totals[(role, stratum)],
            zeros.get((role, stratum), 0),
        )
        for role, stratum in sorted(totals)
    )


def _rows_total(
    rows: tuple[tuple[str, str, int], ...],
) -> int:
    return sum(value for _, _, value in rows)


def _zero_count(
    rows: tuple[tuple[str, str, int, int], ...],
) -> int:
    return sum(zero for _, _, _, zero in rows)


@dataclass(frozen=True)
class CoverageStateCMIFPopulationAudit:
    """Immutable P0-A--D evidence for one cache population."""

    scope: str
    formal_source_bound: bool
    cache_fingerprint: str
    factual_miss_record_ids: tuple[str, ...]
    factual_no_miss_record_ids: tuple[str, ...]
    clean_pair_ids: tuple[str, ...]
    component_pair_ids: tuple[str, ...]
    identity_pair_ids: tuple[str, ...]
    diagnostic_component_pair_ids: tuple[str, ...]
    endpoint_state_count: int
    endpoint_domain_observation_count: int
    endpoint_stratum_counts: tuple[tuple[str, str, int], ...]
    endpoint_partition_failure_count: int
    endpoint_active_observation_count: int
    endpoint_background_observation_count: int
    endpoint_background_scanned_count: int
    endpoint_background_phase_candidate_count: int
    endpoint_background_exact_key_match_count: int
    endpoint_grouped_observation_count: int
    endpoint_exact_key_count: int
    endpoint_singleton_key_count: int
    endpoint_repeated_key_count: int
    endpoint_maximum_group_size: int
    endpoint_conflict_key_count: int
    endpoint_conflict_observation_count: int
    endpoint_lookup_linf_lower_bound_hex: str
    endpoint_zero_feature_rows: tuple[
        tuple[str, str, int, int],
        ...,
    ]
    clean_response_pixel_count: int
    reachable_clean_response_pixel_count: int
    unreachable_clean_response_pixel_count: int
    response_distance_histogram: tuple[tuple[int, int], ...]
    transition_observation_count: int
    transition_role_counts: tuple[tuple[str, str, int], ...]
    transition_exact_key_count: int
    transition_singleton_key_count: int
    transition_repeated_key_count: int
    transition_maximum_group_size: int
    transition_conflict_key_count: int
    transition_conflict_observation_count: int
    transition_nonzero_zero_conflict_key_count: int
    transition_lookup_linf_lower_bound_hex: str
    transition_response_zero_feature_rows: tuple[
        tuple[str, str, int, int],
        ...,
    ]
    component_nonzero_response_pixel_count: int
    exact_digest_collision_count: int
    endpoint_partition_fingerprint: str
    endpoint_group_stream_fingerprint: str
    response_stream_fingerprint: str
    transition_stream_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.scope, str) or not self.scope:
            raise ValueError("population audit scope must be non-empty")
        if not isinstance(self.formal_source_bound, bool):
            raise TypeError("formal_source_bound must be bool")
        _require_sha256(
            self.cache_fingerprint,
            name="cache_fingerprint",
        )
        for name in (
            "endpoint_partition_fingerprint",
            "endpoint_group_stream_fingerprint",
            "response_stream_fingerprint",
            "transition_stream_fingerprint",
        ):
            _require_sha256(getattr(self, name), name=name)
        identity_groups = (
            self.factual_miss_record_ids,
            self.factual_no_miss_record_ids,
            self.clean_pair_ids,
            self.component_pair_ids,
            self.identity_pair_ids,
            self.diagnostic_component_pair_ids,
        )
        if any(
            values != tuple(sorted(set(values)))
            for values in identity_groups
        ):
            raise ValueError("population identities must be sorted and unique")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for name, value in self.__dict__.items()
            if name.endswith("_count")
        ):
            raise ValueError("population counts must be non-negative integers")
        for name, rows in (
            ("endpoint_stratum_counts", self.endpoint_stratum_counts),
            ("transition_role_counts", self.transition_role_counts),
        ):
            identities = tuple((role, stratum) for role, stratum, _ in rows)
            if (
                identities != tuple(sorted(set(identities)))
                or any(
                    not isinstance(role, str)
                    or not role
                    or not isinstance(stratum, str)
                    or not stratum
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or count < 0
                    for role, stratum, count in rows
                )
            ):
                raise ValueError(f"{name} is not canonical")
        for name, rows in (
            ("endpoint_zero_feature_rows", self.endpoint_zero_feature_rows),
            (
                "transition_response_zero_feature_rows",
                self.transition_response_zero_feature_rows,
            ),
        ):
            identities = tuple(
                (role, stratum)
                for role, stratum, _, _ in rows
            )
            if (
                identities != tuple(sorted(set(identities)))
                or any(
                    not isinstance(role, str)
                    or not role
                    or not isinstance(stratum, str)
                    or not stratum
                    or isinstance(total, bool)
                    or not isinstance(total, int)
                    or total < 0
                    or isinstance(zero, bool)
                    or not isinstance(zero, int)
                    or zero < 0
                    or zero > total
                    for role, stratum, total, zero in rows
                )
            ):
                raise ValueError(f"{name} is not canonical")
        if (
            self.response_distance_histogram
            != tuple(sorted(self.response_distance_histogram))
            or len(dict(self.response_distance_histogram))
            != len(self.response_distance_histogram)
            or any(
                isinstance(distance, bool)
                or not isinstance(distance, int)
                or distance < 0
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
                for distance, count in self.response_distance_histogram
            )
            or sum(
                count for _, count in self.response_distance_histogram
            )
            != self.clean_response_pixel_count
        ):
            raise ValueError("response distance histogram is inconsistent")
        for name in (
            "endpoint_lookup_linf_lower_bound_hex",
            "transition_lookup_linf_lower_bound_hex",
        ):
            try:
                value = float.fromhex(getattr(self, name))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must be a finite hex float") from error
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative value")
        if (
            self.endpoint_state_count
            != len(self.factual_miss_record_ids)
            + len(self.factual_no_miss_record_ids)
            + 2 * len(self.clean_pair_ids)
            + 2 * len(self.component_pair_ids)
        ):
            raise ValueError("endpoint state accounting changed")
        if (
            self.endpoint_exact_key_count
            != self.endpoint_singleton_key_count
            + self.endpoint_repeated_key_count
            or self.transition_exact_key_count
            != self.transition_singleton_key_count
            + self.transition_repeated_key_count
        ):
            raise ValueError("exact-key group accounting changed")
        if (
            self.endpoint_background_phase_candidate_count
            < self.endpoint_background_exact_key_match_count
            or self.endpoint_grouped_observation_count
            < self.endpoint_exact_key_count
            or self.transition_observation_count
            < self.transition_exact_key_count
        ):
            raise ValueError("group observation accounting changed")
        if self.endpoint_domain_observation_count != _rows_total(
            self.endpoint_stratum_counts
        ):
            raise ValueError("endpoint stratum accounting is incomplete")
        expected_endpoint_rows = tuple(
            sorted(
                (role, stratum)
                for role in (
                    "factual_miss",
                    "factual_no_miss",
                    "clean_plus",
                    "clean_minus",
                    "component_plus",
                    "component_minus",
                )
                for stratum in COVERAGE_STATE_CMIF_ENDPOINT_STRATA
            )
        )
        if tuple(
            (role, stratum)
            for role, stratum, _ in self.endpoint_stratum_counts
        ) != expected_endpoint_rows:
            raise ValueError("endpoint role-by-stratum rows are incomplete")
        if self.endpoint_active_observation_count + (
            self.endpoint_background_observation_count
        ) != self.endpoint_domain_observation_count:
            raise ValueError("endpoint active/background accounting changed")
        if self.endpoint_background_scanned_count != (
            self.endpoint_background_observation_count
        ):
            raise ValueError("not every endpoint background was scanned")
        if self.endpoint_grouped_observation_count != (
            self.endpoint_active_observation_count
            + self.endpoint_background_exact_key_match_count
        ):
            raise ValueError("endpoint exact-group accounting changed")
        if self.transition_observation_count != _rows_total(
            self.transition_role_counts
        ):
            raise ValueError("transition role accounting changed")
        if sum(
            total
            for _, _, total, _ in self.endpoint_zero_feature_rows
        ) != self.endpoint_active_observation_count:
            raise ValueError("endpoint feature-witness accounting changed")
        endpoint_counts = {
            (role, stratum): count
            for role, stratum, count in self.endpoint_stratum_counts
        }
        if any(
            endpoint_counts.get((role, stratum)) != total
            for role, stratum, total, _
            in self.endpoint_zero_feature_rows
        ) or {
            (role, stratum)
            for role, stratum, _, _
            in self.endpoint_zero_feature_rows
        } != {
            (role, stratum)
            for role, stratum in endpoint_counts
            if stratum in {"target", "response_ring"}
        }:
            raise ValueError(
                "endpoint feature-witness strata changed"
            )
        transition_response_observations = sum(
            count
            for _, stratum, count in self.transition_role_counts
            if stratum in {"response_core", "response_ring"}
        )
        if sum(
            total
            for _, _, total, _
            in self.transition_response_zero_feature_rows
        ) != transition_response_observations:
            raise ValueError("transition feature-witness accounting changed")
        transition_counts = {
            (role, stratum): count
            for role, stratum, count in self.transition_role_counts
        }
        if any(
            transition_counts.get((role, stratum)) != total
            for role, stratum, total, _
            in self.transition_response_zero_feature_rows
        ) or {
            (role, stratum)
            for role, stratum, _, _
            in self.transition_response_zero_feature_rows
        } != {
            (role, stratum)
            for role, stratum in transition_counts
            if stratum in {"response_core", "response_ring"}
        }:
            raise ValueError(
                "transition feature-witness strata changed"
            )
        if self.reachable_clean_response_pixel_count + (
            self.unreachable_clean_response_pixel_count
        ) != self.clean_response_pixel_count:
            raise ValueError("clean response reachability accounting changed")

    @property
    def endpoint_zero_feature_count(self) -> int:
        return _zero_count(self.endpoint_zero_feature_rows)

    @property
    def transition_zero_feature_count(self) -> int:
        return _zero_count(
            self.transition_response_zero_feature_rows
        )

    @property
    def necessary_conditions_passed(self) -> bool:
        transition_counts = {
            (role, stratum): count
            for role, stratum, count in self.transition_role_counts
        }
        return (
            bool(self.factual_miss_record_ids)
            and bool(self.factual_no_miss_record_ids)
            and bool(self.clean_pair_ids)
            and bool(self.component_pair_ids)
            and self.endpoint_state_count > 0
            and self.endpoint_domain_observation_count > 0
            and self.endpoint_partition_failure_count == 0
            and self.endpoint_background_scanned_count
            == self.endpoint_background_observation_count
            and self.endpoint_conflict_key_count == 0
            and float.fromhex(
                self.endpoint_lookup_linf_lower_bound_hex
            )
            == 0.0
            and self.endpoint_zero_feature_count == 0
            and self.clean_response_pixel_count > 0
            and self.unreachable_clean_response_pixel_count == 0
            and self.transition_observation_count > 0
            and transition_counts.get(
                ("clean_zero", "zero_response"),
                0,
            )
            > 0
            and transition_counts.get(
                ("component_zero", "zero_response"),
                0,
            )
            > 0
            and sum(
                count
                for (role, stratum), count
                in transition_counts.items()
                if role == "clean_nonzero"
                and stratum in {"response_core", "response_ring"}
            )
            > 0
            and self.transition_conflict_key_count == 0
            and self.transition_nonzero_zero_conflict_key_count == 0
            and float.fromhex(
                self.transition_lookup_linf_lower_bound_hex
            )
            == 0.0
            and self.transition_zero_feature_count == 0
            and self.component_nonzero_response_pixel_count == 0
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "formal_source_bound": self.formal_source_bound,
            "cache_fingerprint": self.cache_fingerprint,
            "population": {
                "factual_miss_record_ids": list(
                    self.factual_miss_record_ids
                ),
                "factual_no_miss_record_ids": list(
                    self.factual_no_miss_record_ids
                ),
                "clean_pair_ids": list(self.clean_pair_ids),
                    "component_pair_ids": list(self.component_pair_ids),
                    "identity_pair_ids": list(self.identity_pair_ids),
                    "diagnostic_component_pair_ids": list(
                        self.diagnostic_component_pair_ids
                    ),
                "counts": {
                    "factual_miss": len(
                        self.factual_miss_record_ids
                    ),
                    "factual_no_miss": len(
                        self.factual_no_miss_record_ids
                    ),
                    "clean": len(self.clean_pair_ids),
                    "component": len(self.component_pair_ids),
                    "identity_diagnostic": len(self.identity_pair_ids),
                    "component_diagnostic_only": len(
                        self.diagnostic_component_pair_ids
                    ),
                },
                "endpoint_exclusion_policy": {
                    "identity_null_optimizer_exposure": 0,
                    "diagnostic_only_component_optimizer_exposure": 0,
                    "excluded_from_endpoint_and_transition_gates": True,
                },
            },
            "endpoint": {
                "states": self.endpoint_state_count,
                "domain_observations": (
                    self.endpoint_domain_observation_count
                ),
                "stratum_counts": [
                    {
                        "role": role,
                        "stratum": stratum,
                        "count": count,
                    }
                    for role, stratum, count
                    in self.endpoint_stratum_counts
                ],
                "partition_failures": (
                    self.endpoint_partition_failure_count
                ),
                "active_observations": (
                    self.endpoint_active_observation_count
                ),
                "background_observations": (
                    self.endpoint_background_observation_count
                ),
                "background_scanned": (
                    self.endpoint_background_scanned_count
                ),
                "background_phase_candidates": (
                    self.endpoint_background_phase_candidate_count
                ),
                "background_exact_active_key_matches": (
                    self.endpoint_background_exact_key_match_count
                ),
                "grouped_observations": (
                    self.endpoint_grouped_observation_count
                ),
                "exact_keys": self.endpoint_exact_key_count,
                "singleton_keys": self.endpoint_singleton_key_count,
                "repeated_keys": self.endpoint_repeated_key_count,
                "maximum_group_size": (
                    self.endpoint_maximum_group_size
                ),
                "conflict_keys": self.endpoint_conflict_key_count,
                "conflict_observations": (
                    self.endpoint_conflict_observation_count
                ),
                "lookup_linf_lower_bound_hex": (
                    self.endpoint_lookup_linf_lower_bound_hex
                ),
                "zero_feature_rows": [
                    {
                        "role": role,
                        "stratum": stratum,
                        "observations": observations,
                        "zero_feature_patches": zeros,
                    }
                    for role, stratum, observations, zeros
                    in self.endpoint_zero_feature_rows
                ],
                "partition_fingerprint": (
                    self.endpoint_partition_fingerprint
                ),
                "group_stream_fingerprint": (
                    self.endpoint_group_stream_fingerprint
                ),
                "background_policy": (
                    "all endpoint background is scanned; only exact "
                    "matches to active keys need grouping because all "
                    "background targets are the same exact +0.9"
                ),
            },
            "reachability": {
                "clean_response_pixels": (
                    self.clean_response_pixel_count
                ),
                "reachable": (
                    self.reachable_clean_response_pixel_count
                ),
                "unreachable": (
                    self.unreachable_clean_response_pixel_count
                ),
                "distance_histogram": [
                    {"distance": distance, "count": count}
                    for distance, count
                    in self.response_distance_histogram
                ],
                "stream_fingerprint": (
                    self.response_stream_fingerprint
                ),
            },
            "transition": {
                "observations": self.transition_observation_count,
                "role_counts": [
                    {
                        "role": role,
                        "stratum": stratum,
                        "count": count,
                    }
                    for role, stratum, count
                    in self.transition_role_counts
                ],
                "exact_keys": self.transition_exact_key_count,
                "singleton_keys": (
                    self.transition_singleton_key_count
                ),
                "repeated_keys": self.transition_repeated_key_count,
                "maximum_group_size": (
                    self.transition_maximum_group_size
                ),
                "conflict_keys": self.transition_conflict_key_count,
                "conflict_observations": (
                    self.transition_conflict_observation_count
                ),
                "nonzero_zero_conflict_keys": (
                    self.transition_nonzero_zero_conflict_key_count
                ),
                "lookup_linf_lower_bound_hex": (
                    self.transition_lookup_linf_lower_bound_hex
                ),
                "response_zero_feature_rows": [
                    {
                        "role": role,
                        "stratum": stratum,
                        "observations": observations,
                        "zero_feature_patches": zeros,
                    }
                    for role, stratum, observations, zeros
                    in self.transition_response_zero_feature_rows
                ],
                "component_nonzero_response_pixels": (
                    self.component_nonzero_response_pixel_count
                ),
                "stream_fingerprint": (
                    self.transition_stream_fingerprint
                ),
            },
            "exact_digest_collision_count": (
                self.exact_digest_collision_count
            ),
            "necessary_conditions_passed": (
                self.necessary_conditions_passed
            ),
            "claim_boundary": {
                "deterministic_local_contradiction_not_found": (
                    self.necessary_conditions_passed
                ),
                "learnability_proven": False,
                "performance_supported": False,
                "singleton_keys_are_not_positive_evidence": True,
            },
        }


def _audit_coverage_state_cmif_population(
    cache: CoverageStateScalarCache,
    *,
    scope: str,
    formal_source_bound: bool,
) -> CoverageStateCMIFPopulationAudit:
    if not isinstance(cache, CoverageStateScalarCache):
        raise TypeError("cache must be CoverageStateScalarCache")
    if not isinstance(scope, str) or not scope:
        raise ValueError("scope must be non-empty")
    if not isinstance(formal_source_bound, bool):
        raise TypeError("formal_source_bound must be bool")
    cache.verify_unchanged()
    stride = cache.raw_catalog.feature_stride
    builder = _PatchKeyBuilder(stride=stride)

    endpoint_states: list[_EndpointState] = []
    factual_miss_ids: list[str] = []
    factual_no_miss_ids: list[str] = []
    clean_ids: list[str] = []
    component_ids: list[str] = []
    identity_ids: list[str] = []
    diagnostic_component_ids: list[str] = []

    for cached in sorted(
        cache.natural_records,
        key=lambda value: value.record.record_id,
    ):
        record = cached.record
        if record.state_kind == "factual_miss":
            factual_miss_ids.append(record.record_id)
            role = "factual_miss"
        elif record.state_kind == "factual_no_miss":
            factual_no_miss_ids.append(record.record_id)
            role = "factual_no_miss"
        else:
            continue
        endpoint_states.append(
            _EndpointState(
                state_id=f"natural:{record.record_id}",
                role=role,
                source=builder.register_feature(record.feature),
                state=builder.register_state(record.occupancy),
                target=record.target,
                target_field=cached.targets.target_field,
                domain=cached.targets.loss_valid_mask,
            )
        )

    ordered_pairs = tuple(
        sorted(
            cache.pair_records,
            key=lambda value: value.record.pair_id,
        )
    )
    for cached in ordered_pairs:
        if cached.optimizer_role == "identity_diagnostic":
            identity_ids.append(cached.record.pair_id)
        elif (
            cached.optimizer_role == "diagnostic_only"
            and cached.record.pair_kind == "component_null"
        ):
            diagnostic_component_ids.append(cached.record.pair_id)

    optimization_pairs = tuple(
        cached
        for cached in ordered_pairs
        if cached.optimizer_role in {
            "clean_positive",
            "component_null",
        }
    )
    for cached in optimization_pairs:
        record = cached.record
        if cached.optimizer_role == "clean_positive":
            clean_ids.append(record.pair_id)
            base_role = "clean"
        else:
            component_ids.append(record.pair_id)
            base_role = "component"
        source = builder.register_feature(record.feature)
        plus = builder.register_state(record.occupancy_plus)
        minus = builder.register_state(record.occupancy_minus)
        endpoint_states.extend(
            (
                _EndpointState(
                    state_id=f"pair:{record.pair_id}:plus",
                    role=f"{base_role}_plus",
                    source=source,
                    state=plus,
                    target=record.target_plus,
                    target_field=(
                        cached.joint_targets.target_field_plus
                    ),
                    domain=cached.joint_targets.valid_mask,
                ),
                _EndpointState(
                    state_id=f"pair:{record.pair_id}:minus",
                    role=f"{base_role}_minus",
                    source=source,
                    state=minus,
                    target=record.target_minus,
                    target_field=(
                        cached.joint_targets.target_field_minus
                    ),
                    domain=cached.joint_targets.valid_mask,
                ),
            )
        )

    endpoint_groups: dict[
        tuple[int, ...],
        _ValueAccumulator,
    ] = {}
    active_phases_by_state_patch: dict[int, set[int]] = {}
    stratum_counts: dict[tuple[str, str], int] = {}
    active_totals: dict[tuple[str, str], int] = {}
    active_zeros: dict[tuple[str, str], int] = {}
    partition_rows: list[dict[str, object]] = []
    endpoint_stream = sha256()
    partition_failures = 0
    endpoint_domain_count = 0
    endpoint_active_count = 0
    endpoint_background_count = 0

    # Pass one: build every non-background exact endpoint key.
    for state in endpoint_states:
        target = state.domain & state.target
        ring = (
            state.domain
            & ~state.target
            & state.target_field.ne(CSLF_FIELD_AMPLITUDE)
        )
        background = (
            state.domain
            & state.target_field.eq(CSLF_FIELD_AMPLITUDE)
        )
        union = target | ring | background
        disjoint = not bool(
            torch.any(target & ring)
            or torch.any(target & background)
            or torch.any(ring & background)
        )
        complete = torch.equal(union, state.domain)
        target_negative = not bool(
            torch.any(target & ~(state.target_field < 0.0))
        )
        if not (disjoint and complete and target_negative):
            partition_failures += 1
        masks = {
            "target": target,
            "response_ring": ring,
            "background": background,
        }
        counts = {
            name: int(torch.count_nonzero(mask).item())
            for name, mask in masks.items()
        }
        for name, count in counts.items():
            stratum_counts[(state.role, name)] = (
                stratum_counts.get((state.role, name), 0) + count
            )
        domain_count = int(torch.count_nonzero(state.domain).item())
        endpoint_domain_count += domain_count
        endpoint_active_count += counts["target"] + counts["response_ring"]
        endpoint_background_count += counts["background"]
        partition_rows.append(
            {
                "state_id": state.state_id,
                "role": state.role,
                "domain_count": domain_count,
                "domain_fingerprint": tensor_content_fingerprint(
                    state.domain
                ),
                "target_fingerprint": tensor_content_fingerprint(target),
                "ring_fingerprint": tensor_content_fingerprint(ring),
                "background_fingerprint": tensor_content_fingerprint(
                    background
                ),
                "counts": counts,
                "partition_complete": complete,
                "partition_disjoint": disjoint,
                "target_negative": target_negative,
            }
        )
        for stratum in ("target", "response_ring"):
            mask = masks[stratum]
            active_totals[(state.role, stratum)] = (
                active_totals.get((state.role, stratum), 0)
                + counts[stratum]
            )
            for row, column in torch.nonzero(
                mask[0, 0],
                as_tuple=False,
            ).tolist():
                coarse_row, coarse_column = _coarse_coordinate(
                    row,
                    column,
                    stride=stride,
                )
                phase = _phase_index(row, column, stride=stride)
                result = builder.endpoint_key(
                    source=state.source,
                    state=state.state,
                    row=coarse_row,
                    column=coarse_column,
                    phase=phase,
                    create=True,
                )
                if result is None:
                    raise AssertionError("active endpoint key vanished")
                key, feature_nonzero = result
                value = _float_hex(
                    state.target_field[0, 0, row, column]
                )
                _add_group_value(
                    endpoint_groups,
                    key=key,
                    value=value,
                    role=state.role,
                    stratum=stratum,
                )
                active_phases_by_state_patch.setdefault(
                    key[1],
                    set(),
                ).add(phase)
                _update_stream(
                    endpoint_stream,
                    key=key,
                    value=value,
                    role=state.role,
                    stratum=stratum,
                )
                if not feature_nonzero:
                    active_zeros[(state.role, stratum)] = (
                        active_zeros.get((state.role, stratum), 0)
                        + 1
                    )

    # Pass two: scan all far background, grouping only exact active matches.
    frozen_active_phases = {
        token: tuple(sorted(phases))
        for token, phases in active_phases_by_state_patch.items()
    }
    background_scanned = 0
    background_phase_candidates = 0
    background_exact_matches = 0
    state_candidate_cells: dict[
        int,
        tuple[tuple[int, int, tuple[int, ...]], ...],
    ] = {}
    for state in endpoint_states:
        background = (
            state.domain
            & state.target_field.eq(CSLF_FIELD_AMPLITUDE)
        )
        background_scanned += int(torch.count_nonzero(background).item())
        height, width = background.shape[-2:]
        coarse_height = height // stride
        coarse_width = width // stride
        candidate_cells = state_candidate_cells.get(state.state)
        if candidate_cells is None:
            values: list[tuple[int, int, tuple[int, ...]]] = []
            for coarse_row in range(coarse_height):
                for coarse_column in range(coarse_width):
                    phase_token = builder.lookup_phase_patch(
                        state.state,
                        coarse_row,
                        coarse_column,
                        cache=False,
                    )
                    if phase_token is None:
                        continue
                    candidate_phases = frozen_active_phases.get(
                        phase_token
                    )
                    if candidate_phases:
                        values.append(
                            (
                                coarse_row,
                                coarse_column,
                                candidate_phases,
                            )
                        )
            candidate_cells = tuple(values)
            state_candidate_cells[state.state] = candidate_cells
        for coarse_row, coarse_column, candidate_phases in candidate_cells:
                for phase in candidate_phases:
                    row = coarse_row * stride + phase // stride
                    column = (
                        coarse_column * stride + phase % stride
                    )
                    if not bool(background[0, 0, row, column]):
                        continue
                    background_phase_candidates += 1
                    result = builder.endpoint_key(
                        source=state.source,
                        state=state.state,
                        row=coarse_row,
                        column=coarse_column,
                        phase=phase,
                        create=False,
                    )
                    if result is None:
                        continue
                    key, _ = result
                    if key not in endpoint_groups:
                        continue
                    background_exact_matches += 1
                    value = _float_hex(
                        state.target_field[0, 0, row, column]
                    )
                    if value != _FIELD_AMPLITUDE_FP32_HEX:
                        raise AssertionError(
                            "background target is not exact FP32 +0.9"
                        )
                    _add_group_value(
                        endpoint_groups,
                        key=key,
                        value=value,
                        role=state.role,
                        stratum="background",
                    )
                    _update_stream(
                        endpoint_stream,
                        key=key,
                        value=value,
                        role=state.role,
                        stratum="background",
                    )

    (
        endpoint_key_count,
        endpoint_singletons,
        endpoint_repeated,
        endpoint_maximum,
        endpoint_conflicts,
        endpoint_conflict_observations,
        endpoint_lower_bound,
    ) = _group_statistics(endpoint_groups)

    transition_groups: dict[
        tuple[int, ...],
        _ValueAccumulator,
    ] = {}
    transition_role_counts: dict[tuple[str, str], int] = {}
    response_totals: dict[tuple[str, str], int] = {}
    response_zeros: dict[tuple[str, str], int] = {}
    transition_stream = sha256()
    response_stream = sha256()
    response_distance_counts: dict[int, int] = {}
    clean_response_count = 0
    reachable_response_count = 0
    component_nonzero_count = 0

    for cached in optimization_pairs:
        record = cached.record
        source = builder.register_feature(record.feature)
        plus = builder.register_state(record.occupancy_plus)
        minus = builder.register_state(record.occupancy_minus)
        phase_plus = pixel_unshuffle_bool_occupancy(
            record.occupancy_plus,
            stride=stride,
        )
        phase_minus = pixel_unshuffle_bool_occupancy(
            record.occupancy_minus,
            stride=stride,
        )
        changed = (phase_plus != phase_minus).any(
            dim=1,
            keepdim=True,
        )
        changed_cells = torch.nonzero(
            changed[0, 0],
            as_tuple=False,
        )
        if changed_cells.numel() == 0:
            raise ValueError("optimization pair has no phase-visible change")
        affected = F.max_pool2d(
            changed.to(dtype=torch.float32),
            kernel_size=2 * CMIF_COARSE_RADIUS + 1,
            stride=1,
            padding=CMIF_COARSE_RADIUS,
        ).to(dtype=torch.bool)
        target_response = (
            cached.joint_targets.target_field_minus
            - cached.joint_targets.target_field_plus
        )
        valid = cached.joint_targets.valid_mask
        response_mask = target_response.ne(0.0) & valid
        response_core = (
            record.target_minus
            & ~record.target_plus
            & response_mask
        )
        response_ring = response_mask & ~response_core
        if cached.optimizer_role == "clean_positive":
            for row, column in torch.nonzero(
                response_mask[0, 0],
                as_tuple=False,
            ).tolist():
                coarse_row, coarse_column = _coarse_coordinate(
                    row,
                    column,
                    stride=stride,
                )
                distance = _minimum_changed_cell_distance(
                    changed_cells,
                    row=coarse_row,
                    column=coarse_column,
                )
                clean_response_count += 1
                response_distance_counts[distance] = (
                    response_distance_counts.get(distance, 0) + 1
                )
                if distance <= CMIF_COARSE_RADIUS:
                    reachable_response_count += 1
                response_stream.update(
                    bytes.fromhex(
                        stable_fingerprint(
                            {
                                "pair_id": record.pair_id,
                                "row": row,
                                "column": column,
                                "distance": distance,
                                "target_response_hex": _float_hex(
                                    target_response[
                                        0,
                                        0,
                                        row,
                                        column,
                                    ]
                                ),
                            }
                        )
                    )
                )
        else:
            component_nonzero_count += int(
                torch.count_nonzero(response_mask).item()
            )

        for coarse_row, coarse_column in torch.nonzero(
            affected[0, 0],
            as_tuple=False,
        ).tolist():
            for phase in range(stride**2):
                row = coarse_row * stride + phase // stride
                column = coarse_column * stride + phase % stride
                if (
                    row >= valid.shape[-2]
                    or column >= valid.shape[-1]
                    or not bool(valid[0, 0, row, column])
                ):
                    continue
                key, feature_nonzero = builder.transition_key(
                    source=source,
                    plus=plus,
                    minus=minus,
                    row=coarse_row,
                    column=coarse_column,
                    phase=phase,
                )
                if key[1] == key[2]:
                    # max-pool is only a candidate generator; exact equal
                    # local patches imply an identically-zero transition.
                    continue
                nonzero = bool(response_mask[0, 0, row, column])
                if cached.optimizer_role == "clean_positive":
                    role = "clean_nonzero" if nonzero else "clean_zero"
                else:
                    role = (
                        "component_nonzero"
                        if nonzero
                        else "component_zero"
                    )
                if nonzero:
                    stratum = (
                        "response_core"
                        if bool(response_core[0, 0, row, column])
                        else "response_ring"
                    )
                    response_totals[(role, stratum)] = (
                        response_totals.get((role, stratum), 0) + 1
                    )
                    if not feature_nonzero:
                        response_zeros[(role, stratum)] = (
                            response_zeros.get((role, stratum), 0) + 1
                        )
                else:
                    stratum = "zero_response"
                transition_role_counts[(role, stratum)] = (
                    transition_role_counts.get((role, stratum), 0) + 1
                )
                value = _float_hex(
                    target_response[0, 0, row, column]
                )
                _add_group_value(
                    transition_groups,
                    key=key,
                    value=value,
                    role=role,
                    stratum=stratum,
                )
                _update_stream(
                    transition_stream,
                    key=key,
                    value=value,
                    role=role,
                    stratum=stratum,
                )

    (
        transition_key_count,
        transition_singletons,
        transition_repeated,
        transition_maximum,
        transition_conflicts,
        transition_conflict_observations,
        transition_lower_bound,
    ) = _group_statistics(transition_groups)
    transition_nonzero_zero_conflicts = sum(
        bool(
            group.roles
            & {"clean_nonzero", "component_nonzero"}
            and group.roles
            & {"clean_zero", "component_zero"}
        )
        for group in transition_groups.values()
    )

    partition_fingerprint = stable_fingerprint(
        {
            "schema": "cmif-p0-endpoint-partitions-v1",
            "states": partition_rows,
        }
    )
    result = CoverageStateCMIFPopulationAudit(
        scope=scope,
        formal_source_bound=formal_source_bound,
        cache_fingerprint=cache.cache_fingerprint,
        factual_miss_record_ids=tuple(sorted(factual_miss_ids)),
        factual_no_miss_record_ids=tuple(
            sorted(factual_no_miss_ids)
        ),
        clean_pair_ids=tuple(sorted(clean_ids)),
        component_pair_ids=tuple(sorted(component_ids)),
        identity_pair_ids=tuple(sorted(identity_ids)),
        diagnostic_component_pair_ids=tuple(
            sorted(diagnostic_component_ids)
        ),
        endpoint_state_count=len(endpoint_states),
        endpoint_domain_observation_count=endpoint_domain_count,
        endpoint_stratum_counts=_counter_rows(stratum_counts),
        endpoint_partition_failure_count=partition_failures,
        endpoint_active_observation_count=endpoint_active_count,
        endpoint_background_observation_count=endpoint_background_count,
        endpoint_background_scanned_count=background_scanned,
        endpoint_background_phase_candidate_count=(
            background_phase_candidates
        ),
        endpoint_background_exact_key_match_count=(
            background_exact_matches
        ),
        endpoint_grouped_observation_count=sum(
            group.count for group in endpoint_groups.values()
        ),
        endpoint_exact_key_count=endpoint_key_count,
        endpoint_singleton_key_count=endpoint_singletons,
        endpoint_repeated_key_count=endpoint_repeated,
        endpoint_maximum_group_size=endpoint_maximum,
        endpoint_conflict_key_count=endpoint_conflicts,
        endpoint_conflict_observation_count=(
            endpoint_conflict_observations
        ),
        endpoint_lookup_linf_lower_bound_hex=endpoint_lower_bound,
        endpoint_zero_feature_rows=_zero_rows(
            active_totals,
            active_zeros,
        ),
        clean_response_pixel_count=clean_response_count,
        reachable_clean_response_pixel_count=reachable_response_count,
        unreachable_clean_response_pixel_count=(
            clean_response_count - reachable_response_count
        ),
        response_distance_histogram=tuple(
            sorted(response_distance_counts.items())
        ),
        transition_observation_count=sum(
            group.count for group in transition_groups.values()
        ),
        transition_role_counts=_counter_rows(
            transition_role_counts
        ),
        transition_exact_key_count=transition_key_count,
        transition_singleton_key_count=transition_singletons,
        transition_repeated_key_count=transition_repeated,
        transition_maximum_group_size=transition_maximum,
        transition_conflict_key_count=transition_conflicts,
        transition_conflict_observation_count=(
            transition_conflict_observations
        ),
        transition_nonzero_zero_conflict_key_count=(
            transition_nonzero_zero_conflicts
        ),
        transition_lookup_linf_lower_bound_hex=transition_lower_bound,
        transition_response_zero_feature_rows=_zero_rows(
            response_totals,
            response_zeros,
        ),
        component_nonzero_response_pixel_count=(
            component_nonzero_count
        ),
        exact_digest_collision_count=builder.digest_collision_count,
        endpoint_partition_fingerprint=partition_fingerprint,
        endpoint_group_stream_fingerprint=endpoint_stream.hexdigest(),
        response_stream_fingerprint=response_stream.hexdigest(),
        transition_stream_fingerprint=transition_stream.hexdigest(),
    )
    return result


def audit_coverage_state_cmif_population(
    cache: CoverageStateScalarCache,
    *,
    scope: str = "generic_or_toy_population",
) -> CoverageStateCMIFPopulationAudit:
    """Run a non-authoritative generic/toy P0-A--D audit."""

    return _audit_coverage_state_cmif_population(
        cache,
        scope=scope,
        formal_source_bound=False,
    )


def _formal_source_checks(
    *,
    real_inputs: CoverageStateRealDRInputs,
    bounded_population: CoverageStateBoundedPopulation,
) -> dict[str, bool]:
    return {
        "formal_dataset": (
            real_inputs.source_binding.dataset
            == COVERAGE_STATE_CMIF_FORMAL_DATASET
            and real_inputs.manifest.dataset
            == COVERAGE_STATE_CMIF_FORMAL_DATASET
        ),
        "formal_split": (
            real_inputs.source_binding.split
            == COVERAGE_STATE_CMIF_FORMAL_SPLIT
            and real_inputs.bundle.split
            == COVERAGE_STATE_CMIF_FORMAL_SPLIT
            and real_inputs.raw_catalog.split
            == COVERAGE_STATE_CMIF_FORMAL_SPLIT
        ),
        "scalar_cache_identity": (
            real_inputs.scalar_cache is bounded_population.source_cache
            and bounded_population.cache.raw_catalog.split
            == COVERAGE_STATE_CMIF_FORMAL_SPLIT
        ),
        "real_build_fingerprint": (
            real_inputs.current_fingerprint
            == real_inputs.build_fingerprint
        ),
        "source_binding_fingerprint": (
            stable_fingerprint(
                real_inputs.source_binding.canonical_payload()
            )
            == real_inputs.source_binding.binding_fingerprint
        ),
        "frozen_source_file_sha256": all(
            getattr(real_inputs.source_binding, name) == expected
            for name, expected
            in COVERAGE_STATE_CMIF_FROZEN_SOURCE_FILE_SHA256
        ),
    }


def recompute_coverage_state_cmif_p0_single_checks(
    *,
    dataset_free_receipt: CoverageStateCMIFDatasetFreeReceipt,
    real_inputs: CoverageStateRealDRInputs,
    bounded_population_source: CoverageStateBoundedPopulation,
    full_population: CoverageStateCMIFPopulationAudit,
    bounded_population: CoverageStateCMIFPopulationAudit,
    implementation_binding: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, bool], ...]:
    checks = {
        "dataset_free_gate_passed": dataset_free_receipt.all_pass,
        "implementation_closure_bound": (
            tuple(path for path, _ in implementation_binding)
            == COVERAGE_STATE_CMIF_P0_IMPLEMENTATION_PATHS
            and len(set(path for path, _ in implementation_binding))
            == len(implementation_binding)
            and all(
                len(digest) == 64
                and all(
                    character in _HEX_DIGITS
                    for character in digest
                )
                for _, digest in implementation_binding
            )
        ),
        **_formal_source_checks(
            real_inputs=real_inputs,
            bounded_population=bounded_population_source,
        ),
        "full_audit_source_bound": full_population.formal_source_bound,
        "bounded_audit_source_bound": (
            bounded_population.formal_source_bound
        ),
        "full_cache_binding": (
            full_population.cache_fingerprint
            == real_inputs.scalar_cache.cache_fingerprint
        ),
        "bounded_cache_binding": (
            bounded_population.cache_fingerprint
            == bounded_population_source.cache.cache_fingerprint
        ),
        "full_factual_miss_population": (
            len(full_population.factual_miss_record_ids)
            == COVERAGE_STATE_CMIF_EXPECTED_FULL_FACTUAL_MISS
        ),
        "full_factual_no_miss_population": (
            len(full_population.factual_no_miss_record_ids)
            == COVERAGE_STATE_CMIF_EXPECTED_FULL_FACTUAL_NO_MISS
        ),
        "full_clean_population": (
            len(full_population.clean_pair_ids)
            == COVERAGE_STATE_CMIF_EXPECTED_FULL_CLEAN_PAIRS
        ),
        "full_component_population": (
            len(full_population.component_pair_ids)
            == COVERAGE_STATE_CMIF_EXPECTED_FULL_COMPONENT_PAIRS
        ),
        "full_identity_diagnostic_population": (
            len(full_population.identity_pair_ids)
            == COVERAGE_STATE_CMIF_EXPECTED_FULL_IDENTITY_PAIRS
        ),
        "full_component_diagnostic_only_population": (
            len(full_population.diagnostic_component_pair_ids)
            == COVERAGE_STATE_CMIF_EXPECTED_FULL_DIAGNOSTIC_COMPONENT_PAIRS
        ),
        "full_response_population": (
            full_population.clean_response_pixel_count
            == COVERAGE_STATE_CMIF_EXPECTED_FULL_RESPONSE_PIXELS
        ),
        "full_necessary_conditions": (
            full_population.necessary_conditions_passed
        ),
        "bounded_factual_miss_population": (
            len(bounded_population.factual_miss_record_ids)
            == COVERAGE_STATE_BOUNDED_ROLE_COUNT
        ),
        "bounded_factual_no_miss_population": (
            len(bounded_population.factual_no_miss_record_ids)
            == COVERAGE_STATE_BOUNDED_ROLE_COUNT
        ),
        "bounded_clean_population": (
            len(bounded_population.clean_pair_ids)
            == COVERAGE_STATE_BOUNDED_ROLE_COUNT
        ),
        "bounded_component_population": (
            len(bounded_population.component_pair_ids)
            == COVERAGE_STATE_BOUNDED_ROLE_COUNT
        ),
        "bounded_identity_diagnostic_population": (
            len(bounded_population.identity_pair_ids)
            == COVERAGE_STATE_BOUNDED_ROLE_COUNT
        ),
        "bounded_component_diagnostic_only_population": (
            len(bounded_population.diagnostic_component_pair_ids)
            == COVERAGE_STATE_CMIF_EXPECTED_FULL_DIAGNOSTIC_COMPONENT_PAIRS
        ),
        "bounded_necessary_conditions": (
            bounded_population.necessary_conditions_passed
        ),
        "full_and_bounded_distinct": (
            full_population.cache_fingerprint
            != bounded_population.cache_fingerprint
        ),
    }
    return tuple(sorted(checks.items()))


@dataclass(frozen=True, eq=False)
class CoverageStateCMIFP0SingleRunReceipt:
    """One formal P0 run, eligible for replay but never for training."""

    dataset_free_receipt: CoverageStateCMIFDatasetFreeReceipt
    dataset_free_receipt_fingerprint: str
    real_inputs: CoverageStateRealDRInputs
    real_inputs_build_fingerprint: str
    source_binding_fingerprint: str
    bounded_population_source: CoverageStateBoundedPopulation
    bounded_population_fingerprint: str
    full_population: CoverageStateCMIFPopulationAudit
    bounded_population: CoverageStateCMIFPopulationAudit
    implementation_binding: tuple[tuple[str, str], ...]
    checks: tuple[tuple[str, bool], ...]
    evidence_fingerprint: str

    def __post_init__(self) -> None:
        self.verify_unchanged()

    def _evidence_payload(self) -> dict[str, object]:
        return {
            "dataset_free_receipt_fingerprint": (
                self.dataset_free_receipt_fingerprint
            ),
            "real_inputs_build_fingerprint": (
                self.real_inputs_build_fingerprint
            ),
            "source_binding_fingerprint": (
                self.source_binding_fingerprint
            ),
            "bounded_population_fingerprint": (
                self.bounded_population_fingerprint
            ),
            "full_population": self.full_population.canonical_payload(),
            "bounded_population": (
                self.bounded_population.canonical_payload()
            ),
            "implementation_binding": dict(
                self.implementation_binding
            ),
        }

    def verify_unchanged(self) -> None:
        self.dataset_free_receipt.verify_unchanged()
        self.real_inputs.verify_unchanged()
        self.bounded_population_source.verify_unchanged()
        expected_checks = recompute_coverage_state_cmif_p0_single_checks(
            dataset_free_receipt=self.dataset_free_receipt,
            real_inputs=self.real_inputs,
            bounded_population_source=self.bounded_population_source,
            full_population=self.full_population,
            bounded_population=self.bounded_population,
            implementation_binding=self.implementation_binding,
        )
        if (
            self.dataset_free_receipt.receipt_fingerprint
            != self.dataset_free_receipt_fingerprint
            or self.real_inputs.build_fingerprint
            != self.real_inputs_build_fingerprint
            or self.real_inputs.source_binding.binding_fingerprint
            != self.source_binding_fingerprint
            or self.bounded_population_source.population_fingerprint
            != self.bounded_population_fingerprint
            or self.real_inputs.scalar_cache
            is not self.bounded_population_source.source_cache
            or self.full_population.cache_fingerprint
            != self.real_inputs.scalar_cache.cache_fingerprint
            or self.bounded_population.cache_fingerprint
            != self.bounded_population_source.cache.cache_fingerprint
            or self.implementation_binding
            != _current_implementation_binding()
            or self.checks != expected_checks
            or stable_fingerprint(self._evidence_payload())
            != self.evidence_fingerprint
        ):
            raise RuntimeError("CMIF P0 single-run evidence changed")

    @property
    def eligible_for_replay(self) -> bool:
        self.verify_unchanged()
        return bool(self.checks) and all(
            value for _, value in self.checks
        )

    @property
    def training_authorized(self) -> bool:
        self.verify_unchanged()
        return False

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        return {
            "schema_version": COVERAGE_STATE_CMIF_P0_SCHEMA,
            "representation": CMIF_INPUT_REPRESENTATION,
            "endpoint_key_policy": (
                COVERAGE_STATE_CMIF_ENDPOINT_KEY_POLICY
            ),
            "transition_key_policy": (
                COVERAGE_STATE_CMIF_TRANSITION_KEY_POLICY
            ),
            "padding_policy": COVERAGE_STATE_CMIF_PATCH_PADDING_POLICY,
            "dataset": COVERAGE_STATE_CMIF_FORMAL_DATASET,
            "split": COVERAGE_STATE_CMIF_FORMAL_SPLIT,
            "runtime_splits": ["D_R"],
            "dataset_free_receipt_fingerprint": (
                self.dataset_free_receipt_fingerprint
            ),
            "real_D_R_binding": {
                "build_fingerprint": (
                    self.real_inputs_build_fingerprint
                ),
                "source_binding_fingerprint": (
                    self.source_binding_fingerprint
                ),
                "scalar_cache_fingerprint": (
                    self.real_inputs.scalar_cache_fingerprint
                ),
                "source_files_sha256": {
                    "manifest": (
                        self.real_inputs.source_binding
                        .manifest_file_sha256
                    ),
                    "state_index": (
                        self.real_inputs.source_binding
                        .state_index_file_sha256
                    ),
                    "geometry_config": (
                        self.real_inputs.source_binding
                        .geometry_config_file_sha256
                    ),
                    "geometry_receipt": (
                        self.real_inputs.source_binding
                        .geometry_receipt_file_sha256
                    ),
                    "observability_config": (
                        self.real_inputs.source_binding
                        .observability_config_file_sha256
                    ),
                },
                "legacy_container_representation": "scalar_max",
                "legacy_observability_is_CMIF_authority": False,
            },
            "bounded_population_fingerprint": (
                self.bounded_population_fingerprint
            ),
            "implementation_binding": dict(
                self.implementation_binding
            ),
            "full_population": self.full_population.canonical_payload(),
            "bounded_population": (
                self.bounded_population.canonical_payload()
            ),
            "checks": dict(self.checks),
            "evidence_fingerprint": self.evidence_fingerprint,
            "eligible_for_replay": (
                bool(self.checks)
                and all(value for _, value in self.checks)
            ),
            "training_authorized": False,
            "execution_accounting": {
                "dataset_free_generated_gradient_probe_optimizer_steps": 1,
                "D_R_dataset_optimizer_steps": 0,
                "D_R_training_performed": False,
                "calibration_performed": False,
                "inference_performed": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
            },
            "claim_boundary": {
                "single_run_can_authorize_training": False,
                "local_necessary_conditions_only": True,
                "learnability_proven": False,
                "performance_supported": False,
            },
        }

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def run_coverage_state_cmif_p0_single(
    *,
    dataset_free_receipt: CoverageStateCMIFDatasetFreeReceipt,
    real_inputs: CoverageStateRealDRInputs,
    bounded_population: CoverageStateBoundedPopulation,
) -> CoverageStateCMIFP0SingleRunReceipt:
    """Run one formally bound P0 audit without authorizing training."""

    if not isinstance(
        dataset_free_receipt,
        CoverageStateCMIFDatasetFreeReceipt,
    ):
        raise TypeError(
            "dataset_free_receipt must be "
            "CoverageStateCMIFDatasetFreeReceipt"
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
    if not dataset_free_receipt.all_pass:
        raise PermissionError("CMIF dataset-free gate did not pass")
    real_inputs.verify_unchanged()
    bounded_population.verify_unchanged()
    if (
        real_inputs.source_binding.dataset
        != COVERAGE_STATE_CMIF_FORMAL_DATASET
        or real_inputs.source_binding.split
        != COVERAGE_STATE_CMIF_FORMAL_SPLIT
        or real_inputs.scalar_cache is not bounded_population.source_cache
    ):
        raise PermissionError("CMIF P0 is not bound to formal real D_R")
    rebuilt_population = build_coverage_state_bounded_population(
        real_inputs.scalar_cache
    )
    rebuilt_population.verify_unchanged()
    if (
        rebuilt_population.population_fingerprint
        != bounded_population.population_fingerprint
        or rebuilt_population.canonical_payload()
        != bounded_population.canonical_payload()
    ):
        raise PermissionError(
            "bounded population was not produced by the frozen selector"
        )

    full = _audit_coverage_state_cmif_population(
        real_inputs.scalar_cache,
        scope="formal_D_R_full_population",
        formal_source_bound=True,
    )
    bounded = _audit_coverage_state_cmif_population(
        bounded_population.cache,
        scope="formal_D_R_bounded_16_per_role_population",
        formal_source_bound=True,
    )
    implementation_binding = _current_implementation_binding()
    checks = recompute_coverage_state_cmif_p0_single_checks(
        dataset_free_receipt=dataset_free_receipt,
        real_inputs=real_inputs,
        bounded_population_source=bounded_population,
        full_population=full,
        bounded_population=bounded,
        implementation_binding=implementation_binding,
    )
    evidence = {
        "dataset_free_receipt_fingerprint": (
            dataset_free_receipt.receipt_fingerprint
        ),
        "real_inputs_build_fingerprint": real_inputs.build_fingerprint,
        "source_binding_fingerprint": (
            real_inputs.source_binding.binding_fingerprint
        ),
        "bounded_population_fingerprint": (
            bounded_population.population_fingerprint
        ),
        "full_population": full.canonical_payload(),
        "bounded_population": bounded.canonical_payload(),
        "implementation_binding": dict(implementation_binding),
    }
    return CoverageStateCMIFP0SingleRunReceipt(
        dataset_free_receipt=dataset_free_receipt,
        dataset_free_receipt_fingerprint=(
            dataset_free_receipt.receipt_fingerprint
        ),
        real_inputs=real_inputs,
        real_inputs_build_fingerprint=real_inputs.build_fingerprint,
        source_binding_fingerprint=(
            real_inputs.source_binding.binding_fingerprint
        ),
        bounded_population_source=bounded_population,
        bounded_population_fingerprint=(
            bounded_population.population_fingerprint
        ),
        full_population=full,
        bounded_population=bounded,
        implementation_binding=implementation_binding,
        checks=checks,
        evidence_fingerprint=stable_fingerprint(evidence),
    )


def recompute_coverage_state_cmif_p0_replay_checks(
    *,
    first: CoverageStateCMIFP0SingleRunReceipt,
    second: CoverageStateCMIFP0SingleRunReceipt,
    first_canonical_sha256: str,
    second_canonical_sha256: str,
) -> tuple[tuple[str, bool], ...]:
    first_bytes = _canonical_json_bytes(first.canonical_payload())
    second_bytes = _canonical_json_bytes(second.canonical_payload())
    return tuple(
        sorted(
            {
                "distinct_single_run_receipt_objects": first is not second,
                "first_eligible_for_replay": first.eligible_for_replay,
                "second_eligible_for_replay": second.eligible_for_replay,
                "single_runs_never_authorize": (
                    not first.training_authorized
                    and not second.training_authorized
                ),
                "canonical_bytes_identical": first_bytes == second_bytes,
                "canonical_sha256_identical": (
                    first_canonical_sha256
                    == second_canonical_sha256
                    == sha256(first_bytes).hexdigest()
                    == sha256(second_bytes).hexdigest()
                ),
                "receipt_fingerprints_identical": (
                    first.receipt_fingerprint
                    == second.receipt_fingerprint
                ),
                "formal_source_identical": (
                    first.real_inputs_build_fingerprint
                    == second.real_inputs_build_fingerprint
                    and first.source_binding_fingerprint
                    == second.source_binding_fingerprint
                    and first.bounded_population_fingerprint
                    == second.bounded_population_fingerprint
                ),
                "implementation_binding_identical": (
                    first.implementation_binding
                    == second.implementation_binding
                ),
            }.items()
        )
    )


@dataclass(frozen=True, eq=False)
class CoverageStateCMIFP0ReplayCandidate:
    """In-memory consistency replay; persisted r1/r2 is still required."""

    first: CoverageStateCMIFP0SingleRunReceipt
    second: CoverageStateCMIFP0SingleRunReceipt
    first_canonical_sha256: str
    second_canonical_sha256: str
    checks: tuple[tuple[str, bool], ...]
    evidence_fingerprint: str

    def __post_init__(self) -> None:
        self.verify_unchanged()

    def _evidence_payload(self) -> dict[str, object]:
        return {
            "first_receipt_fingerprint": (
                self.first.receipt_fingerprint
            ),
            "second_receipt_fingerprint": (
                self.second.receipt_fingerprint
            ),
            "first_canonical_sha256": self.first_canonical_sha256,
            "second_canonical_sha256": self.second_canonical_sha256,
            "checks": dict(self.checks),
        }

    def verify_unchanged(self) -> None:
        self.first.verify_unchanged()
        self.second.verify_unchanged()
        expected = recompute_coverage_state_cmif_p0_replay_checks(
            first=self.first,
            second=self.second,
            first_canonical_sha256=self.first_canonical_sha256,
            second_canonical_sha256=self.second_canonical_sha256,
        )
        if (
            self.checks != expected
            or stable_fingerprint(self._evidence_payload())
            != self.evidence_fingerprint
        ):
            raise RuntimeError("CMIF P0 replay evidence changed")

    @property
    def replay_consistency_passed(self) -> bool:
        self.verify_unchanged()
        return bool(self.checks) and all(
            value for _, value in self.checks
        )

    @property
    def training_authorized(self) -> bool:
        self.verify_unchanged()
        return False

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        return {
            "schema_version": COVERAGE_STATE_CMIF_P0_REPLAY_SCHEMA,
            "dataset": COVERAGE_STATE_CMIF_FORMAL_DATASET,
            "split": COVERAGE_STATE_CMIF_FORMAL_SPLIT,
            "runtime_splits": ["D_R"],
            "first_receipt_fingerprint": (
                self.first.receipt_fingerprint
            ),
            "second_receipt_fingerprint": (
                self.second.receipt_fingerprint
            ),
            "first_canonical_sha256": self.first_canonical_sha256,
            "second_canonical_sha256": self.second_canonical_sha256,
            "checks": dict(self.checks),
            "evidence_fingerprint": self.evidence_fingerprint,
            "in_memory_replay_consistent": (
                bool(self.checks)
                and all(value for _, value in self.checks)
            ),
            "persisted_independent_r1_r2_required": True,
            "training_authorized": False,
            "bounded_400_authorized": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "execution_accounting": {
                "D_R_dataset_optimizer_steps": 0,
                "D_R_training_performed": False,
                "calibration_performed": False,
                "inference_performed": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
            },
            "claim_boundary": {
                "in_memory_replay_is_not_persisted_independent_replay": True,
                "necessary_conditions_and_replay_candidate_only": True,
                "learnability_proven": False,
                "performance_supported": False,
            },
        }

    @property
    def authorization_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def replay_coverage_state_cmif_p0_in_memory(
    *,
    dataset_free_receipt: CoverageStateCMIFDatasetFreeReceipt,
    real_inputs: CoverageStateRealDRInputs,
    bounded_population: CoverageStateBoundedPopulation,
) -> CoverageStateCMIFP0ReplayCandidate:
    """Run two in-memory audits without authorizing bounded-400."""

    first = run_coverage_state_cmif_p0_single(
        dataset_free_receipt=dataset_free_receipt,
        real_inputs=real_inputs,
        bounded_population=bounded_population,
    )
    second = run_coverage_state_cmif_p0_single(
        dataset_free_receipt=dataset_free_receipt,
        real_inputs=real_inputs,
        bounded_population=bounded_population,
    )
    first_bytes = _canonical_json_bytes(first.canonical_payload())
    second_bytes = _canonical_json_bytes(second.canonical_payload())
    first_sha256 = sha256(first_bytes).hexdigest()
    second_sha256 = sha256(second_bytes).hexdigest()
    checks = recompute_coverage_state_cmif_p0_replay_checks(
        first=first,
        second=second,
        first_canonical_sha256=first_sha256,
        second_canonical_sha256=second_sha256,
    )
    evidence = {
        "first_receipt_fingerprint": first.receipt_fingerprint,
        "second_receipt_fingerprint": second.receipt_fingerprint,
        "first_canonical_sha256": first_sha256,
        "second_canonical_sha256": second_sha256,
        "checks": dict(checks),
    }
    return CoverageStateCMIFP0ReplayCandidate(
        first=first,
        second=second,
        first_canonical_sha256=first_sha256,
        second_canonical_sha256=second_sha256,
        checks=checks,
        evidence_fingerprint=stable_fingerprint(evidence),
    )


__all__ = [
    "COVERAGE_STATE_CMIF_ENDPOINT_KEY_POLICY",
    "COVERAGE_STATE_CMIF_EXPECTED_FULL_CLEAN_PAIRS",
    "COVERAGE_STATE_CMIF_EXPECTED_FULL_COMPONENT_PAIRS",
    "COVERAGE_STATE_CMIF_EXPECTED_FULL_DIAGNOSTIC_COMPONENT_PAIRS",
    "COVERAGE_STATE_CMIF_EXPECTED_FULL_FACTUAL_MISS",
    "COVERAGE_STATE_CMIF_EXPECTED_FULL_FACTUAL_NO_MISS",
    "COVERAGE_STATE_CMIF_EXPECTED_FULL_IDENTITY_PAIRS",
    "COVERAGE_STATE_CMIF_FROZEN_SOURCE_FILE_SHA256",
    "COVERAGE_STATE_CMIF_EXPECTED_FULL_RESPONSE_PIXELS",
    "COVERAGE_STATE_CMIF_FORMAL_DATASET",
    "COVERAGE_STATE_CMIF_FORMAL_SPLIT",
    "COVERAGE_STATE_CMIF_P0_IMPLEMENTATION_PATHS",
    "COVERAGE_STATE_CMIF_P0_REPLAY_SCHEMA",
    "COVERAGE_STATE_CMIF_P0_SCHEMA",
    "COVERAGE_STATE_CMIF_TRANSITION_KEY_POLICY",
    "CoverageStateCMIFP0ReplayCandidate",
    "CoverageStateCMIFP0SingleRunReceipt",
    "CoverageStateCMIFPopulationAudit",
    "audit_coverage_state_cmif_population",
    "recompute_coverage_state_cmif_p0_replay_checks",
    "recompute_coverage_state_cmif_p0_single_checks",
    "replay_coverage_state_cmif_p0_in_memory",
    "run_coverage_state_cmif_p0_single",
]
