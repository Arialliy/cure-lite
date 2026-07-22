from __future__ import annotations

import math

import pytest

from cure_lite.provenance import (
    BaseTrainingProvenanceError,
    deterministic_base_checkpoint_selection,
)
from cure_lite.splits import SplitManifest, SplitRecord


def _manifest(d_b_records: tuple[SplitRecord, ...]) -> SplitManifest:
    held_out = (
        SplitRecord("dr", "D_R", "dr-group", "dr.png"),
        SplitRecord("dv", "D_V", "dv-group", "dv.png"),
        SplitRecord("dt", "D_T", "dt-group", "dt.png"),
    )
    return SplitManifest(dataset="toy", records=d_b_records + held_out)


def _assert_no_grouping_key_crosses(
    manifest: SplitManifest,
    *,
    fit_ids: tuple[str, ...],
    select_ids: tuple[str, ...],
) -> None:
    role_by_id = {sample_id: "fit" for sample_id in fit_ids}
    role_by_id.update({sample_id: "select" for sample_id in select_ids})
    grouping_roles: dict[tuple[str, str], str] = {}
    for record in manifest.records_for("D_B"):
        role = role_by_id[record.sample_id]
        for key in record.grouping_keys():
            assert grouping_roles.setdefault(key, role) == role


def test_builder_is_deterministic_exhaustive_and_manifest_order_invariant() -> None:
    d_b = tuple(
        SplitRecord(
            f"db-{index:02d}",
            "D_B",
            f"group-{index:02d}",
            f"db-{index:02d}.png",
        )
        for index in range(10)
    )
    forward = _manifest(d_b)
    reversed_rows = _manifest(tuple(reversed(d_b)))

    first = deterministic_base_checkpoint_selection(
        forward,
        select_fraction=0.3,
        seed=17,
    )
    repeated = deterministic_base_checkpoint_selection(
        forward,
        select_fraction=0.3,
        seed=17,
    )
    reordered = deterministic_base_checkpoint_selection(
        reversed_rows,
        select_fraction=0.3,
        seed=17,
    )

    assert first == repeated == reordered
    assert len(first.select_sample_ids) == 3
    assert set(first.fit_sample_ids).isdisjoint(first.select_sample_ids)
    assert set(first.fit_sample_ids) | set(first.select_sample_ids) == {
        record.sample_id for record in d_b
    }
    _assert_no_grouping_key_crosses(
        forward,
        fit_ids=first.fit_sample_ids,
        select_ids=first.select_sample_ids,
    )


def test_builder_keeps_transitively_connected_grouping_keys_indivisible() -> None:
    # a --group_id--> b --scene_id--> c is one connected component, even
    # though a and c share no grouping key directly.
    manifest = _manifest(
        (
            SplitRecord("a", "D_B", "g-ab", "a.png", scene_id="scene-a"),
            SplitRecord("b", "D_B", "g-ab", "b.png", scene_id="scene-bc"),
            SplitRecord("c", "D_B", "g-c", "c.png", scene_id="scene-bc"),
            SplitRecord("d", "D_B", "g-d", "d.png"),
            SplitRecord("e", "D_B", "g-e", "e.png"),
        )
    )

    selection = deterministic_base_checkpoint_selection(
        manifest,
        select_fraction=0.6,
        seed=9,
    )

    # Three selected samples can only be the size-three transitive component.
    assert selection.select_sample_ids == ("a", "b", "c")
    assert selection.fit_sample_ids == ("d", "e")
    _assert_no_grouping_key_crosses(
        manifest,
        fit_ids=selection.fit_sample_ids,
        select_ids=selection.select_sample_ids,
    )


def test_builder_uses_nearest_feasible_component_count_and_honors_minima() -> None:
    manifest = _manifest(
        (
            *tuple(
                SplitRecord(
                    f"large-{index}",
                    "D_B",
                    "large",
                    f"large-{index}.png",
                )
                for index in range(4)
            ),
            *tuple(
                SplitRecord(
                    f"medium-{index}",
                    "D_B",
                    "medium",
                    f"medium-{index}.png",
                )
                for index in range(2)
            ),
            SplitRecord("single", "D_B", "single", "single.png"),
        )
    )

    selection = deterministic_base_checkpoint_selection(
        manifest,
        select_fraction=0.8,
        seed=3,
        min_fit_samples=3,
        min_select_samples=2,
    )

    # Requested 6/7 is clamped to four selectable rows; four is attainable only
    # by taking the complete "large" component.
    assert len(selection.select_sample_ids) == 4
    assert len(selection.fit_sample_ids) == 3
    assert set(selection.select_sample_ids) == {
        "large-0",
        "large-1",
        "large-2",
        "large-3",
    }


def test_builder_rejects_impossible_group_disjoint_partitions() -> None:
    one_component = _manifest(
        tuple(
            SplitRecord(f"db-{index}", "D_B", "shared", f"db-{index}.png")
            for index in range(4)
        )
    )
    with pytest.raises(BaseTrainingProvenanceError, match="fewer than two"):
        deterministic_base_checkpoint_selection(one_component)

    incompatible_minima = _manifest(
        (
            *tuple(
                SplitRecord(f"a-{index}", "D_B", "a", f"a-{index}.png")
                for index in range(4)
            ),
            *tuple(
                SplitRecord(f"b-{index}", "D_B", "b", f"b-{index}.png")
                for index in range(2)
            ),
        )
    )
    with pytest.raises(BaseTrainingProvenanceError, match="cannot satisfy"):
        deterministic_base_checkpoint_selection(
            incompatible_minima,
            min_fit_samples=3,
            min_select_samples=3,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"select_fraction": 0.0}, "select_fraction"),
        ({"select_fraction": 1.0}, "select_fraction"),
        ({"select_fraction": math.inf}, "select_fraction"),
        ({"seed": True}, "seed"),
        ({"min_fit_samples": 0}, "min_fit_samples"),
        ({"min_select_samples": False}, "min_select_samples"),
    ),
)
def test_builder_rejects_invalid_configuration(
    kwargs: dict[str, object], message: str
) -> None:
    manifest = _manifest(
        (
            SplitRecord("fit", "D_B", "fit", "fit.png"),
            SplitRecord("select", "D_B", "select", "select.png"),
        )
    )
    with pytest.raises(BaseTrainingProvenanceError, match=message):
        deterministic_base_checkpoint_selection(manifest, **kwargs)  # type: ignore[arg-type]
