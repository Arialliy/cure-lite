"""Exact deterministic one-to-one matching for CURE-Lite components.

The solver implements the specification's objectives in order: maximum
cardinality, minimum quantized centroid distance, maximum quantized IoU, and
finally the lexicographically smallest sorted ``(gt_id, pred_id)`` pair list.
All optimization costs are Python integers; floating-point values are used only
to construct the explicitly quantized edge attributes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

from .config import MatchConfig
from .instances import centroid_distance, mask_iou
from .types import InstanceMap, MatchPair, MatchResult


@dataclass(frozen=True)
class _Candidate:
    gt_id: int
    pred_id: int
    distance: float
    iou: float
    distance_q: int
    iou_q: int

    @property
    def identity(self) -> tuple[int, int]:
        return self.gt_id, self.pred_id


@dataclass
class _Arc:
    to: int
    reverse: int
    capacity: int
    cost: int


def _quantize_nonnegative(value: float, scale: int) -> int:
    """Use documented half-up fixed-point quantization for nonnegative values."""

    if value < 0.0:
        raise ValueError("cannot quantize a negative matching attribute")
    return int(floor(value * scale + 0.5))


def _candidate_edges(pred: InstanceMap, gt: InstanceMap, config: MatchConfig) -> tuple[_Candidate, ...]:
    if pred.shape != gt.shape:
        raise ValueError("prediction and GT instance maps must have the same shape")
    result: list[_Candidate] = []
    for target in gt.instances:
        for component in pred.instances:
            distance = centroid_distance(component, target)
            if distance >= config.max_distance:  # strict < d_max is the contract
                continue
            iou = mask_iou(component.mask, target.mask)
            result.append(
                _Candidate(
                    gt_id=target.instance_id,
                    pred_id=component.instance_id,
                    distance=distance,
                    iou=iou,
                    distance_q=_quantize_nonnegative(distance, config.distance_quantization),
                    iou_q=_quantize_nonnegative(iou, config.iou_quantization),
                )
            )
    return tuple(sorted(result, key=lambda edge: edge.identity))


def _add_arc(graph: list[list[_Arc]], source: int, target: int, capacity: int, cost: int) -> _Arc:
    forward = _Arc(to=target, reverse=len(graph[target]), capacity=capacity, cost=cost)
    backward = _Arc(to=source, reverse=len(graph[source]), capacity=0, cost=-cost)
    graph[source].append(forward)
    graph[target].append(backward)
    return forward


def _minimum_cost_flow(
    edges: tuple[_Candidate, ...],
    requested_flow: int,
    costs: tuple[int, ...],
) -> tuple[int, tuple[int, ...]]:
    """Send up to ``requested_flow`` units through a bipartite unit network."""

    if requested_flow < 0:
        raise ValueError("requested_flow may not be negative")
    if len(edges) != len(costs):
        raise ValueError("every candidate edge must have one cost")
    if requested_flow == 0 or not edges:
        return 0, ()

    gt_ids = tuple(sorted({edge.gt_id for edge in edges}))
    pred_ids = tuple(sorted({edge.pred_id for edge in edges}))
    source = 0
    gt_node = {gt_id: 1 + index for index, gt_id in enumerate(gt_ids)}
    pred_offset = 1 + len(gt_ids)
    pred_node = {pred_id: pred_offset + index for index, pred_id in enumerate(pred_ids)}
    sink = pred_offset + len(pred_ids)
    graph: list[list[_Arc]] = [[] for _ in range(sink + 1)]

    for gt_id in gt_ids:
        _add_arc(graph, source, gt_node[gt_id], 1, 0)
    for pred_id in pred_ids:
        _add_arc(graph, pred_node[pred_id], sink, 1, 0)

    candidate_arcs: list[_Arc] = []
    for edge, cost in zip(edges, costs, strict=True):
        if cost < 0:
            raise ValueError("forward matching costs must be nonnegative")
        candidate_arcs.append(
            _add_arc(graph, gt_node[edge.gt_id], pred_node[edge.pred_id], 1, cost)
        )

    flow = 0
    node_count = len(graph)
    while flow < requested_flow:
        distances: list[int | None] = [None] * node_count
        predecessor: list[tuple[int, int] | None] = [None] * node_count
        distances[source] = 0

        # Bellman-Ford is intentionally used here.  Residual reverse arcs can
        # have negative integer costs, while the local ambiguity graphs in this
        # application are small.  Strict relaxation keeps ties deterministic.
        for _ in range(node_count - 1):
            changed = False
            for node, outgoing in enumerate(graph):
                base = distances[node]
                if base is None:
                    continue
                for arc_index, arc in enumerate(outgoing):
                    if arc.capacity == 0:
                        continue
                    candidate_distance = base + arc.cost
                    current = distances[arc.to]
                    if current is None or candidate_distance < current:
                        distances[arc.to] = candidate_distance
                        predecessor[arc.to] = (node, arc_index)
                        changed = True
            if not changed:
                break

        if distances[sink] is None:
            break
        node = sink
        steps = 0
        while node != source:
            previous = predecessor[node]
            if previous is None:
                raise RuntimeError("broken augmenting path in matching solver")
            parent, arc_index = previous
            arc = graph[parent][arc_index]
            arc.capacity -= 1
            graph[node][arc.reverse].capacity += 1
            node = parent
            steps += 1
            if steps > node_count:
                raise RuntimeError("cyclic augmenting path in matching solver")
        flow += 1

    selected = tuple(index for index, arc in enumerate(candidate_arcs) if arc.capacity == 0)
    if len(selected) != flow:
        raise RuntimeError("matching flow and selected-edge count disagree")
    return flow, selected


def _maximum_cardinality(edges: tuple[_Candidate, ...]) -> int:
    if not edges:
        return 0
    upper_bound = min(
        len({edge.gt_id for edge in edges}),
        len({edge.pred_id for edge in edges}),
    )
    flow, _ = _minimum_cost_flow(edges, upper_bound, (0,) * len(edges))
    return flow


def _best_distance_and_iou(
    edges: tuple[_Candidate, ...],
    cardinality: int,
    config: MatchConfig,
) -> tuple[int, int] | None:
    if cardinality == 0:
        return (0, 0)
    if not edges:
        return None
    # With fixed cardinality, this multiplier makes one distance-quantum more
    # important than the largest possible total IoU difference.
    distance_multiplier = cardinality * config.iou_quantization + 1
    costs = tuple(
        edge.distance_q * distance_multiplier + (config.iou_quantization - edge.iou_q)
        for edge in edges
    )
    flow, selected = _minimum_cost_flow(edges, cardinality, costs)
    if flow != cardinality:
        return None
    return (
        sum(edges[index].distance_q for index in selected),
        sum(edges[index].iou_q for index in selected),
    )


def _lexicographic_optimum(edges: tuple[_Candidate, ...], config: MatchConfig) -> tuple[_Candidate, ...]:
    cardinality = _maximum_cardinality(edges)
    if cardinality == 0:
        return ()
    optimum = _best_distance_and_iou(edges, cardinality, config)
    if optimum is None:
        raise RuntimeError("failed to solve a feasible matching problem")
    target_distance, target_iou = optimum

    chosen_indices: list[int] = []
    chosen_gt: set[int] = set()
    chosen_pred: set[int] = set()
    fixed_distance = 0
    fixed_iou = 0
    next_index = 0

    # Greedily choose the smallest possible next pair.  For each proposed
    # prefix, an exact residual optimization proves whether that prefix can be
    # completed without changing the already frozen distance/IoU objectives.
    while len(chosen_indices) < cardinality:
        accepted = False
        for index in range(next_index, len(edges)):
            edge = edges[index]
            if edge.gt_id in chosen_gt or edge.pred_id in chosen_pred:
                continue
            proposed_gt = chosen_gt | {edge.gt_id}
            proposed_pred = chosen_pred | {edge.pred_id}
            remaining_count = cardinality - len(chosen_indices) - 1
            remaining_edges = tuple(
                candidate
                for later, candidate in enumerate(edges)
                if later > index
                and candidate.gt_id not in proposed_gt
                and candidate.pred_id not in proposed_pred
            )
            residual = _best_distance_and_iou(remaining_edges, remaining_count, config)
            if residual is None:
                continue
            residual_distance, residual_iou = residual
            if fixed_distance + edge.distance_q + residual_distance != target_distance:
                continue
            if fixed_iou + edge.iou_q + residual_iou != target_iou:
                continue
            chosen_indices.append(index)
            chosen_gt.add(edge.gt_id)
            chosen_pred.add(edge.pred_id)
            fixed_distance += edge.distance_q
            fixed_iou += edge.iou_q
            next_index = index + 1
            accepted = True
            break
        if not accepted:
            raise RuntimeError("failed to construct the lexicographic matching optimum")

    return tuple(edges[index] for index in chosen_indices)


def match_components(
    pred: InstanceMap,
    gt: InstanceMap,
    config: MatchConfig = MatchConfig(),
) -> MatchResult:
    """Return the exact deterministic CURE-Lite one-to-one matching."""

    if not isinstance(pred, InstanceMap) or not isinstance(gt, InstanceMap):
        raise TypeError("pred and gt must be InstanceMap objects")
    if not isinstance(config, MatchConfig):
        raise TypeError("config must be MatchConfig")
    edges = _candidate_edges(pred, gt, config)
    selected = _lexicographic_optimum(edges, config)
    pairs = tuple(
        MatchPair(
            gt_id=edge.gt_id,
            pred_id=edge.pred_id,
            distance=edge.distance,
            iou=edge.iou,
        )
        for edge in selected
    )
    return MatchResult(
        pairs=pairs,
        pred_ids=tuple(sorted(pred.ids)),
        gt_ids=tuple(sorted(gt.ids)),
    )
