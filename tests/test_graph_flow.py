import itertools

import numpy as np
import pytest

from parvaneh.graph_flow import (ArcCost, QuadraticCost, build_dual_network,
                                 convex_flow, curl_flow, residues_of_cycles)


class CoupledCost(ArcCost):
    def evaluate(self, flow):
        flow = np.asarray(flow)
        if flow.shape != (4,):
            raise ValueError("coupled cost requires the complete flow")
        total = flow[0] + flow[2] - 1
        return np.array([4 * total * total + np.dot(flow, flow),
                         0.0, 0.0, 0.0])

    def marginal_up(self, flow):
        base = self.evaluate(flow).sum()
        return np.array([self.evaluate(flow + np.eye(4, dtype=int)[i]).sum()
                         - base for i in range(4)])

    def marginal_down(self, flow):
        base = self.evaluate(flow).sum()
        return np.array([self.evaluate(flow - np.eye(4, dtype=int)[i]).sum()
                         - base for i in range(4)])

    def start(self):
        return np.zeros(4, dtype=int)


def divergence(tail, head, flow, nodes):
    return (np.bincount(tail, weights=flow, minlength=nodes)
            - np.bincount(head, weights=flow, minlength=nodes))


def test_dual_orientation_closes_both_adjacent_cycles_b12():
    cycles = [(np.array([0, 1]), np.array([1, 1])),
              (np.array([1, 2]), np.array([-1, 1]))]
    residues = np.array([1, 0])
    network = build_dual_network(cycles, residues, 3)
    flow = convex_flow(3, network.tail, network.head, network.supply,
                       QuadraticCost(np.ones(3), np.zeros(3)))
    jumps = np.zeros(3, dtype=int)
    jumps[network.arc_edge] = flow
    assert np.array_equal(np.array([np.dot(s, jumps[e])
                                    for e, s in cycles]), -residues)


def test_all_coupled_marginals_are_refreshed_after_augmentation_b14():
    tail = np.array([0, 1, 0, 2])
    head = np.array([1, 3, 2, 3])
    supply = np.array([1, 0, 0, -1])
    cost = CoupledCost()
    flow, info = convex_flow(4, tail, head, supply, cost, return_info=True)
    feasible = []
    for values in itertools.product(range(-1, 4), repeat=4):
        values = np.array(values)
        if np.array_equal(divergence(tail, head, values, 4), supply):
            feasible.append(cost.evaluate(values).sum())
    assert info.objective == min(feasible)
    assert np.array_equal(divergence(tail, head, flow, 4), supply)


def test_convex_flow_accounts_for_an_unbalanced_cost_start_b18():
    tail = np.array([0, 0])
    head = np.array([1, 1])
    supply = np.array([1, -1])
    cost = QuadraticCost(np.ones(2), np.array([0.6, 0.6]))
    flow, info = convex_flow(2, tail, head, supply, cost, return_info=True)
    assert np.array_equal(divergence(tail, head, flow, 2), supply)
    feasible = [x for x in itertools.product(range(-2, 4), repeat=2)
                if sum(x) == 1]
    expected = min(np.sum(cost.evaluate(np.array(x))) for x in feasible)
    assert info.objective == pytest.approx(expected)


def test_curl_flow_closes_cycles_and_rejects_invalid_inputs():
    cycles = [(np.array([0, 1, 2]), np.array([1, 1, -1]))]
    gradients = np.array([2.5, 2.5, -1.2831853071795862])
    flow = curl_flow(3, np.array([0, 1, 0]), np.array([1, 2, 2]),
                     gradients, cycles=cycles, method="flow")
    residue = residues_of_cycles(gradients, cycles)
    assert np.array_equal(np.array([np.dot(s, flow[e]) for e, s in cycles]),
                          -residue)
    with pytest.raises(ValueError):
        convex_flow(2, [0], [1], [1, 0], QuadraticCost([1], [0]))
