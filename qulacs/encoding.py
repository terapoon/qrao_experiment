from typing import DefaultDict, List
from collections import defaultdict

from qulacs import Observable
from docplex.mp.model import Model
import numpy as np
import retworkx as rx


class RandomAccessEncoder:
    # Quanrum Random Access Encoding using (m, n, p)-QRAC
    PAULI_OPERATORS = defaultdict({
        (1, 1): ("Z",),
        (2, 1): ("Z", "X"),
        (3, 1): ("Z", "X", "Y"),
        # TODO: (3, 2): (),
        # TODO: (5, 2): (),
    }, lambda: ())

    def __init__(self, m: int, n: int):
        self.pauli_operators = defaultdict({
            (1, 1): ("Z",),             # (1,1,1)-QRAC
            (2, 1): ("Z", "X"),         # (2,1,p)-QRAC, p~0.85
            (3, 1): ("Z", "X", "Y"),    # (3,1,p)-RQAC, p~0.79
            # TODO: (3, 2): (),               # (3,2,p)-QRAC, p~??
            # TODO: (5, 2): (),               # (5,2,p)-QRAC, p~??
        }, lambda: ())
        self.operators = self.pauli_operators[(m, n)]
    
    def _partition_vertices(edges: np.ndarray) -> DefaultDict[int, List[int]]:
        num_nodes = edges.shape[0]
        assert edges.shape == (num_nodes, num_nodes)
        graph = rx.PyGraph()
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from_no_data(list(zip(*np.where(edges != 0))))
        node_color_map = rx.graph_greedy_color(graph)
        color_node_map = defaultdict(lambda: [])
        for node, color in node_color_map.items():
            color_node_map[color].append(node)
        return color_node_map
    
    def _generate_hamiltonian(self, problem_instance: Model) -> Observable:
        # TODO: implement here.
        return Observable()
