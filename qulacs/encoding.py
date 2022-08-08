from typing import DefaultDict, List, Tuple
from collections import defaultdict

from qulacs import Observable
from docplex.mp.model import Model
import numpy as np
import retworkx as rx


class RandomAccessEncoder:
    # Quanrum Random Access Encoding using (m, n, p)-QRAC

    def __init__(self, m: int, n: int):
        self.__pauli_operators = defaultdict(
            {
                (1, 1): ("Z",),  # (1,1,1)-QRAC
                (2, 1): ("X", "Z"),  # (2,1,p)-QRAC, p~0.85
                (3, 1): ("X", "Y", "Z"),  # (3,1,p)-RQAC, p~0.79
                # TODO: (3, 2): (),               # (3,2,p)-QRAC, p~??
                # TODO: (5, 2): (),               # (5,2,p)-QRAC, p~??
            },
            lambda: (),
        )
        self.__m = m
        self.__n = n
        self.__operators = self.__pauli_operators[(self.__m, self.__n)]
        assert self.__operators != (), f"({m},{n},p)-QRAC is not supported now."
        self.__qubit_to_vertex_map = defaultdict(lambda: [])
        self.__vertex_to_op_map = defaultdict(lambda: ())
    
    @property
    def qrac_type(self) -> Tuple[int, int]:
        return (self.__m, self.__n)
    
    @property
    def qubit_to_vertex_map(self) -> DefaultDict[int, List[int]]:
        return self.__qubit_to_vertex_map
    
    @property
    def vertex_to_op_map(self) -> DefaultDict[int, Tuple[int, str]]:
        return self.__vertex_to_op_map

    @staticmethod
    def _partition_vertices(edges: np.ndarray) -> DefaultDict[int, List[int]]:
        num_nodes = edges.shape[0]
        assert edges.shape == (num_nodes, num_nodes)
        graph = rx.PyGraph()
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from_no_data(list(zip(*np.where(edges != 0))))
        node_to_color_map = rx.graph_greedy_color(graph)
        color_to_node_map = defaultdict(lambda: [])
        for node, color in node_to_color_map.items():
            color_to_node_map[color].append(node)
        return color_to_node_map

    def _add_vertices(self, vertices: List[int]):
        num_op_kind = len(self.__operators)
        num_qubits = len(self.__qubit_to_vertex_map)
        for idx, vertex in enumerate(vertices):
            offset, op_idx = divmod(idx, num_op_kind)
            new_qubit_idx = num_qubits + offset
            self.__vertex_to_op_map[vertex] = (new_qubit_idx, self.__operators[op_idx])
            self.__qubit_to_vertex_map[new_qubit_idx].append(vertex)
    
    def _generate_term(self, i: int, j: int) -> str:
        # FIXME: generalize to self.__n >= 2.
        op_i = self.__vertex_to_op_map[i][1]
        qubit_idx_i = self.__vertex_to_op_map[i][0]
        op_j = self.__vertex_to_op_map[j][1]
        qubit_idx_j = self.__vertex_to_op_map[j][0]
        return f"{op_i} {qubit_idx_i} {op_j} {qubit_idx_j}"

    def _adjust_weight(self, weight: int) -> float:
        # FIXME: generalize to self.__n >= 2.
        adjust_ratio = 0.5 * self.__m
        return adjust_ratio * weight
    
    def _get_edges(problem_instance: Model) -> np.ndarray:
        num_vertices = problem_instance.number_of_binary_variables
        edges = np.zeros((num_vertices, num_vertices))
        for (i, j), coef in problem_instance.q
        return edges

    def generate_hamiltonian(
        self, edges: np.ndarray, problem_instance: Model
    ) -> Observable:
        hamiltonian = Observable(len(edges))
        color_to_node_map = self._partition_vertices(edges)
        for _, vertices in sorted(color_to_node_map.items()):
            self._add_vertices(sorted(vertices))
        for i in range(len(edges)):
            for j in range(len(edges)):
                weight = edges[i, j]
                if weight != 0:
                    term = self._generate_term(i, j)
                    adjusted_weight = self._adjust_weight(weight)
                    hamiltonian.add_operator(weight, term)

        return hamiltonian
