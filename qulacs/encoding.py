from typing import DefaultDict, List, Tuple, Dict
from collections import defaultdict

from qulacs import Observable
from docplex.mp.model import Model
import numpy as np
import retworkx as rx


class RandomAccessEncoder:
    # Quanrum Random Access Encoding using (m, n, p)-QRAC

    def __init__(self, m: int, n: int):
        self.__pauli_operators = defaultdict(
            lambda: (),
            {
                (1, 1): ("Z",),  # (1,1,1)-QRAC
                (2, 1): ("Z", "X"),  # (2,1,p)-QRAC, p~0.85
                (3, 1): ("Z", "X", "Y"),  # (3,1,p)-RQAC, p~0.79
                # TODO: (3, 2): (),               # (3,2,p)-QRAC, p~??
                # TODO: (5, 2): (),               # (5,2,p)-QRAC, p~??
            },
        )
        self.__m = m
        self.__n = n
        self.__operators = self.__pauli_operators[(self.__m, self.__n)]
        assert self.__operators != (), f"({m},{n},p)-QRAC is not supported now."
        self.__qubit_to_vertex_map = defaultdict(lambda: [])
        self.__vertex_to_op_map = defaultdict(lambda: ())
        self.__node_to_color_map = {}
        self.__color_to_node_map = defaultdict(lambda: [])

    @property
    def qrac_type(self) -> Tuple[int, int]:
        return (self.__m, self.__n)

    @property
    def qubit_to_vertex_map(self) -> DefaultDict[int, List[int]]:
        return self.__qubit_to_vertex_map

    @property
    def vertex_to_op_map(self) -> DefaultDict[int, Tuple[int, str]]:
        return self.__vertex_to_op_map

    @property
    def node_to_color_map(self) -> Dict[int, int]:
        return self.__node_to_color_map

    @property
    def color_to_node_map(self) -> DefaultDict[int, List[int]]:
        return self.__color_to_node_map

    def _partition_vertices(
        self, edge_weights: np.ndarray
    ) -> DefaultDict[int, List[int]]:
        num_nodes = edge_weights.shape[0]
        assert edge_weights.shape == (num_nodes, num_nodes)
        graph = rx.PyGraph()
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from_no_data(list(zip(*np.where(edge_weights != 0))))
        node_to_color_map = rx.graph_greedy_color(graph)
        color_to_node_map = defaultdict(lambda: [])
        for node, color in node_to_color_map.items():
            color_to_node_map[color].append(node)
        self.__node_to_color_map = node_to_color_map
        self.__color_to_node_map = color_to_node_map
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

    @staticmethod
    def _convert_into_ising_model(
        problem_instance: Model,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        num_variables = problem_instance.number_of_binary_variables

        # We are trying to construct minimization problem here.
        sense = -1 if problem_instance.is_maximized() else 1
        constant_term = problem_instance.objective_expr.get_constant()

        # c x -> c/2 (x' + 1)
        linear_terms_coef = np.zeros(num_variables)
        for var, coef in problem_instance.objective_expr.get_linear_part().iter_terms():
            var_idx = int(var.name[1:])
            adjusted_coef = coef * sense / 2
            linear_terms_coef[var_idx] += adjusted_coef
            constant_term += adjusted_coef

        # c x y -> c/4 (x' + y' + x' y' + 1)
        # c x x -> c/4 (2 x' + x' ^ 2 + 1) = c/4 (2 x' + 2)
        quad_terms_coef = np.zeros((num_variables, num_variables))
        for var_i, var_j, coef in problem_instance.objective_expr.iter_quad_triplets():
            var_i_idx = int(var_i.name[1:])
            var_j_idx = int(var_j.name[1:])
            adjusted_coef = coef * sense / 4
            if var_i_idx == var_j_idx:
                linear_terms_coef += 2 * adjusted_coef
                constant_term += 2 * adjusted_coef
            else:
                quad_terms_coef[var_i_idx, var_j_idx] += adjusted_coef
                quad_terms_coef[var_j_idx, var_i_idx] += adjusted_coef
                linear_terms_coef[var_i_idx] += adjusted_coef
                linear_terms_coef[var_j_idx] += adjusted_coef
                constant_term += adjusted_coef

        return constant_term, linear_terms_coef, quad_terms_coef

    def generate_hamiltonian(self, problem_instance: Model) -> Observable:
        constant_term, _, quad_terms_coef = self._convert_into_ising_model(
            problem_instance
        )
        color_to_node_map = self._partition_vertices(quad_terms_coef)
        for _, vertices in sorted(color_to_node_map.items()):
            self._add_vertices(sorted(vertices))

        num_nodes = problem_instance.number_of_binary_variables
        hamiltonian = Observable(len(self.__qubit_to_vertex_map))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                weight = quad_terms_coef[i, j]
                if weight != 0:
                    term = self._generate_term(i, j)
                    adjusted_weight = self._adjust_weight(weight)
                    hamiltonian.add_operator(adjusted_weight, term)
        hamiltonian.add_operator(constant_term, "I 0")

        return hamiltonian

    @staticmethod
    def print_hamiltonian(hamiltonian: Observable):
        pauli_table = {
            0: "I",
            1: "X",
            2: "Y",
            3: "Z",
        }
        for i in range(hamiltonian.get_term_count()):
            term = hamiltonian.get_term(i)
            coef = term.get_coef()
            index_list = term.get_index_list()
            pauli_id_list = term.get_pauli_id_list()
            term_str = str(coef)
            for j in range(hamiltonian.get_qubit_count()):
                if j in index_list:
                    idx = index_list.index(j)
                    term_str += pauli_table[pauli_id_list[idx]]
                else:
                    term_str += "I"
            print(term_str)
