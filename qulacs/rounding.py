from typing import List, Dict
from collections import defaultdict, Counter

from qulacs import QuantumCircuit
import numpy as np
from docplex.mp.model import Model
from qiskit_optimization.translators import from_docplex_mp

from vqe import VQEForQRAO
from encoding import RandomAccessEncoder


class MagicRounding:
    def __init__(
        self,
        m: int,
        n: int,
        shots: int,
        vqe_instance: VQEForQRAO,
        encoder: RandomAccessEncoder,
        basis_sampling_method: str = "uniform",
    ):
        self.__decording_rules = defaultdict(
            lambda: (),
            {
                # (1,1,1)-QRAC
                (1, 1): ({"0": [0], "1": [1]},),
                # (2,1,p)-QRAC, p~0.85
                (2, 1): (
                    # measurement with {I xi+ I, I xi- I}
                    {"0": [0, 0], "1": [1, 1]},
                    # measurement with {X xi+ X, X xi- X}
                    {"0": [0, 1], "1": [1, 0]},
                ),
                # (3,1,p)-QRAC, p~0.79
                (3, 1): (
                    # measurement with {I mu+ I, I mu- I}
                    {"0": [0, 0, 0], "1": [1, 1, 1]},
                    # measurement with {X mu+ X, X mu- X}
                    {"0": [0, 1, 1], "1": [1, 0, 0]},
                    # measurement with {Y mu+ Y, Y mu- Y}
                    {"0": [1, 0, 1], "1": [0, 1, 0]},
                    # measurement with {Z mu+ Z, Z mu- Z}
                    {"0": [1, 1, 0], "1": [0, 0, 1]},
                ),
                # TODO (3,2,p)-QRAC, p~??
                # TODO (5,2,p)-QRAC, p~??
            },
        )
        self.__operator_indices = defaultdict(
            lambda: (),
            {
                # (1,1,1)-QRAC
                (1, 1): {"Z": 0},
                # (2,1,p)-QRAC, p~0.85
                (2, 1): {"X": 0, "Z": 1},
                # (3,1,p)-QRAC, p~0.79
                (3, 1): {"X": 0, "Y": 1, "Z": 2},
                # TODO (3,2,p)-QRAC, p~??
                # TODO (5,2,p)-QRAC, p~??
            },
        )

        self.__m = m
        self.__n = n
        self.__shots = shots
        self.__decording_rule = self.__decording_rules[(self.__m, self.__n)]
        self.__operator_index = self.__operator_indices[(self.__m, self.__n)]
        assert self.__decording_rule != (), f"({m},{n},p)-QRAC is not supported now."
        self.__vqe_instance = vqe_instance
        self.__encoder = encoder
        if basis_sampling_method not in ["uniform", "weighted"]:
            raise ValueError(
                f"basis_sampling_method: {basis_sampling_method} is not supported"
            )
        self.basis_sampling_method = basis_sampling_method

    def _circuit_converting_qrac_basis_to_z_basis(self, basis: List[int]):
        num_qubits = len(self.__encoder.qubit_to_vertex_map)
        assert len(basis) == num_qubits
        circuit = QuantumCircuit(num_qubits)

        if (self.__m, self.__n) == (1, 1):
            pass

        elif (self.__m, self.__n) == (2, 1):
            for i, base in enumerate(basis):
                if base == 0:
                    phi = -np.pi / 4
                    theta = -np.pi / 2
                    circuit.add_RX_gate(i, -np.cos(phi) * theta)
                    circuit.add_RY_gate(i, -np.sin(phi) * theta)

                elif base == 1:
                    phi = -3 * np.pi / 4
                    theta = -np.pi / 2
                    circuit.add_RX_gate(i, -np.cos(phi) * theta)
                    circuit.add_RY_gate(i, -np.sin(phi) * theta)

                else:
                    raise ValueError

        elif (self.__m, self.__n) == (3, 1):
            BETA = np.arccos(1 / np.sqrt(3))
            for i, base in enumerate(basis):
                if base == 0:
                    phi = -BETA
                    theta = -np.pi / 4
                    circuit.add_RX_gate(i, -np.cos(phi) * theta)
                    circuit.add_RY_gate(i, -np.sin(phi) * theta)

                elif base == 1:
                    phi = np.pi - BETA
                    theta = np.pi / 4
                    circuit.add_RX_gate(i, -np.cos(phi) * theta)
                    circuit.add_RY_gate(i, -np.sin(phi) * theta)

                elif base == 2:
                    phi = np.pi + BETA
                    theta = np.pi / 4
                    circuit.add_RX_gate(i, -np.cos(phi) * theta)
                    circuit.add_RY_gate(i, -np.sin(phi) * theta)

                elif base == 3:
                    phi = BETA
                    theta = -np.pi / 4
                    circuit.add_RX_gate(i, -np.cos(phi) * theta)
                    circuit.add_RY_gate(i, -np.sin(phi) * theta)

                else:
                    raise ValueError

        else:
            # TODO: (3, 2) and (5, 2)
            raise ValueError

        return circuit

    def _sample_bases_uniform(self):
        if (self.__m, self.__n) == (1, 1):
            total_bases_num = 1
        elif (self.__m, self.__n) == (2, 1):
            total_bases_num = 2
        elif (self.__m, self.__n) == (3, 1):
            total_bases_num = 4
        else:
            # TODO: implement the case (3, 2) and (5, 2)
            raise NotImplementedError

        bases = [
            np.random.choice(
                total_bases_num, size=len(self.__encoder.qubit_to_vertex_map)
            ).tolist()
            for _ in range(self.__shots)
        ]
        bases, basis_shots = np.unique(bases, axis=0, return_counts=True)
        return bases, basis_shots

    def _sample_bases_weighted(self):
        # TODO: implement here
        raise NotImplementedError

    def _unpack_measurement_outcome(
        self,
        bits: str,
        basis: List[int],
    ):
        output_bits = []
        for vertex in range(len(self.__encoder.vertex_to_op_map)):
            qubit, operator = self.__encoder.vertex_to_op_map[vertex]
            operator_index = self.__operator_index[operator]
            bit_outcomes = self.__decording_rule[basis[qubit]]
            magic_bits = bit_outcomes[bits[qubit]]
            vertex_value = magic_bits[operator_index]
            output_bits.append(vertex_value)
        return output_bits

    def round(self, best_theta_list: List[float]):
        """Perform magic rounding"""
        if self.basis_sampling_method == "uniform":
            bases, basis_shots = self._sample_bases_uniform()
        elif self.basis_sampling_method == "weighted":
            bases, basis_shots = self._sample_bases_weighted()
        else:
            raise ValueError

        assert self.__shots == np.sum(basis_shots)

        # measure the relaxed state and get the measurement results.
        counts_list = []
        num_qubits = len(self.__encoder.qubit_to_vertex_map)
        for basis, shots in zip(bases, basis_shots):
            state = self.__vqe_instance._make_state(best_theta_list)
            decoding_circuit = self._circuit_converting_qrac_basis_to_z_basis(basis)
            decoding_circuit.update_quantum_state(state)
            result = [
                bin(sample)[2:].zfill(num_qubits)[::-1]
                for sample in state.sampling(shots)
            ]
            counts = dict(Counter(result))
            counts_list.append(counts)

        # decode the measurment outcomes into solution of the quadratic programming.
        solution_counts = defaultdict(lambda: 0, {})
        for basis, counts in zip(bases, counts_list):
            for meas_outcome, count in counts.items():
                solution = self._unpack_measurement_outcome(meas_outcome, basis)
                sol_key = "".join([str(int(bit)) for bit in solution])
                solution_counts[sol_key] += count

        return dict(solution_counts)

    @staticmethod
    def get_objective_value_counts(
        problem_instance: Model, solution_counts: Dict[str, int]
    ):
        objective_value_counts = defaultdict(lambda: 0, {})
        objective = from_docplex_mp(problem_instance).objective
        for key, count in solution_counts.items():
            objective_value = objective.evaluate(np.asarray([int(bit) for bit in key]))
            objective_value_counts[objective_value] += count

        objective_value_counts = {
            key: val
            for key, val in sorted(
                dict(objective_value_counts).items(),
                key=lambda item: item[1],
                reverse=True,
            )
        }

        return objective_value_counts


class PauliRounding:
    def __init__(
        self,
        m: int,
        n: int,
        shots: int,
        vqe_instance: VQEForQRAO,
        encoder: RandomAccessEncoder,
    ):
        self.__decording_rules = defaultdict(
            lambda: (),
            {
                # (1,1,1)-QRAC
                (1, 1): ({"0": [0], "1": [1]},),
                # (2,1,p)-QRAC, p~0.85
                (2, 1): (
                    # measurement with {I xi+ I, I xi- I}
                    {"0": [0, 0], "1": [1, 1]},
                    # measurement with {X xi+ X, X xi- X}
                    {"0": [0, 1], "1": [1, 0]},
                ),
                # (3,1,p)-QRAC, p~0.79
                (3, 1): (
                    # measurement with {I mu+ I, I mu- I}
                    {"0": [0, 0, 0], "1": [1, 1, 1]},
                    # measurement with {X mu+ X, X mu- X}
                    {"0": [0, 1, 1], "1": [1, 0, 0]},
                    # measurement with {Y mu+ Y, Y mu- Y}
                    {"0": [1, 0, 1], "1": [0, 1, 0]},
                    # measurement with {Z mu+ Z, Z mu- Z}
                    {"0": [1, 1, 0], "1": [0, 0, 1]},
                ),
                # TODO (3,2,p)-QRAC, p~??
                # TODO (5,2,p)-QRAC, p~??
            },
        )
        self.__operator_indices = defaultdict(
            lambda: (),
            {
                # (1,1,1)-QRAC
                (1, 1): {"Z": 0},
                # (2,1,p)-QRAC, p~0.85
                (2, 1): {"X": 0, "Z": 1},
                # (3,1,p)-QRAC, p~0.79
                (3, 1): {"X": 0, "Y": 1, "Z": 2},
                # TODO (3,2,p)-QRAC, p~??
                # TODO (5,2,p)-QRAC, p~??
            },
        )

        self.__m = m
        self.__n = n
        self.__shots = shots
        self.__decording_rule = self.__decording_rules[(self.__m, self.__n)]
        self.__operator_index = self.__operator_indices[(self.__m, self.__n)]
        assert self.__decording_rule != (), f"({m},{n},p)-QRAC is not supported now."
        self.__vqe_instance = vqe_instance
        self.__encoder = encoder

    def round(self, best_theta_list: List[float]):
        """Perform pauli rounding"""
        num_qubits = len(self.__encoder.qubit_to_vertex_map)
        state_x = self.__vqe_instance.make_state(best_theta_list)
        state_y = self.__vqe_instance.make_state(best_theta_list)
        state_z = self.__vqe_instance.make_state(best_theta_list)

        circuit_x = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            circuit_x.add_H_gate(i)
        circuit_x.update_quantum_state(state_x)

        circuit_y = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            circuit_y.add_Sdg_gate(i)
            circuit_y.add_H_gate(i)
        circuit_y.update_quantum_state(state_y)

        results = [
            [
                bin(sample)[2:].zfill(num_qubits)[::-1]
                for sample in state_x.sampling(self.__shots)
            ],
            [
                bin(sample)[2:].zfill(num_qubits)[::-1]
                for sample in state_y.sampling(self.__shots)
            ],
            [
                bin(sample)[2:].zfill(num_qubits)[::-1]
                for sample in state_z.sampling(self.__shots)
            ],
        ]

        counts = []
        for result in results:
            count = []
            for i in range(num_qubits):
                count.append(
                    defaultdict(lambda: 0, dict(Counter([val[0] for val in result])))
                )
            counts.append(count)

        decoded_results = []
        for count in counts:
            decoded_result = []
            for cnt in count:
                decoded_val = 0 if cnt["0"] > cnt["1"] else 1
                if cnt["0"] == cnt["1"]:
                    decoded_val = np.random.randint(2)
                decoded_result.append(decoded_val)
            decoded_results.append(decoded_result)

        solution = ""
        for vertex in range(len(self.__encoder.vertex_to_op_map)):
            qubit, operator = self.__encoder.vertex_to_op_map[vertex]
            operator_index = self.__operator_index[operator]
            vertex_value = decoded_results[operator_index][qubit]
            solution += str(vertex_value)

        return solution

    @staticmethod
    def get_objective_value_counts(
        problem_instance: Model,
        solution: str,
    ) -> int:
        objective = from_docplex_mp(problem_instance).objective
        objective_value = objective.evaluate(np.asarray([int(bit) for bit in solution]))

        return objective_value
