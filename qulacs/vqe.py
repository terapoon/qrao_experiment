from typing import List, Set, Tuple

from qulacs import QuantumState, QuantumCircuit, Observable
from qulacs.gate import CZ, RY
import numpy as np


class VQEForQRAO:
    def __init__(
        self,
        hamiltonian: Observable,
        rotation_gate: str = "normal",
        entanglement: str = "compatible",
        num_layer: int = 0,
        qubit_pairs: Set[Tuple[int, int]] = set(),
    ):
        self.__hamiltonian = hamiltonian
        self.__num_qubits = hamiltonian.get_qubit_count()

        if rotation_gate not in ["normal", "freeaxis"]:
            raise ValueError(f"rotation_gate: {rotation_gate} is not supported.")
        self.__rotation_gate = rotation_gate

        if entanglement not in ["compatible", "linear", "random"]:
            raise ValueError(f"entanglement {entanglement} is not supported.")
        self.__entanglement = entanglement

        if num_layer < 0:
            raise ValueError("num_layer should be nonnegative.")
        self.__num_layer = num_layer

        self.__qubit_pairs = qubit_pairs

    def _cost_function(self, theta_list):
        # Prepare |00...0>.
        state = QuantumState(self.__num_qubits)
        state.set_zero_state()

        # Construct ansatz circuit.
        if self.__rotation_gate == "normal":
            ansatz = self._normal_ansatz_circuit(theta_list)
        else:
            ansatz = self._free_axis_ansatz_circuit(theta_list)

        # Operate quantum circuit on a prepared state.
        ansatz.update_quantum_state(state)

        return self.__hamiltonian.get_expectation_value(state)

    def _normal_ansatz_circuit(self, theta_list: List[float]):
        circuit = QuantumCircuit(self.__num_qubits)
        # First Layer (l = 0)
        for i in range(self.__num_qubits):
            circuit.add_gate(RY(i, theta_list[i]))

        # Add Layers (l > 0)
        for layer in range(self.__num_layer):
            # Add RY gates.
            for i in range(self.__num_qubits):
                circuit.add_gate(RY(i, theta_list[(layer + 1) * self.__num_qubits + i]))

            # Add CZ gates (entanglements)
            if self.__entanglement == "compatilbe":
                # Compatible Entanglement
                for i, j in self.__qubit_pairs:
                    circuit.add_gate(CZ(i, j))

            elif self.__entanglement == "linear":
                # Linear entanglement
                for i in range(self.__num_qubits - 1):
                    circuit.add_gate(CZ(i, i + 1))

            elif self.__entanglement == "random":
                # Random entanglement
                for _ in range(len(self.__qubit_pairs)):
                    i = np.random.randint(0, self.__num_qubits)
                    while True:
                        j = np.random.randint(0, self.__num_qubits)
                        if i != j:
                            break
                    circuit.add_gate(CZ(i, j))

        return circuit

    def _free_axis_ansatz_circuit(self):
        # TODO: implement here.
        pass