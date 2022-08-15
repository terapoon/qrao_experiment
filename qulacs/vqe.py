from qulacs import QuantumState, QuantumCircuit, Observable
from qulacs.gate import CZ, RY


class VQEForQRAO:
    def __init__(
        self,
        hamiltonian: Observable,
        rotation_gate: str = "normal",
        entanglement: str = "compatible",
        init_state: str = "zero",
        num_layer: int = 0,
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
            raise ValueError(f"num_layer should be nonnegative.")
        self.__num_layer = num_layer
        
        if init_state not in ["zero", "plus"]:
            raise ValueError(f"init_state {init_state} is not supported.")
        self.__init_state = init_state

    def _cost_function(self):
        state = QuantumState(self.__num_qubits)
        if self.__init_state == "plus":

            
        if self.__rotation_gate == "normal":
            theta_list, ansatz = self._normal_ansatz_circuit()
        else:
            theta_list, ansatz = self._free_axis_ansatz_circuit()

        circuit.update_quantum_state(state)

    def _normal_ansatz_circuit(self):
    

    def _free_axis_ansatz_circuit(self):
