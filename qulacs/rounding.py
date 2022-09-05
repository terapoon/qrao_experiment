from typing import List
from collections import defaultdict

from qulacs import QuantumState, QuantumCircuit
import numpy as np

# from vqe import VQEForQRAO
from encoding import RandomAccessEncoder


class MagicRounding:
    def __init__(
        self,
        m: int,
        n: int,
        # vqe_instance: VQEForQRAO,
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

        self.__m = m
        self.__n = n
        self.__decording_rule = self.__decording_rules[(self.__m, self.__n)]
        assert self.__decording_rule != (), f"({m},{n},p)-QRAC is not supported now."
        # self.__vqe_instance = vqe_instance
        self.__encoder = encoder

    def _circuit_converting_qrac_basis_to_z_basis(self, basis: List[int]):
        num_qubits = len(self.__encoder.qubit_to_vertex_map())
        assert len(basis) == num_qubits
        circuit = QuantumCircuit(num_qubits)

        if (self.__m, self.__n) == (1, 1):
            pass

        elif (self.__m, self.__n) == (2, 1):
            for i, base in enumerate(basis):
                if base == 0:
                    circuit.add_gate()
                elif base == 1:
                    circuit.add_gate()
                else:
                    raise ValueError

        elif (self.__m, self.__n) == (3, 1):
            BETA = np.arccos(1 / np.sqrt(3))
            for i, base in enumerate(basis):
                if base == 0:
                    circuit.add_gate(
                        RY(
                            i,
                        )
                    )
                elif base == 1:
                    circuit.add_gate()
                elif base == 2:
                    circuit.add_gate()
                elif base == 3:
                    circuit.add_gate()
                else:
                    raise ValueError

        else:
            # TODO: (3, 2) and (5, 2)
            raise ValueError

        return circuit

    def round():
        """Perform magic rounding"""
