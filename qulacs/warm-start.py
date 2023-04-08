# import packages
from maxcut_instance_generator import regular_graph
from encoding import RandomAccessEncoder
from vqe import VQEForQRAO
from rounding import MagicRounding, PauliRounding

from tqdm.auto import tqdm

import numpy as np
from numpy.linalg import eigh

from qutip import qeye, sigmax, sigmay, sigmaz, tensor, basis

# function to encode classical bits to qubit with (3,1)-QRAC
def state(b_1, b_2, b_3):
    ket_0, ket_1 = basis(2, 0), basis(2, 1)
    cos_theta = np.sqrt(1 / 2 + np.sqrt(3) / 6)
    sin_theta = np.sqrt(1 / 2 - np.sqrt(3) / 6)
    bits_pattern = (b_3, b_1, b_2)
    if bits_pattern == (0, 0, 0):
        state = cos_theta * ket_0 + np.exp(np.pi * 1j / 4) * sin_theta * ket_1
    elif bits_pattern == (0, 0, 1):
        state = cos_theta * ket_0 + np.exp(-np.pi * 1j / 4) * sin_theta * ket_1
    elif bits_pattern == (0, 1, 0):
        state = cos_theta * ket_0 + np.exp(np.pi * 3j / 4) * sin_theta * ket_1
    elif bits_pattern == (0, 1, 1):
        state = cos_theta * ket_0 + np.exp(-np.pi * 3j / 4) * sin_theta * ket_1
    elif bits_pattern == (1, 0, 0):
        state = sin_theta * ket_0 + np.exp(np.pi * 1j / 4) * cos_theta * ket_1
    elif bits_pattern == (1, 0, 1):
        state = sin_theta * ket_0 + np.exp(-np.pi * 1j / 4) * cos_theta * ket_1
    elif bits_pattern == (1, 1, 0):
        state = sin_theta * ket_0 + np.exp(np.pi * 3j / 4) * cos_theta * ket_1
    elif bits_pattern == (1, 1, 1):
        state = sin_theta * ket_0 + np.exp(-np.pi * 3j / 4) * cos_theta * ket_1
    else:
        raise ValueError
    return state


# function to make quantum state from decoded results
def make_state_from_decoded_results(decoded_results):
    ws_state = None
    for (b_1, b_2, b_3) in zip(
        decoded_results[0], decoded_results[1], decoded_results[2]
    ):
        if ws_state is None:
            ws_state = state(b_1, b_2, b_3)
        else:
            ws_state = tensor(ws_state, state(b_1, b_2, b_3))
    return ws_state


# function to run warm-start QRAO
def run_ws_qrao(
    m,
    n,
    instance,
    max_level,
    root_path,
    shots,
    sweeps,
    entanglements=["compatible", "linear", "random"],
):
    qrac = RandomAccessEncoder(m, n)
    hamiltonian = qrac.generate_hamiltonian(instance)
    num_qubit = len(qrac.qubit_to_vertex_map)

    if num_qubit <= 13:
        hamiltonian_matrix = qrac.get_hamiltonian_matrix(hamiltonian)
        eigvals, eigvecs = eigh(hamiltonian_matrix)
    else:
        eigvals = None
        eigvecs = None

    edges = qrac.calculte_edge_among_qubits(instance)

    for entanglement in tqdm(entanglements):
        start = 0 if entanglement == "compatible" else 1
        for level in tqdm(range(start, max_level)):
            # calculate initial state for warm-start.
            vqe = VQEForQRAO(
                hamiltonian,
                entanglement=entanglement,
                num_layer=level,
                qubit_pairs=edges,
                rotation_gate="efficientSU2",
                method="NFT",
                options={"maxfev": (sweeps + 1) * num_qubit * 2 * 2 * (level + 1)},
            )
            cost_history, best_theta_list = vqe.minimize()
            pauli_rounding = PauliRounding(m, n, shots, vqe, qrac)
            solution_pauli = pauli_rounding.round(best_theta_list)

            # calculate warm-start state from the results of Pauli rounding.
            ws_state = None

            # run warm-start VQE.
            ws_vqe = VQEForQRAO(
                hamiltonian,
                entanglement=entanglement,
                num_layer=level,
                qubit_pairs=edges,
                rotation_gate="efficientSU2",
                method="NFT",
                options={"maxfev": (sweeps + 1) * num_qubit * 2 * 2 * (level + 1)},
                initial_point=ws_state,
            )
