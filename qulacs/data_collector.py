# import packages
from maxcut_instance_generator import regular_graph
from encoding import RandomAccessEncoder
from vqe import VQEForQRAO
from rounding import MagicRounding, PauliRounding

import os
import pickle
from tqdm.auto import tqdm

from networkx import node_link_data
from numpy.linalg import eigh


# function to run QRAO
def run_qrao(
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
    # print(f"Hamiltonian is {hamiltonian}")
    num_qubit = len(qrac.qubit_to_vertex_map)

    if num_qubit <= 1:
        hamiltonian_matrix = qrac.get_hamiltonian_matrix(hamiltonian)
        eigvals, eigvecs = eigh(hamiltonian_matrix)
    else:
        eigvals = None
        eigvecs = None

    num_edge = len(qrac.calculate_edge_among_qubits(instance))
    # print(f"{num_qubit} qubits, {num_edge} edges")

    for (idx, entanglement) in tqdm(enumerate(entanglements)):
        start = 0 if idx == 0 else 1
        for level in tqdm(range(start, max_level)):
            vqe = VQEForQRAO(
                hamiltonian,
                entanglement=entanglement,
                num_layer=level,
                qubit_pairs=qrac.calculate_edge_among_qubits(instance),
                rotation_gate="efficientSU2",
                method="NFT",
                options={"maxfev": (sweeps + 1) * num_qubit * 2 * 2 * (level + 1)},
            )
            cost_history, best_theta_list = vqe.minimize()
            magic_rounding = MagicRounding(m, n, shots, vqe, qrac)
            solution_counts_magic = magic_rounding.round(best_theta_list)
            maxcut_values_magic = magic_rounding.get_objective_value_counts(
                instance, solution_counts_magic
            )

            pauli_rounding = PauliRounding(m, n, shots, vqe, qrac)
            solution_pauli = pauli_rounding.round(best_theta_list)
            maxcut_value_pauli = pauli_rounding.get_objective_value(
                instance, solution_pauli
            )

            # result of the experiment
            result = {
                "solution_counts_magic": solution_counts_magic,
                "maxcut_values_magic": maxcut_values_magic,
                "solution_pauli": solution_pauli,
                "maxcut_value_pauli": maxcut_value_pauli,
                "num_qubit": num_qubit,
                "num_edge": num_edge,
                "entanglement": entanglement,
                "level": level,
                "optimum_solution": instance.solve().get_objective_value(),
                "cost_history": cost_history,
                "best_theta_list": best_theta_list,
                "eigvals": eigvals,
                "eigvecs": eigvecs,
            }

            # save experiment result
            save_path = f"{root_path}/{m}-{n}/{entanglement}/"
            os.makedirs(save_path, exist_ok=True)
            save_file_name = f"{save_path}/level{level}.pkl"
            with open(save_file_name, "wb") as f:
                pickle.dump(result, f)


# search pattern
search_pattern = {deg: [48] for deg in [3]}
qrao_patterns = [(3, 1)]
# qrao_patterns = [(2, 1)]
# qrao_patterns = [(1, 1)]
MAX_LEVEL = 3
MIN_TRIAL = 95
MAX_TRIAL = 100
ROUNDING_SHOTS = 1000
SWEEPS = 15
WEIGHTED = True
entanglements = ["compatible", "linear", "random"]

for deg, num_vertices in search_pattern.items():
    for num in num_vertices:
        for m, n in qrao_patterns:
            for i in tqdm(range(MIN_TRIAL, MAX_TRIAL)):
                graph, instance = regular_graph(num, deg, seed=i, weighted=WEIGHTED)
                root_path = f"results3/weighted/deg{deg}/nodes{num}/trial{i}"
                os.makedirs(root_path, exist_ok=True)
                with open(f"{root_path}/graph_data.pkl", "wb") as f:
                    pickle.dump(node_link_data(graph), f)

                run_qrao(
                    m,
                    n,
                    instance,
                    MAX_LEVEL,
                    root_path,
                    ROUNDING_SHOTS,
                    SWEEPS,
                    entanglements,
                )
