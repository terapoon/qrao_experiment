# import packages
from maxcut_instance_generator import regular_graph
from encoding import RandomAccessEncoder
from vqe import VQEForQRAO
from rounding import MagicRounding

import os
import pickle
from tqdm.auto import tqdm

from networkx import node_link_data

# function to run QRAO
def run_qrao(m, n, instance, max_level, root_path, shots):
    qrac = RandomAccessEncoder(m, n)
    hamiltonian = qrac.generate_hamiltonian(instance)
    # print(f"Hamiltonian is {hamiltonian}")
    num_qubit = len(qrac.qubit_to_vertex_map)
    num_edge = len(qrac.calculate_edge_among_qubits(instance))
    # print(f"{num_qubit} qubits, {num_edge} edges")

    for entanglement in tqdm(["compatible", "linear", "random"]):
        for level in tqdm(range(max_level)):
            vqe = VQEForQRAO(
                hamiltonian,
                entanglement=entanglement,
                num_layer=level,
                qubit_pairs=qrac.calculate_edge_among_qubits(instance),
                rotation_gate="efficientSU2",
                method="NFT",
                options=None,
            )
            cost_history, best_theta_list = vqe.minimize()
            rounding = MagicRounding(m, n, shots, vqe, qrac)
            solution_counts = rounding.round(best_theta_list)
            maxcut_values = rounding.get_objective_value_counts(
                instance, solution_counts
            )

            # result of the experiment
            result = {
                "solution_counts": solution_counts,
                "maxcut_values": maxcut_values,
                "num_qubit": num_qubit,
                "num_edge": num_edge,
                "entanglement": entanglement,
                "level": level,
                "optimum_solution": instance.solve().get_objective_value(),
                "cost_history": cost_history,
                "best_theta_list": best_theta_list,
            }

            # save experiment result
            save_path = f"{root_path}/{m}-{n}/{entanglement}/"
            os.makedirs(save_path, exist_ok=True)
            save_file_name = f"{save_path}/level{level}.pkl"
            with open(save_file_name, "wb") as f:
                pickle.dump(result, f)


# search pattern
search_pattern = {3: [42, 44, 46, 48, 50]}
qrao_patterns = [(3, 1)]
# qrao_patterns = [(2, 1)]
# qrao_patterns = [(1, 1)]
MAX_LEVEL = 5
TRIAL = 10
ROUNDING_SHOTS = 1000

for deg, num_vertices in search_pattern.items():
    for num in num_vertices:
        for m, n in qrao_patterns:
            for i in tqdm(range(TRIAL)):
                graph, instance = regular_graph(num, deg)
                root_path = f"results_debug/regular/deg{deg}/nodes{num}/trial{i}"
                os.makedirs(root_path, exist_ok=True)
                with open(f"{root_path}/graph_data.pkl", "wb") as f:
                    pickle.dump(node_link_data(graph), f)

                run_qrao(
                    m,
                    n,
                    instance,
                    MAX_LEVEL,
                    f"results_debug/regular/deg{deg}/nodes{num}/trial{i}",
                    ROUNDING_SHOTS,
                )
