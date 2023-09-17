import os

from collections import defaultdict
from tqdm.auto import tqdm

from analysis import (
    load_results,
    extract_vqe_eigvals,
    extract_vqe_cost_history,
    result_data,
)

graphs_dict = defaultdict(lambda: {}, {})
compatible_results_dict = defaultdict(lambda: {}, {})
linear_results_dict = defaultdict(lambda: {}, {})
random_results_dict = defaultdict(lambda: {}, {})

degs = [3]
num_nodes_list = [30, 32, 34, 36, 38, 40, 42, 44, 46, 48]
num_trials = 100
num_levels = 3
# dir_name = "results3/unweighted"
dir_name = "results3/weighted"

for deg in tqdm(degs):
    for num_nodes in tqdm(num_nodes_list):
        graphs, compatible_results, linear_results, random_results = load_results(
            deg, num_nodes, num_trials, num_levels, dir_name
        )
        graphs_dict[deg][num_nodes] = graphs
        compatible_results_dict[deg][num_nodes] = compatible_results
        linear_results_dict[deg][num_nodes] = linear_results
        random_results_dict[deg][num_nodes] = random_results

eigvals_dict = defaultdict(lambda: {}, {})
cost_history_compatible_dict = defaultdict(lambda: {}, {})
cost_history_linear_dict = defaultdict(lambda: {}, {})
cost_history_random_dict = defaultdict(lambda: {}, {})
for deg in tqdm(degs):
    for num_nodes in tqdm(num_nodes_list):
        compatible_eigvals = extract_vqe_eigvals(
            compatible_results_dict, deg, num_nodes
        )
        # TODO: it's better to take assertion here.
        eigvals_dict[deg][num_nodes] = compatible_eigvals
        cost_history_compatible_dict[deg][num_nodes] = extract_vqe_cost_history(
            compatible_results_dict, deg, num_nodes
        )
        cost_history_linear_dict[deg][num_nodes] = extract_vqe_cost_history(
            linear_results_dict, deg, num_nodes
        )
        cost_history_random_dict[deg][num_nodes] = extract_vqe_cost_history(
            random_results_dict, deg, num_nodes
        )

# base_path = "/work/gs54/s54005/qrao_experiment/results4_data/unweighted/"
base_path = "/work/gs54/s54005/qrao_experiment/results4_data/weighted/"
for deg in tqdm(degs):
    deg_path = base_path + f"deg{deg}/"
    for num_nodes in tqdm(num_nodes_list):
        nodes_path = deg_path + f"nodes{num_nodes}/"
        os.makedirs(nodes_path, exist_ok=True)
        result_data(
            cost_history_compatible_dict,
            cost_history_linear_dict,
            cost_history_random_dict,
            compatible_results_dict,
            linear_results_dict,
            random_results_dict,
            deg,
            num_nodes,
            num_levels,
            num_trials,
            [
                -compatible_results_dict[deg][num_nodes][i][0]["optimum_solution"]
                for i in range(num_trials)
            ],
            nodes_path,
        )
