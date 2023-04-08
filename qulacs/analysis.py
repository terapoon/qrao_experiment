import pickle
from typing import List
from networkx import node_link_graph
import numpy as np
import matplotlib.pyplot as plt


def load_results(
    deg: int,
    num_nodes: int,
    num_trials: int,
    num_levels: int,
    dir_name: str = "results/regular",
):
    graphs = []
    compatible_results = []
    linear_results = []
    random_results = []
    root_path = (
        f"/work/gs54/s54005/qrao_experiment/{dir_name}/deg{deg}/nodes{num_nodes}/"
    )

    for trial in range(num_trials):
        trial_path = root_path + f"trial{trial}/"
        with open(trial_path + "graph_data.pkl", "rb") as file:
            data = pickle.load(file)
        graph = node_link_graph(data)
        graphs.append(graph)
        compatible_result = []
        linear_result = []
        random_result = []
        compatible_root = trial_path + "3-1/compatible/"
        linear_root = trial_path + "3-1/linear/"
        random_root = trial_path + "3-1/random/"

        for level in range(num_levels):
            with open(compatible_root + f"level{level}.pkl", "rb") as file:
                compatible = pickle.load(file)
            compatible_result.append(compatible)

            if level == 0:
                linear_result.append(compatible)
                random_result.append(compatible)
            else:
                with open(linear_root + f"level{level}.pkl", "rb") as file:
                    linear = pickle.load(file)
                with open(random_root + f"level{level}.pkl", "rb") as file:
                    random = pickle.load(file)
                linear_result.append(linear)
                random_result.append(random)

        compatible_results.append(compatible_result)
        linear_results.append(linear_result)
        random_results.append(random_result)

    return graphs, compatible_results, linear_results, random_results


def calc_exp(maxcut_values):
    shots = 1000
    sum_val = 0
    for key, val in maxcut_values.items():
        sum_val += key * val
    return sum_val / shots


def get_mean(values):
    values_mean = []
    for level in range(len(values[0])):
        sum_val = 0
        for instance in range(len(values)):
            sum_val += values[instance][level]
        sum_val /= len(values)
        values_mean.append(sum_val)
    return values_mean


def extract_vqe_eigvals(results_dict, deg, num_nodes):
    results = results_dict[deg][num_nodes]
    eigvals_list = []
    for result in results:
        eigvals = []
        for level_result in result:
            eigvals.append(level_result["eigvals"])
        eigvals_list.append(eigvals)
    return eigvals_list


def extract_vqe_cost_history(results_dict, deg, num_nodes):
    results = results_dict[deg][num_nodes]
    cost_history_list = []
    for result in results:
        cost_history = []
        for level_result in result:
            parameter_num = len(level_result["best_theta_list"])
            cost_history.append(level_result["cost_history"][::parameter_num])
        cost_history_list.append(cost_history)
    return cost_history_list


def plot_cost_history_by_entanglement(
    eigvals_dict,
    cost_history_compatible_dict,
    cost_history_linear_dict,
    cost_history_random_dict,
    deg: int,
    num_nodes: int,
    num_trial: int = 0,
    entanglement: str = "compatible",
    plot_eigvals: bool = True,
):
    # load values
    eigvals = eigvals_dict[deg][num_nodes][num_trial]
    if entanglement == "compatible":
        history = cost_history_compatible_dict[deg][num_nodes][num_trial]
    elif entanglement == "linear":
        history = cost_history_linear_dict[deg][num_nodes][num_trial]
    elif entanglement == "random":
        history = cost_history_random_dict[deg][num_nodes][num_trial]
    else:
        raise ValueError

    max_len = 0
    for level, hist in enumerate(history):
        plt.plot(hist, label=f"{level} layer")
        max_len = max(max_len, len(hist))

    if plot_eigvals:
        min_eigval = eigvals[0][0]
        second_min_eigval = eigvals[0][1]
        third_min_eigval = eigvals[0][2]
        forth_min_eigval = eigvals[0][3]

        plt.plot(
            [min_eigval for _ in range(max_len)],
            color="black",
            label="1st min relax cut",
            linestyle="dashed",
        )
        plt.plot(
            [second_min_eigval for _ in range(max_len)],
            color="gray",
            label="2nd min relax cut",
            linestyle="dashdot",
        )
        plt.plot(
            [third_min_eigval for _ in range(max_len)],
            color="blue",
            label="3rd min relax cut",
            linestyle="dotted",
        )
        plt.plot(
            [forth_min_eigval for _ in range(max_len)],
            color="purple",
            label="4th relax cut",
            linestyle="dashed",
        )

    plt.xlabel("Num of sweep")
    plt.ylabel("Relax maxcut value")
    plt.legend()
    plt.show()


def plot_relax_maxcut_results_for_instance(
    eigvals_dict,
    cost_history_compatible_dict,
    cost_history_linear_dict,
    cost_history_random_dict,
    deg: int,
    num_nodes: int,
    num_levels: int,
    maxcut_vals: List[int],
    num_trial: int = 0,
    plot_eigvals: bool = False,
    save_file_name: str = None,
):
    # load values
    maxcut_val = maxcut_vals[num_trial]
    eigvals = eigvals_dict[deg][num_nodes][num_trial]
    compatible_hist = cost_history_compatible_dict[deg][num_nodes][num_trial]
    linear_hist = cost_history_linear_dict[deg][num_nodes][num_trial]
    random_hist = cost_history_random_dict[deg][num_nodes][num_trial]

    compatible = [hist[-1] / maxcut_val for hist in compatible_hist]
    linear = [hist[-1] / maxcut_val for hist in linear_hist]
    random = [hist[-1] / maxcut_val for hist in random_hist]

    plt.rcParams["font.size"] = 16
    plt.plot(
        [-0.3, num_levels - 0.7],
        [1, 1],
        color="black",
        label="opt maxcut",
        linestyle="-",
        linewidth=0.5,
    )
    plt.plot(
        [i - 0.1 for i in range(num_levels)],
        compatible,
        marker="o",
        markersize=8,
        linestyle="dotted",
        color="blue",
        label="compatible",
    )
    plt.plot(
        [i for i in range(num_levels)],
        linear,
        marker="o",
        markersize=8,
        linestyle="dotted",
        color="red",
        label="linear",
    )
    plt.plot(
        [i + 0.1 for i in range(num_levels)],
        random,
        marker="o",
        markersize=8,
        linestyle="dotted",
        color="green",
        label="random",
    )

    if plot_eigvals:
        min_eigval = eigvals[0][0]
        second_min_eigval = eigvals[0][1]
        third_min_eigval = eigvals[0][2]
        forth_min_eigval = eigvals[0][3]
        plt.plot(
            [-0.3, num_levels - 0.7],
            [min_eigval / maxcut_val for _ in range(2)],
            color="orange",
            label="1st relaxed-maxcut",
            linestyle="dashed",
            linewidth=0.5,
        )
        plt.plot(
            [-0.3, num_levels - 0.7],
            [second_min_eigval / maxcut_val for _ in range(2)],
            color="gray",
            label="2nd relaxed-maxcut",
            linestyle="dashed",
            linewidth=0.5,
        )
        plt.plot(
            [-0.3, num_levels - 0.7],
            [third_min_eigval / maxcut_val for _ in range(2)],
            color="blue",
            label="3rd relaxed-maxcut",
            linestyle="dashed",
            linewidth=0.5,
        )
        plt.plot(
            [-0.3, num_levels - 0.7],
            [forth_min_eigval / maxcut_val for _ in range(2)],
            color="purple",
            label="4th relaxed-maxcut",
            linestyle="dashed",
            linewidth=0.5,
        )

    plt.xticks([i for i in range(num_levels)])
    plt.xlim(-0.3, num_levels - 0.7)
    plt.ylim(0.9, 1.25)
    plt.xlabel("Num of entanglement layer")
    plt.ylabel("Relax maxcut / Optimal maxcut")
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    if save_file_name is not None:
        plt.savefig(save_file_name, bbox_inches="tight")
    plt.show()


def result_data(
    cost_history_compatible_dict,
    cost_history_linear_dict,
    cost_history_random_dict,
    compatible_results_dict,
    linear_results_dict,
    random_results_dict,
    deg: int,
    num_nodes: int,
    num_levels: int,
    num_trials: int,
    maxcut_vals: List[int],
    save_base_path: str = None,
):
    # load values
    compatible_relax = np.array(
        [
            [hist[-1] / maxcut_val for hist in trial]
            for (trial, maxcut_val) in zip(
                cost_history_compatible_dict[deg][num_nodes], maxcut_vals
            )
        ]
    )
    linear_relax = np.array(
        [
            [hist[-1] / maxcut_val for hist in trial]
            for (trial, maxcut_val) in zip(
                cost_history_linear_dict[deg][num_nodes], maxcut_vals
            )
        ]
    )
    random_relax = np.array(
        [
            [hist[-1] / maxcut_val for hist in trial]
            for (trial, maxcut_val) in zip(
                cost_history_random_dict[deg][num_nodes], maxcut_vals
            )
        ]
    )
    compatible_results = compatible_results_dict[deg][num_nodes]
    linear_results = linear_results_dict[deg][num_nodes]
    random_results = random_results_dict[deg][num_nodes]
    opts = [compatible_results[i][0]["optimum_solution"] for i in range(num_trials)]
    compatible_magic_results = []
    linear_magic_results = []
    random_magic_results = []
    compatible_pauli_results = []
    linear_pauli_results = []
    random_pauli_results = []

    # iteration for instances
    for trial, (compatible_result, linear_result, random_result) in enumerate(
        zip(compatible_results, linear_results, random_results)
    ):
        compatible_magic = []
        linear_magic = []
        random_magic = []
        compatible_pauli = []
        linear_pauli = []
        random_pauli = []
        # iteration for num of layers
        for compatible, linear, random in zip(
            compatible_result, linear_result, random_result
        ):
            compatible_magic.append(
                max(list(compatible["maxcut_values_magic"].keys())) / opts[trial]
            )
            compatible_pauli.append(compatible["maxcut_value_pauli"] / opts[trial])
            linear_magic.append(
                max(list(linear["maxcut_values_magic"].keys())) / opts[trial]
            )
            linear_pauli.append(linear["maxcut_value_pauli"] / opts[trial])
            random_magic.append(
                max(list(random["maxcut_values_magic"].keys())) / opts[trial]
            )
            random_pauli.append(random["maxcut_value_pauli"] / opts[trial])
        compatible_magic_results.append(compatible_magic)
        linear_magic_results.append(linear_magic)
        random_magic_results.append(random_magic)
        compatible_pauli_results.append(compatible_pauli)
        linear_pauli_results.append(linear_pauli)
        random_pauli_results.append(random_pauli)

    compatible_magic_results = np.array(compatible_magic_results)
    linear_magic_results = np.array(linear_magic_results)
    random_magic_results = np.array(random_magic_results)
    compatible_pauli_results = np.array(compatible_pauli_results)
    linear_pauli_results = np.array(linear_pauli_results)
    random_pauli_results = np.array(random_pauli_results)

    np.savez(
        save_base_path + "data",
        compatible_relax=compatible_relax,
        linear_relax=linear_relax,
        random_relax=random_relax,
        compatible_magic=compatible_magic_results,
        linear_magic=linear_magic_results,
        random_magic=random_magic_results,
        compatible_pauli=compatible_pauli_results,
        linear_pauli=linear_pauli_results,
        random_pauli=random_pauli_results,
    )


def plot_relax_maxcut_results_with_error_bar(
    cost_history_compatible_dict,
    cost_history_linear_dict,
    cost_history_random_dict,
    deg: int,
    num_nodes: int,
    num_levels: int,
    maxcut_vals: List[int],
    save_file_name: str = None,
):
    # load values
    compatible = np.array(
        [
            [hist[-1] / maxcut_val for hist in trial]
            for (trial, maxcut_val) in zip(
                cost_history_compatible_dict[deg][num_nodes], maxcut_vals
            )
        ]
    )
    linear = np.array(
        [
            [hist[-1] / maxcut_val for hist in trial]
            for (trial, maxcut_val) in zip(
                cost_history_linear_dict[deg][num_nodes], maxcut_vals
            )
        ]
    )
    random = np.array(
        [
            [hist[-1] / maxcut_val for hist in trial]
            for (trial, maxcut_val) in zip(
                cost_history_random_dict[deg][num_nodes], maxcut_vals
            )
        ]
    )

    compatible_err = [
        compatible.mean(axis=0) - compatible.min(axis=0),
        compatible.max(axis=0) - compatible.mean(axis=0),
    ]
    linear_err = [
        linear.mean(axis=0) - linear.min(axis=0),
        linear.max(axis=0) - linear.mean(axis=0),
    ]
    random_err = [
        random.mean(axis=0) - random.min(axis=0),
        random.max(axis=0) - random.mean(axis=0),
    ]

    plt.rcParams["font.size"] = 16
    plt.plot(
        [-0.3, num_levels - 0.7],
        [1, 1],
        color="black",
        label="opt maxcut",
        linestyle="-",
        linewidth=0.5,
    )
    plt.errorbar(
        [i - 0.1 for i in range(num_levels)],
        compatible.mean(axis=0),
        compatible_err,
        marker="o",
        markerfacecolor="blue",
        markeredgecolor="blue",
        markersize=8,
        linestyle="dotted",
        color="blue",
        label="compatible",
        capsize=8,
    )
    plt.errorbar(
        [i for i in range(num_levels)],
        linear.mean(axis=0),
        linear_err,
        marker="o",
        markerfacecolor="red",
        markeredgecolor="red",
        markersize=8,
        linestyle="dotted",
        color="red",
        label="linear",
        capsize=8,
    )
    plt.errorbar(
        [i + 0.1 for i in range(num_levels)],
        random.mean(axis=0),
        random_err,
        marker="o",
        markerfacecolor="green",
        markeredgecolor="green",
        markersize=8,
        linestyle="dotted",
        color="green",
        label="random",
        capsize=8,
    )

    plt.xticks([i for i in range(num_levels)])
    plt.xlim(-0.3, num_levels - 0.7)
    plt.ylim(0.9, 1.25)
    plt.xlabel("Num of entanglement layer")
    plt.ylabel("Relax maxcut / Optimal maxcut")
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    if save_file_name is not None:
        plt.savefig(save_file_name, bbox_inches="tight")
    plt.show()


def calc_statistics(
    graphs_dict,
    compatible_results_dict,
    linear_results_dict,
    random_results_dict,
    deg,
    num_nodes,
):
    compatible_values_max_pauli = []
    linear_values_max_pauli = []
    random_values_max_pauli = []
    compatible_values_max_magic = []
    linear_values_max_magic = []
    random_values_max_magic = []
    compatible_values_exp_magic = []
    linear_values_exp_magic = []
    random_values_exp_magic = []
    compatible_values_freq_magic = []
    linear_values_freq_magic = []
    random_values_freq_magic = []
    opt_values = []

    graphs = graphs_dict[deg][num_nodes]
    compatible_results = compatible_results_dict[deg][num_nodes]
    linear_results = linear_results_dict[deg][num_nodes]
    random_results = random_results_dict[deg][num_nodes]

    # iteration for instances
    for graph, compatible_result, linear_result, random_result in zip(
        graphs, compatible_results, linear_results, random_results
    ):
        opt_value = -1
        compatible_value_max_pauli = []
        linear_value_max_pauli = []
        random_value_max_pauli = []
        compatible_value_max_magic = []
        linear_value_max_magic = []
        random_value_max_magic = []
        compatible_value_exp_magic = []
        linear_value_exp_magic = []
        random_value_exp_magic = []
        compatible_value_freq_magic = []
        linear_value_freq_magic = []
        random_value_freq_magic = []

        # iteration for levels
        for compatible, linear, random in zip(
            compatible_result, linear_result, random_result
        ):
            compatible_maxcut_max_pauli = compatible["maxcut_value_pauli"]
            linear_maxcut_max_pauli = linear["maxcut_value_pauli"]
            random_maxcut_max_pauli = random["maxcut_value_pauli"]
            compatible_maxcut_max_magic = max(
                list(compatible["maxcut_values_magic"].keys())
            )
            linear_maxcut_max_magic = max(list(linear["maxcut_values_magic"].keys()))
            random_maxcut_max_magic = max(list(random["maxcut_values_magic"].keys()))
            compatible_maxcut_exp_magic = calc_exp(compatible["maxcut_values_magic"])
            linear_maxcut_exp_magic = calc_exp(linear["maxcut_values_magic"])
            random_maxcut_exp_magic = calc_exp(random["maxcut_values_magic"])
            compatible_maxcut_freq_magic = list(
                compatible["maxcut_values_magic"].keys()
            )[0]
            linear_maxcut_freq_magic = list(linear["maxcut_values_magic"].keys())[0]
            random_maxcut_freq_magic = list(random["maxcut_values_magic"].keys())[0]
            assert (
                compatible["optimum_solution"]
                == linear["optimum_solution"]
                == random["optimum_solution"]
            )
            assert opt_value == -1 or opt_value == compatible["optimum_solution"]
            opt_value = max(opt_value, compatible["optimum_solution"])
            compatible_value_max_pauli.append(compatible_maxcut_max_pauli)
            linear_value_max_pauli.append(linear_maxcut_max_pauli)
            random_value_max_pauli.append(random_maxcut_max_pauli)
            compatible_value_max_magic.append(compatible_maxcut_max_magic)
            linear_value_max_magic.append(linear_maxcut_max_magic)
            random_value_max_magic.append(random_maxcut_max_magic)
            compatible_value_exp_magic.append(compatible_maxcut_exp_magic)
            linear_value_exp_magic.append(linear_maxcut_exp_magic)
            random_value_exp_magic.append(random_maxcut_exp_magic)
            compatible_value_freq_magic.append(compatible_maxcut_freq_magic)
            linear_value_freq_magic.append(linear_maxcut_freq_magic)
            random_value_freq_magic.append(random_maxcut_freq_magic)

        compatible_values_max_pauli.append(compatible_value_max_pauli)
        linear_values_max_pauli.append(linear_value_max_pauli)
        random_values_max_pauli.append(random_value_max_pauli)
        compatible_values_max_magic.append(compatible_value_max_magic)
        linear_values_max_magic.append(linear_value_max_magic)
        random_values_max_magic.append(random_value_max_magic)
        compatible_values_exp_magic.append(compatible_value_exp_magic)
        linear_values_exp_magic.append(linear_value_exp_magic)
        random_values_exp_magic.append(random_value_exp_magic)
        compatible_values_freq_magic.append(compatible_value_freq_magic)
        linear_values_freq_magic.append(linear_value_freq_magic)
        random_values_freq_magic.append(random_value_freq_magic)
        opt_values.append(opt_value)

    return (
        compatible_values_max_pauli,
        linear_values_max_pauli,
        random_values_max_pauli,
        compatible_values_max_magic,
        linear_values_max_magic,
        random_values_max_magic,
        compatible_values_exp_magic,
        linear_values_exp_magic,
        random_values_exp_magic,
        compatible_values_freq_magic,
        linear_values_freq_magic,
        random_values_freq_magic,
        opt_values,
    )


def plot_maxcut_results(
    compatible_values_max_pauli_dict,
    linear_values_max_pauli_dict,
    random_values_max_pauli_dict,
    compatible_values_max_magic_dict,
    linear_values_max_magic_dict,
    random_values_max_magic_dict,
    compatible_values_exp_magic_dict,
    linear_values_exp_magic_dict,
    random_values_exp_magic_dict,
    compatible_values_freq_magic_dict,
    linear_values_freq_magic_dict,
    random_values_freq_magic_dict,
    opt_values_dict,
    deg: int,
    num_nodes: int,
    calc_mean: bool = True,
    num_trial: int = 0,
):
    # load calculated statistics
    compatible_values_max_pauli = compatible_values_max_pauli_dict[deg][num_nodes]
    linear_values_max_pauli = linear_values_max_pauli_dict[deg][num_nodes]
    random_values_max_pauli = random_values_max_pauli_dict[deg][num_nodes]
    compatible_values_max_magic = compatible_values_max_magic_dict[deg][num_nodes]
    linear_values_max_magic = linear_values_max_magic_dict[deg][num_nodes]
    random_values_max_magic = random_values_max_magic_dict[deg][num_nodes]
    compatible_values_exp_magic = compatible_values_exp_magic_dict[deg][num_nodes]
    linear_values_exp_magic = linear_values_exp_magic_dict[deg][num_nodes]
    random_values_exp_magic = random_values_exp_magic_dict[deg][num_nodes]
    compatible_values_freq_magic = compatible_values_freq_magic_dict[deg][num_nodes]
    linear_values_freq_magic = linear_values_freq_magic_dict[deg][num_nodes]
    random_values_freq_magic = random_values_freq_magic_dict[deg][num_nodes]
    opt_values = opt_values_dict[deg][num_nodes]

    if calc_mean:
        # calculate mean for instances
        compatible_values_max_pauli_for_plot = get_mean(compatible_values_max_pauli)
        linear_values_max_pauli_for_plot = get_mean(linear_values_max_pauli)
        random_values_max_pauli_for_plot = get_mean(random_values_max_pauli)
        compatible_values_max_magic_for_plot = get_mean(compatible_values_max_magic)
        linear_values_max_magic_for_plot = get_mean(linear_values_max_magic)
        random_values_max_magic_for_plot = get_mean(random_values_max_magic)
        compatible_values_exp_magic_for_plot = get_mean(compatible_values_exp_magic)
        linear_values_exp_magic_for_plot = get_mean(linear_values_exp_magic)
        random_values_exp_magic_for_plot = get_mean(random_values_exp_magic)
        compatible_values_freq_magic_for_plot = get_mean(compatible_values_freq_magic)
        linear_values_freq_magic_for_plot = get_mean(linear_values_freq_magic)
        random_values_freq_magic_for_plot = get_mean(random_values_freq_magic)

        # FIXME: fig this logic
        opt_values_for_plot = opt_values[0]

    else:
        # retrieve the result for the instance identified by num_trial
        compatible_values_max_pauli_for_plot = compatible_values_max_pauli[num_trial]
        linear_values_max_pauli_for_plot = linear_values_max_pauli[num_trial]
        random_values_max_pauli_for_plot = random_values_max_pauli[num_trial]
        compatible_values_max_magic_for_plot = compatible_values_max_magic[num_trial]
        linear_values_max_magic_for_plot = linear_values_max_magic[num_trial]
        random_values_max_magic_for_plot = random_values_max_magic[num_trial]
        compatible_values_exp_magic_for_plot = compatible_values_exp_magic[num_trial]
        linear_values_exp_magic_for_plot = linear_values_exp_magic[num_trial]
        random_values_exp_magic_for_plot = random_values_exp_magic[num_trial]
        compatible_values_freq_magic_for_plot = compatible_values_freq_magic[num_trial]
        linear_values_freq_magic_for_plot = linear_values_freq_magic[num_trial]
        random_values_freq_magic_for_plot = random_values_freq_magic[num_trial]
        opt_values_for_plot = opt_values[num_trial]

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle("MaxCut values")

    ax1.plot(compatible_values_max_magic_for_plot, color="red", label="compatible")
    ax1.plot(linear_values_max_magic_for_plot, color="green", label="linear")
    ax1.plot(random_values_max_magic_for_plot, color="blue", label="random")
    ax1.plot(
        [opt_values_for_plot for i in range(len(compatible_values_max_magic_for_plot))],
        color="black",
        label="opt",
    )
    ax1.set_xlabel("Num of entanglement layers")
    ax1.set_ylabel("Found maxcut value (max)")
    ax1.legend()

    ax2.plot(compatible_values_exp_magic_for_plot, color="red", label="compatible")
    ax2.plot(linear_values_exp_magic_for_plot, color="green", label="linear")
    ax2.plot(random_values_exp_magic_for_plot, color="blue", label="random")
    ax2.plot(
        [opt_values_for_plot for i in range(len(compatible_values_max_magic_for_plot))],
        color="black",
        label="opt",
    )
    ax2.set_xlabel("Num of entanglement layers")
    ax2.set_ylabel("Found maxcut value (exp)")
    ax2.legend()

    ax3.plot(compatible_values_freq_magic_for_plot, color="red", label="compatible")
    ax3.plot(linear_values_freq_magic_for_plot, color="green", label="linear")
    ax3.plot(random_values_freq_magic_for_plot, color="blue", label="random")
    ax3.plot(
        [opt_values_for_plot for i in range(len(compatible_values_max_magic_for_plot))],
        color="black",
        label="opt",
    )
    ax3.set_xlabel("Num of entanglement layers")
    ax3.set_ylabel("Found maxcut value (freq)")
    ax3.legend()

    ax4.plot(
        compatible_values_max_pauli_for_plot,
        color="red",
        label="Pauli Rounding (compatible)",
    )
    ax4.plot(
        compatible_values_max_magic_for_plot,
        color="blue",
        label="Magic Rounding (compatible)",
    )
    ax4.plot(
        [opt_values_for_plot for i in range(len(compatible_values_max_magic_for_plot))],
        color="black",
        label="opt",
    )
    ax4.set_xlabel("Num of entanglement layers")
    ax4.set_ylabel("Found maxcut value (max)")
    ax4.legend()

    ax5.plot(
        linear_values_max_pauli_for_plot, color="red", label="Pauli Rounding (linear)"
    )
    ax5.plot(
        linear_values_max_magic_for_plot, color="blue", label="Magic Rounding (linear)"
    )
    ax5.plot(
        [opt_values_for_plot for i in range(len(compatible_values_max_magic_for_plot))],
        color="black",
        label="opt",
    )
    ax5.set_xlabel("Num of entanglement layers")
    ax5.set_ylabel("Found maxcut value (max)")
    ax5.legend()

    ax6.plot(
        random_values_max_pauli_for_plot, color="red", label="Pauli Rounding (random)"
    )
    ax6.plot(
        random_values_max_magic_for_plot, color="blue", label="Magic Rounding (random)"
    )
    ax6.plot(
        [opt_values_for_plot for i in range(len(compatible_values_max_magic_for_plot))],
        color="black",
        label="opt",
    )
    ax6.set_xlabel("Num of entanglement layers")
    ax6.set_ylabel("Found maxcut value (max)")
    ax6.legend()

    fig.show()


def plot_rounding_results_for_instance(
    compatible_results_dict,
    linear_results_dict,
    random_results_dict,
    deg: int,
    num_nodes: int,
    num_levels: int,
    num_trial: int = 0,
    save_file_name: str = None,
):
    compatible_results = compatible_results_dict[deg][num_nodes][num_trial]
    linear_results = linear_results_dict[deg][num_nodes][num_trial]
    random_results = random_results_dict[deg][num_nodes][num_trial]
    opt = compatible_results[0]["optimum_solution"]
    compatible_plots = []
    linear_plots = []
    random_plots = []
    compatible_magic = []
    linear_magic = []
    random_magic = []
    compatible_pauli = []
    linear_pauli = []
    random_pauli = []
    for compatible, linear, random in zip(
        compatible_results, linear_results, random_results
    ):
        compatible_magic_results = compatible["maxcut_values_magic"]
        compatible_magic.append(max(list(compatible_magic_results.keys())) / opt)
        compatible_pauli.append(compatible["maxcut_value_pauli"] / opt)
        compatible_magic_dist = []
        linear_magic_results = linear["maxcut_values_magic"]
        linear_magic.append(max(list(linear_magic_results.keys())) / opt)
        linear_pauli.append(linear["maxcut_value_pauli"] / opt)
        linear_magic_dist = []
        random_magic_results = random["maxcut_values_magic"]
        random_magic.append(max(list(random_magic_results.keys())) / opt)
        random_pauli.append(random["maxcut_value_pauli"] / opt)
        random_magic_dist = []
        for val, cnt in compatible_magic_results.items():
            compatible_magic_dist.extend([val / opt] * cnt)
        for val, cnt in linear_magic_results.items():
            linear_magic_dist.extend([val / opt] * cnt)
        for val, cnt in random_magic_results.items():
            random_magic_dist.extend([val / opt] * cnt)
        compatible_plots.append(compatible_magic_dist)
        linear_plots.append(linear_magic_dist)
        random_plots.append(random_magic_dist)

    plt.rcParams["font.size"] = 16
    compatible_vp = plt.violinplot(
        compatible_plots,
        [i - 0.125 for i in range(num_levels)],
        widths=0.15,
        showextrema=False,
    )
    for compatible_body in compatible_vp["bodies"]:
        compatible_body.set_facecolor("blue")
        compatible_body.set_linewidth(0)
        compatible_body.set_alpha(0.3)

    linear_vp = plt.violinplot(
        linear_plots,
        [i - 0.025 for i in range(num_levels)],
        widths=0.15,
        showextrema=False,
    )
    for linear_body in linear_vp["bodies"]:
        linear_body.set_facecolor("red")
        linear_body.set_linewidth(0)
        linear_body.set_alpha(0.3)

    random_vp = plt.violinplot(
        random_plots,
        [i + 0.075 for i in range(num_levels)],
        widths=0.15,
        showextrema=False,
    )
    for random_body in random_vp["bodies"]:
        random_body.set_facecolor("green")
        random_body.set_linewidth(0)
        linear_body.set_alpha(0.3)

    plt.plot(
        [-0.3, num_levels - 0.7],
        [1, 1],
        color="black",
        label="opt",
        linestyle="-",
        linewidth=0.5,
    )
    plt.plot(
        [-0.3, num_levels - 0.7],
        [5 / 9, 5 / 9],
        color="purple",
        label="lower bound",
        linestyle="-",
        linewidth=0.5,
    )
    plt.plot(
        [i - 0.125 for i in range(num_levels)],
        compatible_magic,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        color="blue",
        label="compatible (magic)",
        linestyle="dashed",
    )
    plt.plot(
        [i - 0.025 for i in range(num_levels)],
        linear_magic,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        color="red",
        label="linear (magic)",
        linestyle="dashed",
    )
    plt.plot(
        [i + 0.075 for i in range(num_levels)],
        random_magic,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        color="green",
        label="random (magic)",
        linestyle="dashed",
    )
    plt.plot(
        [i - 0.075 for i in range(num_levels)],
        compatible_pauli,
        marker="o",
        markersize=8,
        color="blue",
        label="compatible (pauli)",
        linestyle="dotted",
    )
    plt.plot(
        [i + 0.025 for i in range(num_levels)],
        linear_pauli,
        marker="o",
        markersize=8,
        color="red",
        label="linear (pauli)",
        linestyle="dotted",
    )
    plt.plot(
        [i + 0.125 for i in range(num_levels)],
        random_pauli,
        marker="o",
        markersize=8,
        color="green",
        label="random (pauli)",
        linestyle="dotted",
    )

    plt.xticks([i for i in range(num_levels)])
    plt.xlim(-0.3, num_levels - 0.7)
    plt.ylim(0.35, 1.05)
    plt.ylabel("Approximation ratio")
    plt.xlabel("Number of entanglement layers")
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    if save_file_name is not None:
        plt.savefig(save_file_name, bbox_inches="tight")
    plt.show()


def plot_rounding_results_with_error_bar(
    compatible_results_dict,
    linear_results_dict,
    random_results_dict,
    deg: int,
    num_nodes: int,
    num_levels: int,
    num_trials: int,
    save_file_name: str = None,
):
    compatible_results = compatible_results_dict[deg][num_nodes]
    linear_results = linear_results_dict[deg][num_nodes]
    random_results = random_results_dict[deg][num_nodes]
    opts = [compatible_results[i][0]["optimum_solution"] for i in range(num_trials)]
    compatible_magic_results = []
    linear_magic_results = []
    random_magic_results = []
    compatible_pauli_results = []
    linear_pauli_results = []
    random_pauli_results = []

    # iteration for instances
    for trial, (compatible_result, linear_result, random_result) in enumerate(
        zip(compatible_results, linear_results, random_results)
    ):
        compatible_magic = []
        linear_magic = []
        random_magic = []
        compatible_pauli = []
        linear_pauli = []
        random_pauli = []
        # iteration for num of layers
        for compatible, linear, random in zip(
            compatible_result, linear_result, random_result
        ):
            compatible_magic.append(
                max(list(compatible["maxcut_values_magic"].keys())) / opts[trial]
            )
            compatible_pauli.append(compatible["maxcut_value_pauli"] / opts[trial])
            linear_magic.append(
                max(list(linear["maxcut_values_magic"].keys())) / opts[trial]
            )
            linear_pauli.append(linear["maxcut_value_pauli"] / opts[trial])
            random_magic.append(
                max(list(random["maxcut_values_magic"].keys())) / opts[trial]
            )
            random_pauli.append(random["maxcut_value_pauli"] / opts[trial])
        compatible_magic_results.append(compatible_magic)
        linear_magic_results.append(linear_magic)
        random_magic_results.append(random_magic)
        compatible_pauli_results.append(compatible_pauli)
        linear_pauli_results.append(linear_pauli)
        random_pauli_results.append(random_pauli)

    compatible_magic_results = np.array(compatible_magic_results)
    linear_magic_results = np.array(linear_magic_results)
    random_magic_results = np.array(random_magic_results)
    compatible_pauli_results = np.array(compatible_pauli_results)
    linear_pauli_results = np.array(linear_pauli_results)
    random_pauli_results = np.array(random_pauli_results)
    compatible_magic_err = [
        compatible_magic_results.mean(axis=0) - compatible_magic_results.min(axis=0),
        compatible_magic_results.max(axis=0) - compatible_magic_results.mean(axis=0),
    ]
    linear_magic_err = [
        linear_magic_results.mean(axis=0) - linear_magic_results.min(axis=0),
        linear_magic_results.max(axis=0) - linear_magic_results.mean(axis=0),
    ]
    random_magic_err = [
        random_magic_results.mean(axis=0) - random_magic_results.min(axis=0),
        random_magic_results.max(axis=0) - random_magic_results.mean(axis=0),
    ]
    compatible_pauli_err = [
        compatible_pauli_results.mean(axis=0) - compatible_pauli_results.min(axis=0),
        compatible_pauli_results.max(axis=0) - compatible_pauli_results.mean(axis=0),
    ]
    linear_pauli_err = [
        linear_pauli_results.mean(axis=0) - linear_pauli_results.min(axis=0),
        linear_pauli_results.max(axis=0) - linear_pauli_results.mean(axis=0),
    ]
    random_pauli_err = [
        random_pauli_results.mean(axis=0) - random_pauli_results.min(axis=0),
        random_pauli_results.max(axis=0) - random_pauli_results.mean(axis=0),
    ]

    plt.rcParams["font.size"] = 16
    plt.plot(
        [-0.3, num_levels - 0.7],
        [1, 1],
        color="black",
        label="opt",
        linestyle="-",
        linewidth=0.5,
    )
    plt.plot(
        [-0.3, num_levels - 0.7],
        [5 / 9, 5 / 9],
        color="purple",
        label="lower bound",
        linestyle="-",
        linewidth=0.5,
    )
    plt.errorbar(
        [i - 0.125 for i in range(num_levels)],
        compatible_magic_results.mean(axis=0),
        compatible_magic_err,
        marker="o",
        markerfacecolor="white",
        markeredgecolor="blue",
        markersize=8,
        linestyle="dashed",
        color="blue",
        label="compatible (magic)",
        capsize=8,
    )
    plt.errorbar(
        [i - 0.025 for i in range(num_levels)],
        linear_magic_results.mean(axis=0),
        linear_magic_err,
        marker="o",
        markerfacecolor="white",
        markeredgecolor="red",
        markersize=8,
        linestyle="dashed",
        color="red",
        label="linear (magic)",
        capsize=8,
    )
    plt.errorbar(
        [i + 0.075 for i in range(num_levels)],
        random_magic_results.mean(axis=0),
        random_magic_err,
        marker="o",
        markerfacecolor="white",
        markeredgecolor="green",
        markersize=8,
        linestyle="dashed",
        color="green",
        label="random (magic)",
        capsize=8,
    )
    plt.errorbar(
        [i - 0.075 for i in range(num_levels)],
        compatible_pauli_results.mean(axis=0),
        compatible_pauli_err,
        marker="o",
        markerfacecolor="blue",
        markeredgecolor="blue",
        markersize=8,
        linestyle="dotted",
        color="blue",
        label="compatible (pauli)",
        capsize=8,
    )
    plt.errorbar(
        [i + 0.025 for i in range(num_levels)],
        linear_pauli_results.mean(axis=0),
        linear_pauli_err,
        marker="o",
        markerfacecolor="red",
        markeredgecolor="red",
        markersize=8,
        linestyle="dotted",
        color="red",
        label="linear (pauli)",
        capsize=8,
    )
    plt.errorbar(
        [i + 0.125 for i in range(num_levels)],
        random_pauli_results.mean(axis=0),
        random_pauli_err,
        marker="o",
        markerfacecolor="green",
        markeredgecolor="green",
        markersize=8,
        linestyle="dotted",
        color="green",
        label="random (pauli)",
        capsize=8,
    )

    plt.xticks([i for i in range(num_levels)])
    plt.xlim(-0.3, num_levels - 0.7)
    plt.ylim(0.35, 1.05)
    plt.ylabel("Approximation ratio")
    plt.xlabel("Number of entanglement layers")
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    if save_file_name is not None:
        plt.savefig(save_file_name, bbox_inches="tight")
    plt.show()
