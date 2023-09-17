import numpy as np
import networkx as nx
from networkx import Graph
from docplex.mp.model import Model

"""
Functions to generate docplex model of maxcut problem.
"""


def _generate_model_from_graph(
    graph: Graph,
    edge_weights: np.ndarray,
    num_nodes: int,
) -> Model:
    model = Model("maxcut")
    nodes = list(range(num_nodes))
    edges = graph.edges()
    var = [model.binary_var(name="x" + str(i)) for i in nodes]
    model.maximize(
        model.sum(
            edge_weights[i, j] * (var[i] + var[j] - 2 * var[i] * var[j])
            for i, j in edges
        )
    )

    return model


def regular_graph(
    num_nodes: int,
    degree: int,
    seed: int = 0,
    weighted: bool = False,
    max_weight: int = 1,
    draw: bool = False,
) -> Model:
    seed = np.random.RandomState(seed)
    graph = nx.random_regular_graph(d=degree, n=num_nodes, seed=seed)
    edge_weights = np.zeros((num_nodes, num_nodes))
    for i, j in graph.edges():
        if weighted and (max_weight <= 1):
            # assign 1 or -1 randomly.
            weight = seed.choice((-1, 1))
        elif weighted and (max_weight > 1):
            # assign weight from [1, max_weight] randomly
            weight = seed.randint(1, max_weight + 1)
        else:
            weight = 1
        edge_weights[i, j] = edge_weights[j, i] = weight

    if draw:
        nx.draw(graph, with_labels=True, font_color="whitesmoke")

    return graph, _generate_model_from_graph(graph, edge_weights, num_nodes)
