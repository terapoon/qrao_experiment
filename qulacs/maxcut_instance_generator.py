from ast import NodeTransformer
from re import A
import numpy as np
import networkx as nx
from networkx import Graph
from docplex.mp.model import Model

"""
Functions to generate docplex model of maxcut problem.
"""


def _generate_model_from_graph(
    edges: np.ndarray,
    num_nodes: int,
) -> Model:
    model = Model("maxcut")
    nodes = list(range(num_nodes))
    var = [model.binary_var(name="x" + str(i)) for i in nodes]
    model.maximize(
        model.sum(
            0.5 * edges[i, j] * (var[i] + var[j] - 2 * var[i] * var[j])
            for i in nodes
            for j in nodes
        )
    )

    return model


def regular_graph(
    num_nodes: int,
    degree: int,
    seed: int = 0,
    min_weight: int = 1,
    max_weight: int = 1,
    draw: bool = False,
) -> Model:
    assert max_weight >= min_weight
    seed = np.random.RandomState(seed)
    graph = nx.random_regular_graph(d=degree, n=num_nodes, seed=seed)
    edges = np.zeros((num_nodes, num_nodes))
    for i, j in graph.edges():
        if min_weight == max_weight:
            weight = 1
        else:
            weight = seed.randint(min_weight, max_weight)
        edges[i, j] = edges[j, i] = weight

    if draw:
        nx.draw(graph, with_labels=True, font_color="whitesmoke")

    return _generate_model_from_graph(edges, num_nodes)
