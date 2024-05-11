import numpy as np
import networkx as nx
import string
import pickle

from itertools import chain, product, islice, count

from numpy.random import default_rng
from pgmpy import models
from pgmpy.factors.continuous import LinearGaussianCPD


def sample_erdos_renyi_graph(
    num_variables,
    p=None,
    num_edges=None,
    nodes=None,
    create_using=models.BayesianNetwork,
    rng=default_rng(),
):
    if p is None:
        if num_edges is None:
            raise ValueError("One of p or num_edges must be specified.")
        p = num_edges / ((num_variables * (num_variables - 1)) / 2.0)

    if nodes is None:
        uppercase = string.ascii_uppercase
        iterator = chain.from_iterable(product(uppercase, repeat=r) for r in count(1))
        nodes = ["".join(letters) for letters in islice(iterator, num_variables)]

    adjacency = rng.binomial(1, p=p, size=(num_variables, num_variables))
    adjacency = np.tril(adjacency, k=-1)  # Only keep the lower triangular part

    # Permute the rows and columns
    perm = rng.permutation(num_variables)
    adjacency = adjacency[perm, :]
    adjacency = adjacency[:, perm]

    graph = nx.from_numpy_array(adjacency, create_using=create_using)
    mapping = dict(enumerate(nodes))
    nx.relabel_nodes(graph, mapping=mapping, copy=False)

    return graph


def sample_erdos_renyi_linear_gaussian(
    num_variables,
    p=None,
    num_edges=None,
    nodes=None,
    loc_edges=0.0,
    scale_edges=1.0,
    obs_noise=0.1,
    rng=default_rng(),
):

    with open(
        "../model/model1.pkl",
        "rb",
    ) as file:
        graph = pickle.load(file)
        # graph.add_node("SEVERITY")
        graph.remove_node("TOT_INJ")
        graph.add_node("ACCTYPE")
        nodes = list(graph.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                graph.add_edge(nodes[i], nodes[j])
                graph.add_edge(nodes[j], nodes[i])
    # # Create graph structure
    # graph = sample_erdos_renyi_graph(
    #     num_variables,
    #     p=p,
    #     num_edges=num_edges,
    #     nodes=nodes,
    #     create_using=models.LinearGaussianBayesianNetwork,
    #     rng=rng
    # )

    # # Create the model parameters
    # factors = []
    # for node in graph.nodes:
    #     parents = list(graph.predecessors(node))

    #     # Sample random parameters (from Normal distribution)
    #     theta = rng.normal(loc_edges, scale_edges, size=(len(parents) + 1,))
    #     theta[0] = 0.  # There is no bias term

    #     # Create factor
    #     factor = LinearGaussianCPD(node, theta, obs_noise, parents)
    #     factors.append(factor)

    # graph.add_cpds(*factors)
    return graph
