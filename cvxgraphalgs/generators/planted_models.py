import random
import itertools
import numpy as np
import networkx as nx


def bernoulli_planted_independent(size, independent_size, probability):
    """
    :param size: (int) The number of total vertices in the graph.
    :param independent_size: (int) The size of the planted independent set.
    :param probability: (float) The probability each "allowed" edge exists;
        must be between 0 and 1 inclusive.
    :return: (nx.classes.graph.Graph, set) A tuple consisting of a graph
        drawn from the planted independent set distribution on vertices
        {0, ..., SIZE - 1} and corresponding independent set, represented as
        a set of vertex labels. No edges exist between any vertices in the
        independent set. All other edges are present independently with
        probability PROBABILITY.
    """
    vertices = list(range(size))
    isolated = np.random.choice(size, independent_size, replace=False)
    isolated = set(isolated)

    graph = nx.Graph()
    graph.add_nodes_from(vertices)
    for start, end in itertools.combinations(graph.nodes, 2):
        if start not in isolated or end not in isolated:
            if random.random() < probability:
                graph.add_edge(start, end)
    return graph, isolated
