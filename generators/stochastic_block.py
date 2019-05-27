import random
import itertools
import networkx as nx


def stochastic_block_on_cut(cut, within, between):
    """
    Returns a graph drawn from the Stochastic Block Model, on the vertices
    in CUT. Every edge between pairs of vertices in CUT.LEFT and CUT.RIGHT is
    present independently with probability WITHIN; edges between sides are
    similarly present independently with probability BETWEEN.

    :param cut: (structures.cut.Cut) A cut which represents the vertices in
        each of the two communities. Traditionally, the size of each side is
        exactly half the total number of vertices in the graph, denoted n.
    :param within: (float) The probability an edge exists between two vertices
        in the same community, denoted p. Must be between 0 and 1 inclusive.
    :param between: (float) The probability of each edge between two vertices
        in different communities, denoted q. Must be between 0 and 1 inclusive.
    :return: (nx.classes.graph.Graph) A graph drawn according to the Stochastic
        Block Model over the cut.
    """
    graph = nx.Graph()
    graph.add_nodes_from(cut.vertices)

    for side in (cut.left, cut.right):
        for start, end in itertools.combinations(side, 2):
            if random.random() < within:
                graph.add_edge(start, end)

    for start in cut.left:
        for end in cut.right:
            if random.random() < between:
                graph.add_edge(start, end)

    return graph
