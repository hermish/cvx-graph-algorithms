import numpy as np
import networkx as nx

WEIGHT = 'weight'


def greedy_max_cut(graph):
    """
    Runs a greedy MAX-CUT approximation algorithm to partition the vertices of
    the graph into two sets. This greedy approach delivers an approximation
    ratio of 0.5.

    :param graph: (NetworkX graph) An undirected graph with no self-loops or
        multiple edges. The graph can either be weighted or unweighted, where
        each edge present is assigned an equal weight of 1.
    :return: (tuple of two sets) The cut returned by the algorithm as two
        sets, where each corresponds to a different side of the cut. Together,
        both lists contain all vertices in the graph, and each vertex is in
        exactly one of the two lists.
    """
    left, right = set(), set()
    for vertex in graph.nodes():
        left_neighbors = sum(adj in left for adj in graph.neighbors(vertex))
        right_neighbors = sum(adj in right for adj in graph.neighbors(vertex))
        if left_neighbors < right_neighbors:
            left.add(vertex)
        else:
            right.add(vertex)
    return left, right


def random_cut(graph, probability):
    """
    :param graph: (graph) A NetworkX graph.
    :param probability: (float) A number in [0, 1] which gives the probability
        each vertex lies on the right side of the cut.
    :return: (tuple of two sets) The random cut which results from randomly
        assigning vertices to either side independently at random according
        to the probability given above.
    """
    size = len(graph)
    sides = np.random.binomial(1, probability, size)

    nodes = list(graph.nodes())
    left = {vertex for side, vertex in zip(sides, nodes) if side == 0}
    right = {vertex for side, vertex in zip(sides, nodes) if side == 1}
    return left, right


def evaluate_cut(graph, left, right):
    """
    :param graph: (graph) A NetworkX graph.
    :param left: (set) Vertices on the left side of the cut.
    :param right: (set) Vertices on the right side of the cut.
    :return: (number) Returns the size of the cut, or more precisely, the sum
        of the weights of the edges between right and left. When the graph is
        unweighted, edge weights are taken to be 1, so this counts the total
        edges between sides of the cute.
    """
    _validate_cut(graph, left, right)
    graph_weighted = nx.is_weighted(graph)
    total, weight = 0, 1

    for edge in graph.edges():
        start, end = edge
        forward_order = start in left and end in right
        reverse_order = start in right and end in left
        if forward_order or reverse_order:
            if graph_weighted:
                weight = graph[start][end][WEIGHT]
            total += weight
    return total


def _validate_cut(graph, left, right):
    """
    :param graph: (graph) A NetworkX graph.
    :param left: (set) Vertices on the left side of the cut.
    :param right: (set) Vertices on the right side of the cut.
    :return: (none) Ensures the left and right compose a valid cut of the graph
        so each vertex in the graph is in exactly one of these two sets.
    """
    size = len(graph)
    left_size, right_size = len(left), len(right)
    assert left_size + right_size == size

    for vertex in graph.nodes():
        assert vertex in left or vertex in right
