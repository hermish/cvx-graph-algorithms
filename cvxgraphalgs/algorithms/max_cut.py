import numpy as np
import cvxpy as cp
import networkx as nx

from cvxgraphalgs.structures.cut import Cut


def greedy_max_cut(graph):
    """
    Runs a greedy MAX-CUT approximation algorithm to partition the vertices of
    the graph into two sets. This greedy approach delivers an approximation
    ratio of 0.5.

    :param graph: (nx.classes.graph.Graph) An undirected graph with no
        self-loops or multiple edges. The graph can either be weighted or
        unweighted, where each edge present is assigned an equal weight of 1.
    :return: (structures.cut.Cut) The cut returned by the algorithm as two
        sets, where each corresponds to a different side of the cut. Together,
        both sets contain all vertices in the graph, and each vertex is in
        exactly one of the two sets.
    """
    cut = Cut(set(), set())
    for vertex in graph.nodes:
        l_neighbors = sum((adj in cut.left) for adj in graph.neighbors(vertex))
        r_neighbors = sum((adj in cut.right) for adj in graph.neighbors(vertex))
        if l_neighbors < r_neighbors:
            cut.left.add(vertex)
        else:
            cut.right.add(vertex)
    return cut


def random_cut(graph, probability):
    """
    :param graph: (nx.classes.graph.Graph) A NetworkX graph.
    :param probability: (float) A number in [0, 1] which gives the probability
        each vertex lies on the right side of the cut.
    :return: (structures.cut.Cut) The random cut which results from randomly
        assigning vertices to either side independently at random according
        to the probability given above.
    """
    size = len(graph)
    sides = np.random.binomial(1, probability, size)

    nodes = list(graph.nodes)
    left = {vertex for side, vertex in zip(sides, nodes) if side == 0}
    right = {vertex for side, vertex in zip(sides, nodes) if side == 1}
    return Cut(left, right)


def goemans_williamson_weighted(graph):
    """
    Runs the Goemans-Williamson randomized 0.87856-approximation algorithm for
    MAX-CUT on the graph instance, returning the cut.

    :param graph: (nx.classes.graph.Graph) An undirected graph with no
        self-loops or multiple edges. The graph can either be weighted or
        unweighted, where each edge present is assigned an equal weight of 1.
    :return: (structures.cut.Cut) The cut returned by the algorithm as two
        sets, where each corresponds to a different side of the cut. Together,
        both sets contain all vertices in the graph, and each vertex is in
        exactly one of the two sets.
    """
    adjacency = nx.linalg.adjacency_matrix(graph)
    adjacency = adjacency.toarray()
    solution = _solve_cut_vector_program(adjacency)
    sides = _recover_cut(solution)

    nodes = list(graph.nodes)
    left = {vertex for side, vertex in zip(sides, nodes) if side < 0}
    right = {vertex for side, vertex in zip(sides, nodes) if side >= 0}
    return Cut(left, right)


def _solve_cut_vector_program(adjacency):
    """
    :param adjacency: (np.ndarray) A square matrix representing the adjacency
        matrix of an undirected graph with no self-loops. Therefore, the matrix
        must be symmetric with zeros along its diagonal.
    :return: (np.ndarray) A matrix whose columns represents the vectors assigned
        to each vertex to maximize the MAX-CUT semi-definite program (SDP)
        objective.
    """
    size = len(adjacency)
    ones_matrix = np.ones((size, size))
    products = cp.Variable((size, size), PSD=True)
    cut_size = 0.5 * cp.sum(cp.multiply(adjacency, ones_matrix - products))

    objective = cp.Maximize(cut_size)
    constraints = [cp.diag(products) == 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    eigenvalues, eigenvectors = np.linalg.eigh(products.value)
    eigenvalues = np.maximum(eigenvalues, 0)
    diagonal_root = np.diag(np.sqrt(eigenvalues))
    assignment = diagonal_root @ eigenvectors.T
    return assignment


def _recover_cut(solution):
    """
    :param solution: (np.ndarray) A vector assignment of vertices, where each
        SOLUTION[:,i] corresponds to the vector associated with vertex i.
    :return: (np.ndarray) The cut from probabilistically rounding the
        solution, where -1 signifies left, +1 right, and 0 (which occurs almost
        surely never) either.
    """
    size = len(solution)
    partition = np.random.normal(size=size)
    projections = solution.T @ partition

    sides = np.sign(projections)
    return sides
