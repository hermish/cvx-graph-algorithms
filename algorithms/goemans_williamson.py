import numpy as np
import cvxpy as cvx
import networkx as nx


def goemans_williamson_weighted(graph):
    """
    Runs the Goemans-Williamson randomized 0.87856-approximation algorithm for
    MAX-CUT on the graph instance, returning the cut.

    :param graph: (NetworkX graph) An undirected graph with no self-loops or
        multiple edges. The graph can either be weighted or unweighted, where
        each edge present is assigned an equal weight of 1.
    :return: (tuple of two sets) The cut returned by the algorithm as two
        sets, where each corresponds to a different side of the cut. Together,
        both lists contain all vertices in the graph, and each vertex is in
        exactly one of the two lists.
    """
    adjacency = nx.linalg.adjacency_matrix(graph)
    adjacency = adjacency.toarray()
    solution = _solve_cut_vector_program(adjacency)
    sides = _recover_cut(solution)

    nodes = list(graph.nodes())
    left = {vertex for side, vertex in zip(sides, nodes) if side < 0}
    right = {vertex for side, vertex in zip(sides, nodes) if side >= 0}
    return left, right


def _solve_cut_vector_program(adjacency):
    """
    :param adjacency: (matrix) A square matrix representing the adjacency matrix
        of an undirected graph with no self-loops. Therefore, the matrix must be
        symmetric with zeros along its diagonal.
    :return: (matrix) A matrix whose columns represents the vectors assigned to
        each vertex to maximize the MAX-CUT semi-definite program (SDP)
        objective.
    """
    size = len(adjacency)
    ones_matrix = np.ones((size, size))
    products = cvx.Variable((size, size), PSD=True)
    cut_size = 0.5 * cvx.sum(cvx.multiply(adjacency, ones_matrix - products))

    objective = cvx.Maximize(cut_size)
    constraints = [cvx.diag(products) == 1]
    problem = cvx.Problem(objective, constraints)
    problem.solve()

    eigenvalues, eigenvectors = np.linalg.eigh(products.value)
    eigenvalues = np.maximum(eigenvalues, 0)
    diagonal_root = np.diag(np.sqrt(eigenvalues))
    assignment = diagonal_root @ eigenvectors.T
    return assignment


def _recover_cut(solution):
    """
    :param solution: (matrix) A vector assignment of vertices, where each
        SOLUTION[:,i] corresponds to the vector associated with vertex i.
    :return: (array) The cut from probabilistically rounding the
        solution, where -1 signifies left, +1 right, and 0 (which occurs almost
        surely never) either.
    """
    size = len(solution)
    partition = np.random.normal(size=size)
    projections = solution.T @ partition

    sides = np.sign(projections)
    return sides
