import numpy as np
import cvxpy as cp
import networkx as nx
from scipy import linalg


def greedy_independent_set(graph):
    """
    :param graph: (nx.classes.graph.Graph) An undirected graph with no
        self-loops or multiple edges. The graph can either be weighted or
        unweighted, although the problem only differentiates between zero and
        non-zero weight edges.
    :return: (set) The independent set the algorithm outputs, represented as
        a set of vertices.
    """
    independent = set()
    for vertex in graph.nodes:
        if not any(graph.has_edge(vertex, element) for element in independent):
            independent.add(vertex)
    return independent


def crude_sdp_independent_set(graph):
    """
    :param graph: (nx.classes.graph.Graph) An undirected graph with no
        self-loops or multiple edges. The graph can either be weighted or
        unweighted, although the problem only differentiates between zero and
        non-zero weight edges.
    :return: (set) The independent set the algorithm outputs, represented as
        a set of vertices.
    """
    solution = _solve_vector_program(graph)
    labels = list(graph.nodes)
    candidates = _get_vector_clusters(labels, solution, 1.0)
    best = max(candidates, key=lambda cluster: len(cluster))
    return best


def _solve_vector_program(graph):
    """
    :param graph: (nx.classes.graph.Graph) An undirected graph with no
        self-loops or multiple edges. The graph can either be weighted or
        unweighted, although the problem only differentiates between zero and
        non-zero weight edges.
    :return: (np.ndarray) A matrix whose columns represents the vectors assigned
        to each vertex to maximize the crude semi-definite program (C-SDP)
        objective.
    """
    size = len(graph)
    products = cp.Variable((size, size), PSD=True)

    objective_matrix = size * np.eye(size) - np.ones((size, size))
    objective = cp.Minimize(cp.trace(objective_matrix @ products))

    adjacency = nx.linalg.adjacency_matrix(graph)
    adjacency = adjacency.toarray()
    constraints = [
        cp.diag(products) == 1,
        products >= 0,
        cp.multiply(products, adjacency) == 0
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()
    assert problem.status == 'optimal'

    eigenvalues, eigenvectors = np.linalg.eigh(products.value)
    eigenvalues = np.maximum(eigenvalues, 0)
    diagonal_root = np.diag(np.sqrt(eigenvalues))
    assignment = diagonal_root @ eigenvectors.T
    return assignment


def _get_vector_clusters(labels, vectors, threshold):
    """
    :param labels: (list) A list of labels.
    :param vectors: (np.ndarray) A matrix whose columns are the vectors
        corresponding to each label. Therefore, the label LABELS[i] references
        the vector VECTORS[:,i]; both lengths must be exactly the same, so
        len(VECTORS.T) == len(LABELS).
    :param threshold: (float | int) The closeness threshold.
    :return: (list) Return a list of sets. For each vector, this list includes a
        set which contains the labels of all vectors within a THRESHOLD-ball
        of the original. The list will contain exactly len(LABELS) entries,
        in the same order as LABELS.
    """
    total = len(labels)
    clusters = []

    for current in range(total):
        output = set()
        for other in range(total):
            if np.linalg.norm(vectors[:,current] - vectors[:,other]) <= threshold:
                output.add(labels[other])
        clusters.append(output)
    return clusters


def planted_spectral_algorithm(graph):
    """
    :param graph: (nx.classes.graph.Graph) An undirected graph with no
        self-loops or multiple edges. The graph can either be weighted or
        unweighted, although the problem only differentiates between zero and
        non-zero weight edges.
    :return: (set) The independent set the algorithm outputs, represented as
        a set of vertices.
    """
    size = len(graph)
    labels = list(graph.nodes)
    adjacency = nx.linalg.adjacency_matrix(graph)
    adjacency = adjacency.toarray()
    co_adjacency = 1 - adjacency

    ones_matrix = np.ones((size, size))
    normalized = co_adjacency - 0.5 * ones_matrix
    _, eigenvector = linalg.eigh(normalized, eigvals=(size - 1, size - 1))

    indices = list(range(size))
    indices.sort(key=lambda num: eigenvector[num], reverse=True)

    output = set()
    for index in indices:
        vertex = labels[index]
        if not any(graph.has_edge(vertex, element) for element in output):
            output.add(vertex)
    return output
