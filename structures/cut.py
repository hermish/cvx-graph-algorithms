import itertools
import networkx as nx

WEIGHT = 'weight'


class Cut:
    """
    A data structure representing a cut in a graph; formally, for a graph G =
    (V, E), this is a partition of the vertices (S, V - S). This class stores
    a cut as a pair or sets, corresponding to the left and right side of the
    cut respectively.
    """

    def __init__(self, left, right):
        """
        :param left: (set) Vertices on the left side of the cut.
        :param right: (set) Vertices on the right side of the cut.
        """
        self.left = left
        self.right = right
        self.vertices = list(itertools.chain(left, right))

    def validate_cut(self, graph):
        """
        :param graph: (nx.classes.graph.Graph) A NetworkX graph.
        :return: (NoneType) Ensures the left and right compose a valid cut of
            the graph so each vertex in the graph is in exactly one of these two
            sets.
        """
        size = len(graph)
        left_size, right_size = len(self.left), len(self.right)
        assert left_size + right_size == size

        for vertex in graph.nodes():
            assert vertex in self.left or vertex in self.right

    def evaluate_cut_size(self, graph):
        """
        :param graph: (nx.classes.graph.Graph) A NetworkX graph.
        :return: (float | int) Returns the size of the cut, or more precisely
            the sum of the weights of the edges between right and left. When the
            graph is unweighted, edge weights are taken to be 1, so this counts
            the total edges between sides of the cut.
        """
        self.validate_cut(graph)
        graph_weighted = nx.is_weighted(graph)
        total, weight = 0, 1

        for edge in graph.edges():
            start, end = edge
            forward_order = start in self.left and end in self.right
            reverse_order = start in self.right and end in self.left
            if forward_order or reverse_order:
                if graph_weighted:
                    weight = graph[start][end][WEIGHT]
                total += weight
        return total
