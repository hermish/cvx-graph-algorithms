# CVX Graph Algorithms

## Introduction

> Modern convex optimization-based graph algorithms.

Convex optimization presents an exciting new direction in designing exact and
approximate graph algorithms. However, these algorithms are often overlooked in
practice due to limitations in solving large convex programs quickly.
Convex optimization-based graph algorithms nonetheless achieve impressive 
theoretical performance, and often provide a beautiful geometric interpretation.
This package implements some of these algorithms and provides corresponding 
graph generators to test performance---hopefully highlighting how simple, 
elegant and effective these can be for many real-world problems.

## Details

In this package, we provide implementations of the following algorithms. Note
featured convex optimization-based algorithms are in bold and references are
provided when available.

This package also provides functions to generate graphs drawn from the planted
independent set distribution and stochastic block model.

1. Maximum Cut Problem
    1. **Goemans-Williamson MAX-CUT Algorithm** [1]
    2. Random MAX-CUT Algorithm
    3. Greedy MAX-CUT Algorithm
2. Independent Set Algorithm
    1. **Crude SDP-based Independent Set** [2]
    2. Greedy Independent Set Algorithm
    3. Spectral Algorithm for Independent Set

## Install and Usage

You can install this directly from the Python Package Index (PyPI).

```
pip install cvxgraphalgs
```

Below, we show how to run the Goemans-Williamson MAX-CUT Algorithm on a graph
drawn from the stochastic block model distribution. For more examples, explore 
the jupyter notebooks available with the package documentation available
[here](https://github.com/hermish/cvx-graph-algorithms/).

```
>>> import cvxgraphalgs as cvxgr
>>> graph, _ = cvxgr.generators.bernoulli_planted_independent(
...     size=50, independent_size=15, probability=0.5
... )
>>> recovered = cvxgr.algorithms.crude_sdp_independent_set(graph)
>>> len(recovered)
15
```

## References
[1]: Goemans, Michel X., and David P. Williamson. "Improved approximation 
algorithms for maximum cut and satisfiability problems using semidefinite 
programming." *Journal of the ACM (JACM)* 42, no. 6 (1995): 1115-1145.

[2]: McKenzie, Theo, Hermish Mehta, and Luca Trevisan. "A New Algorithm for the
Robust Semi-random Independent Set Problem." *arXiv:1808.03633* (2018).
