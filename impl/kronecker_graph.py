#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import igraph as ig
import numpy as np
from scipy import sparse


def kronecker_graph(v: int, density: float) -> ig.Graph:
    logger = logging.getLogger("Kronecker Graph")
    num_mult = int(np.floor(np.log10(v)))
    initial_mat = np.random.random(100).reshape((10, 10))
    np.fill_diagonal(initial_mat, 1.0)

    initial_mat[initial_mat <= density] = 1.0
    initial_mat[initial_mat < 1.0] = 0.0

    initial_mat = sparse.csr_matrix(initial_mat)

    adjacency = sparse.csr_matrix(initial_mat)

    for _ in range(num_mult - 1):
        adjacency = sparse.kron(adjacency, initial_mat)

    nonzero = adjacency.nonzero()
    edge_list = zip(nonzero[0], nonzero[1])

    g = ig.Graph(n=adjacency.shape[0], edges=edge_list)
    logger.info("Create KGM graph on {:d} vertexes with {:d} edges.".format(
        g.vcount(), g.ecount()))

    return g
