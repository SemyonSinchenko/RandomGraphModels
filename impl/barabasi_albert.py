#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from random import choice

import igraph as ig


def barabasi_albert(v: int, e: int) -> ig.Graph:
    g = ig.Graph()
    logger = logging.getLogger("Barabasi-Albert")
    probs = []

    v0 = g.add_vertex().index
    edges = []
    edges.append((v0, v0))
    probs.append(v0)

    num_edges_by_v = e // v

    for _ in range(v - 1):
        n = g.add_vertex().index
        edges.append((n, n))
        probs.append(n)

        for _ in range(num_edges_by_v):
            dst = choice(probs)
            edges.append((n, dst))
            probs.append(n)
            probs.append(dst)

    g.add_edges(edges)
    logger.info(
        "Create BA graph on {:d} vertexes with {:d} edges.".format(
            g.vcount(), g.ecount()
        )
    )

    return g
