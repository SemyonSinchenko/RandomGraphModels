#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from random import choice, random

import igraph as ig


def copy_model(v: int, alpha: float) -> ig.Graph:
    logger = logging.getLogger("Duplication-Divergence")
    g = ig.Graph(2)

    # Create starter
    g.add_edge(0, 1)

    # Grow net
    while g.vcount() < v:
        zero_deg = True
        rnd_node = choice(g.vs)

        n = g.add_vertex()
        edges = []

        for nbr in rnd_node.neighbors():
            if random() <= alpha:
                zero_deg = False
                edges.append((n.index, nbr.index))

        if zero_deg:
            g.delete_vertices(n.index)
        else:
            g.add_edges(edges)
    logger.info(
        "Create DD graph on {:d} vertexes with {:d} edges.".format(
            g.vcount(), g.ecount()
        )
    )

    return g
