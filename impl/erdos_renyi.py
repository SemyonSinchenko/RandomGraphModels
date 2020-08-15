#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from random import random

import igraph as ig


def erdos_renyi(v: int, e: int) -> ig.Graph:
    g = ig.Graph(n=v)
    logger = logging.getLogger("Erdos-Renyi")

    p_edge = 2 * e / (v * (v - 1))
    edges = []

    for src in g.vs.indices:
        for dst in g.vs.indices:
            if src > dst and random() <= p_edge:
                edges.append((src, dst))

    g.add_edges(edges)
    logger.info("Create ER graph on {:d} vertexes with {:d} edges.".format(
        g.vcount(), g.ecount()))

    return g
