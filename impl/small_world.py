#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from random import choice, random

import igraph as ig


def small_world(v: int, e: int, beta: float) -> ig.Graph:
    g = ig.Graph(v)
    logger = logging.getLogger("Small World")

    num_edges = e // v
    edges = []

    for src in g.vs.indices:
        k = num_edges
        d = 1

        while k > 0:
            r_ = src + d
            if r_ >= v:
                r_ -= v

            if random() <= beta:
                while True:
                    new_d = choice(g.vs.indices)
                    if new_d != src and new_d != r_:
                        r_ = new_d
                        break

            l_ = src - d
            if l_ < 0:
                l_ += v

            if random() <= beta:
                while True:
                    new_d = choice(g.vs.indices)
                    if new_d != src and new_d != l_:
                        l_ = new_d
                        break

            k -= 2
            edges.append((src, r_))
            edges.append((src, l_))
            d += 1

    g.add_edges(edges)
    logger.info("Create SW graph on {:d} vertexes with {:d} edges.".format(
        g.vcount(), g.ecount()))

    return g
