#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import igraph as ig
import numpy as np

from impl.barabasi_albert import barabasi_albert


def linear_pref_attach(v: int, e: int, alpha: float) -> ig.Graph:
    num_edges = e // v
    logger = logging.getLogger("Linear PA")

    g = barabasi_albert(100, 5000)
    probs = np.zeros(v)

    for i, deg in enumerate(g.vs.degree()):
        probs[i] = deg

    edges = []
    g.add_vertices(v - 100)

    for src in range(100, v - 100):
        probs[src] = 1
        edges.append((src, src))

        for _ in range(num_edges):
            dst = np.argmax(
                np.random.multinomial(
                    1,
                    pvals=(probs + alpha * (probs > 0.0))
                    / (probs + alpha * (probs > 0.0)).sum(),
                    size=1,
                )
            )

            probs[src] += 1
            probs[dst] += 1
            edges.append((src, dst))

    g.add_edges(edges)
    logger.info(
        "Create LPA graph on {:d} vertexes with {:d} edges.".format(
            g.vcount(), g.ecount()
        )
    )

    return g
