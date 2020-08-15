#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from random import randint

import igraph as ig
import numpy as np

MAX_DEG = 50_000


def _powerlaw(n: int, kmin: int, alpha: float) -> np.ndarray:
    np.random.seed(42)
    return kmin * (1 - np.random.random(n))**(-1 / (np.abs(alpha) - 1))


def configuration_model(v: int, alpha: float) -> ig.Graph:
    logger = logging.getLogger("Configuration Model")
    degrees = _powerlaw(v, 10, alpha)
    degrees = degrees[degrees > 0]
    degrees = degrees[degrees <= MAX_DEG]

    degrees = np.vstack([degrees, np.arange(degrees.shape[0])]).T

    g = ig.Graph(v)
    edges = []

    while degrees.shape[0] >= 2:
        a = randint(0, degrees.shape[0] - 1)
        b = randint(0, degrees.shape[0] - 1)

        while a == b:
            b = randint(0, degrees.shape[0] - 1)

        edges.append((int(degrees[a, 1]), int(degrees[b, 1])))
        degrees[a, 0] -= 1
        degrees[b, 0] -= 1

        degrees = degrees[degrees[:, 0] > 0, :]

    g.add_edges(edges)
    logger.info("Create CFGM graph on {:d} vertexes with {:d} edges.".format(
        g.vcount(), g.ecount()))

    return g
