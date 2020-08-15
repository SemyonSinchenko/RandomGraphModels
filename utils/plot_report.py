#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
import logging
from pathlib import Path
from random import choice, random

import igraph as ig
import matplotlib.pylab as plt
import numpy as np


def plot_all_distributions(g: ig.Graph, path_prefix: Path) -> None:
    path_prefix.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("Plotter")

    logger.info("\tEstimate degree distribution...")
    _plot_degree_distribution(g, path_prefix)
    logger.info("\tDone.")

    logger.info("\tEstimate clsutering...")
    _plot_clustering_coeff(g, path_prefix)
    logger.info("\tDone.")

    logger.info("\tEstimate paths...")
    _plot_shortest_paths(g, path_prefix)
    logger.info("\tDone.")

    logger.info("\tEstimate correlations...")
    _plot_degrees_correlations(g, path_prefix)
    logger.info("\tDone.")


def _plot_degree_distribution(g: ig.Graph, path_prefix: Path) -> None:
    file_path = path_prefix.joinpath("degrees.png")

    # Count degrees
    degrees = g.vs.degree()
    degree_distr = Counter(degrees)

    log_deg = np.zeros(len(degree_distr), dtype=np.float32)
    log_cnt = np.zeros(len(degree_distr), dtype=np.float32)

    for i, kv in enumerate(degree_distr.items()):
        log_deg[i] = np.log(kv[0]) if kv[0] > 0 else 0
        log_cnt[i] = np.log(kv[1])

    # Fit linear regression
    solution = np.polyfit(log_deg[log_cnt > 1.5], log_cnt[log_cnt > 1.5], deg=1)

    # Plot distributions
    plt.style.use("ggplot")
    f: plt.Figure = plt.Figure(figsize=(8, 5), dpi=150)
    ax: plt.Axes = f.add_subplot()
    ax.plot(log_deg, log_cnt, ".", label="Real Degrees")
    ax.plot(
        log_deg,
        log_deg * solution[0] + solution[1],
        "-",
        label="Linear fit: {:.4f}x + {:.2f}".format(solution[0], solution[1]),
    )
    ax.legend()
    ax.set_xlabel("Log degree")
    ax.set_ylabel("Log count degrees")
    ax.set_ylim(bottom=np.min(log_cnt) - 1e-5)

    f.savefig(str(file_path.absolute()))
    plt.close(f)


def _plot_clustering_coeff(g: ig.Graph, path_prefix: Path) -> None:
    file_path = path_prefix.joinpath("clustcoeff.png")
    g = g.copy().simplify()

    # Get clustering coefficient based on 50% of vertices
    subset = [vidx for vidx in g.vs.indices if random() <= 0.5]
    lcc = g.transitivity_local_undirected(vertices=subset, mode="zero")
    degrees = g.degree(vertices=subset)
    avg_clustering_coeff = np.mean(lcc)

    # Plot
    plt.style.use("ggplot")
    f: plt.Figure = plt.Figure(figsize=(8, 5), dpi=150)
    ax: plt.Axes = f.add_subplot()
    ax.plot(
        degrees, lcc, ".", label="ClustCoeff. Avg = {:.2e}".format(avg_clustering_coeff)
    )
    ax.legend()
    ax.set_xlabel("Degree")
    ax.set_ylabel("Avg Clustering of Degree")

    f.savefig(str(file_path.absolute()))
    plt.close(f)


def _plot_shortest_paths(g: ig.Graph, path_prefix: Path) -> None:
    file_path = path_prefix.joinpath("shortest_paths.png")

    # Estimate diameter and paths
    paths = g.get_all_shortest_paths(0, mode=ig.ALL)
    path_lens = list(map(len, paths))

    diam = max(path_lens)
    eff_diam = np.percentile(path_lens, 90)
    cnts = Counter(path_lens)

    # Plot
    plt.style.use("ggplot")
    f: plt.Figure = plt.Figure(figsize=(8, 5), dpi=150)
    ax: plt.Axes = f.add_subplot()
    ax.plot(
        cnts.keys(),
        cnts.values(),
        ".",
        label="Path.Len.Distr.\nDiam.: {:d}\nEff.Diam.: {:.2f}".format(diam, eff_diam),
    )
    ax.legend()
    ax.set_xlabel("Path. length")
    ax.set_ylabel("Count paths")

    f.savefig(str(file_path.absolute()))
    plt.close(f)


def _plot_degrees_correlations(g: ig.Graph, path_prefix: Path) -> None:
    file_path = path_prefix.joinpath("degree_correlations.png")

    # Compute correlations
    num_nodes = int(g.vcount() * 0.35)
    visited = set()
    corrs = np.zeros((num_nodes, 2), dtype=np.float32)

    # Compute coefficient
    assortativity = g.assortativity_degree(directed=False)

    for i in range(num_nodes):
        while True:
            rnd_id = choice(g.vs)
            if rnd_id.index not in visited:
                visited.add(rnd_id.index)
                break

        deg = rnd_id.degree()

        for nbh in rnd_id.neighbors():
            corrs[i, 1] += nbh.degree() / deg

        corrs[i, 0] = deg

    # Drop heavy tail
    corrs = corrs[np.argsort(corrs[:, 0])[:-50], :]

    # Fit line
    solution = np.polyfit(corrs[:, 0], corrs[:, 1], deg=1)

    # Plot scatter
    plt.style.use("ggplot")
    f: plt.Figure = plt.Figure(figsize=(8, 5), dpi=150)
    ax: plt.Axes = f.add_subplot()

    ax.plot(corrs[:, 0], corrs[:, 1], ".", label="Real data")
    ax.plot(
        corrs[np.argsort(corrs[:, 0]), 0],
        corrs[np.argsort(corrs[:, 0]), 0] * solution[0] + solution[1],
        "-",
        label="Fitted line: {:2f}x + {:2f}\nAssortativity coeff.: {:.3f}".format(
            solution[0], solution[1], assortativity
        ),
    )

    ax.legend()
    ax.set_xlabel("Node degree")
    ax.set_ylabel("Neighborhood avg degree")

    f.savefig(str(file_path.absolute()))

    plt.close(f)
