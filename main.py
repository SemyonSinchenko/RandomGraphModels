#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path

from impl import (barabasi_albert, configuration_model, copy_model,
                  erdos_renyi, kronecker_graph, small_world,
                  linear_pref_attach)
from utils import plot_all_distributions

GLOBAL_PREFIX = Path("results")


def run_er_experiments(v: int, e: int) -> None:
    g = erdos_renyi(v, e)
    plot_all_distributions(g, GLOBAL_PREFIX.joinpath("erdos-renyi"))


def run_sw_experiments(v: int, e: int) -> None:
    sw_prefix = GLOBAL_PREFIX.joinpath("small-world")
    for beta in [0.01, 0.05, 0.1]:
        g = small_world(v, e, beta)
        plot_all_distributions(g,
                               sw_prefix.joinpath("beta_{:.0e}".format(beta)))


def run_ba_experiments(v: int, e: int) -> None:
    g = barabasi_albert(v, e)
    plot_all_distributions(g, GLOBAL_PREFIX.joinpath("barabasi-albert"))


def run_cm_experiments(v: int, e: int) -> None:
    cm_prefix = GLOBAL_PREFIX.joinpath("copy-model")
    for p in [0.5, 0.6, 0.7]:
        g = copy_model(v, p)
        plot_all_distributions(g,
                               cm_prefix.joinpath("alpha_{:.0e}_cm".format(p)))


def run_cfgm_experiments(v: int) -> None:
    cfgm_prefix = GLOBAL_PREFIX.joinpath("configuration-model")
    for alpha in [-2.2, -2.5, -2.8]:
        g = configuration_model(v, alpha)
        plot_all_distributions(
            g, cfgm_prefix.joinpath("alpha_{:.1e}_cfgm".format(-alpha)))


def run_kg_experiments() -> None:
    kg_prefix = GLOBAL_PREFIX.joinpath("kronecker-graph")
    for density in [0.3, 0.4, 0.5]:
        g = kronecker_graph(1000, density=density)
        plot_all_distributions(
            g, kg_prefix.joinpath("density_{:0e}_kgm".format(density)))


def run_lpa_experiments(v: int, e: int) -> None:
    lpa_prefix = GLOBAL_PREFIX.joinpath("linear_pref_attach")
    for alpha in [-0.85, -0.5, 0.0, 5.0, 100.0]:
        g = linear_pref_attach(v, e, alpha)
        plot_all_distributions(
            g, lpa_prefix.joinpath("alpha_{:0e}_lpa".format(alpha)))


if __name__ == "__main__":
    logging.basicConfig(filename="mainLog.log",
                        filemode="w",
                        level=logging.INFO)
    v = 3000
    e = 90000

    run_er_experiments(v, e)
    run_sw_experiments(v, e)
    run_ba_experiments(v, e)
    run_cm_experiments(v, e)
    run_cfgm_experiments(v // 2)
    run_kg_experiments()
    run_lpa_experiments(v, e * 2)
