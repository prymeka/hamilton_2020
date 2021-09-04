"""File containing functions to compare the execution time 
for the recursion and inversion GF methods."""

from greens_function import GreensFunctionAnalysis

import numpy as np

import time

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


def time_RGF(lattice_type: str, max_iteration: int, runs_per_iter: int
             ) -> np.ndarray:
    """
    Time the execution of RGF for lattices of up to max_iteration.
    """
    time_list = np.zeros((max_iteration, runs_per_iter))
    for i in range(1, max_iteration+1):
        for j in range(runs_per_iter):
            gfa = GreensFunctionAnalysis(lattice_type, i)
            start = time.perf_counter()
            gfa.get_RGF()
            end = time.perf_counter()
            time_list[i-1, j] = end-start

    return time_list


def time_IGF(lattice_type: str, max_iteration: int, runs_per_iter: int
             ) -> np.ndarray:
    """
    Time the execution of IGF for lattices of up to max_iteration.
    """
    time_list = np.zeros((max_iteration, runs_per_iter))
    for i in range(1, max_iteration+1):
        for j in range(runs_per_iter):
            gfa = GreensFunctionAnalysis(lattice_type, i)
            start = time.perf_counter()
            gfa.get_IGF()
            end = time.perf_counter()
            time_list[i-1, j] = end-start

    return time_list


def plot_execution_time(lattice_type: str, max_iteration: int,
                        runs_per_iter: int) -> None:
    """
    Plot execution time comparison between RGF and IGF. 
    """
    # get the times
    time_list_rgf = time_RGF(lattice_type, max_iteration, runs_per_iter)
    time_list_igf = time_IGF(lattice_type, max_iteration, runs_per_iter)
    # find means and stand. dev.s
    rgf_mean = [np.mean(time_list_rgf[i, :]) for i in range(max_iteration)]
    rgf_std = [np.std(time_list_rgf[i, :]) for i in range(max_iteration)]
    igf_mean = [np.mean(time_list_igf[i, :]) for i in range(max_iteration)]
    igf_std = [np.std(time_list_igf[i, :]) for i in range(max_iteration)]
    # plot
    fig, ax = plt.subplots()
    n = np.arange(1, max_iteration+1)
    ax.errorbar(n, rgf_mean, yerr=rgf_std, capsize=3, label='RGF')
    ax.errorbar(n, igf_mean, yerr=igf_std, capsize=3, label='IGF')
    ax.set_xticks(range(1, max_iteration+1))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Execution Time (sec)')
    ax.set_title(f'Execution Time for Type {lattice_type} Lattice')
    plt.legend()
    plt.savefig('./figures/execution_time.png')
    plt.show()


if __name__ == '__main__':
    plot_execution_time('C', 5, 5)
