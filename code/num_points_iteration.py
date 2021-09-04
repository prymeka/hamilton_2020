"""File containing functions to generate a plot of number of points 
versus the iteration of the lattice."""

from sierpinski import LatticeTypeC, LatticeTypeN, LatticeTypeV

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


def plot_number_of_points(max_iter: int) -> None:
    """
    Generate a plot of number of points versus iteration of the fractal
    lattice.
    """
    # calculate the values
    num_points_list = []
    iter_arr = np.arange(1, max_iter+1)
    for i in iter_arr:
        num_c = LatticeTypeC.get_num_points(i)
        num_n = LatticeTypeN.get_num_points(i)
        num_v = LatticeTypeV.get_num_points(i)
        num_points_list.append((num_c, num_n, num_v))
    num_points_list = np.array(num_points_list)
    # plot
    fig, ax = plt.subplots()
    ax.plot(iter_arr, num_points_list[:, 0], 'o--', label='Type C')
    ax.plot(iter_arr, num_points_list[:, 1], 'o--', label='Type N')
    ax.plot(iter_arr, num_points_list[:, 2], 'o--', label='Type V')
    ax.set_xticks(range(1, max_iter+1))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of Points')
    ax.set_title('Number of Points in the Lattice')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./figures/num_points_iteration.png')
    plt.show()


if __name__ == "__main__":
    plot_number_of_points(5)
