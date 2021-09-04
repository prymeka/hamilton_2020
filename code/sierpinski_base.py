"""File containing the base FractalLattice class used to model 
the Sierpinski triangle fractal lattice."""

from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import scipy.spatial as spatial

import random
from math import hypot


class FractalLattice:
    """Base class for all three types of Sierpinski fractals containing 
    methods to generate and slice the lattice into cells."""

    def constructor(self, x_initial: float, y_initial: float, width: int,
                    top_vertex_index: int, t: float, eps: np.ndarray,
                    frac_num_defects: float, eps_defect: float) -> None:
        """
        Constructor that should only be called by __init__ indside 
        the child class.
        """
        # number of points in the lattice
        self.num_points = self.get_num_points(self.iteration)
        # the hopping term
        self.t = t
        # array with energies
        self.eps = eps if eps is not None else np.zeros(self.num_points)
        # generate the lattice
        self.positions = self.generate_lattice(x_initial, y_initial, width)
        # find neighbours
        self.neighbours, self.tree = self.get_neighbours()
        # starting cell (after slicing into cells the order will be
        # reversed and this will be the last cell)
        self.starting_points_idx = [0, top_vertex_index, self.num_points-1]
        # divide lattice into cells
        self.cells = self.slice_lattice(self.starting_points_idx)
        self.num_cells = len(self.cells)
        # add defects/impurities to the lattice
        self.frac_num_defects = frac_num_defects
        self.eps_defect = eps_defect
        self.add_defects()

    # create the lattice

    def generate_lattice(self, x_initial: float, y_initial: float, width: int
                         ) -> np.ndarray:
        """
        Generate a list of coordinates defining all the points 
        on the lattice.
        """
        # initialise x and y coordinate holder list
        self.xy_list = [(0, 0)]
        # generate all points and save it to the holder array
        self.sierpinski(x_initial, y_initial, width)
        # convert the list to np.ndarray
        positions_with_dups = np.array(self.xy_list)
        # remove duplicates from the list
        positions = self.remove_duplicates(positions_with_dups)

        return positions

    def sierpinski(self, x: float, y: float, width: int) -> None:
        """
        Recursive function that will generate all the points 
        on the lattice and save them to a global holder list.

        It is assumed that self.xy_list already exists, i.e. this method
        is called from within self.generate_lattice(...).
        """
        if width == 1:
            # the central point of a single triangular unit
            self.xy_list.append((x, y))
            # the top and left vertices of a single triangular unit
            # adding the right vertex is not necessary as units share
            # some of the points
            self.xy_list.append((x+0.5, y-np.sqrt(3)/6))
            self.xy_list.append((x, y+np.sqrt(3)/3))

        else:
            width = width/2
            self.sierpinski(x, y, width)
            self.sierpinski(x+width, y, width)
            self.sierpinski(x+width/2, y+width*np.sqrt(3)/2, width)

    def remove_duplicates(self, arr_with_dups: np.ndarray) -> np.ndarray:
        """
        Remove duplicates from a numpy array with dimensions (N, 2),
        and sort it.
        """
        # convert array to pandas
        df_with_dups = pd.DataFrame(arr_with_dups, columns=['x', 'y'])
        # sort data
        df_with_dups = df_with_dups.sort_values(by=['x', 'y'])
        # because of nature of floats multiply all entries by 1e6
        # and convert to int to remove duplicates
        df_with_dups = (df_with_dups*1e6).astype(int)
        df_no_dups = df_with_dups.drop_duplicates(keep='first')
        # convert back to floats
        df_no_dups = (df_no_dups.astype(float))/1e6

        return df_no_dups.to_numpy()

    def get_neighbours(self) -> Tuple[List[List[int]], spatial.kdtree.KDTree]:
        """
        Find all the nearest neighbours of each point.
        """
        # distance between neighbours
        hopping_distance = hypot(self.positions[1, 0], self.positions[1, 1])
        # find neighbours using kd tree
        tree = spatial.KDTree(self.positions)
        neighbours = tree.query_ball_tree(tree, hopping_distance*1.1)

        return neighbours, tree

    def slice_lattice(self, starting_points_idx: List[int]) -> List[int]:
        """
        Divide the lattice into cells. First cell will containt 
        the three vertices and each next will contain the neighbours 
        of the previous one. Before returning reverse the order 
        of the cells.

        It is assumed that the self.get_neighbours() method has been 
        called before calling this one and self.neighbours exists.
        """
        assert hasattr(self, 'neighbours')
        # list holding points not yet in a cell
        not_assigned = list(range(0, self.num_points))
        # holder list with initial cell
        cells_idx = [starting_points_idx]
        # remove from not_assigned the first cell
        not_assigned = list(set(not_assigned) - set(cells_idx[0]))
        # loop while there are unassigned points
        while not_assigned:
            # find neighbours of the last cell
            new_cell = []
            for i in cells_idx[-1]:
                new_cell += self.neighbours[i]
            # append the new cell but remove any point that may already
            # have been assigned
            cells_idx.append(list(set(not_assigned) & set(new_cell)))
            # update the not_assigned list
            not_assigned = list(set(not_assigned) - set(cells_idx[-1]))
        # reverse the order of cell
        cells_idx = cells_idx[::-1]

        return cells_idx

    def add_defects(self) -> None:
        """
        Add defects to the lattice, i.e. change the value of eps (site
        energy) at randomly selected sites.
        """
        # re-assign the eps of the pristine lattice
        self.eps_pristine = self.eps.copy()
        # find the integer number of defects
        num_defected_points = int(self.num_points * self.frac_num_defects)
        # for convenience in deriving the equations for the GF and DOS
        # defects will not be added to the three vertices
        # hence they are excluded
        potential_defect_points = list(
            set(range(self.num_points))-set(self.starting_points_idx)
        )
        # sample sites that will be impure
        self.defect_idx = random.sample(potential_defect_points,
                                        num_defected_points)
        # apply the defect/impurity by changing potential at the site
        self.eps[self.defect_idx] = self.eps_defect

    # plots

    def plot_lattice(self) -> None:
        """
        Plot the lattice.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax = plt.subplot(111)
        ax.scatter(self.positions[:, 0], self.positions[:, 1])
        # plot lines joining neighbours
        for i in range(self.num_points):
            point_i = self.tree.data[i]
            for j in self.neighbours[i]:
                point_j = self.tree.data[j]
                ax.plot([point_i[0], point_j[0]], [point_i[1], point_j[1]],
                        c='#DBDBDB', zorder=0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_title(f'Type {self.lat_type}\nG({self.iteration})')
        fig.tight_layout()
        plt.show()

    def plot_lattice_with_defects(self) -> None:
        """
        Plot the lattice with the impure sites coloured.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax = plt.subplot(111)
        ax.scatter(self.positions[:, 0], self.positions[:, 1])
        # plot lines joining neighbours
        for i in range(self.num_points):
            point_i = self.tree.data[i]
            for j in self.neighbours[i]:
                point_j = self.tree.data[j]
                ax.plot([point_i[0], point_j[0]], [point_i[1], point_j[1]],
                        c='#DBDBDB', zorder=0)
        # plot the defected sites in a different colour
        defect_points = self.positions[self.defect_idx]
        ax.scatter(defect_points[:, 0], defect_points[:, 1], c='k')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        perc = self.frac_num_defects*100
        ax.set_title(f'Type {self.lat_type} with ' +
                     f'{perc}% impurities\nG({self.iteration})')
        fig.tight_layout()
        plt.show()

    def plot_cells(self) -> None:
        """
        Plot the lattice with colour coded cells.
        """
        fig, ax = plt.subplots()
        for i in range(self.num_cells):
            points = self.tree.data[self.cells[i]]
            ax.scatter(points[:, 0], points[:, 1])
        # plot lines joining neighbours
        for i in range(self.num_points):
            point_i = self.tree.data[i]
            for j in self.neighbours[i]:
                point_j = self.tree.data[j]
                ax.plot([point_i[0], point_j[0]], [point_i[1], point_j[1]],
                        c='#DBDBDB', zorder=0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_title(f'Type{self.lat_type}\nG({self.iteration})')
        fig.tight_layout()
        plt.show()
