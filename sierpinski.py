"""File containing the daughter class of the base FractalLattice 
class used to generate explicit types of the lattice."""

from typing import Tuple, List

import numpy as np

import scipy.spatial as spatial
from math import hypot

from sierpinski_base import FractalLattice


class LatticeTypeC(FractalLattice):
    """Class that generates the sierpinski fractal type C 
    (with the centre points)."""

    def __init__(self, iteration: int, t: float = -1.0,
                 eps: np.ndarray = None, frac_num_defects: float = 0,
                 eps_defect: float = 0) -> None:
        """
        Constructor.
        """
        # iteration of the lattice
        self.iteration = iteration
        # the 'width' of the lattice, parameter used in
        # the self.sierpinski() method
        self.width = 2**self.iteration
        # initial coordinates for generating lattice
        self.x_initial = 0.5
        self.y_initial = np.sqrt(3)/6
        # type of the sierpinski fractal lattice
        self.lat_type = 'C'
        # call the constructor from parent class
        self.constructor(self.x_initial, self.y_initial, self.width,
                         self.get_top_vertex(self.iteration), t, eps,
                         frac_num_defects, eps_defect)

    @staticmethod
    def get_num_points(iteration: int) -> int:
        """
        Given the iteration of the lattice return the number of points
        in the lattice.
        """
        return int((5*3**iteration+3)/2)

    @classmethod
    def get_top_vertex(cls, iteration: int) -> int:
        """
        Return the index of the point that is the top vertex 
        of the lattice.
        """
        return int((cls.get_num_points(iteration) + iteration)/2)


class LatticeTypeN(FractalLattice):
    """Class that generates the sierpinski fractal type N 
    (without the centre points)."""

    def __init__(self, iteration: int, t: float = -1.0,
                 eps: np.ndarray = None, frac_num_defects: float = 0,
                 eps_defect: float = 0) -> None:
        """
        Constructor.
        """
        # iteration of the lattice
        self.iteration = iteration
        # the 'width' of the lattice
        self.width = 2**self.iteration
        # initial coordinates for generating lattice
        self.x_initial = 0.5
        self.y_initial = np.sqrt(3)/6
        # type of the sierpinski lattice
        self.lat_type = 'N'
        # extra array to hold coordinates of 'centre' points
        self.xy_centre = []
        # call the constructor from parent class
        self.constructor(self.x_initial, self.y_initial, self.width,
                         self.get_top_vertex(self.iteration), t, eps,
                         frac_num_defects, eps_defect)

    @staticmethod
    def get_num_points(iteration: int) -> int:
        """
        Given the iteration of the lattice return the number of points
        in the lattice.
        """
        return int((3**(iteration+1)+3)/2)

    @classmethod
    def get_top_vertex(cls, iteration: int) -> int:
        """
        Return the index of the point that is the top vertex 
        of the lattice.
        """
        return int((cls.get_num_points(iteration) + iteration)/2)

    # override some parent methods

    def sierpinski(self, x: float, y: float, width: int) -> None:
        """
        Recursive function that will generate all the points 
        on the lattice and save them to a global holder list.

        It is assumed that self.xy_list self.xy_centre already exist, 
        i.e. this method is called from within 
        self.generate_lattice(...).
        """
        if width == 1:
            # in this type of latticce the centre points are not plotted
            # to do this they are saved to another holder as they will
            # be needed to find neighbours
            self.xy_centre.append((x, y))
            self.xy_list.append((x+0.5, y-np.sqrt(3)/6))
            self.xy_list.append((x, y+np.sqrt(3)/3))

        else:
            width = width/2
            self.sierpinski(x, y, width)
            self.sierpinski(x+width, y, width)
            self.sierpinski(x+width/2, y+width*np.sqrt(3)/2, width)

    def get_neighbours(self) -> Tuple[List[List[int]], spatial.kdtree.KDTree]:
        """
        Find all the nearest neighbours of each point. Only points
        with common centre point can be neighbours. 
        """
        xy_centre = np.array(self.xy_centre)
        distance_to_centre = hypot(xy_centre[0, 0], xy_centre[0, 1])
        all_positions = np.concatenate((self.positions, xy_centre), axis=0)
        # neighbours for all the points including the centre points
        tree = spatial.KDTree(all_positions)
        neighbours_temp = tree.query_ball_tree(tree, distance_to_centre*1.1)
        # for each point find neighbours of the centre points
        # it neighbours
        neighbours = []
        for i in range(self.num_points):
            temp = []
            for j in neighbours_temp[i]:
                temp = temp + neighbours_temp[j]
            temp = list(set(temp) - set(neighbours_temp[i]))
            neighbours.append(temp)

        return neighbours, tree


class LatticeTypeV(FractalLattice):
    """Class that generates the sierpinski fractal type V
    (with spaced verices)."""

    def __init__(self, iteration: int, t: float = -1.0,
                 eps: np.ndarray = None, frac_num_defects: float = 0,
                 eps_defect: float = 0) -> None:
        """
        Constructor.
        """
        # iteration of the lattice
        self.iteration = iteration
        # the 'width' of the lattice
        self.width = 2**(iteration+1)
        # initial coordinates for generating lattice
        self.x_initial = 0
        self.y_initial = 0
        # type of the sierpinski lattice
        self.lat_type = 'V'
        # call the constructor from parent class
        self.constructor(self.x_initial, self.y_initial, self.width,
                         self.get_top_vertex(self.iteration), t, eps,
                         frac_num_defects, eps_defect)

    @staticmethod
    def get_num_points(iteration: int) -> int:
        """
        Given the iteration of the lattice return the number of points
        in the lattice.
        """
        return int(3**(iteration+1))

    @classmethod
    def get_top_vertex(cls, iteration: int) -> int:
        """
        Return the index of the point that is the top vertex 
        of the lattice.
        """
        return int((cls.get_num_points(iteration) - 1)/2)

    # override some parent methods

    def generate_lattice(self, x_initial: float, y_initial: float, width: int
                         ) -> np.ndarray:
        """
        Generate a list of coordinates defining all the points 
        on the lattice.
        """
        # initialise x and y coordinate holder list
        self.xy_list = []
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
            # for this type it is enough to just save the centre points
            self.xy_list.append((x, y))

        else:
            width = width/2
            self.sierpinski(x, y, width)
            self.sierpinski(x+width, y, width)
            self.sierpinski(x+width/2, y+width*np.sqrt(3)/2, width)


def main(lat_type: str, iteration: int) -> None:
    lat_dict = {'C': LatticeTypeC, 'N': LatticeTypeN, 'V': LatticeTypeV}
    lat_obj = lat_dict[lat_type](iteration)
    lat_obj.plot_cells()


if __name__ == '__main__':
    main('C', 5)
