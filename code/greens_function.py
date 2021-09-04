"""File containing the class containing methods that implement 
the Recursive Green's Function method and the brute force, Inversion 
Green's Function method."""

from typing import Tuple, List

from sierpinski import LatticeTypeC, LatticeTypeN, LatticeTypeV

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')  # style of plot


class GreensFunctionAnalysis:
    """Class with methods used to find the Hamiltonian, Green's 
    function using both recursive and inversion methods, and DOS."""

    # values for which GF and DOS will be calculated and plotted
    MAX_ENER: float = 3
    MIN_ENER: float = -MAX_ENER
    ENER_STEPS: int = 1000
    ENERGY_ARRAY = np.linspace(MIN_ENER, MAX_ENER, ENER_STEPS)
    # eta*j appearing in the definition of the GF
    ETA_J: complex = 1e-3j

    def __init__(self, lattice_type: str, iteration: int, t: float = -1.0,
                 eps: np.ndarray = None, frac_num_defects: float = 0,
                 eps_defect: float = 0) -> None:
        """
        Constructor.
        """
        # the type of the lattice to be used
        self.lattice_type = lattice_type
        # iteration of the fractal lattice
        self.iteration = iteration
        # initalise the lattice
        self.init_lattice(lattice_type, t, eps, frac_num_defects, eps_defect)
        # find the Hamiltonian of the whole lattice
        self.lattice_hamiltonian = self.get_lattice_hamiltonian()

    def init_lattice(self, lattice_type: str, t: float, eps: np.ndarray,
                     frac_num_defects, eps_defects) -> None:
        """
        Initialise the lattice to be analysed.
        """
        # dictionary with the lattices to simulate the cpp
        # switch/case statement
        lat_dict = {'C': LatticeTypeC, 'N': LatticeTypeN, 'V': LatticeTypeV}
        lat_cls = lat_dict[lattice_type]
        # generate the lattice
        self.t = t
        self.eps = (eps if eps is not None
                    else np.zeros(lat_cls.get_num_points(self.iteration)))
        self.lattice_obj = lat_cls(self.iteration, t, self.eps,
                                   frac_num_defects, eps_defects)

    def get_hamiltonian(self, points_idx: List[int], t: float, eps: np.ndarray
                        ) -> np.ndarray:
        """
        Find the matrix Hamiltonian of a (portion of a) lattice. 

        The diagonal entries are equal to the on-site energy of each 
        point. The ij-th entry is equal to the hopping term between 
        the i-th and j-th sites (if i-th and j-th sites are not 
        the nearest-neighbours the value is zero.)
        """
        # the number of points in the system
        size = len(points_idx)
        # Hamiltonian is a square N x N-matrix where N is
        # the number of points
        H = np.zeros((size, size))
        # loop over the points in the system
        for i in range(size):
            # loop over the neighbours of a point (do not include any
            # points outside of the system)
            s1 = set(self.lattice_obj.neighbours[points_idx[i]])
            s2 = set(points_idx)
            good_neighbours = list(s1 & s2)
            for j in good_neighbours:
                # set i,j entry to t if ith and jth sites are neighbours
                H[i, points_idx.index(j)] = t
            # set diagonal entries to eps at i
            H[i, i] = eps[i]

        return H

    def get_potential(self, cells_idx: np.ndarray, step: int) -> np.ndarray:
        """
        Find the potential between two systems, i.e. the coupling term
        between two different cells/two partitions in a system.

        The coupling term is given by the off-diagonal block element
        in the Hamiltonian of the joined system. In this case we find 
        the coupling of (step-1) and (step) cells.
        """
        # if system consists of no or only one cell return 0
        if step < 2:
            return 0
        else:
            # find the number of points in cells joind so far,
            # not including the cell number (step)
            d = len([item for sublist in cells_idx[:step-1]
                    for item in sublist])
            # return the part of Hamiltonian that joins
            # cells (step-1) and (step)
            potential = self.lattice_hamiltonian[d-len(cells_idx[step-2]):d,
                                                 d:d+len(cells_idx[step-1])]

            return potential

    def get_lattice_hamiltonian(self) -> np.ndarray:
        """
        Return the Hamiltonian of the whole lattce.
        """
        flat_cells = [val for sublist in self.lattice_obj.cells
                      for val in sublist]

        return self.get_hamiltonian(flat_cells, self.t, self.eps)

    def RGF(self, energy: float) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Calculate the Green's functions and the density of states 
        using the recursive method. 
        """
        # for convenience re-assign some variables
        cells = self.lattice_obj.cells
        steps = self.lattice_obj.num_cells
        # consider the system containing only the first cell,
        # find the Hamiltonian...
        h = self.get_hamiltonian(cells[0], self.t, self.eps)
        # the Green's function,...
        g = np.linalg.inv(np.eye(len(h)) * (energy+self.ETA_J) - h)
        # and the density of states
        dos = -(1/np.pi) * np.imag(np.trace(g))

        # loop over the rest of the cells
        for i in range(1, steps):
            # get potential joining the cell i and the cell i+1
            V = self.get_potential(cells, i+1)
            # find the A_(N+1) term - note that A_0 = 0
            if i == 1:
                A = np.linalg.multi_dot((V.T, g, g, V))
            else:
                A = np.linalg.multi_dot((V.T, g, A+np.eye(len(A)), g, V))
            # find the Hamiltonian with cell i joined
            h = self.get_hamiltonian(cells[i], self.t, self.eps)
            # find the GF of only cell i
            gnplusone = np.linalg.inv(np.eye(len(h)) * (energy+self.ETA_J) - h)
            # find the new GF using formula for G_r^(N+1)
            numerator = gnplusone
            denominator = np.eye(len(gnplusone)) - \
                np.linalg.multi_dot((gnplusone, V.T, g, V))
            g = np.dot(np.linalg.inv(denominator), numerator)
            # update DOS
            dos = dos - (1/np.pi) * \
                np.imag(np.trace(np.matmul(g, A + np.eye(len(A)))))

        return g, dos, A

    def IGF(self, energy: float) -> Tuple[np.ndarray, float]:
        """
        Calculate the Green's functions and the density of states 
        using the inversion method. 
        """
        H = self.lattice_hamiltonian
        # find Green's function using the standard formula
        g = np.linalg.inv(np.eye(len(H)) * (energy+self.ETA_J) - H)
        # DOS is defined to be proportional to the imaginary
        # part of trace of GF
        dos = -(1/np.pi) * np.imag(np.trace(g))

        return g[-3:, -3:], dos

    def get_RGF(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the GF and DOS using the recursion method for a number
        of energy values.
        """
        g_list, dos_list = [], []
        for energy in self.ENERGY_ARRAY:
            g_temp, dos_temp, _ = self.RGF(energy)
            g_list.append(g_temp)
            dos_list.append(dos_temp)

        return g_list, dos_list

    def get_IGF(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the GF and DOS using the inversion method for a number
        of energy values.
        """
        g_list, dos_list = [], []
        for energy in self.ENERGY_ARRAY:
            g_temp, dos_temp = self.IGF(energy)
            g_list.append(g_temp)
            dos_list.append(dos_temp)

        return g_list, dos_list

    def plot_dos(self, dos: List[float]) -> None:
        """
        Plot the density of states.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.ENERGY_ARRAY, dos)
        ax.set_xlabel(f'Energy (steps={self.ENER_STEPS})')
        ax.set_ylabel('DOS')
        ax.set_title('DOS as a Function of Energy - Recursion')
        fig.suptitle('Generation {} of Type {} Lattice'.format(self.iteration,
                                                               self.lattice_type))
        fig.patch.set_facecolor('xkcd:silver')
        plt.show()

    def plot_RGF(self, g: List[np.ndarray]) -> None:
        """
        Plot the real and imaginary values of the Green's function
        for the three vertices.
        """
        # finding GF for each vertex
        g_right = [g[i][0, 0] for i in range(self.ENER_STEPS)]
        g_top = [g[i][0, 1] for i in range(self.ENER_STEPS)]
        g_left = [g[i][0, 2] for i in range(self.ENER_STEPS)]
        # plotting
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        # right vertex
        ax[0].plot(self.ENERGY_ARRAY, np.real(g_right), c='r',
                   label='Real Part')
        ax[0].plot(self.ENERGY_ARRAY, np.imag(g_right), c='b',
                   label='Imaginary Part')
        ax[0].set_xlabel('g')
        ax[0].set_ylabel('Energy')
        ax[0].set_title('GF for the Left Vertex of the Lattice')
        ax[0].legend(loc='lower left')
        # top vertex
        ax[1].plot(self.ENERGY_ARRAY, np.real(g_top), c='r',
                   label='Real Part')
        ax[1].plot(self.ENERGY_ARRAY, np.imag(g_top), c='b',
                   label='Imaginary Part')
        ax[1].set_xlabel('g')
        ax[1].set_ylabel('Energy')
        ax[1].set_title('GF for the Top Vertex of the Lattice')
        ax[1].legend(loc='lower left')
        # left vertex
        ax[2].plot(self.ENERGY_ARRAY, np.real(g_left), c='r',
                   label='Real Part')
        ax[2].plot(self.ENERGY_ARRAY, np.imag(g_left), c='b',
                   label='Imaginary Part')
        ax[2].set_xlabel('g')
        ax[2].set_ylabel('Energy')
        ax[2].set_title('GF for the Right Vertex of the Lattice')
        ax[2].legend(loc='lower left')
        fig.tight_layout()
        plt.show()


def main() -> None:
    gfa = GreensFunctionAnalysis('C', 3)
    g, dos = gfa.get_RGF()
    gfa.plot_dos(dos)
    gfa.plot_RGF(g)


if __name__ == '__main__':
    main()
