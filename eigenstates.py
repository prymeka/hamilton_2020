"""File containing the class used to calculate the eigenstates, IPR 
and DOS of pristine and impure lattices."""

from typing import Tuple, Dict

from conductance import LatticeConductance

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')  # style of plot


class LatticeEigenstates(LatticeConductance):
    """Class that calculates the eigenstates of pristine and impure
    lattices."""

    def __init__(self, lattice_type: str, iteration: int, t: float = -1,
                 eps: np.ndarray = None, frac_num_defects: float = 0,
                 eps_defect: float = 0) -> None:
        super().__init__(lattice_type, iteration, t, eps, frac_num_defects,
                         eps_defect)
        # get the prisitine lattice Hamiltonian
        self.pristine_hamiltonian = self.get_pristine_hamiltonian()
        # find the eigenvalues and eigenvectors
        self.eigen_dict = self.get_eigen_all()
        # find IPR
        self.clean_IPR = self.get_IPR(self.eigen_dict['clean_vecs'])
        self.dirty_IPR = self.get_IPR(self.eigen_dict['dirty_vecs'])

    def get_eigen_all(self) -> Dict[str, np.ndarray]:
        """
        Get the eigenvalues and eigenvectors for both impure 
        and pristine lattices.
        """
        eigen = dict()
        clean_h = self.pristine_hamiltonian
        (eigen['clean_vals'],
         eigen['clean_vecs']) = self.get_eigenvalues_and_eigenvectors(clean_h)
        dirty_h = self.lattice_hamiltonian
        (eigen['dirty_vals'],
         eigen['dirty_vecs']) = self.get_eigenvalues_and_eigenvectors(dirty_h)

        return eigen

    def get_eigenvalues_and_eigenvectors(self, hamiltonian: np.ndarray
                                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the eigenvalues and eigenvectors of a Hamiltonian.
        """
        vals, vecs = np.linalg.eig(hamiltonian)
        idx = vals.argsort()
        vals = vals[idx]
        vecs = vecs[:, idx]

        return vals, vecs

    def get_IPR(self, eigenvectors: np.ndarray) -> np.ndarray:
        """
        Calculate the IPR for entire set of the eigenvectors. 
        """
        num_states = len(eigenvectors)
        ipr = [self.get_IPR_for_one_state(eigenvectors[:, i])
               for i in range(num_states)]

        return ipr

    def get_IPR_for_one_state(self, p: np.ndarray) -> float:
        """
        Calculate the Inverse Participation Ratio defined by:
        IPR = sum(N*p_i^4), 
        where p_i is the probability of a site being in the state i.
        """
        return np.real(sum(np.power(p, 4)))

    def get_pristine_hamiltonian(self) -> np.ndarray:
        """
        Calculate the Hamiltonian of a pristine lattice.
        """
        flat_cells = [val for sublist in self.lattice_obj.cells
                      for val in sublist]
        h = self.get_hamiltonian(flat_cells, self.t,
                                 self.lattice_obj.eps_pristine)

        return h

    def plot_IPR_pristine_and_impure(self) -> None:
        """
        Plot the IPR for both pristine and impure lattices.
        """
        perc = self.lattice_obj.frac_num_defects*100
        x = np.arange(1, self.lattice_obj.num_points+1)
        fig, ax = plt.subplots(figsize=(13, 8))
        # pristine lattice
        ax.plot(x, self.clean_IPR, label='IPR of Pristine Lattice')
        # impure lattice
        ax.plot(x, self.dirty_IPR,
                label=f'IPR of Lattice with {perc}% Defects')
        # other
        ax.axhline(y=0, c='k', zorder=0)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel('Eigentates')
        ax.set_ylabel('IPR')
        ax.set_title(f'Generation {self.iteration} of ' +
                     f'{self.lattice_type} type Sierpinski Fractal')
        ax.legend()
        plt.savefig(f'./figures/IPR_{self.lattice_type}{self.iteration}')
        plt.show()

    def plot_eigenstates(self, state: int) -> None:
        """
        Plot the eigenstates. 
        """
        flat_cells = [val for sublist in self.lattice_obj.cells
                      for val in sublist]
        # coordinates of all the points in the same order as eigenvalues
        positions_sorted = self.lattice_obj.tree.data[flat_cells]
        # plotting
        fig, ax = plt.subplots(ncols=2, figsize=(15, 5))
        # re-normalisation factor
        renorm = 5000
        # pristin lattice
        size = renorm*abs(self.eigen_dict['clean_vecs'][:, state-1])**2
        ax[0].scatter(positions_sorted[:, 0], positions_sorted[:, 1],
                      s=size, c='k')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title(
            f'Eigenstate {state}/{self.lattice_obj.num_points} - Pristine')
        # impure lattice
        size = renorm*abs(self.eigen_dict['dirty_vecs'][:, state])**2
        ax[1].scatter(positions_sorted[:, 0],
                      positions_sorted[:, 1], s=size, c='k')
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        perc = self.lattice_obj.frac_num_defects*100
        ax[1].set_title(
            f' Eigenstate {state}/{self.lattice_obj.num_points} - {perc}% of Defects')
        # plot the lines between neighbours
        for i in range(self.lattice_obj.num_points):
            point_i = self.lattice_obj.tree.data[i]
            for j in self.lattice_obj.neighbours[i]:
                point_j = self.lattice_obj.tree.data[j]
                ax[0].plot([point_i[0], point_j[0]], [
                           point_i[1], point_j[1]], c='#B0B0B0', zorder=0)
                ax[1].plot([point_i[0], point_j[0]], [
                           point_i[1], point_j[1]], c='#B0B0B0', zorder=0)
        fig.suptitle(
            f'Generation {self.iteration} of {self.lattice_type} Lattice')
        fig.patch.set_facecolor('xkcd:silver')
        plt.show()


def main() -> None:
    state = 1
    e = LatticeEigenstates('C', 3, frac_num_defects=0.2, eps_defect=1e2)
    e.plot_eigenstates(state)
    e.plot_IPR_pristine_and_impure()


if __name__ == '__main__':
    main()
