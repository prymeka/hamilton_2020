"""File containing the class used to connect the leads to the lattice 
and find the conductance of the connected lattice."""

from typing import Tuple, List, Union

from greens_function import GreensFunctionAnalysis

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')  # style of plot


class LatticeConductance(GreensFunctionAnalysis):
    """Class inheriting from the GreensFunctionAnalysis. Used to find 
    the conductance and DOS of a fractal lattice."""

    # parameters of leads (it is assumed that leads are connected
    # to the left (_L/_l) and right (_R/_r) vertices)
    EPS_LEAD_L = 0
    EPS_LEAD_R = 0
    T_LEAD_L = -1
    T_LEAD_R = -1
    QUANTUM_CONDUCTANCE = 7.748091729E-5

    def __init__(self, lattice_type: str, iteration: int, t: float = -1,
                 eps: np.ndarray = None, frac_num_defects: float = 0,
                 eps_defect: float = 0) -> None:
        """
        Constructor.
        """
        super().__init__(lattice_type, iteration, t, eps, frac_num_defects,
                         eps_defect)

    def lead_self_energy_retarded(self, eps_lead: float, t_lead: float,
                                  energy: float) -> complex:
        """
        Calculate the retarded self-energy of a lead (it is equivalent
        to the surface GF of a semi-finite chain).
        """
        temp = energy + self.ETA_J - eps_lead
        return (temp + np.sqrt((temp)**2 - 4*t_lead**2)) / (2*t_lead**2)

    def lead_self_energy_advanced(self, eps_lead: float, t_lead: float,
                                  energy: float) -> complex:
        """
        Calculate the advanced self-energy of a lead, which is given 
        as complex conjugate of the retarded self-energy.
        """
        return np.conj(self.lead_self_energy_retarded(eps_lead, t_lead, energy))

    def broadening_matrix_lead(self, eps_lead: float, t_lead: float,
                               energy: float) -> complex:
        """
        Calculate the Gamma broadening matrix (in our case it is 
        actually a scalar) which captures the effect of adding the lead 
        to the device. It is defined by:
        Gamma_p = i*(Sigma^R_p - Sigma^A_p).
        """
        sigma_ret = self.lead_self_energy_retarded(eps_lead, t_lead, energy)
        sigma_adv = self.lead_self_energy_advanced(eps_lead, t_lead, energy)

        return 1.0j * (sigma_ret - sigma_adv)

    def get_gamma_matrix(self, energy: float) -> np.ndarray:
        """
        Calculate the gamma matrix.
        """
        sigma_l = self.broadening_matrix_lead(self.EPS_LEAD_L, self.T_LEAD_L,
                                              energy)
        sigma_r = self.broadening_matrix_lead(self.EPS_LEAD_R, self.T_LEAD_R,
                                              energy)
        sigma_mat = np.zeros((3, 3), dtype=np.complex_)
        sigma_mat[0, 0], sigma_mat[1, 1] = sigma_l, sigma_r

        return sigma_mat

    def get_lead_gf(self, energy: float) -> Tuple[complex, complex]:
        """
        Return the GF of the leads. 
        """
        g_l = self.lead_self_energy_retarded(self.EPS_LEAD_L, self.T_LEAD_L,
                                             energy)
        g_r = self.lead_self_energy_retarded(self.EPS_LEAD_R, self.T_LEAD_R,
                                             energy)
        return g_l, g_r

    def update_by_single_lead(self, g_lead_disconnected: float, V: np.ndarray,
                              g_lattice: np.ndarray, dos: float,
                              An: Union[np.ndarray, complex]
                              ) -> Tuple[np.ndarray, float, complex]:
        """
        Find the GF and DOS of the lattice after a lead was connected.
        """
        # first we find the GF of the lead after being connected
        # to the lattice
        temp_denom = np.linalg.multi_dot((V.T, g_lattice, V))
        # assert that we get the right shape so that we can safely
        # convert the 1x1-matrix to complex number
        assert temp_denom.shape == (1, 1)
        g_lead_connected = g_lead_disconnected / \
            (1 - g_lead_disconnected*temp_denom[0, 0])
        # second we find the GF of the lattice after being connected
        # to the lead
        temp = np.linalg.multi_dot(
            (g_lattice, V, V.T, g_lattice)
        ) * g_lead_connected
        g_lattice_connected = g_lattice + temp
        # find the new DOS
        if isinstance(An, np.ndarray):
            An_con = np.linalg.multi_dot(
                (V.T, g_lattice, An+np.eye(len(An)), g_lattice, V)
            )
        else:
            An_con = np.linalg.multi_dot(
                (V.T, g_lattice, g_lattice, V)
            ) * (An+1)
        # assert that we get the right shape so that we can safely
        # convert the 1x1-matrix to complex number
        assert An_con.shape == (1, 1)
        An_con = An_con[0, 0]
        dos_con = dos - (1/np.pi)*np.imag(g_lead_connected * (An_con+1))

        return g_lattice_connected, dos_con, An_con

    def update_by_leads(self, energy: float, g_lattice: np.ndarray,
                        dos_lattice: float, An_lattice: np.ndarray
                        ) -> Tuple[np.ndarray, float]:
        """
        Find the GF and DOS of the lattice after both leads were 
        connected.
        """
        # the GFs of both leads
        g_lead_left, g_lead_right = self.get_lead_gf(energy)
        # the potential connecting the left lead to the left vertex
        V_left = np.zeros((3, 1))
        V_left[0, 0] = self.t
        # the potential connecting the right lead to the right vertex
        V_right = np.zeros((3, 1))
        V_right[2, 0] = self.t
        # update the GF of the disconnected lattice by the left lead
        g_con_left, dos_con_left, An_con_left = self.update_by_single_lead(
            g_lead_left, V_left, g_lattice, dos_lattice, An_lattice)
        # update the GF of the lattice connected to the left lead
        # by the right lead
        g_con_both, dos_con_both, _ = self.update_by_single_lead(
            g_lead_right, V_right, g_con_left, dos_con_left, An_con_left)

        return g_con_both, dos_con_both

    def get_conductance(self) -> Tuple[List[complex], List[complex],
                                       List[float]]:
        """
        Calculate the conductance of the lattice.
        """
        # holder lists
        gf_list = []
        dos_list = []
        conductance_list = []
        # loop over energy values
        for energy in self.ENERGY_ARRAY:
            # get the GF of the disconnected system
            g_lat_dis, dos_lat_dis, An_lat_dis = self.RGF(energy)
            # get the broadening matrices
            gamma_mat = self.get_gamma_matrix(energy)
            # get the values of the connected system
            g_con, dos_con = self.update_by_leads(
                energy, g_lat_dis, dos_lat_dis, An_lat_dis)
            # save new GF and DOS
            gf_list.append(g_con)
            dos_list.append(dos_con)
            # calucluate conductance
            g_ret = g_con[0, 1]
            g_adv = np.conj(g_con[1, 0])
            conductance = self.QUANTUM_CONDUCTANCE * \
                gamma_mat[0, 0] * g_ret * gamma_mat[1, 1] * g_adv
            conductance_list.append(conductance)

        return conductance_list, gf_list, dos_list

    def plot_conductance(self, conduct_list: List[complex],
                         dos_list: List[float]) -> None:
        """
        Plot the conductance and dos. 
        """
        fig, ax = plt.subplots(ncols=2, figsize=(15, 5))

        ax[0].plot(self.ENERGY_ARRAY, dos_list)
        ax[0].set_ylabel('DOS')
        ax[0].set_xlabel('Energy')
        ax[0].set_title('Density of States')

        ax[1].plot(self.ENERGY_ARRAY, np.real(conduct_list))
        ax[1].set_ylabel('Conductance (S)')
        ax[1].set_xlabel('Energy')
        ax[1].set_title('Conductance')

        fig.suptitle(
            f'Generation {self.iteration} of Type {self.lattice_type} Lattice'
        )
        plt.savefig(f'./figures/cond_{self.lattice_type}{self.iteration}')
        plt.show()


def main() -> None:
    cd = LatticeConductance('C', 3)
    conductance, _, dos = cd.get_conductance()
    cd.plot_conductance(conductance, abs(np.array(dos)))


if __name__ == '__main__':
    main()
