"""Module for implementing a modulated V-Level system."""
import numpy as np
import qutip

from typing import Tuple

import pulse
import modulated_quantum_system as mqs


class ModulatedVLevelSystem(mqs.ModulatedQuantumSystem):
    """Implements a class for the modulated V-Level system."""
    def __init__(self,
                 gammas: Tuple[float, float],
                 omegas: Tuple[pulse.Pulse, pulse.Pulse],
                 coup_const: pulse.Pulse,
                 period: float) -> None:
        """Creates a new ModulatedVLevelSystem object.

        Args:
            gammas: The decays associated with the transition from the excited
                states of the two transition to their ground states.
            omegas: The frequencies of the two excited states.
            coup_const: The coupling constant between the two excited states.
            period: The overall period of the hamiltonian.
        """
        self._gammas = gammas
        self._omegas = omegas
        self._coup_const = coup_const
        self._kappa = kappa
        self._period = period

    @property
    def period(self) -> float:
        return self._period

    @property
    def subspace_dim(self, num_ex: int) -> int:
        """Returns the dimensionality of each excitation subspace.

        Args:
            num_ex: The number of excitations in the subspace being considered.

        Returns:
            An integer corresponding to the dimensionality of the subspace
            being considered.
        """
        if num_ex == 1:
            return 2
        else:
            return 0

    def compute_decay_op(self, num_ex: int) -> np.ndarray:
        if num_ex == 1:
            return np.sqrt(np.array([[self._kappas[0], self._kappas[1]]]))
        else:
            raise ValueError("A V-Level system only contains one excitation.")

    def compute_hamiltonian(self, t: float, num_ex: int) -> np.ndarray:
        # Get the values of various coefficients in the Hamiltonian.
        omega_1 = self._omegas[0](t)
        omega_2 = self._omegas[1](t)
        kappa_1 = self._kappas[0](t)
        kappa_2 = self._kappas[1](t)
        g = self._coup_const(t)
        # Setup the hamiltonian.
        return np.array(
                [[omega_1 - 0.5j * kappa_1,
                  g - 0.5j * np.sqrt(kappa_1 * kappa_2)],
                 [np.conj(g) - 0.5j * np.sqrt(kappa_1 * kappa_2),
                  omega_2 - 0.5j * kappa_2]])

