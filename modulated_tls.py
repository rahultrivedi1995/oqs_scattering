"""Module for implementing a modulated two-level system."""
import numpy as np
import qutip

from typing import Tuple, Optional

import modulated_quantum_system as mqs
import pulse


class ModulatedTwoLevelSystem(mqs.ModulatedQuantumSystem):
    """Implements a modulated two-level system."""
    def __init__(
            self,
            omega_e: float,
            gamma: float,
            delta_e: pulse.Pulse) -> None:
        """Creates a new `ModulatedTwoLevelSystem` object.

        Args:
            omega_e: The frequency of the emitter.
            gamma: The decay rate of the emitter.
            delta_e: The modulation applied on the frequency of the emitter.
            period: The period of the hamiltonian. If not specified, it is
                assumed to be the period of `delta_e`.
        """
        self._omega_e = omega_e
        self._gamma = gamma
        self._delta_e = delta_e

    @property
    def period(self) -> float:
        return self._delta_e.period

    def subspace_dim(self, num_ex: int) -> int:
        if num_ex == 1:
            return 1
        else:
            return 0

    def compute_decay_op(self, num_ex: int) -> np.ndarray:
        if num_ex == 1:
            return np.sqrt(self._gamma) * np.array([[1]])
        else:
            raise ValueError("A two-level system only has single-excitation.")

    def compute_hamiltonian(self, t: float, num_ex: int) -> np.ndarray:
        if num_ex == 1:
            omega_e = self._omega_e + self._delta_e(t)
            return (omega_e - 0.5j * self._gamma) * np.array([[1]])
        else:
            raise ValueError("A two-level system only has single-excitation.")

    def get_qutip_operators(
            self,
            inp_freq: float,
            inp_amp: float):
        def _pulse_value(t: float, args) -> float:
            return self._delta_e(t)
        # Set the de-excitation operator for the two-level system.
        sigma = qutip.destroy(2)
        # The hamiltonian without any modulation.
        H_0 = (self._omega_e - inp_freq) * sigma.dag() * sigma
        H_drive = inp_amp * (sigma.dag() + sigma)
        # Setup the full time-dependent hamiltonian.
        H = [H_0 + H_drive, [sigma.dag() * sigma, _pulse_value]]
        return H, np.sqrt(self._gamma) * sigma
