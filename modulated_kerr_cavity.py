"""Module for implementing modulated Kerr system."""
import numpy as np

from typing import Tuple, Optional

import modulated_quantum_system as mqs
import pulse


class ModulatedKerrCavity(mqs.ModulatedQuantumSystem):
    """Implements a modulated Kerr cavity."""
    def __init__(self,
                 omega_c: float,
                 gamma: float,
                 chi: float,
                 delta_c: pulse.Pulse) -> None:
        self._omega_c = omega_c
        self._gamma = gamma
        self._chi = chi
        self._delta_c = delta_c

    @property
    def period(self) -> float:
        return self._delta_c.period

    def subspace_dim(self, num_ex: int) -> int:
        return 1

    def compute_decay_op(self, num_ex: int) -> int:
        return np.sqrt(num_ex * self._gamma) * np.array([[1]])

    def compute_hamiltonian(self, t: float, num_ex: int) -> np.ndarray:
        omega_c = self._omega_c + self._delta_c(t)
        return (num_ex * (omega_c - 0.5j * self._gamma) +
                self._chi * num_ex * (num_ex - 1)) * np.array([[1]])

    def get_qutip_operators(self, inp_freq: float, inp_amp: float) -> float:
        pass

