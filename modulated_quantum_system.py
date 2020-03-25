"""Module to define an interface for a modulated open quantum system."""
import abc
import numpy as np
import qutip

from typing import Tuple


class ModulatedQuantumSystem(metaclass=abc.ABCMeta):
    """Interface for a modulated open quantum system."""
    @property
    @abc.abstractmethod
    def period(self) -> float:
        """Returns the period of the modulated system."""
        raise NotImplementedError()

    @abc.abstractmethod
    def subspace_dim(self, num_ex: int) -> int:
        """Returns the dimensionality of each excitation subspace.

        Args:
            num_ex: The number of excitations of the subspace being considered.

        Returns:
            The dimensionality of the subspace being considered.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def compute_decay_op(self, num_ex: int) -> np.ndarray:
        """Compute the decay operator corresponding to the quantum system.

        Args:
            num_ex: The number of excitation to consider. Note that the operator
                constructed here maps the subspace with `num_ex` excitations to
                the subspace with `num_ex - 1` excitations.

        Returns:
            The decay operator as a numpy array.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def compute_hamiltonian(self, t: float, num_ex: int) -> np.ndarray:
        """Computes the hamiltonian of the modulated quantum system.

        Args:
            t: The time at which ot compute the hamiltonian.
            num_ex: The number of excitation in the excitation subspace being
                considered.

        Returns:
            The hamiltonian with the excitation subspace.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_qutip_operators(
            self,
            inp_freq: float,
            inp_amp: float) -> Tuple[qutip.Qobj, qutip.Qobj]:
        """Get the qutip hamiltonian and decay operator for the given system.

        This function is helpful in simulating single and two-photon
        scattering using qutip to match the scattering matrix methods against.

        Args:
            inp_freq: The input frequency of the applied continuous-wave
                laser drive.
            inp_amp: The amplitude of the applied continuous-wave laser drive.

        Returns:
            The qutip hamiltonian as well as the qutip decay operator needed
            for the system.
        """
        raise NotImplementedError()
