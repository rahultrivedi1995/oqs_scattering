"""Module for implementing a modulated Jaynes-Cumming system."""
import numpy as np
import qutip

from typing import NamedTuple, Tuple

import pulse
import modulated_quantum_system as mqs


class ModulatedJCSystem(mqs.ModulatedQuantumSystem):
    """Implements a class for the modulated JC system."""
    def __init__(
            self,
            kappa: float,
            omega_e: pulse.Pulse,
            omega_c: pulse.Pulse,
            coup_const: pulse.Pulse,
            period: float) -> None:
        """Creates a new `ModulatedJCSystem` object.

        Args:
            omega_e: The frequency of the emitter.
            omega_c: The frequency of the cavity mode.
            coup_const: The coupling constant between the emitter and the
                cavity mode.
            delta_e: The modulation applied on the frequency of the emitter.
            delta_c: The modulation applied on the frequency of the cavity.
            period: The overall period of the hamiltonian.
        """
        self._omega_e = omega_e
        self._omega_c = omega_c
        self._coup_const = coup_const
        self._kappa = kappa
        self._period = period

    @property
    def period(self) -> float:
        """Returns the period of the modulated JC Hamiltonian."""
        return self._period

    def subspace_dim(self, num_ex: int) -> int:
        """Returns the dimensionality of the each excitation subspace.

        Args:
            num_ex: The number of excitations of the subspace being considered.

        Returns:
            The dimensionality of the considered subspace.
        """
        return 2

    def compute_decay_op(self, num_ex: int) -> np.ndarray:
        """Compute the decay operator corresponding to the JC system.

        Args:
            num_ex: The number of excitation to consider. Note that the operator
                constructed here maps the subspace with `num_ex` excitations to
                `num_ex - 1` excitations.

        Returns:
            The decay operator as a numpy array.
        """
        # Note that if `num_ex` is 1, then this operator is a row vector with
        # two elements, else it is a 2 x 2 matrix.
        if num_ex == 1:
            return np.sqrt(self._kappa) * np.array([[0, 1]], dtype=complex)
        else:
            return np.sqrt(self._kappa) * np.array(
                    [[np.sqrt(num_ex - 1), 0],
                     [0, np.sqrt(num_ex)]], dtype=complex)

    def compute_hamiltonian(self, t: float, num_ex: int) -> np.ndarray:
        """Computes the hamiltonian of the modulated JC system.

        Args:
            t: The time at which to evaluate the hamiltonian.
            num_ex: The number of excitation in the excitation subspace which to
                consider for the hamiltonian.

        Returns:
            The hamiltonian within the excitation subspace.
        """
        # Compute the decay operator.
        decay_op = self.compute_decay_op(num_ex)

        # Setup the hamiltonian. Within each excitation subspace, it is going to
        # be a `2 x 2` hamiltonian.
        H = np.zeros((2, 2), dtype=complex)
        H[0, 0] = (num_ex - 1) * self._omega_c(t) + self._omega_e(t)
        H[1, 1] = num_ex * self._omega_c(t)
        H[0, 1] = np.sqrt(num_ex) * self._coup_const(t)
        H[1, 0] = np.sqrt(num_ex) * np.conj(self._coup_const(t))

        return H - 0.5j * decay_op.T.conj() @ decay_op

    def get_qutip_operators(
            self,
            inp_freq: float,
            inp_amp: float) -> Tuple[qutip.Qobj, qutip.Qobj]:
        # Setup the operators for the cavity mode and the two-level system.
        # We note that since we are only interested in single and two-photon
        # transport, we will restrict the cavity fock state to two photons.
        sigma = qutip.tensor(qutip.destroy(2), qutip.qeye(3))
        a = qutip.tensor(qutip.qeye(2), qutip.destroy(3))

        # Helper functions for the detuning of the cavity and the emitter. This
        # is just to get around how qutip handles time-dependent hamiltonians.
        def _omega_c(t: float, args) -> float:
            return self._omega_c(t)
        def _omega_e(t: float, args) -> float:
            return self._omega_e(t)
        def _coup_const(t: float, args) -> complex:
            return self._coup_const(t)
        def _coup_const_conj(t: float, args) -> complex:
            return np.conj(self._coup_const(t))

        # Setup the Jaynes Cumming hamiltonian. We set it up in three different
        # parts. One is the static part, modulation on the emitter and
        # modulation on the cavity mode.
        H_0 = -inp_freq * (sigma.dag() * sigma + a.dag() * a)
        H_drive = inp_amp * (a.dag() + a)
        # Setup the full hamiltonian.
        H = [H_0 + H_drive,
             [sigma.dag() * sigma, _omega_e],
             [a.dag() * a, _omega_c],
             [a * sigma.dag(), _coup_const],
             [a.dag() * sigma, _coup_const_conj]]
        return H, np.sqrt(self._kappa) * a
