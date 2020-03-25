import functools
import numpy as np
import scipy
from typing import Callable, Tuple

import modulated_quantum_system as mqs


def compute_floquet_states(
        hamiltonian: Callable[[float], np.ndarray],
        period: float,
        num_dt: int) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the floquet states of the given hamiltonian.

    Args:
        system: The object describing the system being considered.
        num_ex: The number of excitations in the subspace within which to
            perform the Floquet analysis.
        num_dt: The number of time-steps to use within one period of the system
            Hamitlonian for the floquet decomposition.

    Returns:
        The floquet eigenvalues and the floquet spectrum of the given
        Hamiltonian.
    """
    # Time step to use for computation of the propagator.
    dt = period / num_dt

    # Compute the propagator for each time-step within the period. We do this
    # once and store them.
    prop_tsteps = []
    for k in range(num_dt - 1):
        # Current hamiltonian. Since we are computing the propagator that maps
        # the state at `k * dt` to `(k + 1) * dt`, we use the hamiltonian at
        # `(k + 0.5) * dt` for the propagator.
        H = hamiltonian((k + 0.5) * dt)
        # Compute the propagator corresponding to this hamiltonian.
        prop_dt = scipy.linalg.expm(-1.0j * dt * H)
        prop_tsteps.append(prop_dt)

    # Compute the propagator for one period of the system.
    prop = np.eye(prop_tsteps[0].shape[0], dtype=complex)
    for prop_dt in prop_tsteps:
        # Cascade the propagator.
        prop = prop_dt @ prop
        prop_dt_eigvals, prop_dt_eigvecs = np.linalg.eig(prop_dt)

    # We now diagonalize the propagator. It is assumed that the propagator has
    # eigenvectors that are orthogonal to each other since it is quite hard to
    # find parameter regimes where this is not the case.
    prop_eigvals, prop_eigvecs = np.linalg.eig(prop)

    # Compute the floquet eigenvalues.
    floquet_eigvals_decay = -np.log(np.abs(prop_eigvals)) / period
    floquet_eigvals_freq = -np.arctan2(
            np.imag(prop_eigvals), np.real(prop_eigvals)) / period
    floquet_eigfreqs = floquet_eigvals_freq - 1.0j * floquet_eigvals_decay

    # Compute the floquet eigenvectors.
    floquet_eigvecs = [prop_eigvecs]
    prop_eigvals_dt = (prop_eigvals[np.newaxis, :])**(1.0 / (num_dt - 1))
    for prop_dt in prop_tsteps:
        # Propagate the eigenvector of the full propagator within a period.
        # At the same time, also remove the time-evolution due to the floquet
        # eigenvalues.
        propagated_eigvec = (prop_dt @ floquet_eigvecs[-1]) / prop_eigvals_dt
        floquet_eigvecs.append(propagated_eigvec)

    return floquet_eigfreqs, np.array(floquet_eigvecs)


class FloquetAnalyzer:
    """Object for setting up various operators useful for floquet analysis."""
    def __init__(
            self,
            system: mqs.ModulatedQuantumSystem,
            num_floquet_bands: int) -> None:
        """Creates a new `FloquetAnalyzer` object.

        Args:
            system: The system under consideration.
            num_floquet_bands: The number of floquet bands to use for the
                analysis.
        """
        self._system = system
        self._num_floquet_bands = num_floquet_bands

        # Setup the cache for the eigenvectors and eigenvalues. We use a simple
        # dictionary that has as keys the number of excitations being use. Each
        # element of this dictionary is a tuple with the first element being the
        # floquet values and floquet eigenvectors.
        self._floquet_decomp = {}

        # Setup the cache for the decay operator within the single-excitation
        # subspace. This is a tuple of two arrays, the first being the decay
        # operator within the single excitation subspace in the frame of the
        # floquet states, and the second being its adjoint.
        self._decay_ops = {}
        self._ex_ops = {}

        # Setup the cache for the convolved excitation operator.
        self._ex_ops_conv = {}

        # Caching tolerance.
        self._cache_tol = 1e-5

    @property
    def num_floquet_bands(self) -> int:
        """Number of floquet bands used for the simulation."""
        return self._num_floquet_bands

    @property
    def num_dt(self) -> int:
        """Returns the number of time-steps to use."""
        return 2 * self._num_floquet_bands + 1

    @property
    def dt(self) -> int:
        """Returns the mesh size for coarse graining the time axis."""
        return self._system.period / self.num_dt

    @property
    def system(self) -> mqs.ModulatedQuantumSystem:
        """Returns the system that the floquet object is analyzer."""
        return self._system

    def floquet_decomposition(
            self, num_ex: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes the floquet decomposition for the given excitation subspace.

        Args:
            num_ex: The excitation number subspace that is to be considered.

        Returns:
            The floquet decomposition as a tuple of an array of floquet
            eigenvalues, floquet eigenvectors and the inverse of floquet
            eigenvectors.
        """
        if not num_ex in self._floquet_decomp.keys():
            # Calculate the floquet eigenvalues and eigenvectors.
            eigvals, eigvecs = compute_floquet_states(
                    functools.partial(self._system.compute_hamiltonian,
                                      num_ex=num_ex),
                    period=self._system.period,
                    num_dt=self.num_dt)
            # Calculate the inverse of the eigenvectors.
            eigvecs_inv = np.array([np.linalg.inv(e) for e in eigvecs])
            # Cache the floquet decomposition.
            self._floquet_decomp[num_ex] = (eigvals, eigvecs, eigvecs_inv)

        return self._floquet_decomp[num_ex]

    def decay_op(self, num_ex) -> np.ndarray:
        """Compute the decay operator as a function of time.""" 
        if not num_ex in self._decay_ops.keys():
            # Setup the two transforming matrices to sandwich the decay operator
            # in between.
            _, mat_right, _ = self.floquet_decomposition(num_ex)
            if num_ex > 1:
                _, _, mat_left = self.floquet_decomposition(num_ex - 1)
            else:
                mat_left = np.array([[1]])
            # Calculate the decay operator.
            decay_op = self._system.compute_decay_op(num_ex)
            self._decay_ops[num_ex] = mat_left @ decay_op @ mat_right

        return self._decay_ops[num_ex]

    def ex_op(self, num_ex) -> np.ndarray:
        """Compute the excitation operator as a function of time.""" 
        if not num_ex in self._ex_ops.keys():
            # Setup the two transforming matrices to sandwich the decay operator
            # in between.
            _, _, mat_left = self.floquet_decomposition(num_ex)
            if num_ex > 1:
                _, mat_right, _ = self.floquet_decomposition(num_ex - 1)
            else:
                mat_right = np.array([[1]])
            # Calculate the decay operator.
            ex_op = self._system.compute_decay_op(num_ex).T.conj()
            self._ex_ops[num_ex] = mat_left @ ex_op @ mat_right

        return self._ex_ops[num_ex]

    def ex_op_conv(
            self, num_ex: int, freqs: np.ndarray) -> np.ndarray:
        """Computes the convolved decay operator."""
        # Check if the computation has been performed earlier.
        if num_ex in self._ex_ops_conv.keys():
            # Check if the input frequencies match.
            if freqs.size == self._ex_ops_conv[num_ex][0].size:
                error = np.linalg.norm(self._ex_ops_conv[num_ex][0] - freqs)
                if error < self._cache_tol:
                    return self._ex_ops_conv[num_ex][1]
        # Consider the full computation and caching. Compute the fft of the
        # excitation operators.
        ex_op = self.ex_op(num_ex)
        ex_op_fft = np.fft.fftshift(
                np.fft.fft(ex_op, axis=0), axes=0) / self.num_dt
        # Multiply the above vector with the convolving lorentzian.
        harms = (np.arange(-self._num_floquet_bands,
                           self._num_floquet_bands + 1)
                ) * (2 * np.pi / self._system.period)
        eigvals, _, _ = self.floquet_decomposition(num_ex)
        lor = 1 / (1.0j * harms[np.newaxis, :, np.newaxis] +
                   1.0j * eigvals[np.newaxis, np.newaxis, :] -
                   1.0j * freqs[:, np.newaxis, np.newaxis])
        ex_op_conv_fft = (ex_op_fft[np.newaxis, :, :, :] *
                          lor[:, :, :, np.newaxis])
        # Inverse FFT the resulting vector. Note that while performing inverse
        # fft, we first perform a fft shift to shift the vector back into the
        # "digital" format.
        ex_op_conv = np.fft.ifft(np.fft.ifftshift(
                    ex_op_conv_fft, axes=1), axis=1) * self.num_dt
        # Cache the resulting vector.
        self._ex_ops_conv[num_ex] = (freqs, ex_op_conv)
        return ex_op_conv
