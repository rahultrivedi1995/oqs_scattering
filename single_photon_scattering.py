"""Module to implement single-photon scattering from a modulated system."""
import numpy as np
from typing import List

import floquet_analysis
import modulated_quantum_system as mqs


def compute_single_ph_gfunc(
        obj: floquet_analysis.FloquetAnalyzer,
        freqs: np.array) -> np.ndarray:
    """Computes the single-photon scattering matrix for modulated systems.

    Args:
        system: The system under consideration.
        freqs: The frequencies under consideration.
        num_bands: The number of side bands to compute the scattering matrix
            at.
        num_dt: Number of time-steps within one period to use for the
            computation of the floquet states.

    Returns:
        The single-photon Green's function as a function of frequency for the
        various sidebands.
    """
    # Compute the decay operator and the convolved excitation operator within
    # the single-excitation subspace.
    decay_op = obj.decay_op(1)
    ex_op_conv = obj.ex_op_conv(1, freqs)
    gfunc_time = np.matmul(
            decay_op[np.newaxis, :, :, :], ex_op_conv)[:, :, 0, 0]
    return gfunc_time, np.fft.fftshift(np.fft.ifft(gfunc_time, axis=1), axes=1)


def compute_single_ph_gfunc_adiabatically(
        sys: mqs.ModulatedQuantumSystem,
        freqs: np.ndarray,
        num_dt: float) -> np.ndarray:
    """Computes the single-photon scattering matrix in the adiabatic limit.

    This computes the Green's function by simply averaging over the Green's
    function computed with the instantaneous operators over one period.

    Args:
        obj: The floquet analyzer object corresponding to the system under
            consideration.
        freqs: The frequencies at which to compute the Green's function.
        dt: The time-step to use while averaging the transmission through the
            modulated Hamiltonian.

    Returns:
        The single-photon Green's function as a function of the input frequency
        in the adiabatic approximation.
    """
    # We first diagonalize the effective Hamiltonian within the
    # single-excitatoin subspace as a function of time.
    eig_vals = []
    eig_vecs = []
    eig_vecs_inv = []
    for tstep in range(num_dt):
        t = tstep * sys.period / (num_dt - 1)
        eig_val, eig_vec = np.linalg.eig(sys.compute_hamiltonian(t, 1))
        eig_vals.append(eig_val)
        eig_vecs.append(eig_vec)
        eig_vecs_inv.append(np.linalg.inv(eig_vec))

    # Setup the time-independent coupling operator.
    decay_op = sys.compute_decay_op(1)
    ex_op = decay_op.conj().T

    # Calculate the transmission as a function of time.
    trans = []
    for eig_val, eig_vec, eig_vec_inv in zip(
            eig_vals, eig_vecs, eig_vecs_inv):
        # Compute the left and right vectors.
        vec_left = (decay_op @ eig_vec_inv)[0]
        vec_right = (eig_vec @ ex_op)[:, 0]
        # Compute the transmission at all frequencies.
        lor = 1 / (freqs[:, np.newaxis] - eig_val[np.newaxis, :])
        trans_amp = np.sum(vec_left * (lor * vec_right), axis=1)
        trans.append(np.abs(trans_amp)**2)

    return np.mean(trans, axis=0)



