"""Module to implement two-photon scattering from a modulated system."""
import numpy as np
import scipy.interpolate
from typing import List, Tuple

import floquet_analysis
import modulated_quantum_system as mqs
import single_photon_scattering


def _map_to_time_diffs(
        g2: np.ndarray,
        num_periods: int,
        num_dt: int) -> np.ndarray:
    """Map the correlation function to a function of initial time and delay.

    It is more convenient to compute the two-photon correlation in terms of the
    initial and final time from an implementation standpoint. However, the
    result of the computation is more interpretable if the expressed in terms
    of the initial time and the delay between the initial and final time.
    This function implements such a transformation. It is assumed that the
    times that are considered are on a uniformly sampled grid with `num_dt`
    time steps per period and `num_periods` periods. The given `g2` is
    assumed to be computed within one period for the initial time instant and
    `num_periods` final time instant.

    Args:
        g2: The two-photon correlation function. It is assumed that this is
            a 3D array, with the first index corresponding to the frequency at
            which it is computed, the second index corresponding to the final
            time instant and the third index corresponding to the initial time
            instant. It is assumed that the intial time instant lies in between
            `0` and one period, while the final time instant is varied over
            `num_periods` periods.
        num_periods: The number of periods for which the `g2` function is
            computed.
        num_dt: The number of time-steps within one period.

    Returns:
        The `g2` function as a 3D array where the first index corresponds to
        frequency, the second index corresponds to the initial time instant
        and the third index corresponds to the delay.
    """
    # Setup grids for the initial time and the time-differences.
    init_times = np.arange(num_dt)
    delays = np.arange(0, num_dt * (num_periods - 1))
    init_times_grid, delays_grid = np.meshgrid(
            init_times, delays, indexing="ij")
    final_times_grid = init_times_grid + delays_grid
    # Map the correlation function as a function of initial time instant and
    # the delay.
    g2_init_delay = g2[:, final_times_grid, init_times_grid]
    # Now downsample the delay axis by a factor of 2. This is to ensure that
    # the transformation to the central time is easy to perform.
    g2_ds = g2_init_delay[:, :, 0::2]
    # Finally, transform to the central time and the delay.
    delays_ds = delays[0::2]
    cen_times = np.arange(0, num_dt)
    cen_times_grid, delays_ds_grid = np.meshgrid(
            cen_times, delays_ds, indexing="ij")
    # Note that since `g2` is a periodic function of the initial time, we can
    # compute it modulo `num_dt` while setting up the map from `g2_ds` to the
    # final result.
    init_times_grid = (cen_times_grid - delays_ds_grid // 2) % num_dt
    return g2_ds[:, init_times_grid, delays_ds_grid // 2]


def _compute_two_ph_output_single_ex_uncorr(
        obj: floquet_analysis.FloquetAnalyzer,
        freqs: np.ndarray,
        num_periods: int) -> np.ndarray:
    """Calculate the two-photon output state without the correlations."""
    # You at least need more than one period.
    if num_periods <= 1:
        raise ValueError("Expected number of periods to be more than 1")
    # Compute the decay operator and the convolved excitation operator applied
    # on the ground state (since there is only one ground state, this is
    # equivalent to accessing the element corresponding to the ground state).
    decay_op = obj.decay_op(1)[:, 0, :]
    ex_op_conv = obj.ex_op_conv(1, freqs)[:, :, :, 0]

    # Construct the decay operator over all the periods for which we want to
    # compute the two-photon correlation function. Note that we introduce one
    # axis corresponding to the frequency, and one axis corresponding to the
    # second time index.
    decay_op_t1 = np.tile(decay_op[np.newaxis, :, np.newaxis, :],
                          (1, num_periods, 1, 1))
    decay_op_t2 = decay_op[np.newaxis, np.newaxis, :, :]
    ex_op_conv_t1 = np.tile(ex_op_conv[:, :, np.newaxis, :],
                            (1, num_periods, 1, 1))
    ex_op_conv_t2 = ex_op_conv[:, np.newaxis, :, :]
    # Calculate the linear term. This term corresponds to the uncorrelated part
    # of the two-photon wavefunctions.
    psi_out = (np.sum(decay_op_t1 * ex_op_conv_t1, axis=3) *
               np.sum(decay_op_t2 * ex_op_conv_t2, axis=3))
    return _map_to_time_diffs(psi_out, num_periods, obj.num_dt) / np.pi


def _compute_two_ph_output_single_ex_corr(
        obj: floquet_analysis.FloquetAnalyzer,
        freqs: np.ndarray,
        num_periods: int) -> np.ndarray:
    """Calculate the two-photon output state under cw excitation.

    This function considers the problem of exciting a modulated quantum system
    with two continuous wave photons at the given frequencies, and computes
    the output state. (TODO: This function ignores the second-excitation
    subspace, and will thus provide perfect photon blockade).

    Args:
        obj: The floquet analyzer object corresponding to the system being
            considered.
        freqs: The input frequencies at which to compute the two-photon
            correlation at.
        num_periods: The number of periods at which to compute the two-photon
            correlation. The number of periods determine what the value of the
            maximum final time instant can be.

    Returns:
        The two-photon correlation as a function of the initial time-instant
        and the time delay. Note that the initial time-instant is varied over
        only a single period while the delay is varied from `0` to
        `(num_periods - 1) * period`.
    """
    # You at least need more than one period.
    if num_periods <= 1:
        raise ValueError("Expected number of periods to be more than 1")
    # Compute the decay operator and the convolved excitation operator applied
    # on the ground state (since there is only one ground state, this is
    # equivalent to accessing the element corresponding to the ground state).
    decay_op = obj.decay_op(1)[:, 0, :]
    ex_op_conv = obj.ex_op_conv(1, freqs)[:, :, :, 0]

    # Construct the decay operator over all the periods for which we want to
    # compute the two-photon correlation function. Note that we introduce one
    # axis corresponding to the frequency, and one axis corresponding to the
    # second time index.
    decay_op_t1 = np.tile(decay_op[np.newaxis, :, np.newaxis, :],
                          (1, num_periods, 1, 1))
    decay_op_t2 = decay_op[np.newaxis, np.newaxis, :, :]
    ex_op_conv_t1 = np.tile(ex_op_conv[:, :, np.newaxis, :],
                            (1, num_periods, 1, 1))
    ex_op_conv_t2 = ex_op_conv[:, np.newaxis, :, :]

    # Calculate the nonlinear exponentially decaying term. For this, we need
    # the floquet eigenvalues within the single-excitation subspace.
    eigvals, _, _ = obj.floquet_decomposition(1)
    # We also need to setup grids of `t1` and `t2`.
    t2 = np.arange(obj.num_dt) * obj.dt
    t1 = np.arange(obj.num_dt * num_periods) * obj.dt
    t1_grid, t2_grid = np.meshgrid(t1, t2, indexing="ij")
    # Calculate the delay on the grid. We introduce two addtional axes here, one
    # for the number of eigenvalues to multiply this with, and the second for
    # the number of input frequencies.
    delay_grid = (t1_grid - t2_grid)[np.newaxis, :, :, np.newaxis]
    # Reshape the `eigvals` into a 4D array - we introduced three new axis here.
    # The first axis is for the input frequency, the second one is for the
    # initial time instant and the third one is for the final time instant.
    eigvals = eigvals[np.newaxis, np.newaxis, np.newaxis, :]
    # Reshape the `freqs` into a 4D array again by introducing three
    # additional new axis.
    input_freqs = freqs[:, np.newaxis, np.newaxis, np.newaxis]
    # Now calculate the decaying exponential on the grid. This exponential has
    # a contribution from the single-excitation floquet eigenvalues and the
    # input frequencies:
    #       exp(-1.0j * eigvals * (t1 - t2)) * exp(1.0j * nu * (t1 - t2))
    exp = np.exp(1.0j * (input_freqs - eigvals) * delay_grid)
    # Finally, compute the nonlinear contribution to the two-photon correlation.
    psi_corr = -(np.sum(decay_op_t2 * ex_op_conv_t2, axis=3) *
                 np.sum(decay_op_t1 * exp * ex_op_conv_t2, axis=3))

    return _map_to_time_diffs(psi_corr, num_periods, obj.num_dt) / np.pi


def _compute_second_to_first_decay_op(
        obj: floquet_analysis.FloquetAnalyzer,
        freqs: np.ndarray) -> np.ndarray:
    """Computes the operator describing decay from second to first excitation.

    Args:
        obj: The system being considered.
        freqs: The frequencies at which to compute the operator.

    Returns:
        The operator as a numpy array.
    """
    # Compute the decay operator and the convolved excitation operator applied
    # on the ground state (since there is only one ground state, this is
    # equivalent to accessing the element corresponding to the ground state).
    decay_op_1ex = obj.decay_op(1)
    ex_op_conv_1ex = obj.ex_op_conv(1, freqs)
    # Also need the decay operators for the two-excitation subspace.
    ex_op_2ex = obj.ex_op(2)
    decay_op_2ex = obj.decay_op(2)
    # Calculate the multiplication of the convolved excitation operator and the
    # excitation operator.
    mult = np.matmul(ex_op_2ex[np.newaxis, :, :, :], ex_op_conv_1ex)[:, :, :, 0]
    # Compute its fourier transform.
    mult_fft = np.fft.fftshift(np.fft.fft(mult, axis=1), axes=1) / obj.num_dt
    # Multiply it by a lorentzian corresponding to the second excitation
    # subspace.
    eigvals_2ex, _, _ = obj.floquet_decomposition(2)
    mod_freq = 2 * np.pi / obj._system.period
    harms = np.arange(-obj.num_floquet_bands,
                      obj.num_floquet_bands + 1) * mod_freq
    lor_2ex = 1.0 / (1.0j * harms[np.newaxis, :, np.newaxis] -
                     2.0j * freqs[:, np.newaxis, np.newaxis] +
                     1.0j * eigvals_2ex[np.newaxis, np.newaxis, :])
    # Perform the inverse FFT of the result of the product of this lorentzian
    # and the operator multiplication computed above.
    mult_lor_ifft = np.fft.ifft(np.fft.ifftshift(
                lor_2ex * mult_fft, axes=1), axis=1) * obj.num_dt
    # Multiply the result by the decay operator going from the second to the
    # first excitation subspace.
    mult_lor_ifft_decay = np.matmul(decay_op_2ex[np.newaxis, :, :, :],
                                    mult_lor_ifft[:, :, :, np.newaxis])
    return mult_lor_ifft_decay


def _compute_two_ph_output_two_ex(
        obj: floquet_analysis.FloquetAnalyzer,
        freqs: np.ndarray,
        num_periods: int) -> np.ndarray:
    """Computes the two-photon output state due to the two-photon subspace.

    Args:
        obj: The floquet analyzer object corresponding to the system being
            considered.
        freqs: The input frequencies at which to compute the two-photon
            correlation at.
        num_periods: The number of periods at which to compute the two-photon
            correlation. The number of periods determine what the value of the
            maximum final time instant can be.

    Returns:
        The two-photon correlation as a function of the initial time-instant
        and the time delay. Note that the initial time-instant is varied over
        only a single period while the delay is varied from `0` to
        `(num_periods - 1) * period`.
    """
    if num_periods <= 1:
        raise ValueError("Expected `num_periods` to be more than 1.")
    evals_2ex, evec_2ex, inv_evec_2ex = obj.floquet_decomposition(1)

    # Compute the decay operator and the convolved excitation operator applied
    # on the ground state (since there is only one ground state, this is
    # equivalent to accessing the element corresponding to the ground state).
    decay_op_1ex = obj.decay_op(1)
    mult_lor_ifft_decay = _compute_second_to_first_decay_op(obj, freqs)

    # Finally, we latch on the time-dependent factors. Here, we first setup
    # grids in terms of the final time instant `t1` and initial time instant
    # `t2`, where we allow `t2` to only vary over a single period and `t1` to
    # vary over all the required periods.
    t2 = np.arange(obj.num_dt) * obj.dt
    t1 = np.arange(obj.num_dt * num_periods) * obj.dt
    t1_grid, t2_grid = np.meshgrid(t1, t2, indexing="ij")
    # Calculate the delay on the grid. We introduce two addtional axes here, one
    # for the number of eigenvalues to multiply this with, and the second for
    # the number of input frequencies.
    delay_grid = (t1_grid - t2_grid)[np.newaxis, :, :, np.newaxis]
    # Calculate the eigenvalues from the single-excitation subspace.
    eigvals_1ex, _, _ = obj.floquet_decomposition(1)
    # Reshape the `eigvals` into a 4D array - we introduced three new axis here.
    # The first axis is for the input frequency, the second one is for the
    # initial time instant and the third one is for the final time instant.
    eigvals_1ex = eigvals_1ex[np.newaxis, np.newaxis, np.newaxis, :]
    # Reshape the `freqs` into a 4D array again by introducing three
    # additional new axis.
    input_freqs = freqs[:, np.newaxis, np.newaxis, np.newaxis]
    # Now calculate the decaying exponential on the grid. This exponential has
    # a contribution from the single-excitation floquet eigenvalues and the
    # input frequencies:
    #       exp(-1.0j * eigvals * (t1 - t2)) * exp(1.0j * nu * (t1 - t2))
    exp = np.exp(1.0j * (input_freqs - eigvals_1ex) * delay_grid)
    # Finally get the single-excitation operator over `t1`.
    decay_op_1ex_t1 = np.tile(decay_op_1ex[np.newaxis, :, np.newaxis, 0, :],
                              (1, num_periods, 1, 1))
    # Calculate the output state.
    psi_out = np.sum(decay_op_1ex_t1 *
                     exp * mult_lor_ifft_decay[:, np.newaxis, :, :, 0],
                     axis=3)
    # We finally map the output state as a function of the initial time index
    # and the delay in between them.
    return _map_to_time_diffs(psi_out, num_periods, obj.num_dt) / np.pi


def compute_two_ph_output(
        obj: floquet_analysis.FloquetAnalyzer,
        freqs: np.ndarray,
        num_periods: int) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the two-photon output state due to the two-photon subspace.

    Args:
        obj: The floquet analyzer object corresponding to the system being
            considered.
        freqs: The input frequencies at which to compute the two-photon
            correlation at.
        num_periods: The number of periods at which to compute the two-photon
            correlation. The number of periods determine what the value of the
            maximum final time instant can be.

    Returns:
        The two-photon output state under excitation by two photons at the
        same frequency.
    """
    if obj.system.subspace_dim(2) == 0:
        return (_compute_two_ph_output_single_ex_uncorr(obj, freqs, num_periods),
                _compute_two_ph_output_single_ex_corr(obj, freqs, num_periods))
    else:
        return (_compute_two_ph_output_single_ex_uncorr(obj, freqs, num_periods),
                _compute_two_ph_output_single_ex_corr(obj, freqs, num_periods) +
                _compute_two_ph_output_two_ex(obj, freqs, num_periods))


def compute_connected_smat(
        obj: floquet_analysis.FloquetAnalyzer,
        input_freqs: np.ndarray,
        out_deltas: np.ndarray,
        harmonics: List[int],
        num_periods: int = 10) -> List[np.ndarray]:
    """Compute the connected part of the scattering matrix.

    Args:
        obj: The Floquet object under consideration.
        input_freqs: The input frequencies at which to perform this compuation.
        deltas: The frequency gap between the output frequencies at which to
            compute the connected part.
        harmonics: The harmonics at which to compute the connected part of the
            scattering matrix.

    Returns:
        The connected part of the scattering matrix.
    """
    # Modulation frequency.
    freq = 2 * np.pi / obj.system.period

    # Compute the correlated part of the output time-domain wavefunction.
    _, psi_corr = compute_two_ph_output(obj, input_freqs, num_periods)
    # Expand the correlation function to include negative delays.
    psi_corr = np.concatenate((np.flip(psi_corr, axis=2), psi_corr[:, :, 1:]),
                              axis=2)

    # Compute the connected part scattering matrix in frequency domain.
    conn_smat = []
    for k in harmonics:
        # Compute the fourier series with respect to the central time instant.
        exp_k = np.exp(2.0j * np.pi * k * np.arange(obj.num_dt) / obj.num_dt)
        conn_smat_harm = np.sum(
                exp_k[np.newaxis, :, np.newaxis] * psi_corr, axis=1) / obj.num_dt
        # Compute the fourier transform with respect to the delay time.
        dtau = 2 * obj.dt
        delays = np.arange(-(psi_corr.shape[-1] // 2),
                           psi_corr.shape[-1] // 2 + 1) * dtau
        exp_delay = np.exp(1.0j * delays[:, np.newaxis] *
                           out_deltas[np.newaxis, :])
        conn_smat.append(conn_smat_harm @ exp_delay * dtau)

    return conn_smat


def compute_g2(obj: floquet_analysis.FloquetAnalyzer,
               freqs: np.ndarray,
               num_periods: int) -> np.ndarray:
    """Computes the normalized two-photon correlation function.

    This function computes the properly normalized two-photon correlation
    function only as a function of the time-delay.

    Args:
        obj: The system to be considered.
        freqs: The frequencies at which to compute the two-photon correlation
            function.
        num_periods: The number of periods to simulate in terms of the delay.

    Returns:
        The two-photon correlation as a function of the delay between the two
        time instants being considered and the frequencies of the input photon.
    """
    psi_out_uncorr, psi_out_corr = compute_two_ph_output(obj, freqs, num_periods)
    psi_out = psi_out_uncorr + psi_out_corr
    # Now, we average the square of the output state with respect to the intial
    # time-instant to get the unnormalized correlation function.
    two_ph_corr_unnorm = np.mean(np.abs(psi_out)**2, axis=1) * np.pi**2

    # Finally, we normalize against the total single-photon transmission in
    # steady state.
    _, gfunc = single_photon_scattering.compute_single_ph_gfunc(obj, freqs)
    tran = np.sum(np.abs(gfunc)**2, axis=1)
    return two_ph_corr_unnorm / (tran[:, np.newaxis])**2


def compute_equal_time_g2(
        obj: floquet_analysis.FloquetAnalyzer,
        freqs: np.ndarray,
        perform_average: bool) -> np.ndarray:
    """Computes the equal-time correlation function efficiently.

    This function should be used instead of `compute_g2` if it is desired to
    compute the equal-time correlation function at a large number of
    frequencies.

    Args:
        obj: The system to be considered.
        freqs: The frequencies at which to compute the equal time correlation
            function.

    Returns:
        The equal time correlation function as a function of frequency.
    """
    mult_lor_ifft_decay = _compute_second_to_first_decay_op(
            obj, freqs)[:, :, :, 0]
    # Decay operator from the second excitation subspace to first excitation
    # subspace.
    decay_op_1ex = obj.decay_op(1)[np.newaxis, :, 0, :]
    # Calculate the output wavefunction at equal time.
    psi_out = np.sum(mult_lor_ifft_decay * decay_op_1ex, axis=2)
    # Compute the normalized g2.
    # _, gfunc = single_photon_scattering.compute_single_ph_gfunc(obj, freqs)
    # tran = np.sum(np.abs(gfunc)**2, axis=1)
    tran = 1
    # Compute the equal time unnormalized g2.
    if perform_average:
        return np.mean(np.abs(psi_out)**2, axis=1) / tran**2
    else:
        return np.abs(psi_out)**2 / tran[:, np.newaxis]**2


def compute_equal_time_g2_adiabatically(
        sys: mqs.ModulatedQuantumSystem,
        freqs: np.ndarray,
        num_dt: float) -> np.ndarray:
    """Computes the equal time correlation with the adiabatic approximation.

    Args:
        sys: The system under consideration specified as a modulated quantum
            system.
        freqs: The frequencies at which to compute the equal-time correlation.
        num_dt: The number of time-steps to use in the time-averaging integral.
    """
    # Diagonalize the effective Hamiltonian within the single and two
    # excitation subspace as a function of time.
    eig_vals_1ex = []
    eig_vecs_1ex = []
    eig_vecs_inv_1ex = []
    eig_vals_2ex = []
    eig_vecs_2ex = []
    eig_vecs_inv_2ex = []
    for tstep in range(num_dt):
        t = tstep * sys.period / (num_dt - 1)
        eig_val_1ex, eig_vec_1ex = np.linalg.eig(sys.compute_hamiltonian(t, 1))
        eig_val_2ex, eig_vec_2ex = np.linalg.eig(sys.compute_hamiltonian(t, 2))
        eig_vals_1ex.append(eig_val_1ex)
        eig_vals_2ex.append(eig_val_2ex)
        eig_vecs_1ex.append(eig_vec_1ex)
        eig_vecs_2ex.append(eig_vec_2ex)
        eig_vecs_inv_1ex.append(np.linalg.inv(eig_vec_1ex))
        eig_vecs_inv_2ex.append(np.linalg.inv(eig_vec_2ex))

    # Compute the two-photon correlation as a function of time.
    decay_op_1 = sys.compute_decay_op(1)
    decay_op_2 = sys.compute_decay_op(2)
    ex_op_1 = decay_op_1.conj().T
    ex_op_2 = decay_op_2.conj().T
    G2_times = []
    for (eig_val_1ex, eig_val_2ex, eig_vec_1ex, eig_vec_2ex,
         eig_vec_inv_1ex, eig_vec_inv_2ex) in zip(
             eig_vals_1ex, eig_vals_2ex, eig_vecs_1ex, eig_vecs_2ex,
             eig_vecs_inv_1ex, eig_vecs_inv_2ex):
         vec_left = (decay_op_1 @ decay_op_2 @ eig_vec_inv_2ex).T
         vec_right = (eig_vec_1ex @ ex_op_1)
         op_middle = eig_vec_2ex @ ex_op_2 @ eig_vec_inv_1ex
         lor_2ex = 1 / (2 * freqs[np.newaxis, :] - eig_val_2ex[:, np.newaxis])
         lor_1ex = 1 / (freqs[np.newaxis, :] - eig_val_1ex[:, np.newaxis])
         G2_time_amp = np.sum(vec_left * (lor_2ex * (
                         op_middle @ (lor_1ex * vec_right))), axis=0)
         G2_times.append(np.abs(G2_time_amp)**2)

    return np.mean(G2_times, axis=0)


