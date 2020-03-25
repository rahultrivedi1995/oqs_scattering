"""Module to implement qutip calculations."""
import numpy as np
import qutip

import floquet_analysis
import modulated_quantum_system as mqs


def compute_single_ph_expectation(
        system: mqs.ModulatedQuantumSystem,
        inp_freqs: np.ndarray,
        inp_amp: float,
        num_periods: int,
        num_dt: int):
    # The time-mesh over which to solve the master equation.
    dt = system.period / num_dt
    times = np.arange(0, system.period * num_periods, dt)
    # Iterate over all the frequencies and compute the 
    mean_vals = []
    for freq in inp_freqs:
        # Get the hamiltonian and the decay operators.
        H, L = system.get_qutip_operators(freq, inp_amp)
        # Generate an initial state. Note that we randomly generate an initial
        # state and the expectation is that for system with a single ground
        # state, the initial state does not matter in steady state.
        psi_init_as_list = []
        for dim in H[0].dims[0]:
            psi_init_as_list.append(qutip.basis(dim))
        psi_init = qutip.tensor(psi_init_as_list)
        # Solve the master equation.
        out = qutip.mesolve(H, psi_init, times, c_ops=[L], e_ops=[L.dag() * L])
        mean_vals.append(out.expect[0][-num_dt:])

    return np.array(mean_vals) / inp_amp**2

def compute_two_ph_corr(
        obj: floquet_analysis.FloquetAnalyzer,
        num_periods: int,
        inp_freqs: np.ndarray,
        inp_amp: float) -> np.ndarray:
    # Get the time lists.
    init_times = np.arange(obj.num_dt) * obj.dt
    delays = np.arange(obj.num_dt * (num_periods - 1)) * obj.dt
    # Get the qutip operators.
    two_ph_corrs = []
    for freq in inp_freqs:
        H, L = obj._system.get_qutip_operators(freq, inp_amp)
        psi_init_as_list = []
        for dim in H[0].dims[0]:
            psi_init_as_list.append(qutip.basis(dim))
        psi_init = qutip.tensor(psi_init_as_list)
        two_ph_corrs.append(qutip.correlation_3op_2t(
                    H, psi_init, init_times, delays, [L], L, L, L))

    return two_ph_corrs

