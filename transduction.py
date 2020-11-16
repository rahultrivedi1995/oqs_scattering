import pickle
import numpy as np

from typing import Tuple, Optional

import floquet_analysis
import modulated_quantum_system as mqs
import pulse
import single_photon_scattering


class TransductionSystem(mqs.ModulatedQuantumSystem):
    """Implements a transduction system with lambda-level emitter ensembles."""
    def __init__(
            self,
            delta_mu: np.ndarray,
            delta_opt: np.ndarray,
            gamma_mu: float,
            gamma_opt: float,
            G_mu: float,
            G_opt: float,
            drive: pulse.Pulse) -> None:
        """Creates a new `TransductionSystem` object."""
        if delta_mu.size != delta_opt.size:
            raise ValueError("The number of microwave and optical transitions "
                             "are inconsistent.")
        else:
            self._num_em = delta_mu.size
        self._delta_mu = delta_mu
        self._delta_opt = delta_opt
        self._gamma_mu = gamma_mu
        self._gamma_opt = gamma_opt
        self._G_mu = G_mu
        self._G_opt = G_opt
        self._drive = drive

    def subspace_dim(self, num_ex: int) -> int:
        if num_ex == 1:
            return 2 * self._num_em
        else:
            raise NotImplementedError()

    def compute_decay_op(self, num_ex: int) -> np.ndarray:
        if num_ex == 1:
            op = np.zeros(2 * self._num_em, dtype=complex)
            op[1::2] = np.sqrt(self._gamma_opt)
            return op[np.newaxis, :]

        else:
            raise NotImplementedError()

    def compute_ex_op(self, num_ex: int) -> np.ndarray:
        if num_ex == 1:
            op = np.zeros(2 * self._num_em, dtype=complex)
            op[0::2] = np.sqrt(self._gamma_mu)
            return op[:, np.newaxis]
        else:
            raise NotImplementedError()

    def compute_hamiltonian(self, t: float, num_ex: int) -> np.ndarray:
        if num_ex == 1:
            # Indices corresponding to the microwave and optical excited
            # states in the single-excitation state vector.
            mu_inds = np.arange(0, 2 * self._num_em, 2)
            opt_inds = np.arange(1, 2 * self._num_em, 2)
            # Calculate the drive value.
            drive = self._drive(t)
            # Calculate the effective hamiltonian without accounting for the
            # collective microwave and optical channel.
            H = np.zeros((2 * self._num_em, 2 * self._num_em),
                         dtype=complex)
            H[mu_inds, mu_inds] = (self._delta_mu - 0.5j * self._G_mu)
            H[opt_inds, opt_inds] = (self._delta_opt - 0.5j * self._G_opt)
            H[mu_inds, opt_inds] = drive
            H[opt_inds, mu_inds] = drive
            # Excitation and decay operator into microwave and optical
            # channel.
            decay_op = self.compute_decay_op(1)
            ex_op = self.compute_ex_op(1)
            # Setup the total effective Hamiltonian.
            return H - 0.5j * (decay_op.T @ decay_op + ex_op @ ex_op.T)
        else:
            raise NotImplementedError()

    def get_qutip_operators(self, inp_freq: float, inp_amp: float):
        raise NotImplementedError()


class TransductionPulse(pulse.Pulse):
    """Implements the harmonic decomposition of the transducing pulse."""
    def __init__(
            self,
            num_har: int,
            fund_freq: float,
            coeffs: np.ndarray) -> None:
        """Creates a new `TransductionPulse` object."""
        if coeffs.size != (2 * num_har):
            raise ValueError("Coefficients inconsistent with the number of "
                             "harmonics.")

        self._fund_freq = fund_freq
        self._coeffs = coeffs
        self._num_har = num_har

    def __call__(self, t: float) -> float:
        ind = np.arange(self._num_har)
        harmonics = np.zeros(2 * self._num_har, dtype=complex)
        harmonics[0::2] = np.sin(ind * self._fund_freq * t)
        harmonics[1::2] = np.cos(ind * self._fund_freq * t)
        return np.sum(harmonics * self._coeffs)


if __name__ == "__main__":
    # Load the simulation parameters.
    with open("data/params_for_floq_transm.pkl", "rb") as f:
        params_data = pickle.load(f)[0]

    num_sb = 100
    drive = TransductionPulse(data["fund_freq"], data["coeffs"])
    period = 2 * np.pi / data["fund_freq"]
    sys = transduction.TransductionSystem(
            params_data["delta_mu"], params_data["delta_opt"],
            params_data["gamma"], params_data["gamma"],
            params_data["Gamma"], params_data["Gamma"], drive)
    obj = floquet_analysis.FloquetAnalyzer(sys, period, num_sb)
    _, gfuncs = single_photon_scattering.compute_single_ph_gfunc(
            obj, 2 * np.pi * params_data["freqs"])
    trans = np.sqrt(np.sum(np.abs(gfuncs)**2, axis=1))
    with open("data/floq_transm.pkl", "wb") as f:
        pickle.dump(trans)
