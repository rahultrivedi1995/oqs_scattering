"""Module to compute two-photon emission from modulated quantum system."""
import numpy as np
import scipy.linalg

import modulated_quantum_system as mqs
import pulse


def _get_decay_operator(
        sys: mqs.ModulatedQuantumSystem,
        num_ex: int) -> np.array:
    """Compute the decay operator for the system over all subspaces."""
    decay_op_blocks = [[None for _ in range(num_ex + 1)] for _ in range(num_ex + 1)]
    for n in range(1, num_ex + 1):
        decay_op_blocks[n - 1][n] = sys.compute_decay_op(n)

    return np.block(decay_op_blocks)

def _get_coherently_driven_eff_hamil(
        sys: mqs.ModulatedQuantumSystem,
        t: float,
        laser_pulse: pulse.Pulse,
        num_ex: int) -> np.array:
    """Compute the coherently driven effective Hamilitonian."""
    # Setup the Hamiltonian.
    hamil_blocks = [[None for _ in range(num_ex + 1)] for _ in range(num_ex + 1)]
    # Setup the ground state.
    hamil_blocks[0][0] = np.array([[0]])
    hamil_blocks[0][1] = sys.compute_decay_op(1) * np.conj(laser_pulse(t))
    # Setup all the excited state subspaces.
    for n in range(1, num_ex):
        hamil_n = (sys.compute_hamiltonian(t, n) -
                   0.5j * sys.compute_decay_op(n).T.conj() @ sys.compute_decay_op(n))
        hamil_blocks[n][n - 1] = sys.compute_decay_op(n).T.conj() * laser_pulse(t)
        hamil_blocks[n][n] = hamil_n
        hamil_blocks[n][n + 1] = sys.compute_decay_op(n + 1) * np.conj(laser_pulse(t))
    # The last excited state block.
    hamil_blocks[num_ex][num_ex] = (
            sys.compute_hamiltonian(t, num_ex))# -
            #0.5j * sys.compute_decay_op(num_ex).T.conj() @ sys.compute_decay_op(num_ex))
    hamil_blocks[num_ex][num_ex - 1] = (
            sys.compute_decay_op(num_ex).T.conj() * laser_pulse(t))

    return np.block(hamil_blocks)


def zero_photon_emission(
        sys: mqs.ModulatedQuantumSystem,
        times: np.array,
        laser_pulse: pulse.Pulse,
        num_ex: int) -> np.array:
  
    # Start by computing and storing the propagators for all times.
    prop_list = []
    for k in range(times.size - 1):
        t_mean = 0.5 * (times[k + 1] + times[k])
        dt = times[k + 1] - times[k]
        H = _get_coherently_driven_eff_hamil(sys, t_mean, laser_pulse, num_ex)
        prop_list.append(scipy.linalg.expm(-1.0j * H * dt))
    # Compute the cumulated propagator.
    cum_prop_list = [np.eye(prop_list[0].shape[0])]
    inv_cum_prop_list = [np.eye(prop_list[0].shape[0])]
    for prop in prop_list:
        cum_prop_list.append(prop @ cum_prop_list[-1])
        inv_cum_prop_list.append(np.linalg.inv(cum_prop_list[-1]))
    
    Ufinal = cum_prop_list[-1]
    
    return Ufinal[0][0]
    
    

def two_photon_emission(
        sys: mqs.ModulatedQuantumSystem,
        times: np.array,
        laser_pulse: pulse.Pulse,
        num_ex: int) -> np.array:
  
    # Start by computing and storing the propagators for all times.
    prop_list = []
    for k in range(times.size - 1):
        t_mean = 0.5 * (times[k + 1] + times[k])
        dt = times[k + 1] - times[k]
        H = _get_coherently_driven_eff_hamil(sys, t_mean, laser_pulse, num_ex)
        prop_list.append(scipy.linalg.expm(-1.0j * H * dt))
    # Compute the cumulated propagator.
    cum_prop_list = [np.eye(prop_list[0].shape[0])]
    inv_cum_prop_list = [np.eye(prop_list[0].shape[0])]
    for prop in prop_list:
        cum_prop_list.append(prop @ cum_prop_list[-1])
        inv_cum_prop_list.append(np.linalg.inv(cum_prop_list[-1]))

    # Compute the emission-time dependent term. Right now specialized to
    # two-level system.
    L = np.array([[0, sys.compute_decay_op(1)[0][0]], [0, 0]])
    UdagSigmaU = [(inv_cum_prop @ L @ cum_prop)
                   for inv_cum_prop, cum_prop in zip(inv_cum_prop_list, cum_prop_list)]
    
    #bra0Ufinal = cum_prop_list[-1][0][:]
    #ket0 = [1,0]
    #doubleTimeMat = np.einsum('abc,qcd->aqbd', UdagSigmaU, UdagSigmaU)
    #state = np.triu(np.einsum('abi,i->ab' ,np.einsum('i,abid->abd', ket0, doubleTimeMat), bra0Ufinal))
    #state = np.triu(np.einsum('abi,i->ab' ,np.einsum('i,abdi->abd', bra0Ufinal, doubleTimeMat), ket0))
    
    
    Ufinal = cum_prop_list[-1]
    
    gUfsigt2g = [(Ufinal @ UsU)[0][0]
                   for UsU in UdagSigmaU]
    
    gUfsigt2e = [(Ufinal @ UsU)[0][1]
                   for UsU in UdagSigmaU]
    
    gsigt1g = [UsU[0][0] for UsU in UdagSigmaU]
    
    esigt1g = [UsU[1][0] for UsU in UdagSigmaU]
    
    state = np.tril(np.outer(gUfsigt2g, gsigt1g) + np.outer(gUfsigt2e, esigt1g))
    
    return state + state.T - np.diag(np.diag(state))
    
    
def two_photon_emission_gradient(
        sys: mqs.ModulatedQuantumSystem,
        times: np.array,
        laser_pulse: pulse.Pulse,
        num_ex: int,
        objective: np.array) -> np.array:
    
    # compute gradient of output spectrum due to frequency shift at every time tprime
    
    # Start by computing and storing the propagators for all times.
    prop_list = []
    for k in range(times.size - 1):
        t_mean = 0.5 * (times[k + 1] + times[k])
        dt = times[k + 1] - times[k]
        H = _get_coherently_driven_eff_hamil(sys, t_mean, laser_pulse, num_ex)
        prop_list.append(scipy.linalg.expm(-1.0j * H * dt))
    # Compute the cumulated propagator.
    cum_prop_list = [np.eye(prop_list[0].shape[0])]
    inv_cum_prop_list = [np.eye(prop_list[0].shape[0])]
    for prop in prop_list:
        cum_prop_list.append(prop @ cum_prop_list[-1])
        inv_cum_prop_list.append(np.linalg.inv(cum_prop_list[-1]))

    # Compute the emission-time dependent term. Right now specialized to
    # two-level system.
    L = np.array([[0, sys.compute_decay_op(1)[0][0]], [0, 0]])
    UdagSigmaU = [(inv_cum_prop @ L @ cum_prop)
                   for inv_cum_prop, cum_prop in zip(inv_cum_prop_list, cum_prop_list)]
    
    Ufinal = cum_prop_list[-1]
    
    gradList = []
    
    delta = 1e-8
    deltaW = np.array([[0,0],[0,1]])*delta
    
    for tPrime in range(times.size-1):
        
        Heff = _get_coherently_driven_eff_hamil(sys, 0.5 * (times[tPrime + 1] + times[tPrime]), laser_pulse, num_ex)
        Udiv = (scipy.linalg.expm(-1.0j * (Heff+deltaW) * dt) - scipy.linalg.expm(-1.0j * (Heff-deltaW) * dt))/(2*delta)
        Uprime = inv_cum_prop_list[tPrime + 1] @ Udiv @ cum_prop_list[tPrime]
    
        gUfsigt2gRIGHT = [(Ufinal @ Uprime @ UsU)[0][0]
                       for UsU in UdagSigmaU[0:tPrime]]
        
        gUfsigt2gOTHER = [(Ufinal @ UsU)[0][0]
                       for UsU in UdagSigmaU[tPrime:]]

        gUfsigt2eRIGHT = [(Ufinal @ Uprime @ UsU)[0][1]
                       for UsU in UdagSigmaU[0:tPrime]]
        gUfsigt2eOTHER = [(Ufinal @ UsU)[0][1]
                       for UsU in UdagSigmaU[tPrime:]]

        gsigt1gRIGHT = [UsU[0][0] for UsU in UdagSigmaU[0:tPrime]]
        gsigt1gMIDDLE = [(Uprime @ UsU)[0][0] for UsU in UdagSigmaU[0:tPrime]]
        gsigt1gLEFT = [(UsU @ Uprime)[0][0] for UsU in UdagSigmaU[tPrime:]]

        esigt1gRIGHT = [UsU[1][0] for UsU in UdagSigmaU[0:tPrime]]
        esigt1gMIDDLE = [(Uprime @ UsU)[1][0] for UsU in UdagSigmaU[0:tPrime]]
        esigt1gLEFT = [(UsU @ Uprime)[1][0] for UsU in UdagSigmaU[tPrime:]]
        
        state = np.zeros((times.size, times.size), dtype = complex)
        
        state[0:tPrime, 0:tPrime] = np.outer(gUfsigt2gRIGHT, gsigt1gRIGHT) + np.outer(gUfsigt2eRIGHT, esigt1gRIGHT)  
        state[tPrime:, 0:tPrime] = np.outer(gUfsigt2gOTHER, gsigt1gMIDDLE) + np.outer(gUfsigt2eOTHER, esigt1gMIDDLE)
        state[tPrime:, tPrime:] = np.outer(gUfsigt2gOTHER, gsigt1gLEFT) + np.outer(gUfsigt2eOTHER, esigt1gLEFT)
        
        state = np.tril(state)
        
        state = state + state.T - np.diag(np.diag(state))
        
        gradList.append(np.sum(np.sum(np.abs(np.fft.fftshift(np.fft.fft2(state))) * objective)))
        
    gradList.append(0) #tfinal shouldn't affect anything

    return np.array(gradList)
    
    
    
    
    
    
    
    
    
    