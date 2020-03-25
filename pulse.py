import abc
import numpy as np

from typing import List


class Pulse(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def period(self) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def __call__(self, t: float) -> float:
        raise NotImplementedError()


class SinusoidalPulse(Pulse):
    """Implements a sinusoidal modulation."""
    def __init__(self,
                 mean: float,
                 amp: float,
                 phase: float,
                 freq: float) -> None:
        """Creates a new `SinusoidalPulse` object.

        Args:
            mean: The mean value of the pulse.
            amp: The amplitude of the sinusoidal pulse.
            phase: The phase of the sinusoidal pulse.
            freq: The frequency of the sinusoidal pulse.
        """
        self._mean = mean
        self._amp = amp
        self._phase = phase
        self._freq = freq

    @property
    def period(self) -> float:
        return 2 * np.pi / self._freq

    def __call__(self, t: float) -> float:
        return self._mean + self._amp * np.sin(self._freq * t + self._phase)

class MultiSinusoidalPulse(Pulse):
    """Implements a modulation that is sum of many sinusoids."""
    def __init__(self,
                 amps: List[float],
                 phases: List[float],
                 harmonics: List[int],
                 freq: float) -> None:
        self._amps = amps
        self._phases = phases
        self._harmonics = harmonics
        self._freq = freq

    @property
    def period(self) -> float:
        return 2 * np.pi / self._freq

    def __call__(self, t: float) -> float:
        return np.sum([amp * np.cos(harm * self._freq * t + phase) for
                       amp, harm, phase in zip(self._amps,
                                               self._harmonics,
                                               self._phases)])

class ComplexExpPulse(Pulse):
    """Implements a complex exponential modulation."""
    def __init__(self,
                 mean: float,
                 amp: float,
                 phase: float,
                 freq: float) -> None:
        self._mean = mean
        self._amp = amp
        self._phase = phase
        self._freq = freq

    @property
    def period(self) -> float:
        return 2 * np.pi / self._freq

    def __call__(self, t: float) -> complex:
        return self._mean + self._amp * np.exp(1.0j * (self._freq * t + self._phase))


class PolynomialPhasePulse(Pulse):
    """Implements a pulse that only has a phase that has polynomial time dep."""
    def __init__(self,
                 amp: float,
                 poly_ind: int,
                 freq: float) -> None:
        self._amp = amp
        self._poly_ind = poly_ind
        self._freq = freq

    @property
    def period(self) -> float:
        return 2 * np.pi / self._freq

    def __call__(self, t: float) -> complex:
        if t < 0.5 * self.period:
            phase = np.pi * (2 * t / self.period)**self._poly_ind
        else:
            phase = np.pi * (2 - (2 - 2 * t / self.period)**self._poly_ind)
        return self._amp * np.exp(-1.0j * phase)


class PolynomialPhaseCosinePulse(Pulse):
    """Implements a pulse that takes the cosine of the polynomial phase."""
    def __init__(self,
                 amp: float,
                 poly_ind: int,
                 freq: float) -> None:
        self._phase = PolynomialPhasePulse(amp, poly_ind, freq)

    @property
    def period(self) -> float:
        return self._phase.period

    def __call__(self, t: float) -> float:
        return np.real(self._phase(t))


class ConstantPulse(Pulse):
    """Implements a constant amplitude pulse."""
    def __init__(self,
                 amp: complex) -> None:
        """Creates a new `ConstantPulse` object.

        Args:
            amp: The amplitude of the pulse which in general can be a constant
                complex number.
        """
        self._amp = amp

    @property
    def period(self) -> float:
        return None

    def __call__(self, t: float) -> complex:
        return self._amp

class PolynomialAmplitudePhasePulse(Pulse):
    def __init__(
            self,
            dc_amp: float,
            mod_amp: float,
            amp_ind: int,
            poly_ind: int,
            freq: float) -> None:
        self._dc_amp = dc_amp
        self._mod_amp = mod_amp
        self._amp_ind = amp_ind
        self._phase = PolynomialPhasePulse(1, poly_ind, freq)

    @property
    def period(self) -> float:
        return self._phase.period

    def __call__(self, t: float) -> float:
        phase = self._phase(t)
        amp = self._dc_amp + self._mod_amp * np.real(phase)**(2 * self._amp_ind)
        return amp * phase
