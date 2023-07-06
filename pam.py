import numpy as np
from numpy.random import randint
from scipy.stats import norm
import matplotlib.pyplot as plt


def symbol_error_rate_plot(number_of_levels, sequence_size, upper_bound=10):
    """
    Plots a graph of SER by E_b/N_0.

    Generate a random sequence of bits of size sequence_size and
    compare the theoretical symbol error rate and the simulated
    symbol error rate for E_b/N_0 ranging from [0, upper_bound] dB
    for a (number_of_levels)-PAM modulation.
    """
    bit_sequence = randint(2, size=sequence_size)
    theoretical= np.array([])
    simulational = np.array([])
    bit_energy_noise_ratio_db = np.linspace(0, upper_bound, 100)
    noise_density_array = 10**(-bit_energy_noise_ratio_db/10)
    for noise_density in noise_density_array:
        pam = PAM(number_of_levels, noise_density, bit_sequence)
        theoretical = np.append(
            theoretical, pam.theoretical_error_probability()
        )
        simulational = np.append(
            simulational, pam.simulational_error_probability()
        )
    theoretical_db = np.log10(theoretical)
    simulational_db = np.log10(simulational[simulational != 0])
    plt.figure()
    plt.plot(bit_energy_noise_ratio_db, theoretical_db,
            label='Theoretical')
    plt.plot(bit_energy_noise_ratio_db[simulational != 0], simulational_db,
            'r', label='Simulational')
    plt.title(f'SERx($E_b/N_0$) for {number_of_levels}-PAM modulation')
    plt.legend()
    plt.grid()
    plt.ylabel('SNR (dB)')
    plt.xlabel(r'$E_b/N_0 (dB)$')
    plt.show()

class PAM:
    """
    Implements PAM modulation.

    The two important methods of this class is
    theoretical_error_probability and
    simulational_error_probability, the latter being obtained
    through the use of Monte Carlo Method. Note that this has only been
    implemented for 2-PAM and 4-PAM.
    """
    number_of_levels: int
    symbol_energy: int|float
    noise_density: int|float
    bit_sequence: np.ndarray
    delta: float
    amplitude_levels: float
    level_sequence: np.ndarray
    def __init__(self, number_of_levels, noise_density, bit_sequence):
        """
        Initialize class instance.
        
        Note that bit energy is not received. It was decided to fix
        bit energy as E_b = 1 and use that to calculate the symbol
        energy and the E_b/N_0 from the number of levels and noise
        density provided as parameters when initializing the instance.
        """
        self.number_of_levels = number_of_levels
        self.symbol_energy = np.log2(self.number_of_levels)
        self.noise_density = noise_density
        self.bit_sequence = bit_sequence
        self.delta = self._calculate_delta()
        self.amplitude_levels = self._amplitude_levels()
        self.level_sequence = self._map_bits_to_levels()

    def theoretical_error_probability(self):
        """
        Calculate the symbol error rate from the theoretical equation.
        """
        symbol_noise_ratio = self.symbol_energy / self.noise_density
        M = self.number_of_levels
        q_function_argument = np.sqrt(6*symbol_noise_ratio/(M*M-1))
        return 2*(1-1/M)*norm.sf(q_function_argument)

    def _calculate_delta(self):
        """Calculate the distance between two levels."""
        return np.sqrt(12*self.symbol_energy/(self.number_of_levels**2-1))

    def simulational_error_probability(self):
        """
        Calculate the symbol error rate using the Monte Carlo Method.
        """
        decided_levels = self._decide_noisy_levels()
        error_array = np.not_equal(decided_levels, self.level_sequence)
        number_of_errors = error_array.sum()
        return number_of_errors/error_array.size

    def _amplitude_levels(self):
        """Calculate the amplitude levels used in the modulation."""
        M = self.number_of_levels
        return np.linspace(-(M-1)*self.delta/2, (M-1)*self.delta/2, M)

    def _map_bits_to_levels(self):
        """
        Map the bit sequence to amplitude levels.

        This function has been implemented only in the cases where
        the number of levels is 2 and 4.
        """
        bit_sequence = self.bit_sequence
        if self.number_of_levels == 2:
            level_sequence = np.empty(self.bit_sequence.shape)
            level_sequence[bit_sequence == 0] = self.amplitude_levels[0]
            level_sequence[bit_sequence == 1] = self.amplitude_levels[1]
        elif self.number_of_levels == 4:
            level_sequence = np.empty(self.bit_sequence.size//2)
            bit_pairs = np.reshape(bit_sequence, (bit_sequence.size//2, 2))
            level_0 = np.all(bit_pairs[:] == np.array([0, 0]), axis=1)
            level_1 = np.all(bit_pairs[:] == np.array([0, 1]), axis=1)
            level_2 = np.all(bit_pairs[:] == np.array([1, 1]), axis=1)
            level_3 = np.all(bit_pairs[:] == np.array([1, 0]), axis=1)
            level_sequence[level_0] = self.amplitude_levels[0]
            level_sequence[level_1] = self.amplitude_levels[1]
            level_sequence[level_2] = self.amplitude_levels[2]
            level_sequence[level_3] = self.amplitude_levels[3]
        return level_sequence

    def _noisify_levels(self):
        """Introduce noise to the input sequence's levels."""
        n_0 = self.noise_density
        size = self.level_sequence.size
        noise = np.sqrt(n_0/2)*np.random.normal(0, 1, size)
        return self.level_sequence + noise

    def _decide_noisy_levels(self):
        """Decide which level is closest to noisy value."""
        M = self.number_of_levels
        signal = self._noisify_levels()
        decider = np.concatenate(tuple(signal for i in range(M)), axis=0)
        decider = decider.reshape(M, decider.size//M)
        decider = (decider - self.amplitude_levels.reshape(M, 1))**2
        indexes = np.argmin(decider, axis=0)
        return np.array([self.amplitude_levels[i] for i in indexes])
