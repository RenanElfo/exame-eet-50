import numpy as np
from numpy.random import randint
from scipy.stats import norm
import matplotlib.pyplot as plt


def symbol_error_rate_plot(order, sequence_size, upper_bound=10):
    """
    Plots a graph of SER by E_b/N_0.

    Generate a random sequence of bits of size sequence_size and
    compare the theoretical symbol error rate and the simulated
    symbol error rate for E_b/N_0 ranging from [0, upper_bound] dB
    for a (order)-PAM modulation.
    """
    bit_sequence = randint(2, size=sequence_size)
    theoretical= np.array([])
    simulational = np.array([])
    bit_energy_noise_ratio_db = np.linspace(0, upper_bound, 100)
    noise_density_array = 10**(-bit_energy_noise_ratio_db/10)
    for noise_density in noise_density_array:
        pam = PAM(order, noise_density, bit_sequence)
        theoretical = np.append(
            theoretical, pam.theoretical_error_probability()
        )
        simulational = np.append(
            simulational, pam.simulational_error_probability()
        )
    theoretical_db = 10*np.log10(theoretical)
    simulational_db = 10*np.log10(simulational[simulational != 0])
    plt.figure()
    plt.plot(bit_energy_noise_ratio_db, theoretical_db,
            label='Theoretical')
    plt.plot(bit_energy_noise_ratio_db[simulational != 0], simulational_db,
            'r', label='Simulational')
    plt.title(f'SERx($E_b/N_0$) for {order}-PAM modulation')
    plt.legend()
    plt.grid()
    plt.ylabel('SER (dB)')
    plt.xlabel(r'$E_b/N_0 (dB)$')
    plt.show()

class PAM:
    """
    Implementation of 2-PAM and 4-PAM modulations.

    The two important methods of this class is
    theoretical_error_probability and
    simulational_error_probability, the latter being obtained
    through the use of Monte Carlo Method. Note that this has only been
    implemented for 2-PAM, 4-PAM and 8-PAM.
    """
    order: int
    symbol_energy: int|float
    noise_density: int|float
    bit_sequence: np.ndarray
    delta: float
    amplitude_levels: float
    level_sequence: np.ndarray
    def __init__(self, order, noise_density, bit_sequence):
        """
        Initialize class instance.
        
        Note that bit energy is not received. It was decided to fix
        bit energy as E_b = 1 and use that to calculate the symbol
        energy and the E_b/N_0 from the number of levels and noise
        density provided as parameters when initializing the instance.
        """
        self.order = order
        self.symbol_energy = np.log2(self.order)
        self.noise_density = noise_density
        self.bit_sequence = bit_sequence
        self.delta = self._calculate_delta()
        self.amplitude_levels = self._amplitude_levels()
        self.level_sequence = self._map_bits_to_levels()

    def theoretical_error_probability(self):
        """
        Calculate the symbol error rate from the theoretical equation.

        This calculation uses a known equation that was presented in
        the classroom.
        """
        symbol_noise_ratio = self.symbol_energy / self.noise_density
        M = self.order
        q_function_argument = np.sqrt(6*symbol_noise_ratio/(M*M-1))
        return 2*(1-1/M)*norm.sf(q_function_argument)

    def _calculate_delta(self):
        """
        Calculate the distance between two levels.
        
        This calculation can also be retrieved by what was given in
        the classroom.
        """
        return np.sqrt(12*self.symbol_energy/(self.order**2-1))

    def simulational_error_probability(self):
        """
        Calculate the symbol error rate using the Monte Carlo Method.

        We count the number of times that the original signal differs
        from the noisy signal after passing the decider block. That
        number is the number of errors. The ratio between the number
        of errors and the total number of symbols gives us the
        simulational probability of symbolic error.
        """
        decided_levels = self._decide_noisy_levels()
        error_array = np.not_equal(decided_levels, self.level_sequence)
        number_of_errors = error_array.sum()
        return number_of_errors/error_array.size

    def _amplitude_levels(self):
        """
        Calculate the amplitude levels used in the modulation.
        
        From what was teached, we know that the levels in which we will
        map our sequence of bits goes from -(M-1)*delta/2 to
        (M-1)*delta/2 and we have M levels. Using numpy's linspace
        function we obtain an array with the possible levels.
        """
        M = self.order
        return np.linspace(-(M-1)*self.delta/2, (M-1)*self.delta/2, M)

    def _map_bits_to_levels(self):
        """
        Map the bit sequence to amplitude levels.

        This function has been implemented only in the cases where
        the number of levels is 2, 4 and 8. In the case where M = 2, we
        map the bit 0 to the lowest level and the bit 1 to the highest.
        When M = 4, we map 00 to the lowest level, 01 to the second
        lowest level, 11 to the second highest level and 10 to the
        highest level. For M = 8, we do something analogous to M = 4,
        """
        bit_sequence = self.bit_sequence
        if self.order == 2:
            level_sequence = np.empty(self.bit_sequence.shape)
            level_sequence[bit_sequence == 0] = self.amplitude_levels[0]
            level_sequence[bit_sequence == 1] = self.amplitude_levels[1]
        elif self.order == 4:
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
        elif self.order == 8:
            level_sequence = np.empty(self.bit_sequence.size//3)
            bit_trios = np.reshape(bit_sequence, (bit_sequence.size//3, 3))
            level_0 = np.all(bit_trios[:] == np.array([0, 0, 0]), axis=1)
            level_1 = np.all(bit_trios[:] == np.array([0, 0, 1]), axis=1)
            level_2 = np.all(bit_trios[:] == np.array([0, 1, 1]), axis=1)
            level_3 = np.all(bit_trios[:] == np.array([0, 1, 0]), axis=1)
            level_4 = np.all(bit_trios[:] == np.array([1, 1, 0]), axis=1)
            level_5 = np.all(bit_trios[:] == np.array([1, 1, 1]), axis=1)
            level_6 = np.all(bit_trios[:] == np.array([1, 0, 1]), axis=1)
            level_7 = np.all(bit_trios[:] == np.array([1, 0, 0]), axis=1)
            level_sequence[level_0] = self.amplitude_levels[0]
            level_sequence[level_1] = self.amplitude_levels[1]
            level_sequence[level_2] = self.amplitude_levels[2]
            level_sequence[level_3] = self.amplitude_levels[3]
            level_sequence[level_4] = self.amplitude_levels[4]
            level_sequence[level_5] = self.amplitude_levels[5]
            level_sequence[level_6] = self.amplitude_levels[6]
            level_sequence[level_7] = self.amplitude_levels[7]
        else:
            message = 'Code not implemented for 16-PAM and above.'
            raise ValueError(message)
        return level_sequence

    def _noisify_levels(self):
        """Introduce noise to the input sequence's levels."""
        n_0 = self.noise_density
        size = self.level_sequence.size
        noise = np.sqrt(n_0/2)*np.random.normal(0, 1, size)
        return self.level_sequence + noise

    def _decide_noisy_levels(self):
        """
        Decide which level is closest to noisy value.
        
        We replicate the signal M times and subtract from each replica
        the value of each level. Then we square that result and the min
        value will be given by the closest point in the constelation to
        our signal. That's how we can decide which point in the
        constelation is closest to our signal.
        """
        M = self.order
        signal = self._noisify_levels()
        decider = np.concatenate(tuple(signal for i in range(M)), axis=0)
        decider = decider.reshape(M, decider.size//M)
        decider = (decider - self.amplitude_levels.reshape(M, 1))**2
        indexes = np.argmin(decider, axis=0)
        return np.array([self.amplitude_levels[i] for i in indexes])
