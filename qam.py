import numpy as np
from numpy.random import randint
from scipy.stats import norm
import matplotlib.pyplot as plt

from pam import PAM


def symbol_error_rate_plot(order,sequence_size, natural_mapping,
                           upper_bound=10):
    """
    Plots a graph of SER by E_b/N_0.

    Generate a random sequence of bits of size sequence_size and
    compare the theoretical symbol error rate and the simulated
    symbol error rate for E_b/N_0 ranging from [0, upper_bound] dB
    for a M-QAM modulation.
    """
    bit_sequence = randint(2, size=sequence_size)
    theoretical= np.array([])
    simulational = np.array([])
    bit_energy_noise_ratio_db = np.linspace(0, upper_bound, 100)
    noise_density_array = 10**(-bit_energy_noise_ratio_db/10)
    for noise_density in noise_density_array:
        qam = QAM(order, noise_density, bit_sequence, natural_mapping)
        theoretical = np.append(
            theoretical, qam.theoretical_error_probability()
        )
        simulational = np.append(
            simulational, qam.simulational_error_probability()
        )
    theoretical_db = 10*np.log10(theoretical)
    simulational_db = 10*np.log10(simulational[simulational != 0])
    plt.figure()
    plt.plot(bit_energy_noise_ratio_db, theoretical_db,
            label='Theoretical')
    plt.plot(bit_energy_noise_ratio_db[simulational != 0], simulational_db,
            'r', label='Simulational')
    plt.title(f'SERx($E_b/N_0$) for {order}-QAM modulation')
    plt.legend()
    plt.grid()
    plt.ylabel('SER (dB)')
    plt.xlabel(r'$E_b/N_0 (dB)$')
    plt.show()


def natural_vs_gray_plot(sequence_size, upper_bound=10):
    """
    Compare the BER for 4-QAM modulation with gray and natural mapping.

    Similar to the code in the symbol_error_rate_plot function.
    """
    bit_sequence = randint(2, size=sequence_size)
    natural= np.array([])
    gray = np.array([])
    bit_energy_noise_ratio_db = np.linspace(0, upper_bound, 100)
    noise_density_array = 10**(-bit_energy_noise_ratio_db/10)
    for noise_density in noise_density_array:
        qam_natural = QAM(4, noise_density, bit_sequence, True)
        natural = np.append(
            natural, qam_natural.simulational_bit_error_rate()
        )
        qam_gray = QAM(4, noise_density, bit_sequence)
        gray = np.append(
            gray, qam_gray.simulational_bit_error_rate()
        )
    natural_db = 10*np.log10(natural[natural != 0])
    gray_db = 10*np.log10(gray[gray != 0])
    plt.figure()
    plt.plot(bit_energy_noise_ratio_db[natural != 0], natural_db,
            'g', label='Natural')
    plt.plot(bit_energy_noise_ratio_db[gray != 0], gray_db,
            'r', label='Gray')
    plt.title(f'BERx($E_b/N_0$) for {4}-QAM modulation')
    plt.legend()
    plt.grid()
    plt.ylabel('BER (dB)')
    plt.xlabel(r'$E_b/N_0 (dB)$')
    plt.show()


class QAM:
    """
    Implementation of QAM modulation.
    
    Analogous to the PAM modulation, we consider E_b = 1 and only take
    as parameters the noise density and the bit sequence, in addition
    to whether or not we want to use natural mapping. Default is to use
    gray mapping.
    """
    def __init__(self, order, noise_density, bit_sequence,
                 natural_mapping=False):
        self.order = order
        self.bits_per_symbol = int(np.log2(self.order))
        self.noise_density = noise_density
        self.bit_sequence = bit_sequence
        self.natural_mapping = natural_mapping
        self.pam_x, self.pam_y = self._map_bits_to_constelation()

    def theoretical_error_probability(self):
        """
        Calculate the symbol error rate from the theoretical equation.

        For this calculation, we consider delta = 2, because we
        fixed the bit energy as E_b = 1 and the number of levels
        in each independet PAM that forms the 4-QAM is sqrt(4) = 2.
        """
        n_0 = self.noise_density
        inside_root = 3*np.log2(self.order)/((self.order-1)*n_0)
        q_function_argument = np.sqrt(inside_root)
        return 4*(1-1/np.sqrt(self.order))*norm.sf(q_function_argument)

    def simulational_error_probability(self):
        """
        Calculate the symbol error rate using the Monte Carlo Method.

        Analogous to simulational_error_probability method in PAM
        class.
        """
        decided_levels_x = self.pam_x._decide_noisy_levels()
        decided_levels_y = self.pam_y._decide_noisy_levels()
        error_array_x = np.not_equal(decided_levels_x,
                                     self.pam_x.level_sequence)
        error_array_y = np.not_equal(decided_levels_y,
                                     self.pam_y.level_sequence)
        error_array_xy = error_array_x*error_array_y
        number_of_errors_x = error_array_x.sum()
        number_of_errors_y = error_array_y.sum()
        number_of_errors_xy = error_array_xy.sum()
        # the result bellow for number_of_errors is only valid for
        # 4-QAM, but, since number_of_error_xy is an very small,
        # it can be discarded for higher order modulations. Also,
        # considering the use of gray mapping, we can consider every
        # symbol error as a single bit error.
        number_of_errors = (number_of_errors_x + number_of_errors_y
                            - number_of_errors_xy)
        number_of_symbols = self.bit_sequence.size/self.bits_per_symbol
        return number_of_errors/number_of_symbols

    def simulational_bit_error_rate(self):
        """
        Calculate the bit error rate using the Monte Carlo Method.

        Calculate BER using Monte Carlo Method for gray mapping and
        for natural mapping, decided by the natural_mapping parameter.

        Consider the natural mapping bellow:

        11 | 10
        -------
        00 | 01

        And the gray mapping:

        10 | 11
        -------
        00 | 01

        We can think that these constelations as a combination of a
        2-PAM in the x axis (pam_x) and a 2-PAM (pam_y). That said, we
        can affirm that total number of errors of the sequence is:

        N_natural = N_x + 2*N_y - N_xy
        N_gray = N_x + N_y

        where N_x is the number of errors in pam_x, N_y is the
        number of errors in pam_y and N_xy is the number of times
        there was a simultaneous error in pam_x and pam_y.
        """
        decided_levels_x = self.pam_x._decide_noisy_levels()
        decided_levels_y = self.pam_y._decide_noisy_levels()
        error_array_x = np.not_equal(decided_levels_x,
                                     self.pam_x.level_sequence)
        error_array_y = np.not_equal(decided_levels_y,
                                     self.pam_y.level_sequence)
        error_array_xy = error_array_x*error_array_y
        number_of_errors_x = error_array_x.sum()
        number_of_errors_y = error_array_y.sum()
        number_of_errors_xy = error_array_xy.sum()
        if not self.natural_mapping:
            number_of_errors = number_of_errors_x + number_of_errors_y
        else:
            number_of_errors = (number_of_errors_x + 2*number_of_errors_y
                                - number_of_errors_xy)
        return number_of_errors/self.bit_sequence.size

    def _map_bits_to_constelation(self):
        """
        Map each group of k bits in the sequence given to constelation.

        Each bit pair can be mapped to a M-QAM constelation using two
        sqrt(M)-PAM constelations as foundation.
        """
        pam_order = int(np.sqrt(self.order))
        bit_sequence = self.bit_sequence
        bit_pairs = np.reshape(
            bit_sequence,
            (bit_sequence.size//self.bits_per_symbol, self.bits_per_symbol)
        ).T
        y_bits = bit_pairs[:int(self.bits_per_symbol/2)][:]
        x_bits = bit_pairs[int(self.bits_per_symbol/2):][:]
        pam_y = PAM(pam_order, self.noise_density, y_bits)
        pam_x = PAM(pam_order, self.noise_density, x_bits)
        return pam_x, pam_y
