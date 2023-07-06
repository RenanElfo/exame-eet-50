import numpy as np
from numpy.random import randint
from scipy.stats import norm
import matplotlib.pyplot as plt

from pam import PAM


def symbol_error_rate_plot(sequence_size, natural_mapping, upper_bound=10):
    """
    Plots a graph of SER by E_b/N_0.

    Generate a random sequence of bits of size sequence_size and
    compare the theoretical symbol error rate and the simulated
    symbol error rate for E_b/N_0 ranging from [0, upper_bound] dB
    for a 4-QAM modulation.
    """
    bit_sequence = randint(2, size=sequence_size)
    theoretical= np.array([])
    simulational = np.array([])
    bit_energy_noise_ratio_db = np.linspace(0, upper_bound, 100)
    noise_density_array = 10**(-bit_energy_noise_ratio_db/10)
    for noise_density in noise_density_array:
        qam = FourQAM(noise_density, bit_sequence, natural_mapping)
        theoretical = np.append(
            theoretical, qam.theoretical_error_probability()
        )
        simulational = np.append(
            simulational, qam.simulational_error_probability()
        )
    theoretical_db = np.log10(theoretical)
    simulational_db = np.log10(simulational[simulational != 0])
    plt.figure()
    plt.plot(bit_energy_noise_ratio_db, theoretical_db,
            label='Theoretical')
    plt.plot(bit_energy_noise_ratio_db[simulational != 0], simulational_db,
            'r', label='Simulational')
    plt.title(f'SERx($E_b/N_0$) for {4}-QAM modulation')
    plt.legend()
    plt.grid()
    plt.ylabel('SNR (dB)')
    plt.xlabel(r'$E_b/N_0 (dB)$')
    plt.show()


def natural_vs_gray_plot(sequence_size, upper_bound=10):
    """
    Compare the BER for 4-QAM modulation with gray and natural mapping.
    """
    bit_sequence = randint(2, size=sequence_size)
    natural= np.array([])
    gray = np.array([])
    bit_energy_noise_ratio_db = np.linspace(0, upper_bound, 100)
    noise_density_array = 10**(-bit_energy_noise_ratio_db/10)
    for noise_density in noise_density_array:
        qam_natural = FourQAM(noise_density, bit_sequence, True)
        natural = np.append(
            natural, qam_natural.simulational_bit_error_rate()
        )
        qam_gray = FourQAM(noise_density, bit_sequence)
        gray = np.append(
            gray, qam_gray.simulational_bit_error_rate()
        )
    natural_db = np.log10(natural[natural != 0])
    gray_db = np.log10(gray[gray != 0])
    plt.figure()
    plt.plot(bit_energy_noise_ratio_db[natural != 0], natural_db,
            'g', label='Natural')
    plt.plot(bit_energy_noise_ratio_db[gray != 0], gray_db,
            'r', label='Gray')
    plt.title(f'BERx($E_b/N_0$) for {4}-QAM modulation')
    plt.legend()
    plt.grid()
    plt.ylabel('SNR (dB)')
    plt.xlabel(r'$E_b/N_0 (dB)$')
    plt.show()


class FourQAM:
    """Implementation of 4-QAMmodulation."""
    def __init__(self, noise_density, bit_sequence, natural_mapping=False):
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
        q_function_argument = np.sqrt(2/self.noise_density)
        return 2*norm.sf(q_function_argument)

    def simulational_error_probability(self):
        """
        Calculate the symbol error rate using the Monte Carlo Method.
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
        number_of_errors = (number_of_errors_x + number_of_errors_y
                            - number_of_errors_xy)
        number_of_symbols = self.bit_sequence.size/2
        return number_of_errors/number_of_symbols

    def simulational_bit_error_rate(self):
        """
        Calculate the bit error rate using the Monte Carlo Method.

        Calculate BER using Monte Carlo Method for gray mapping and
        for natural mapping, decided by the natural_mapping parameter.
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
        number_of_symbols = self.bit_sequence.size/2
        return number_of_errors/number_of_symbols

    def _map_bits_to_constelation(self):
        """
        Map each pair of bits in the sequence given to constelation.

        Each bit pair can be mapped to a 4-QAM constelation using two
        2-PAM constelations as foundation.
        """
        bit_sequence = self.bit_sequence
        bit_pairs = np.reshape(bit_sequence, (bit_sequence.size//2, 2)).T
        pam_y = PAM(2, self.noise_density, bit_pairs[0][:])
        pam_x = PAM(2, self.noise_density, bit_pairs[1][:])
        return pam_x, pam_y
