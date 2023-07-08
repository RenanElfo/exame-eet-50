import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt

from pam import PAM
from qam import QAM

def transmission_protocol_ser_plot(sequence_size, performance_limit,
                                   upper_bound=20):
    # Generate random bit sequence of size sequence_size.
    bit_sequence = randint(2, size=sequence_size)
    # Initialize arrays for theoretical values of SER
    # and Monte Carlo SER for the modulation that meets the
    # protocol requisites.
    qam_64_theoretical = np.array([])
    qam_16_theoretical = np.array([])
    qam_4_theoretical= np.array([])
    pam_2_theoretical = np.array([])
    monte_carlo = np.array([])
    # Generate array of values for Eb/N0.
    bit_energy_noise_ratio_db = np.linspace(0, upper_bound, 200)
    # We fixed Eb = 1 in all our code, so we generate an array
    # for the noise density given Eb/N0.
    noise_density_array = 10**(-bit_energy_noise_ratio_db/10)
    # Required for plotting vertical lines that separete the opperation
    # bands of the protocol.
    plt.figure()
    count = 1
    for noise_density in noise_density_array:
        # Create instance of classes for each modulation.
        qam_64 = QAM(64, noise_density, bit_sequence)
        qam_16 = QAM(16, noise_density, bit_sequence)
        qam_4 = QAM(4, noise_density, bit_sequence)
        pam_2 = PAM(2, noise_density, bit_sequence)
        # Put them on a list for later reference.
        modulation_list = [
            qam_64,
            qam_16,
            qam_4,
            pam_2
        ]
        # Grab the theoretical error probability for each modulation
        # given the noise density.
        qam_64_theoretical = np.append(
            qam_64_theoretical,
            qam_64.theoretical_error_probability()
        )
        qam_16_theoretical = np.append(
            qam_16_theoretical,
            qam_16.theoretical_error_probability()
        )
        qam_4_theoretical = np.append(
            qam_4_theoretical,
            qam_4.theoretical_error_probability()
        )
        pam_2_theoretical = np.append(
            pam_2_theoretical,
            pam_2.theoretical_error_probability()
        )
        # Modulations with greater spectral efficiency come first
        # in our priority array of modulations.
        priority_modulation = np.array([
            qam_64_theoretical[-1],
            qam_16_theoretical[-1],
            qam_4_theoretical[-1],
            pam_2_theoretical[-1]
        ])
        # We figure which modulatios have a theoretical SER smaller
        # than the given threshold.
        modulation_decider = priority_modulation < performance_limit
        # Plot opperation bands vertical lines separators.
        if count < np.sum(modulation_decider):
            upper =-10*np.log10(noise_density)
            plt.axvline(upper,
                        color='lightgray',
                        linestyle='dashed',
                        label=f'Band {count} upper bound = {upper:.2f} db')
            count += 1
        # Then we grab the modulation from our list of modulations
        # that has the highest spectral efficiency but doens't have
        # a SER higher than the threshold.
        if not np.any(modulation_decider):
            chosen_modulation = modulation_list[-1]
        else:
            chosen_modulation = modulation_list[np.argmax(modulation_decider)]
        # Grab the simulational error probability for the chosen
        # modulation obtained through the criteria explained above.
        monte_carlo = np.append(
            monte_carlo, chosen_modulation.simulational_error_probability()
        )
    # Plot the results.
    qam_64_theoretical_db = 10*np.log10(qam_64_theoretical)
    plot_limiter_1 = bit_energy_noise_ratio_db < 15
    qam_16_theoretical_db = 10*np.log10(qam_16_theoretical[plot_limiter_1])
    plot_limiter_2 = bit_energy_noise_ratio_db < 11
    qam_4_theoretical_db = 10*np.log10(qam_4_theoretical[plot_limiter_2])
    pam_2_theoretical_db = 10*np.log10(pam_2_theoretical[plot_limiter_2])
    monte_carlo_db = 10*np.log10(monte_carlo[monte_carlo != 0])
    plt.plot(bit_energy_noise_ratio_db, qam_64_theoretical_db,
             color='pink', label='64-QAM')
    plt.plot(bit_energy_noise_ratio_db[plot_limiter_1], qam_16_theoretical_db,
             color='gold', label='16-QAM')
    plt.plot(bit_energy_noise_ratio_db[plot_limiter_2], qam_4_theoretical_db,
             color='g', label='4-QAM')
    plt.plot(bit_energy_noise_ratio_db[plot_limiter_2], pam_2_theoretical_db,
             color='b', label='2-PAM')
    plt.plot(bit_energy_noise_ratio_db[monte_carlo != 0], monte_carlo_db,
             'ro', label='Monte Carlo')
    plt.title(f'SERx($E_b/N_0$) for transmission protocol')
    plt.legend()
    plt.grid()
    plt.ylabel('SER (dB)')
    plt.xlabel(r'$E_b/N_0 (dB)$')
    plt.show()

def transmission_protocol_ber_plot(sequence_size, performance_limit,
                                   upper_bound=20):
    # Generate random bit sequence of size sequence_size.
    bit_sequence = randint(2, size=sequence_size)
    # Initialize arrays for theoretical values of BER
    # and Monte Carlo BER for the modulation that meets the
    # protocol requisites.
    qam_64_theoretical = np.array([])
    qam_16_theoretical = np.array([])
    qam_4_theoretical= np.array([])
    pam_2_theoretical = np.array([])
    monte_carlo = np.array([])
    # Generate array of values for Eb/N0.
    bit_energy_noise_ratio_db = np.linspace(0, upper_bound, 200)
    # We fixed Eb = 1 in all our code, so we generate an array
    # for the noise density given Eb/N0.
    noise_density_array = 10**(-bit_energy_noise_ratio_db/10)
    # Required for plotting vertical lines that separete the opperation
    # bands of the protocol.
    plt.figure()
    count = 1
    for noise_density in noise_density_array:
        # Create instance of classes for each modulation.
        qam_64 = QAM(64, noise_density, bit_sequence)
        qam_16 = QAM(16, noise_density, bit_sequence)
        qam_4 = QAM(4, noise_density, bit_sequence)
        pam_2 = PAM(2, noise_density, bit_sequence)
        # Put them on a list for later reference.
        modulation_list = [
            qam_64,
            qam_16,
            qam_4,
            pam_2
        ]
        # Grab the theoretical error probability for each modulation
        # given the noise density.
        qam_64_theoretical = np.append(
            qam_64_theoretical,
            qam_64.theoretical_error_probability()/6
        )
        qam_16_theoretical = np.append(
            qam_16_theoretical,
            qam_16.theoretical_error_probability()/4
        )
        qam_4_theoretical = np.append(
            qam_4_theoretical,
            qam_4.theoretical_error_probability()/2
        )
        pam_2_theoretical = np.append(
            pam_2_theoretical,
            pam_2.theoretical_error_probability()
        )
        # Modulations with greater spectral efficiency come first
        # in our priority array of modulations.
        priority_modulation = np.array([
            qam_64_theoretical[-1],
            qam_16_theoretical[-1],
            qam_4_theoretical[-1],
            pam_2_theoretical[-1]
        ])
        # We figure which modulatios have a theoretical BER smaller
        # than the given threshold.
        modulation_decider = priority_modulation < performance_limit
        if count < np.sum(modulation_decider):
            upper =-10*np.log10(noise_density)
            plt.axvline(upper,
                        color='lightgray',
                        linestyle='dashed',
                        label=f'Band {count} upper bound = {upper:.2f} db')
            count += 1
        # Then we grab the modulation from our list of modulations
        # that has the highest spectral efficiency but doens't have
        # a BER higher than the threshold.
        if not np.any(modulation_decider):
            chosen_modulation = modulation_list[-1]
        else:
            chosen_modulation = modulation_list[np.argmax(modulation_decider)]
        # Grab the simulational error probability for the chosen
        # modulation obtained through the criteria explained above.
        number_of_bits = int(np.log2(chosen_modulation.order))
        monte_carlo = np.append(
            monte_carlo,
            chosen_modulation.simulational_error_probability()/number_of_bits
        )
    # Plot the results.
    qam_64_theoretical_db = 10*np.log10(qam_64_theoretical)
    plot_limiter_1 = bit_energy_noise_ratio_db < 15
    qam_16_theoretical_db = 10*np.log10(qam_16_theoretical[plot_limiter_1])
    plot_limiter_2 = bit_energy_noise_ratio_db < 11
    qam_4_theoretical_db = 10*np.log10(qam_4_theoretical[plot_limiter_2])
    pam_2_theoretical_db = 10*np.log10(pam_2_theoretical[plot_limiter_2])
    monte_carlo_db = 10*np.log10(monte_carlo[monte_carlo != 0])
    plt.plot(bit_energy_noise_ratio_db, qam_64_theoretical_db,
             color='pink', label='64-QAM')
    plt.plot(bit_energy_noise_ratio_db[plot_limiter_1], qam_16_theoretical_db,
             color='gold', label='16-QAM')
    plt.plot(bit_energy_noise_ratio_db[plot_limiter_2], qam_4_theoretical_db,
             color='g', label='4-QAM')
    plt.plot(bit_energy_noise_ratio_db[plot_limiter_2], pam_2_theoretical_db,
             color='b', label='2-PAM')
    plt.plot(bit_energy_noise_ratio_db[monte_carlo != 0], monte_carlo_db,
             'ro', label='Monte Carlo')
    plt.title(f'BERx($E_b/N_0$) for transmission protocol')
    plt.legend()
    plt.grid()
    plt.ylabel('BER (dB)')
    plt.xlabel(r'$E_b/N_0 (dB)$')
    plt.show()
