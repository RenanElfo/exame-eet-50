from pam import symbol_error_rate_plot as pam_ser
from qam import symbol_error_rate_plot as qam_ser
from qam import natural_vs_gray_plot
from protocol import transmission_protocol_ser_plot
from protocol import transmission_protocol_ber_plot

SEQUENCE_SIZE = 3*2**17
SYMBOL_ERROR_RATE_LIMIT = 10**(-2)
BIT_ERROR_RATE_LIMIT = 10**(-2)


def main():
    # Simulation for a 2-PAM
    pam_ser(2, SEQUENCE_SIZE, upper_bound=10)
    # Simulation for a 4-PAM
    pam_ser(4, SEQUENCE_SIZE, upper_bound=14)
    # Simulation for a 4-QAM
    qam_ser(SEQUENCE_SIZE, natural_mapping=False, upper_bound=10)
    # Comparison between gray mapping and natural mapping
    natural_vs_gray_plot(SEQUENCE_SIZE, 10)
    # Plot of SER for a transmission protocol
    transmission_protocol_ser_plot(SEQUENCE_SIZE, SYMBOL_ERROR_RATE_LIMIT, 20)
    # Plot of BER for a transmission protocol
    transmission_protocol_ber_plot(SEQUENCE_SIZE, BIT_ERROR_RATE_LIMIT, 20)


if __name__ == '__main__':
    main()
