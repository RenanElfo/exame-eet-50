import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt

from pam import symbol_error_rate_plot as pam_ser
from qam import symbol_error_rate_plot as qam_ser
from qam import natural_vs_gray_plot

NOISE_DENSITY = 1/8
SEQUENCE_SIZE = 2**18  # 2**18


def main():
    # Simulation for a 2-PAM
    # pam_ser(2, SEQUENCE_SIZE, upper_bound=10)
    # Simulation for a 4-PAM
    # pam_ser(4, SEQUENCE_SIZE, upper_bound=14)
    # Simulation for a 4-QAM
    # qam_ser(SEQUENCE_SIZE, natural_mapping=False, upper_bound=10)
    # Comparison between gray mapping and natural mapping
    # natural_vs_gray_plot(SEQUENCE_SIZE, 10)
    pass


if __name__ == '__main__':
    main()
