import numpy as np


def calculate_dice(input_log, input_lab):
    intersection = input_log * input_lab
    return (2 * np.sum(intersection) + 1e-6) / (np.sum(input_log) + np.sum(input_lab) + 1e-6)