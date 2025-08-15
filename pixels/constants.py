"""
This file contains some constants parameters for the pixels pipeline.
"""
import numpy as np

SAMPLE_RATE = 2000 # Hz

freq_bands = {
    "ap":[300, 9000],
    "lfp":[0.5, 500],
    "theta":[4, 11], # from Tom
    "gamma":[30, 80], # from Tom
    "ripple":[110, 220], # from Tom
}

BEHAVIOUR_HZ = 25000

np.random.seed(BEHAVIOUR_HZ)

REPEATS = 100
