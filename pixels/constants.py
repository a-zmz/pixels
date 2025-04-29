"""
This file contains some constants parameters for the pixels pipeline.
"""
import numpy as np

SAMPLE_RATE = 2000

freq_bands = {
    "ap":[300, 9000],
    "lfp":[0.5, 300],
}

BEHAVIOUR_HZ = 25000

np.random.seed(BEHAVIOUR_HZ)
