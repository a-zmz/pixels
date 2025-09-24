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

# latency of luminance change evoked spike
# NOTE: usually it should be between 40-60ms. now we use 500ms just to test
V1_SPIKE_LATENCY = 500 # ms #60
V1_LFP_LATENCY = 40 # ms
LGN_SPIKE_LATENCY = 30 # ms

# chunking for zarr
SMALL_CHUNKS = 64
BIG_CHUNKS = 1024
