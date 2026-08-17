"""
This file contains some constants parameters for the pixels pipeline.
"""
import numpy as np

SAMPLE_RATE = 2000 # Hz

tom_freq_bands = {
    "theta":[4, 11], # from Tom
    "gamma":[30, 80], # from Tom
    "ripple":[110, 220], # from Tom
}
freq_bands = {
    "ap":[300, 9000],
    "lfp":[0.5, 500],
    "delta":[1, 4], # from Moser
    "theta":[5, 10], # from Moser
    "sharp_wave":[5, 40], # from kempter & maier
    "slow_gamma":[20, 45], # from david dupret 2025
    "mid_gamma":[50, 100], # from david dupret 2025
    "ripple":[100, 250], # i.e., fast gamma, from david dupret 2025
    "ripple_surround":[70, 300], # from david dupret 2025
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

ALPHA = 0.05

# position bin sizes
POSITION_BIN = 1 # cm
BIG_POSITION_BIN = 10 # cm

# power spectrum density esimate params
P_TARGET_PERIOD = 80 # cm
P_NPERSEG = 240
P_SMALL_NPERSEG = 160
p_npersegs = np.array([160, 240, 320])
T_NPERSEG = 256
T_SMALL_NPERSEG = 128
MIN_CYCLE = 3
SMALL_MIN_CYCLE = 2
# number of psd background noise median filter bins
N_MEDIAN_FILT_BINS = 11
# time segment size for psd estimation
T_SEG = 5 # second

# min number of units for analysis
MIN_N_UNITS = 5

# duration of alinging event in second
DURATION = 1
PAD_DURATION = 1
