import logging

from wavpack_numcodecs import WavPack
from numcodecs import Blosc
import spikeinterface as si

# Configure logging to include a timestamp with seconds
logging.basicConfig(
    level=logging.INFO,
    format='''\n%(asctime)s %(levelname)s: %(message)s\
            \n[in %(filename)s:%(lineno)d]''',
    datefmt='%Y%m%d %H:%M:%S',
)

#logging.info('This is an info message.')
#logging.warning('This is a warning message.')
#logging.error('This is an error message.')

# set si job_kwargs
job_kwargs = dict(
    pool_engine="thread", # instead of default "process"
    #pool_engine="process",# does not work on 2025 oct 14
    mp_context="fork", # linux
    #mp_context="spawn", # mac & win
    progress_bar=True,
    n_jobs=0.8,
    chunk_duration='1s',
    max_threads_per_worker=8,
)
si.set_global_job_kwargs(**job_kwargs)

# instantiate WavPack compressor
wv_compressor = WavPack(
    level=3, # high compression
    bps=None, # lossless
)

# use blosc compressor for generic zarr
compressor = Blosc(
    cname="zstd",
    clevel=5,
    shuffle=Blosc.BITSHUFFLE,
)

# kilosort 4 singularity image names
ks4_0_30_image_name = "si103.0_ks4-0-30_with_wavpack.sif"
#ks4_0_30_image_name = "si102.3_ks4-0-30_with_wavpack.sif"
ks4_0_18_image_name = "ks4-0-18_with_wavpack.sif"
ks4_image_name = ks4_0_30_image_name

# quality metrics rule
# TODO nov 26 2024
# wait till noise cutoff implemented and include that.
# also see why sliding rp violation gives loads nan.
#qms_rule = "sliding_rp_violation <= 0.1 & amplitude_median <= -40\
#        & amplitude_cutoff < 0.05 & sd_ratio < 1.5 & presence_ratio > 0.9\
#        & snr > 1.1 & rp_contamination < 0.2 & firing_rate > 0.1"
# use the ibl methods, but amplitude_cutoff rather than noise_cutoff
qms_rule = "snr > 1.1 & rp_contamination < 0.2 & amplitude_median <= -40\
        & presence_ratio > 0.9"

# template metrics rule
tms_rule = "num_positive_peaks <= 2 & num_negative_peaks == 1 &\
exp_decay > 0.01 & exp_decay < 0.1" # bombcell
#peak_to_valley > 0.00018 &\
