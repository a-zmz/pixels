import logging

from wavpack_numcodecs import WavPack
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
    #mp_context="fork", # linux, but does not work still on 2025 aug 12
    mp_context="spawn", # mac & win
    chunk_duration='1s',
    progress_bar=True,
    n_jobs=0.8,
    max_threads_per_worker=1,
)
si.set_global_job_kwargs(**job_kwargs)

# instantiate WavPack compressor
wv_compressor = WavPack(
    level=3, # high compression
    bps=None, # lossless
)

# kilosort 4 singularity image names
ks4_0_30_image_name = "si102.3_ks4-0-30_with_wavpack.sif"
ks4_0_18_image_name = "ks4-0-18_with_wavpack.sif"
ks4_image_name = ks4_0_30_image_name
