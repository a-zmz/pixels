"""
A worker function that sets a new process during spike sorting for each session
using spawn.
"""

def sort_session_worker(session, mc_method, job_kwargs):
    import spikeinterface as si

    # Spawned processes do not inherit SI globals set in the parent.
    si.set_global_job_kwargs(**job_kwargs)

    session.sort_spikes(mc_method=mc_method)
