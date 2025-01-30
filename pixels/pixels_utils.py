import numpy as np

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.curation as sc
import spikeinterface.exporters as sexp
import spikeinterface.preprocessing as spre
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm

# set si job_kwargs
job_kwargs = dict(
    n_jobs=0.8, # 80% core
    chunk_duration='1s',
    progress_bar=True,
)
si.set_global_job_kwargs(**job_kwargs)

def load_raw(paths, stream_id):
    """
    Load raw recording file from spikeglx.
    """
    recs = []
    for p, path in enumerate(paths):
        # NOTE: if it is catgt data, pass directly `catgt_ap_data`
        print(f"\n> Getting the orignial recording...")
        # load # recording # file
        rec = se.read_spikeglx(
            folder_path=path.parent,
            stream_id=stream_id,
            stream_name=path.stem,
            all_annotations=True, # include # all # annotations
        )
        recs.append(rec)
        if len(recs) > 1:
            # concatenate # runs # for # each # probe
            concat_recs = si.concatenate_recordings(recs)
        else:
            concat_recs = recs[0]

    return rec


def preprocess_raw(rec):
    shank_groups = rec.get_channel_groups()
    if not np.all(shank_groups == shank_groups[0]):
        preprocessed = []
        # split by groups
        groups = rec.split_by("group")
        for g, group in enumerate(groups.values()):
            print(f"> Preprocessing shank {g}")
            cleaned = _preprocess_raw(group)
            preprocessed.append(cleaned)
        # aggregate groups together
        preprocessed = si.aggregate_channels(preprocessed)
    else:
        preprocessed = _preprocess_raw(rec)

    # NOTE jan 16 2025:
    # BUG: cannot set dtype back to int16, units from ks4 will have
    # incorrect amp & loc
    if not preprocessed.dtype == np.dtype("int16"):
        preprocessed = spre.astype(preprocessed, dtype=np.int16)

    return preprocessed


def _preprocess_raw(rec):
    """
    Implementation of preprocessing on raw pixels data.
    """
    # correct phase shift
    print("\t> step 1: do phase shift correction.")
    rec_ps = spre.phase_shift(rec)

    # remove bad channels from sorting
    print("\t> step 2: remove bad channels.")
    bad_chan_ids, chan_labels = spre.detect_bad_channels(
        rec_ps,
        outside_channels_location="top",
    )
    labels, counts = np.unique(chan_labels, return_counts=True)

    for label, count in zip(labels, counts):
        print(f"\t\t> Found {count} channels labelled as {label}.")
    rec_clean = rec_ps.remove_channels(bad_chan_ids)

    print("\t> step 3: do common median referencing.")
    # NOTE: dtype will be converted to float32 during motion correction
    cmr = spre.common_reference(
        rec_clean,
    )

    return cmr


def correct_motion(rec, mc_method="dredge"):
    """
    Correct motion of recording.

    params
    ===
    mc_method: str, motion correction method.
        Default: "dredge".
            (as of jan 2025, dredge performs better than ks motion correction.)
        "ks": let kilosort do motion correction.

    return
    ===
    None
    """
    print(f"\t> correct motion with {mc_method}.")
    # reduce spatial window size for four-shank
    estimate_motion_kwargs = {
        "win_step_um": 100,
        "win_margin_um": -150,
    }

    mcd = spre.correct_motion(
        rec,
        preset=mc_method,
        estimate_motion_kwargs=estimate_motion_kwargs,
        #interpolate_motion_kwargs={'border_mode':'force_extrapolate'},
    )

    # convert to int16 to save space
    if not mcd.dtype == np.dtype("int16"):
        mcd = spre.astype(mcd, dtype=np.int16)

    return mcd


def detect_n_localise_peaks(rec):
    """
    Get a sense of possible drifts in the recordings by looking at a
    "positional raster plot", i.e. the depth of the spike as function of
    time. To do so, we need to detect the peaks, and then to localize them
    in space.

    params
    ===
    rec: spikeinterface recording extractor.

    loc_method: str, peak location method.
        Default: "monopolar_triangulation"
        list of methods:
        "center_of_mass", "monopolar_triangulation", "grid_convolution"
        to learn more, check:
        https://spikeinterface.readthedocs.io/en/stable/modules/motion_correction.html
    """
    shank_groups = rec.get_channel_groups()
    if not np.all(shank_groups == shank_groups[0]):
        # split by groups
        groups = rec.split_by("group")
        dfs = []
        for g, group in enumerate(groups.values()):
            print(f"\n> Estimate drift of shank {g}")
            dfs.append(_estimate_drift(group, loc_method))
        # concat shanks
        df = pd.concat(
            dfs,
            axis=1,
            keys=groups.keys(),
            names=["shank", "spike_properties"]
        )
    else:
        df = self._estimate_drift(rec, loc_method)

    return df


def _detect_n_localise_peaks(rec, loc_method="monopolar_triangulation"):
    """
    implementation of drift estimation.
    """
    from spikeinterface.sortingcomponents.peak_detection\
        import detect_peaks
    from spikeinterface.sortingcomponents.peak_localization\
        import localize_peaks

    print("> step 1: detect peaks")
    peaks = detect_peaks(
        recording=rec,
        method="by_channel",
        detect_threshold=5,
        exclude_sweep_ms=0.2,
    )

    print("> step 2: localize the peaks to get a sense of their putative "
            "depths")
    peak_locations = localize_peaks(
        recording=rec,
        peaks=peaks,
        method=loc_method,
    )

    # get sampling frequency
    fs = rec.sampling_frequency

    # save it as df
    df_peaks = pd.DataFrame(peaks)
    df_peak_locs = pd.DataFrame(peak_locations)
    df = pd.concat([df_peaks, df_peak_locs], axis=1)
    # add timestamps and channel ids
    df["timestamp"] = df.sample_index / fs
    df["channel_id"] = rec.get_channel_ids()[df.channel_index.values]

    return df
