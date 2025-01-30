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


def extract_band(rec, freq_min, freq_max):
    band = spre.bandpass_filter(
        rec,
        freq_min=freq_min,
        freq_max=freq_min,
        ftype="butterworth",
    )

    return band


def sort_spikes(rec, output, curated_sa_dir, ks_image_path, ks4_params):
    sorting, recording = _sort_spikes(
        rec,
        output,
        ks_image_path,
        ks4_params,
    )

    sa, curated_sa = _curate_sorting(
        sorting,
        recording,
        output,
    )

    _export_sorting_analyser(
        sa,
        curated_sa,
        output,
        curated_sa_dir,
    )

    return None


def _sort_spikes(rec, output, ks_image_path, ks4_params):
    assert 0
    # run sorter
    sorting = ss.run_sorter(
        sorter_name="kilosort4",
        recording=rec,
        folder=output,
        singularity_image=ks_image_path/"ks4_with_wavpack.sif",
        remove_existing_folder=True,
        verbose=True,
        **ks4_params,
    )

    # load ks preprocessed recording for # sorting analyser
    ks_preprocessed = se.read_binary(
        file_paths=output/"sorter_output/temp_wh.dat",
        sampling_frequency=rec.sampling_frequency,
        dtype=np.int16,
        num_channels=rec.get_num_channels(),
        is_filtered=True,
    )
    # attach probe # to ks4 preprocessed recording, from the raw
    recording = ks_preprocessed.set_probe(rec.get_probe())
    # set properties to make sure sorting & sorting sa have all
    # probe # properties to form correct rec_attributes, esp
    recording._properties = rec._properties

    # >>> annotations >>>
    annotations = rec.get_annotation_keys()
    annotations.remove("is_filtered")
    for ann in annotations:
        recording.set_annotation(
            annotation_key=ann,
            value=rec.get_annotation(ann),
            overwrite=True,
        )
    # <<< annotations <<<

    return sorting, recording


def _curate_n_export(sorting, recording, output):
    # curate sorter output
    # remove spikes exceeding recording number of samples
    sorting = sc.remove_excess_spikes(sorting, recording)
    # remove duplicate spikes
    sorting = sc.remove_duplicated_spikes(
        sorting,
        censored_period_ms=0.3,
        method="keep_first_iterative",
    )
    # remove redundant units created by ks
    sorting = sc.remove_redundant_units(
        sorting,
        duplicate_threshold=0.9, # default is 0.8
        align=False,
        remove_strategy="max_spikes",
    )

    # create sorting analyser
    sa = si.create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
    )

    # calculate all extensions BEFORE further steps
    # list required extensions for redundant units removal and quality
    # metrics
    required_extensions = [
        "random_spikes",
        "waveforms",
        "templates",
        "noise_levels",
        "unit_locations",
        "template_similarity",
        "spike_amplitudes",
        "correlograms",
        "principal_components", # for # phy
    ]
    sa.compute(required_extensions, save=True)

    # make sure to have group id for each unit
    if not "group" in sa.sorting.get_property_keys():
        # get shank id, i.e., group
        group = sa.recording.get_channel_groups()
        # get max peak channel for each unit
        max_chan = si.get_template_extremum_channel(sa).values()
        # get group id for each unit
        unit_group = group[list(max_chan)]
        # set unit group as a property for sorting
        sa.sorting.set_property(
            key="group",
            values=unit_group,
        )

    # calculate quality metrics
    qms = sqm.compute_quality_metrics(sa)

    # >>> get depth of units on each shank >>>
    # get probe geometry coordinates
    coords = sa.get_channel_locations()
    # get coordinates of max channel of each unit on probe, column 0 is
    # x-axis, column 1 is y-axis/depth, 0 at bottom-left channel.
    max_chan_coords = coords[sa.channel_ids_to_indices(max_chan)]
    # set coordinates of max channel of each unit as a property of sorting
    sa.sorting.set_property(
        key="max_chan_coords",
        values=max_chan_coords,
    )
    # <<< get depth of units on each shank <<<

    # remove bad units
    #rule = "sliding_rp_violation <= 0.1 & amplitude_median <= -50\
    #        & amplitude_cutoff < 0.05 & sd_ratio < 1.5 & presence_ratio > 0.9\
    #        & snr > 1.1 & rp_contamination < 0.2 & firing_rate > 0.1"
    # use the ibl methods, but amplitude_cutoff rather than noise_cutoff
    rule = "snr > 1.1 & rp_contamination < 0.2 & amplitude_median <= -50\
            & presence_ratio > 0.9"
    good_qms = qms.query(rule)
    # TODO nov 26 2024
    # wait till noise cutoff implemented and include that.
    # also see why sliding rp violation gives loads nan.
    # get unit ids
    curated_unit_ids = list(good_qms.index)
    # select curated
    curated_sa = sa.select_units(curated_unit_ids)

    return sa, curated_sa


def _export_sorting_analyser(sa, curated_sa, output, curated_sa_dir):

    # export pre curation report
    sexp.export_report(
        sorting_analyzer=sa,
        output_folder=output/"report",
    )

    # save pre-curated analyser to disk
    sa.save_as(
        format="zarr",
        folder=output/"sa.zarr",
    )

    # export curated report
    sexp.export_report(
        sorting_analyzer=curated_sa,
        output_folder=output/"curated_report",
    )

    # export to phy for additional manual curation if needed
    sexp.export_to_phy(
        sorting_analyzer=curated_sa,
        output_folder=output/"phy",
        copy_binary=False,
    )

    # save sa to disk
    curated_sa.save_as(
        format="zarr",
        folder=curated_sa_dir,
    )

    return None
