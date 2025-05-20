"""
This module provides utilities for pixels data.
"""
import multiprocessing as mp
import json

import numpy as np
import pandas as pd

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.curation as sc
import spikeinterface.exporters as sexp
import spikeinterface.preprocessing as spre
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm

import pixels.signal_utils as signal
from pixels.ioutils import write_hdf5, reindex_by_longest
from pixels.error import PixelsError
from pixels.configs import *

from common_utils.math_utils import random_sampling, group_and_aggregate
from common_utils.file_utils import init_memmap, read_hdf5

def load_raw(paths, stream_id):
    """
    Load raw recording file from spikeglx.
    """
    recs = []
    for p, path in enumerate(paths):
        # NOTE: if it is catgt data, pass directly `catgt_ap_data`
        logging.info(f"\n> Getting the orignial recording...")
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


def preprocess_raw(rec, surface_depths):
    group_ids = rec.get_channel_groups()

    if np.unique(group_ids).size < 4:
        # correct group id if not all shanks used
        group_ids = correct_group_id(rec)
        # change the group id
        rec.set_channel_groups(group_ids)

    if not np.all(group_ids == group_ids[0]):
        # if more than one shank used
        preprocessed = []
        # split by groups
        groups = rec.split_by("group")
        for g, group in groups.items():
            logging.info(f"\n> Preprocessing shank {g}")
            # get brain surface depth of shank
            surface_depth = surface_depths[g]
            cleaned = _preprocess_raw(group, surface_depth)
            preprocessed.append(cleaned)
        # aggregate groups together
        preprocessed = si.aggregate_channels(preprocessed)
    else:
        # if only one shank used, check which shank
        unique_id = np.unique(group_ids)[0]
        # get brain surface depth of shank
        surface_depth = surface_depths[unique_id]
        # preprocess
        preprocessed = _preprocess_raw(rec, surface_depth)

    return preprocessed


def _preprocess_raw(rec, surface_depth):
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
    rec_removed = rec_ps.remove_channels(bad_chan_ids)

    # get channel group id and use it to index into brain surface channel depth
    shank_id = np.unique(rec_removed.get_channel_groups())[0]
    # get channel depths
    chan_depths = rec_removed.get_channel_locations()[:, 1]
    # get channel ids
    chan_ids = rec_removed.channel_ids
    # remove channels outside by using identified brain surface depths
    outside_chan_ids = chan_ids[chan_depths > surface_depth]
    rec_clean = rec_removed.remove_channels(outside_chan_ids)
    print(f"\t\t> Removed {outside_chan_ids.size} outside channels.")

    print("\t> step 3: do common median referencing.")
    cmr = spre.common_reference(
        rec_clean,
        dtype=np.int16, # make sure output is int16
    )

    return cmr


def correct_ap_motion(rec, mc_method="dredge"):
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
    logging.info(f"\n> Correcting motion with {mc_method}.")

    # reduce spatial window size for four-shank
    # TODO may 8 2025 "method":"dredge_ap" after it's implemented?
    estimate_motion_kwargs = {
        "method": "decentralized",
        "win_step_um": 100,
        "win_margin_um": -150,
        "verbose": True,
    }

    # make sure recording dtype is float for interpolation
    interpolate_motion_kwargs = {
        "dtype": np.float32,
    }

    mcd = spre.correct_motion(
        rec,
        preset=mc_method,
        estimate_motion_kwargs=estimate_motion_kwargs,
        interpolate_motion_kwargs=interpolate_motion_kwargs,
    )

    # convert to int16 to save space
    if not mcd.dtype == np.dtype("int16"):
        mcd = spre.astype(mcd, dtype=np.int16)

    return mcd


def detect_n_localise_peaks(rec, loc_method="monopolar_triangulation"):
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
    level_names = ["shank", "spike_properties"]

    if not np.all(shank_groups == shank_groups[0]):
        # split by groups
        groups = rec.split_by("group")
        dfs = []
        for g, group in groups.items():
            logging.info(f"\n> Estimate drift of shank {g}")
            dfs.append(_detect_n_localise_peaks(group, loc_method))
        # concat shanks
        df = pd.concat(
            dfs,
            axis=1,
            keys=groups.keys(),
            names=level_names,
        )
    else:
        df = _detect_n_localise_peaks(rec, loc_method)
        # add shank level on top
        shank_id = shank_groups[0]
        df.columns = pd.MultiIndex.from_tuples(
            [(shank_id, col) for col in df.columns],
            names=level_names,
        )

    return df


def _detect_n_localise_peaks(rec, loc_method):
    """
    implementation of drift estimation.
    """
    from spikeinterface.sortingcomponents.peak_detection\
        import detect_peaks
    from spikeinterface.sortingcomponents.peak_localization\
        import localize_peaks

    logging.info("\n> step 1: detect peaks")
    peaks = detect_peaks(
        recording=rec,
        method="by_channel",
        detect_threshold=5,
        exclude_sweep_ms=0.2,
    )

    logging.info(
        "\n> step 2: localize the peaks to get a sense of their putative "
        "depths"
    )
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


def extract_band(rec, freq_min, freq_max, ftype="butter"):
    """
    Band pass filter recording.

    params
    ===
    freq_min: float, high-pass cutoff corner frequency.

    freq_max: float, low-pass cutoff corner frequency.

    ftype: str, filter type.
        since its posthoc, we use 5th order acausal filter, and takes
        second-order sections (SOS) representation of the filter,
        forward-backward. but more filters to choose from, e.g., bessel with
        filter_order=2, presumably preserves waveform better? see lussac.

    return
    ===
    band: spikeinterface recording object.
    """
    band = spre.bandpass_filter(
        rec,
        freq_min=freq_min,
        freq_max=freq_max,
        margin_ms=5.0,
        filter_order=5,
        ftype=ftype,
        direction="forward-backward",
    )

    return band


def whiten(rec):
    whitened = spre.whiten(
        recording=rec,
        dtype=np.float32,
        #dtype=np.int16,
        #int_scale=200, # scale traces value to sd of 200, in line with ks4
        mode="local",
        radius_um=240.0, # 16 nearby chans in line with ks4
    )

    return whitened


def sort_spikes(rec, sa_rec, output, curated_sa_dir, ks_image_path, ks4_params,
                per_shank=False):
    """
    Sort spikes with kilosort 4, curate sorting, save sorting analyser to disk,
    and export results to disk.
    
    params
    ===
    rec: spikeinterface recording object.

    sa_rec: spikeinterface recording object for creating sorting analyser.

    output: path object, directory of output.

    curated_sa_dir: path object, directory to save curated sorting analyser.

    ks_image_path: path object, directory of local kilosort 4 singularity image.

    ks4_params: dict, parameters for kilosort 4.

    return
    ===
    sorting: spikeinterface sorting object.

    recording: spikeinterface recording object.
    """
    # NOTE: jan 30 2025 do we sort shanks separately???
    # if shanks are sorted separately, they will have separate sorter output, we
    # will have to build an analyser for each group...
    # maybe easier to just run all shanks together?
    # the only way to concatenate four temp.dat and only create one sorting
    # analyser is to read temp_wh.dat, set channels separately from raw, and
    # si.aggregate_channels...

    # sort spikes
    if np.unique(rec.get_channel_groups()).size > 1 and per_shank:
        # per shank
        sorting, recording = _sort_spikes_by_group(
            rec,
            sa_rec,
            output,
            ks_image_path,
            ks4_params,
        )
    else:
        # all together
        sorting, recording = _sort_spikes(
            rec,
            sa_rec,
            output,
            ks_image_path,
            ks4_params,
        )

    # curate sorting
    sa, curated_sa = _curate_sorting(
        sorting,
        recording,
        output,
    )

    # export sorting analyser
    _export_sorting_analyser(
        sa,
        curated_sa,
        output,
        curated_sa_dir,
    )

    return None


def _sort_spikes_by_group(rec, sa_rec, output, ks_image_path, ks4_params):
    """
    Sort spikes with kilosort 4 by group/shank.
    
    params
    ===
    rec: spikeinterface recording object.

    sa_rec: spikeinterface recording object for creating sorting analyser.
        if None, then use the temp_wh.dat from ks output.

    output: path object, directory of output.

    ks_image_path: path object, directory of local kilosort 4 singularity image.

    ks4_params: dict, parameters for kilosort 4.

    return
    ===
    sorting: spikeinterface sorting object.

    recording: spikeinterface recording object.
    """
    logging.info("\n> Sorting spikes per shank.")

    # run sorter per shank
    sorting = ss.run_sorter_by_property(
        sorter_name="kilosort4",
        recording=rec,
        grouping_property="group",
        folder=output,
        singularity_image=ks_image_path,
        remove_existing_folder=True,
        verbose=True,
        **ks4_params,
    )

    if not sa_rec:
        recs = []
        groups = rec.split_by("group")
        for g, group in groups.items():
            ks_preprocessed = se.read_binary(
                file_paths=output/f"{g}/sorter_output/temp_wh.dat",
                sampling_frequency=group.sampling_frequency,
                dtype=np.int16,
                num_channels=group.get_num_channels(),
                is_filtered=True,
            )

            # attach probe # to ks4 preprocessed recording, from the raw
            with_probe = ks_preprocessed.set_probe(group.get_probe())
            # set properties to make sure sorting & sorting sa have all
            # probe # properties to form correct rec_attributes, esp
            with_probe._properties = group._properties

            # >>> annotations >>>
            annotations = group.get_annotation_keys()
            annotations.remove("is_filtered")
            for ann in annotations:
                with_probe.set_annotation(
                    annotation_key=ann,
                    value=group.get_annotation(ann),
                    overwrite=True,
                )
            # <<< annotations <<<
            recs.append(with_probe)

        recording = si.aggregate_channels(recs)
    else:
        recording = sa_rec

    return sorting, recording


def _sort_spikes(rec, sa_rec, output, ks_image_path, ks4_params):
    """
    Sort spikes with kilosort 4.
    
    params
    ===
    rec: spikeinterface recording object.

    sa_rec: spikeinterface recording object for creating sorting analyser.

    output: path object, directory of output.

    ks_image_path: path object, directory of local kilosort 4 singularity image.

    ks4_params: dict, parameters for kilosort 4.

    return
    ===
    sorting: spikeinterface sorting object.

    recording: spikeinterface recording object.
    """
    logging.info("\n> Sorting spikes.")

    # run sorter
    sorting = ss.run_sorter(
        sorter_name="kilosort4",
        recording=rec,
        folder=output,
        singularity_image=ks_image_path,
        remove_existing_folder=True,
        verbose=True,
        **ks4_params,
    )

    # TODO apr 10 2025:
    # since this file is whitened, the amplitude of the signal is NOT the same
    # as the original, and this might cause issue in calculating signal
    # amplitude in spikeinterface. cuz in ks4 output, units amplitude is between
    # 0-315, but in si it's between -4000 to 4000.
    # POTENTIAL SOLUTIONS:
    # 1. do what chris does, make another preprocessed recording just to build
    # the sorting analyser, or
    # 2. still use the temp_wh.dat from ks4, but check how ks4 handles amplitude
    # and the unit of amplitude, correct it
    # WHAT TO ACHIEVE:
    # 1. without whitening, peak amplitude median should be ~-70uV

    if not sa_rec:
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
    else:
        recording = sa_rec

    return sorting, recording


def _curate_sorting(sorting, recording, output):
    """
    Curate spike sorting results, and export to disk.
    
    params
    ===
    sorting: spikeinterface sorting object.

    recording: spikeinterface recording object.

    output: path object, directory of output.

    return
    ===
    sa: spikeinterface sorting analyser.

    curated_sa: curated spikeinterface sorting analyser.
    """
    logging.info("\n> Curating sorting.")

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
        try:
            unit_group = group[list(max_chan)]
        except IndexError:
            unit_group = group[sa.channel_ids_to_indices(max_chan)]
        # set unit group as a property for sorting
        sa.sorting.set_property(
            key="group",
            values=unit_group,
        )
    else:
        # get max peak channel for each unit
        max_chan = si.get_template_extremum_channel(sa).values()

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
    #rule = "sliding_rp_violation <= 0.1 & amplitude_median <= -40\
    #        & amplitude_cutoff < 0.05 & sd_ratio < 1.5 & presence_ratio > 0.9\
    #        & snr > 1.1 & rp_contamination < 0.2 & firing_rate > 0.1"
    # use the ibl methods, but amplitude_cutoff rather than noise_cutoff
    rule = "snr > 1.1 & rp_contamination < 0.2 & amplitude_median <= -40\
            & presence_ratio > 0.9"
    good_qms = qms.query(rule)
    # TODO nov 26 2024
    # wait till noise cutoff implemented and include that.
    # also see why sliding rp violation gives loads nan.
    # get unit ids
    curated_unit_ids = list(good_qms.index)
    # select curated
    curated_sorting = sa.sorting.select_units(curated_unit_ids)
    curated_sa = sa.select_units(curated_unit_ids)
    # reattach curated sorting to curated_sa to keep sorting properties
    curated_sa.sorting = curated_sorting

    return sa, curated_sa


def _export_sorting_analyser(sa, curated_sa, output, curated_sa_dir):
    """
    Export sorting analyser to disk.
    
    params
    ===
    sa: spikeinterface sorting analyser.

    curated_sa_dir: path object, directory to save curated sorting analyser.

    output: path object, directory of output.

    return
    ===
    None
    """
    logging.info("\n> Exporting sorting results.")

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

    # save sa to disk
    curated_sa.save_as(
        format="zarr",
        folder=curated_sa_dir,
    )

    return None


def export_sa_to_phy(path, sa):
    # export to phy for additional manual curation if needed
    sexp.export_to_phy(
        sorting_analyzer=sa,
        output_folder=path/"phy",
        copy_binary=False,
    )

    return None


def _permute_spikes_n_convolve_fr(array, sigma, sample_rate):
    """
    Randomly permute spike boolean across time.

    params
    ===
    array: 2D np array, time points x units.

    sigma: int/float, time in millisecond of sigma of gaussian kernel for firing
    rate convolution.

    sample_rate: float/int, sampling rate of signal.

    return
    ===
    random_spiked: shuffled spike boolean for each unit.

    random_fr: convolved firing rate from shuffled spike boolean for each unit.
    """
    # initiate random number generator every time to avoid same results from the
    # same seeding
    rng = np.random.default_rng()
    # permutate columns
    random_spiked = rng.permuted(array, axis=0)
    # convolve into firing rate
    random_fr = signal.convolve_spike_trains(
        times=random_spiked,
        sigma=sigma,
        sample_rate=sample_rate,
    )

    return random_spiked, random_fr


def _chance_worker(i, sigma, sample_rate, spiked_shape, chance_data_shape,
                  spiked_memmap_path, fr_memmap_path, concat_spiked_path):
    """
    Worker that computes one set of spiked and fr values.

    params
    ===
    i: index of current repeat.

    sigma: int/float, time in millisecond of sigma of gaussian kernel for firing
    rate convolution.

    sample_rate: float/int, sampling rate of signal.

    spiked_shape: tuple, shape of spike boolean to initiate memmap.

    chance_data_shape: tuple, shape of chance data.

    spiked_memmap_path: 

    fr_memmap_path: 

    return
    ===
    """
    logging.info(f"\nProcessing repeat {i}...")
    # open readonly memmap
    spiked = init_memmap(
        path=concat_spiked_path,
        shape=spiked_shape,
        dtype=np.int16,
        overwrite=False,
        readonly=True,
    )

    # init appendable memmap
    chance_spiked = init_memmap(
        path=spiked_memmap_path,
        shape=chance_data_shape,
        dtype=np.int16,
        overwrite=False,
        readonly=False,
    )
    # init chance firing rate memmap
    chance_fr = init_memmap(
        path=fr_memmap_path,
        shape=chance_data_shape,
        dtype=np.float32,
        overwrite=False,
        readonly=False,
    )

    # get permuted data
    c_spiked, c_fr = _permute_spikes_n_convolve_fr(spiked, sigma, sample_rate)

    chance_spiked[..., i] = c_spiked
    chance_fr[..., i] = c_fr
    # write to disk
    chance_spiked.flush()
    chance_fr.flush()

    logging.info(f"\nRepeat {i} finished.")

    return None


def save_spike_chance(spiked_memmap_path, fr_memmap_path, spiked_df_path,
                      fr_df_path, sigma, sample_rate, repeats=100, spiked=None,
                      spiked_shape=None, concat_spiked_path=None):
    if fr_df_path.exists():
        # save spike chance data if does not exists
        _save_spike_chance(
            spiked_memmap_path, fr_memmap_path, spiked_df_path, fr_df_path, sigma,
            sample_rate, repeats, spiked, spiked_shape,
            concat_spiked_path)
    else:
        logging.info(f"\n> Spike chance already saved at {fr_df_path}, continue.")

    return None


def _save_spike_chance(spiked_memmap_path, fr_memmap_path, sigma, sample_rate,
                       repeats, spiked, spiked_shape, concat_spiked_path):
    """
    Implementation of saving chance level spike data.
    """
    import concurrent.futures

    # save spiked to memmap if not yet
    # TODO apr 9 2025: if i have temp_spiked, how to get its shape? do i need
    # another input arg??? this is to run it again without get the concat spiked
    # again...
    if spiked is None:
        assert concat_spiked_path.exists()
        assert spiked_shape is not None
    else:
        concat_spiked_path = spiked_memmap_path.parent/"temp_spiked.bin"
        spiked_memmap = init_memmap(
            path=concat_spiked_path,
            shape=spiked.shape,
            dtype=np.int16,
            overwrite=True,
            readonly=False,
        )
        spiked_memmap[:] = spiked.values
        spiked_memmap.flush()
        del spiked_memmap

        # get spiked data shape
        spiked_shape = spiked.shape

    # get export data shape
    d_shape = spiked_shape + (repeats,)
    # TODO apr 9 2025 save dshape to json
    #with open(shape_json, "w") as f:
        #json.dump(shape, f, indent=4)

    if not fr_memmap_path.exists():
        # init chance spiked memmap
        chance_spiked = init_memmap(
            path=spiked_memmap_path,
            shape=d_shape,
            dtype=np.int16,
            overwrite=True,
            readonly=False,
        )
        # init chance firing rate memmap
        chance_fr = init_memmap(
            path=fr_memmap_path,
            shape=d_shape,
            dtype=np.float32,
            overwrite=True,
            readonly=False,
        )
        # write to disk
        chance_spiked.flush()
        chance_fr.flush()
        del chance_spiked, chance_fr

        # Set up the process pool to run the worker in parallel.
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit jobs for each repeat.
            futures = []
            for i in range(repeats):
                future = executor.submit(
                    _chance_worker,
                    i=i,
                    sigma=sigma,
                    sample_rate=sample_rate,
                    spiked_shape=spiked_shape,
                    chance_data_shape=d_shape,
                    spiked_memmap_path=spiked_memmap_path,
                    fr_memmap_path=fr_memmap_path,
                    concat_spiked_path=concat_spiked_path,
                )
                futures.append(future)

            # As each future completes, assign the results into the memmap.
            for future in concurrent.futures.as_completed(futures):
                future.result()
    else:
        logging.info(
            "\n> Memmaps already created, only need to convert into "
            "dataframes and save."
        )

    # convert it to dataframe and save it
    #save_chance(
    #    orig_idx=spiked.index,
    #    orig_col=spiked.columns,
    #    spiked_memmap_path=spiked_memmap_path,
    #    fr_memmap_path=fr_memmap_path,
    #    spiked_df_path=spiked_df_path,
    #    fr_df_path=fr_df_path,
    #    d_shape=d_shape,
    #)
    #logging.info(f"\n> Chance data saved to {fr_df_path}.")

    return None


def _convert_to_df(orig_idx, orig_col, memmap_path, df_path, d_shape, d_type,
                   name):
    """
    Convert 

    orig_idx,
    orig_col,
    memmap_path
    df_path,
    d_shape
    d_type
    name
    """
    # NOTE: shape of memmap is `concatenated trials frames * units * repeats`,
    # saved df has outer most level being `repeat`, then `unit`, and all trials
    # are stacked vertically. 
    # to later use it for analysis, go into each repeat, and do
    # `repeat_df.unstack(level='trial', sort=False)` to get the same structure as
    # data.

    # init readonly chance memmap
    chance_memmap = init_memmap(
        path=memmap_path,
        shape=d_shape,
        dtype=d_type,
        overwrite=False,
        readonly=True,
    )

    # copy to cpu
    c_spiked = chance_memmap.copy()
    # reshape to 2D
    c_spiked_reshaped = c_spiked.reshape(d_shape[0], d_shape[1] * d_shape[2])
    del c_spiked

    # create hierarchical index
    col_idx = pd.MultiIndex.from_product(
        [np.arange(d_shape[2]), orig_col],
        names=["repeat", "unit"],
    )

    # create df
    df = pd.DataFrame(c_spiked_reshaped, columns=col_idx)
    # use the original index
    df.index = orig_idx

    # write h5 to disk
    write_hdf5(
        path=df_path,
        df=df,
        key=name,
        mode="w",
    )
    del df

    return None


def save_chance(orig_idx, orig_col, spiked_memmap_path, fr_memmap_path,
                spiked_df_path, fr_df_path, d_shape):
    """
    Saving chance data to dataframe.

    params
    ===
    orig_idx: pandas 
    """
    logging.info(f"\n> Saving chance data...")

    # get chance spiked df
    _convert_to_df(
        orig_idx=orig_idx,
        orig_col=orig_col,
        memmap_path=spiked_memmap_path,
        df_path=spiked_df_path,
        d_shape=d_shape,
        d_type=np.int16,
        name="spiked",
    )
    # get chance fr df
    _convert_to_df(
        orig_idx=orig_idx,
        orig_col=orig_col,
        memmap_path=fr_memmap_path,
        df_path=fr_df_path,
        d_shape=d_shape,
        d_type=np.float32,
        name="fr",
    )

    return None


def get_spike_chance(sample_rate, positions, time_bin, pos_bin,
                     spiked_memmap_path, fr_memmap_path, memmap_shape_path,
                     idx_path, col_path):
    if not fr_memmap_path.exists():
        raise PixelsError("\nHave you saved spike chance data yet?")
    else:
        # TODO apr 3 2025: we need to get positions here for binning!!!
        # BUT HOW????
        _get_spike_chance(sample_rate, positions, time_bin, pos_bin,
                          spiked_memmap_path, fr_memmap_path, memmap_shape_path,
                          idx_path, col_path)

    return None


def _get_spike_chance(sample_rate, positions, time_bin, pos_bin,
                      spiked_memmap_path, fr_memmap_path, memmap_shape_path,
                      idx_path, col_path):

    # TODO apr 9 2025:
    # i do not need to save shape to file, all i need is unit count, repeat,
    # so i load memmap without defining shape, then directly np.reshape(memmap,
    # (-1, count, repeat))!

    with open(memmap_shape_path, "r") as f:
        shape_data = json.load(f)
    shape_list = shape_data.get("dshape", [])
    d_shape = tuple(shape_list)

    spiked_chance = init_memmap(
        path=spiked_memmap_path,
        shape=d_shape,
        dtype=np.int16,
        overwrite=False,
        readonly=True,
    )

    idx_df = read_hdf5(idx_path, key="multiindex")
    idx = pd.MultiIndex.from_frame(idx_df)
    trials = idx_df["trial"].unique()
    col_df = read_hdf5(col_path, key="cols")
    cols = pd.Index(col_df["unit"])

    binned_shuffle = {}
    temp = {}
    # TODO apr 3 2025: implement multiprocessing here!
    # get each repeat and create df
    for r in range(d_shape[-1]):
        shuffled = spiked_chance[:, :, r]
        # create df
        df = pd.DataFrame(shuffled, index=idx, columns=cols)
        temp[r] = {}
        for t in trials:
            counts = df.xs(t, level="trial", axis=0)
            trial_pos = positions.loc[:, t].dropna()
            temp[r][t] = bin_vr_trial(
                counts,
                trial_pos,
                sample_rate,
                time_bin,
                pos_bin,
                bin_method="sum",
            )
        binned_shuffle[r] = reindex_by_longest(
            dfs=temp[r],
            return_format="array",
        ) 
    # concat 
    binned_shuffle_counts = np.stack(
        list(binned_shuffle.values()),
        axis=-1,
    )
    shuffled_counts = {
        "count": binned_shuffle_counts[:, :-2, ...],
        "pos": binned_shuffle_counts[:, -2:, ...],
    }
    #count_path='/home/amz/running_data/npx/interim/20240812_az_VDCN09/20240812_az_VDCN09_imec0_light_all_spike_counts_shuffled_200ms_10cm.npz'
    count_path='/home/amz/running_data/npx/interim/20240812_az_VDCN09/20240812_az_VDCN09_imec0_dark_all_spike_counts_shuffled_200ms_10cm.npz'

    np.savez_compressed(count_path, **shuffled_counts)
    assert 0

    #    fr_chance = _get_spike_chance(
    #        path=fr_memmap_path,
    #        shape=d_shape,
    #        dtype=np.float32,
    #        overwrite=False,
    #        readonly=True,
    #    )

    # TODO apr 2 2025:
    # for fr chance, use memmap, go to each repeat, unstack, bin, then save it
    # to .npz for andrew
    pass


def bin_vr_trial(data, positions, sample_rate, time_bin, pos_bin,
                 bin_method="mean"):
    """
    Bin virtual reality trials by given temporal bin and positional bin.

    params
    ===
    data: pandas dataframe, neural data needed binning.

    positions: pandas dataframe, position of current trial.

    time_bin: str, temporal bin for neural data.

    pos_bin: int, positional bin for positions.

    bin_method: str, method to concatenate data within each temporal bin.
        "mean": taking the mean of all frames.
        "sum": taking sum of all frames.
    """
    data = data.copy()
    positions = positions.copy()

    # convert index to datetime index for resampling
    isi = (1 / sample_rate) * 1000
    data.index = pd.to_timedelta(
        arg=data.index * isi,
        unit="ms",
    )

    # set position index too
    positions.index = data.index

    # resample to 100ms bin, and get position mean
    mean_pos = positions.resample(time_bin).mean()

    if bin_method == "sum":
        # resample to Xms bin, and get sum
        bin_data = data.resample(time_bin).sum()
    elif bin_method == "mean":
        # resample to Xms bin, and get mean
        bin_data = data.resample(time_bin).mean()

    # add position here to bin together
    bin_data['positions'] = mean_pos.values
    # add bin positions
    bin_pos = mean_pos // pos_bin + 1
    bin_data['bin_pos'] = bin_pos.values

    # use numeric index
    bin_data.reset_index(inplace=True, drop=True)

    return bin_data

def correct_group_id(rec):
    # check probe type
    '''
    npx 1.0: 0
    npx 2.0 alpha: 24
    npx 2.0 commercial: 2013
    '''
    probe_type = int(rec.get_annotation("probes_info")[0]["probe_type"])
    # double check it is multishank probe
    assert probe_type > 0

    # get channel x locations
    shank_x_locs = {
        0: [0, 32],
        1: [250, 282],
        2: [500, 582],
        3: [750, 782],
    }

    # get group ids
    group_ids = rec.get_channel_groups()

    x_locs = rec.get_channel_locations()[:, 0]
    for shank_id, shank_x in shank_x_locs.items():
        # map bool channel x locations
        shank_bool = np.isin(x_locs, shank_x)
        if np.any(shank_bool) == False:
            logging.info(
                f"\n> Recording does not have shank {shank_id}, continue."
            )
            continue
        group_ids[shank_bool] = shank_id

    logging.info(
        "\n> Not all shanks used in multishank probe, change group ids into "
        f"{np.unique(group_ids)}."
    )

    return group_ids


def get_vr_positional_data(trial_data):
    """
    Get positional firing rate and spike count for VR behaviour.

    params
    ===
    trial_data: pandas df, output from align_trials.

    return
    ===
    dict, positional firing rate, positional spike count, positional occupancy,
    data in 1cm resolution.
    """
    pos_fr, occupancy = _get_vr_positional_neural_data(
        positions=trial_data["positions"],
        data_type="spike_rate",
        data=trial_data["fr"],
    )
    pos_fc, _ = _get_vr_positional_neural_data(
        positions=trial_data["positions"],
        data_type="spiked",
        data=trial_data["spiked"],
    )

    return {"pos_fr": pos_fr, "pos_fc": pos_fc, "occupancy": occupancy}


def _get_vr_positional_neural_data(positions, data_type, data):
    """
    Get positional neural data for VR behaviour.

    params
    ===
    positions: pandas df, vr positions of all trials.
        shape: time x trials.
    
    data_type: str, type of neural data.
        "spike_rate": firing rate of each unit in each trial.
        "spiked": spike boolean of each unit in each trial.

    data: pandas df, aligned trial firing rate or spike boolean.
        shape: time x (unit x trial)
        levels: unit, trial

    return
    ===
    pos_data: pandas df, positional neural data.
        shape: position x (num of starting positions x unit x trial)
        levels: start, unit, trial

    occupancy: pandas df, count of each position.
        shape: position x trial
    """
    logging.info(f"\n> Getting positional {data_type}...")

    # get constants from vd
    from vision_in_darkness.constants import TUNNEL_RESET, ZONE_END

    # get the starting index for each trial (column)
    starts = positions.iloc[0, :].astype(int)
    # create position indices
    indices = np.arange(0, TUNNEL_RESET+2)
    # create occupancy array for trials
    occupancy = np.full(
        (TUNNEL_RESET+2, positions.shape[1]),
        np.nan,
    )

    pos_data = {}
    for t, trial in enumerate(positions):
        # get trial position
        trial_pos = positions[trial].dropna()

        # floor pre reward zone and end ceil post zone end
        trial_pos = trial_pos.apply(
            lambda x: np.floor(x) if x <= ZONE_END else np.ceil(x)
        )
        # set to int
        trial_pos = trial_pos.astype(int)

        # exclude positions after tunnel reset
        trial_pos = trial_pos[trial_pos <= TUNNEL_RESET+1]

        # get firing rates for current trial of all units
        trial_data = data.xs(
            key=trial,
            axis=1,
            level="trial",
        ).dropna(how="all").copy()

        # get all indices before post reset
        no_post_reset = trial_data.index.intersection(trial_pos.index)
        # remove post reset rows
        trial_data = trial_data.loc[no_post_reset]
        trial_pos = trial_pos.loc[no_post_reset]

        # put trial positions in trial data df
        trial_data["position"] = trial_pos.values

        if data_type == "spike_rate":
            # group values by position and get mean data
            how = "mean"
        elif data_type == "spiked":
            # group values by position and get sum data
            how = "sum"
        grouped_data = group_and_aggregate(trial_data, "position", how)

        # reindex into full tunnel length
        pos_data[trial] = grouped_data.reindex(indices)
        # get trial occupancy
        pos_count = trial_data.groupby("position").size()
        occupancy[pos_count.index.values, t] = pos_count.values

    # concatenate dfs
    pos_data = pd.concat(pos_data, axis=1, names=["trial", "unit"])
    # convert to df
    occupancy = pd.DataFrame(
        data=occupancy,
        index=indices,
        columns=positions.columns,
    )

    # add another level of starting position
    # group trials by their starting index
    trial_level = pos_data.columns.get_level_values("trial")
    unit_level = pos_data.columns.get_level_values("unit")
    # map start level
    start_level = trial_level.map(starts)
    # define new columns
    new_cols = pd.MultiIndex.from_arrays(
        [start_level, unit_level, trial_level],
        names=["start", "unit", "trial"],
    )
    pos_data.columns = new_cols
    # sort by unit
    pos_data = pos_data.sort_index(level="unit", axis=1)

    return pos_data, occupancy
