"""
This module provides utilities for pixels data.
"""
# annotations not evaluated at runtime
from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import zarr
import gc

import xarray as xr
from numcodecs import Blosc, VLenUTF8

import numpy as np
import pandas as pd

from scipy import stats
from scipy.ndimage import median_filter
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from patsy import build_design_matrices

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.curation as sc
import spikeinterface.exporters as sexp
import spikeinterface.preprocessing as spre
import spikeinterface.postprocessing as spost

import pixels.signal_utils as signal
from pixels import ioutils
from pixels.error import PixelsError
from pixels.configs import *
from pixels.constants import *
from pixels.decorators import _df_to_zarr_via_xarray

from common_utils import math_utils
from vision_in_darkness.constants import landmarks, SPATIAL_SAMPLE_RATE

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


def preprocess_raw(rec, surface_depths, faulty_channels):
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
            cleaned = _preprocess_raw(group, surface_depth, faulty_channels[g])
            preprocessed.append(cleaned)
        # aggregate groups together
        preprocessed = si.aggregate_channels(preprocessed)
    else:
        # if only one shank used, check which shank
        unique_id = np.unique(group_ids)[0]
        # get brain surface depth of shank
        surface_depth = surface_depths[unique_id]
        # preprocess
        preprocessed = _preprocess_raw(
            rec,
            surface_depth,
            faulty_channels[unique_id],
        )

    return preprocessed


def _preprocess_raw(rec, surface_depth, faulty_channels):
    """
    Implementation of preprocessing on raw pixels data.
    """
    # correct phase shift
    print("\t> step 1: do phase shift correction.")
    rec_ps = spre.phase_shift(rec)
    
    # remove bad channels from sorting
    print("\t> step 2: remove bad channels.")
    # remove pre-identified bad channels
    chan_names = rec_ps.get_property("channel_name")
    faulty_ids = rec_ps.channel_ids[np.isin(chan_names, faulty_channels)]
    rec_removed = rec_ps.remove_channels(faulty_ids)

    # detect bad channels
    bad_chan_ids, chan_labels = spre.detect_bad_channels(
        rec_removed,
        outside_channels_location="top",
    )
    labels, counts = np.unique(chan_labels, return_counts=True)
    for label, count in zip(labels, counts):
        print(f"\t\t> Found {count} channels labelled as {label}.")
    rec_removed = rec_removed.remove_channels(bad_chan_ids)

    # get channel group id and use it to index into brain surface channel depth
    shank_id = np.unique(rec_removed.get_channel_groups())[0]
    # get channel depths
    chan_depths = rec_removed.get_channel_locations()[:, 1]
    # get channel ids
    chan_ids = rec_removed.channel_ids
    # remove channels outside by using identified brain surface depths
    outside_chan_ids = chan_ids[chan_depths > surface_depth]
    rec_clean = rec_removed.remove_channels(outside_chan_ids)
    print(
        f"\t\t> Removed {outside_chan_ids.size} outside channels "
        f"above {surface_depth}um."
    )

    return rec_clean


def CMR(rec, dtype=np.int16):
    cmr = spre.common_reference(
        rec,
        operator="median",
        dtype=dtype,
    )
    return cmr


def CAR(rec, dtype=np.int16):
    car = spre.common_reference(
        rec,
        operator="average",
        dtype=np.int16,
    )
    return car


def correct_lfp_motion(rec, mc_method="dredge"):
    if mc_method == "dredge":
        em_method = mc_method+"_lfp"
    else:
        em_method = spre.motion.motion_options_preset[mc_method][
            "estimate_motion_kwargs"
        ]["method"]
    raise NotImplementedError("> Not implemented.")


def correct_ap_motion(rec, output, mc_method="dredge"):
    """
    Correct motion of recording.

    params
    ===
    output: str or path, output path.

    mc_method: str, motion correction method.
        Default: "dredge".
            (as of jan 2025, dredge performs better than ks motion correction.)
        "ks": let kilosort do motion correction.

    return
    ===
    None
    """
    logging.info(f"\n> Correcting motion with {mc_method}.")

    if mc_method == "dredge":
        em_method = mc_method+"_ap"
    else:
        em_method = spre.motion.motion_options_preset[mc_method][
            "estimate_motion_kwargs"
        ]["method"]

    # reduce spatial window size for four-shank
    estimate_motion_kwargs = {
        "method": f"{em_method}",
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

    # save here and load later so that the recording object is an si extractor
    mcd.save(
        format="zarr",
        folder=output,
        compressor=wv_compressor,
        max_threads_per_worker=1, # TODO nov 13 2025: otherwise it hangs
    )

    return None


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

    per_shank: bool, whether to sort recording per shank.
        Default: False (as of may 2025, sort shanks separately by ks4 gives less
        units)

    return
    ===
    sorting: spikeinterface sorting object.

    recording: spikeinterface recording object.
    """

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
    sa = curate_sorting(
        sorting,
        recording,
        output,
    )
    sa, curated_sa = curate_sorting_analyser(sa)

    # export sorting analyser
    export_sorting_analyser(
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

    # NOTE: may 20 2025
    # build sa with non-whitened preprocessed rec gives amp between 0-250uV,
    # which makes sense, and quality metric amp_median is comparable across
    # recordings
    recording = sa_rec

    return sorting, recording


def curate_sorting(sorting, recording, output):
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
        sparse=True,
        format="zarr",
        folder=output/"sa.zarr",
        overwrite=True,
    )

    return sa


def curate_sorting_analyser(sa):
    # calculate all extensions BEFORE further steps
    # basics
    required_extensions = [
        "spike_amplitudes",
        "random_spikes",
        "waveforms",
        "templates",
        "noise_levels",
        "unit_locations",
        "template_similarity",
        "correlograms",
    ]
    sa.compute(
        required_extensions,
        save=True,
    )

    # pca
    spost.compute_principal_components(sa, n_jobs=1)

    # metrics
    ext_params = {
        "template_metrics": {
            "include_multi_channel_metrics": True,
        },
        #"quality_metrics": {
        #    "skip_pc_metrics": True,
        #},
    }
    sa.compute(
        ["quality_metrics", "template_metrics"],
        save=True,
        extension_params=ext_params,
    )

    #import spikeinterface.qualitymetrics as sqm
    # NOTE nov 13 2025: pc metrics only runs if n_jobs=1
    #sqm.compute_quality_metrics(sa, n_jobs=1)

    # get max peak channel for each unit
    max_chan = si.get_template_extremum_channel(sa).values()

    # make sure to have group id for each unit
    if not "group" in sa.sorting.get_property_keys():
        # get shank id, i.e., group
        group = sa.recording.get_channel_groups()
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

    # >>> get depth of units on each shank >>>
    # get probe geometry coordinates
    coords = sa.get_channel_locations()
    # get coordinates of max channel of each unit on probe, column 0 is
    # x-axis, column 1 is y-axis/depth, 0 at bottom-left channel.
    max_chan_idx = sa.channel_ids_to_indices(max_chan)
    max_chan_coords = coords[max_chan_idx]
    # set coordinates of max channel of each unit as a property of sorting
    sa.sorting.set_property(
        key="max_chan_coords",
        values=max_chan_coords,
    )
    # <<< get depth of units on each shank <<<

    # remove bad units using metrics
    good_units, soma_units = curate_units(sa)

    # get unit ids
    curated_unit_ids = np.intersect1d(good_units, soma_units)
    # select curated
    curated_sorting = sa.sorting.select_units(curated_unit_ids)
    curated_sa = sa.select_units(curated_unit_ids)
    # reattach curated sorting to curated_sa to keep sorting properties
    curated_sa.sorting = curated_sorting

    return sa, curated_sa


def curate_units(sa):
    # get quality metrics
    qms = sa.get_extension("quality_metrics").get_data()

    # remove bad units
    good_qms = qms.query(qms_rule)
    logging.info(
        "\n> quality metrics check removed "
        f"{np.setdiff1d(sa.unit_ids, good_qms.index.values)}, "
        f"{len(np.setdiff1d(sa.unit_ids, good_qms.index.values))} "
        "units in total."
    )

    # get template metrics
    tms = sa.get_extension("template_metrics").get_data()

    # remove noise based on waveform
    good_tms = tms.query(tms_rule)
    logging.info(
        "\n> Template metrics check removed "
        f"{np.setdiff1d(sa.unit_ids, good_tms.index.values)}, "
        f"{len(np.setdiff1d(sa.unit_ids, good_tms.index.values))} "
        "units in total."
    )

    # get good units that passed quality metrics & template metrics
    good_units = np.intersect1d(good_qms.index.values, good_tms.index.values)
    good_unit_mask = np.isin(sa.unit_ids, good_units)

    # get max peak channel for each unit
    max_chan = si.get_template_extremum_channel(sa).values()
    max_chan_idx = sa.channel_ids_to_indices(max_chan)

    # get template of each unit on its max channel
    templates = sa.load_extension("templates").get_data()
    unit_idx = sa.sorting.ids_to_indices(good_units)
    max_chan_templates = templates[unit_idx, :, max_chan_idx[good_unit_mask]]

    # filter non somatic units by waveform analysis
    soma_mask = filter_non_somatics(
        good_units,
        max_chan_templates,
        sa.sampling_frequency,
    )
    soma_units = good_units[soma_mask] 

    return good_units, soma_units


def export_sorting_analyser(sa, curated_sa, output, curated_sa_dir,
                             to_phy=False):
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

    # save sa to disk
    curated_sa.save_as(
        format="zarr",
        folder=curated_sa_dir,
    )

    # export curated report
    sexp.export_report(
        sorting_analyzer=curated_sa,
        output_folder=output/"curated_report",
    )

    # export pre curation report
    sexp.export_report(
        sorting_analyzer=sa,
        output_folder=output/"report",
    )

    if to_phy:
        export_sa_to_phy(output, sa)

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


def _worker_write_repeat(i, zarr_path, sigma, sample_rate, spike_meta):
    # attach to shared memory
    spiked, spike_shms = ioutils.import_df_to_shm(spike_meta)
    try:
        # child process re-opens the store to avoid pickling big arrays
        root = zarr.open_group(
            store=zarr.DirectoryStore(zarr_path),
            mode="a",
        )

        # get permuted data
        c_spiked, c_fr = _permute_spikes_n_convolve_fr(
            array=spiked.to_numpy(),
            sigma=sigma,
            sample_rate=sample_rate,
        )
        # write the i-th slice along last axis
        root["chance_spiked"][..., i] = c_spiked
        root["chance_fr"][..., i] = c_fr

        del spiked, c_spiked, c_fr
        gc.collect()

        logging.info(f"\nRepeat {i} finished.")
    finally:
        for shm in spike_shms:
            shm.close()

    return None


def save_spike_chance_zarr(
    zarr_path,
    spiked: np.ndarray,
    sigma: float,
    sample_rate: float,
    repeats: int = 100,
    positions=None,
    meta: dict | None = None,
):
    """
    Create a Zarr store at `zarr_path` with datasets:
      - spiked: base spiked array (read-only reference)
      - chance_spiked: base_shape + (repeats,), int16
      - chance_fr: base_shape + (repeats,), float32
      - positions: optional small array (or vector), stored if provided

    Then fill each repeat slice in parallel processes.
    This function is idempotent: if the target datasets exist and match shape, it skips creation.
    """
    n_workers = 2 ** (mp.cpu_count().bit_length() - 2)

    zarr_path = Path(zarr_path)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)

    # add spiked to shared memory
    spike_meta, spike_shms = ioutils.export_df_to_shm(spiked)

    base_shape = spiked.shape
    d_shape = base_shape + (repeats,)

    chunks = tuple(min(s, BIG_CHUNKS) for s in base_shape) + (1,)

    store = zarr.DirectoryStore(str(zarr_path))
    root = zarr.group(store=store, overwrite=not zarr_path.exists())

    # Metadata
    root.attrs["kind"] = "spike_chance"
    root.attrs["sigma"] = float(sigma)
    root.attrs["sample_rate"] = float(sample_rate)
    root.attrs["repeats"] = int(repeats)
    if meta:
        for k, v in meta.items():
            root.attrs[k] = v

    logging.info(f"\n> Creating zarr dataset.")
    # Base source so workers can read it without pickling
    if "spiked" in root:
        del root["spiked"]

    if isinstance(spiked, pd.DataFrame):
        _df_to_zarr_via_xarray(
            df=spiked,
            store=store,
            group_name="spiked",
            compressor=compressor,
            mode="w",
        )
    else:
        root.create_dataset(
            "spiked",
            data=spiked,
            chunks=chunks[:-1],
            compressor=compressor,
        )
    del spiked
    gc.collect()

    # Outputs
    if "chance_spiked" in root\
        and tuple(root["chance_spiked"].shape) != d_shape:
        del root["chance_spiked"]
    if "chance_fr" in root and tuple(root["chance_fr"].shape) != d_shape:
        del root["chance_fr"]

    if "chance_spiked" not in root:
        root.create_dataset(
            "chance_spiked",
            shape=d_shape,
            dtype="bool",
            chunks=chunks,
            compressor=compressor,
        )
    if "chance_fr" not in root:
        root.create_dataset(
            "chance_fr",
            shape=d_shape,
            dtype="float32",
            chunks=chunks,
            compressor=compressor,
        )

    # save positions
    if positions is not None:
        if "positions" in root:
            del root["positions"]

        if isinstance(positions, pd.DataFrame):
            _df_to_zarr_via_xarray(
                df=positions,
                store=store,
                group_name="positions",
                compressor=compressor,
                mode="w",
            )
        else:
            root.create_dataset(
                "positions",
                data=positions,
                chunks=True,
                compressor=compressor,
            )

    del positions
    gc.collect()

    logging.info(f"\n> Starting process pool.")

    # parallel fill: each worker writes a distinct final-axis slice
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(
                _worker_write_repeat,
                i,
                str(zarr_path),
                sigma,
                sample_rate,
                spike_meta,
            ) for i in range(repeats)
        ]
        for f in as_completed(futures):
            f.result()  # raise on error

    for shm in spike_shms:
        shm.close()
        shm.unlink()

    return None


def bin_spike_chance(
    chance_data, sample_rate, time_bin, pos_bin, arr_path, units, trial_ids,
    event_on_t, event_off_t
):
    # TODO nov 25 2025:
    # 1. allows to align to specific event and select units like
    # `save_chance_psd`; 
    # 2. implement multiprocessing here!

    # extract data from chance
    (idx_shms, cols_shms, positions_shms, mask_shm,
     idx_meta, cols_meta, positions_meta, mask_meta, 
     fr_zarr, spiked_zarr, repeats, unit_ids, n_workers) = prep_chance_data(
        chance_data,
        units,
        trial_ids,
        event_on_t,
        event_off_t,
    )
    del chance_data
    gc.collect()

    futures = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(
                _bin_chance_worker,
                fr_zarr,
                spiked_zarr,
                r,
                positions_meta,
                idx_meta,
                cols_meta,
                mask_meta,
                unit_ids,
                trial_ids,
                sample_rate,
                time_bin,
                pos_bin,
            ) for r in range(repeats)
        ]
        # collect and concat
        results = [f.result() for f in as_completed(futures)]

    for shm in idx_shms + cols_shms + positions_shms + [mask_shm]:
        shm.close()
        shm.unlink()

    # save np array, for andrew
    arr_count_output = np.stack(
        [result["count_arr"] for result in results],
        axis=-1,
        dtype=np.float32,
    )
    arr_fr_output = np.stack(
        [result["fr_arr"] for result in results],
        axis=-1,
        dtype=np.float32,
    )
    arrs = {
        "count": arr_fr_output[:, :-2, ...],
        "fr": arr_count_output[:, :-2, ...],
        "pos": arr_fr_output[:, -2:, ...],
    }
    np.savez_compressed(arr_path, **arrs)

    # output df
    df_fr = pd.concat(
        {r: result["fr_df"] for r, result in enumerate(results)},
        axis=0,
        names=["repeat", "trial", "bin_time"],
    ).iloc[:, :-2]
    temp = pd.concat(
        {r: result["count_df"] for r, result in enumerate(results)},
        axis=0,
        names=["repeat", "trial", "bin_time"],
    )
    df_spiked = temp.iloc[:, :-2]
    df_pos = temp.iloc[:, -2:]
    # remove position columns name 
    df_pos.columns.name = None

    return {"spiked": df_spiked, "fr": df_fr, "positions": df_pos}


def _bin_chance_worker(
    fr_zarr, spiked_zarr, r, positions_meta, idx_meta, cols_meta, mask_meta,
    unit_ids, trial_ids, sample_rate, time_bin, pos_bin,
):
    """
    Worker that bins chance data.

    params
    ===

    return
    ===
    """
    # attach to shared memory and rebuild the indices
    idx, idx_shms = ioutils.import_multiindex_to_shm(idx_meta)
    cols, cols_shms = ioutils.import_index_to_shm(cols_meta)
    positions, positions_shms = ioutils.import_df_to_shm(positions_meta)
    mask, mask_shm = ioutils._from_shm(mask_meta)

    try:
        spiked = pd.DataFrame(
            spiked_zarr[..., r],
            index=idx,
            columns=cols,
        ).loc[mask, unit_ids]

        fr = pd.DataFrame(
           fr_zarr[..., r],
           index=idx,
           columns=cols,
        ).loc[mask, unit_ids]

        temp_spiked = {}
        temp_fr = {}
        for trial in trial_ids:
            # bin fr
            temp_fr[trial] = bin_vr_trial(
                data=fr.xs(trial, level="trial", axis=0),
                positions=positions.xs(trial, level="trial", axis=1).dropna(),
                sample_rate=sample_rate,
                time_bin=time_bin,
                pos_bin=pos_bin,
                bin_method="mean", # fr
            )
            # bin spiked
            temp_spiked[trial] = bin_vr_trial(
                data=spiked.xs(trial, level="trial", axis=0),
                positions=positions.xs(trial, level="trial", axis=1).dropna(),
                sample_rate=sample_rate,
                time_bin=time_bin,
                pos_bin=pos_bin,
                bin_method="sum", # spike count
            )

        fr_df = pd.concat(temp_fr, axis=0)
        count_df = pd.concat(temp_spiked, axis=0)

        # np array for andrew
        fr_arr = ioutils.reindex_by_longest(
            dfs=temp_fr,
            return_format="array",
        )
        count_arr = ioutils.reindex_by_longest(
            dfs=temp_spiked,
            return_format="array",
        )

        del (temp_fr, temp_spiked, positions, idx, cols, mask, trial_ids,
             unit_ids, time_bin, pos_bin)
        gc.collect()

        logging.info(f"\nRepeat {r} finished.")
    finally:
        for shm in idx_shms + cols_shms + positions_shms + [mask_shm]:
            shm.close()

    return {"fr_df": fr_df, "count_df": count_df, "fr_arr": fr_arr,
            "count_arr": count_arr}


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

    # resample to ms bin, and get position mean
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

    # get group ids
    group_ids = rec.get_channel_groups()

    # correct only if it is multishank probe
    if probe_type > 0:
        # get channel x locations
        shank_x_locs = {
            0: [0, 32],
            1: [250, 282],
            2: [500, 532],
            3: [750, 782],
        }

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
    # NOTE: take occupancy from spike count since in we might need to
    # interpolate fr for binned data
    pos_fc, occupancy = _get_vr_positional_neural_data(
        positions=trial_data["positions"],
        data_type="spiked",
        data=trial_data["spiked"],
    )
    pos_fr, _ = _get_vr_positional_neural_data(
        positions=trial_data["positions"],
        data_type="spike_rate",
        data=trial_data["fr"],
    )

    return {"pos_fc": pos_fc, "pos_fr": pos_fr, "occupancy": occupancy}


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
    from pandas.api.types import is_integer_dtype

    if "bin" in positions.index.name:
        logging.info(f"\n> Getting binned positional {data_type}...")
        # create position indices for binned data
        indices_range = [
            positions.min().min(),
            positions.max().max()+1,
        ]
    else:
        logging.info(f"\n> Getting positional {data_type}...")
        # get constants from vd
        from vision_in_darkness.constants import TUNNEL_RESET
        # create position indices
        indices_range = [np.floor(positions.min().min()), TUNNEL_RESET+1]

    # get trial ids
    trial_ids = positions.columns.get_level_values("trial")

    # create position indices
    indices = np.arange(*indices_range, SPATIAL_SAMPLE_RATE).astype(int)
    # create occupancy array for trials
    occupancy = pd.DataFrame(
        data=np.full((len(indices), positions.shape[1]), np.nan),
        index=indices,
        columns=trial_ids,
    )

    pos_data = {}
    for t, trial in enumerate(trial_ids):
        # get trial position
        trial_pos = positions.xs(trial, level="trial", axis=1).dropna()

        # convert to int if float 
        if not trial_pos.dtypes.map(is_integer_dtype).all():
            # floor position and set to int
            trial_pos = trial_pos.apply(lambda x: np.floor(x)).astype(int)
            # exclude positions after tunnel reset
            trial_pos = trial_pos[trial_pos <= indices[-1]]

        # get firing rates for current trial of all units
        try:
            trial_data = data.xs(
                key=trial,
                axis=1,
                level="trial",
            ).dropna(how="all").copy()
        except TypeError:
            # chance data has trial and time on index, not column
            trial_data = data.T.xs(
                key=trial,
                axis=1,
                level="trial",
            ).T.dropna(how="all").copy()

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
        grouped_data = math_utils.group_and_aggregate(
            trial_data,
            "position",
            how,
        )

        # reindex into full tunnel length
        reidxed = grouped_data.reindex(indices)

        # check for missing values in binned data
        if ("bin" in positions.index.name) and (data_type == "spike_rate"):
            # remove alll nan before data actually starts
            start_idx = grouped_data.index[0]
            chunk_data = reidxed.loc[start_idx:, :]
            nan_check = chunk_data.isna().any().any()
            if nan_check:
                # interpolate missing fr
                logging.info(f"\n> trial {trial} has missing values, "
                             "do linear interpolation.")
                reidxed.loc[start_idx:, :] = chunk_data.interpolate(
                    method="linear",
                    axis=0,
                )

        # save to dict
        pos_data[trial] = reidxed

        # get trial occupancy
        pos_count = trial_data.groupby("position").size()
        occupancy.loc[pos_count.index.values, trial] = pos_count.values

    # concatenate dfs
    pos_data = pd.concat(pos_data, axis=1, names=["trial", "unit"])

    # add another level of starting position
    # group trials by their starting index
    trial_level = pos_data.columns.get_level_values("trial")
    unit_level = pos_data.columns.get_level_values("unit")
    # map start level
    starts = positions.columns.get_level_values("start").values
    start_series = pd.Series(
        data=starts,
        index=trial_ids,
        name="start",
    )
    start_level = trial_level.map(start_series)

    # define new columns
    new_cols = pd.MultiIndex.from_arrays(
        [start_level, unit_level, trial_level],
        names=["start", "unit", "trial"],
    )
    pos_data.columns = new_cols

    # sort by unit, starting position, and then trial
    pos_data = pos_data.sort_index(
        axis=1,
        level=["unit", "start", "trial"],
        ascending=[True, False, True],
    ).dropna(how="all")

    occupancy = occupancy.dropna(how="all")
    # remove negative position values
    if occupancy.index.min() < 0:
        occupancy = occupancy.loc[0:, :]
        pos_data = pos_data.loc[0:, :]
    occupancy.index.name = "position"

    return pos_data, occupancy


def get_psd(df, fs, nperseg):
    """
    Compute power spectrum density.

    params
    ===
    df: pandas dataframe. y-axis: position/time, x-axis: units, trial.

    return
    ===
    psd
    """
    def _compute_psd(col):
        x = col.dropna().values.squeeze()
        f, psd = math_utils.estimate_power_spectrum(
            x,
            fs=fs,
            nperseg=nperseg,
            use_welch=True,
        )
        ser = pd.Series(psd, index=f, name=col.name)
        ser.index.name = "frequency"

        return ser

    psd = df.apply(_compute_psd, axis=0)

    return psd


def _psd_chance_worker(
    fr_zarr, r, fs, nperseg_max, positions_meta, idx_meta, cols_meta, mask_meta,
    unit_ids, trial_ids,
):
    """
    Worker that computes one set of psd.

    params
    ===
    i: index of current repeat.

    return
    ===
    """
    # attach to shared memory and rebuild the indices
    idx, idx_shms = ioutils.import_multiindex_to_shm(idx_meta)
    cols, cols_shms = ioutils.import_index_to_shm(cols_meta)
    positions, positions_shms = ioutils.import_df_to_shm(positions_meta)
    mask, mask_shm = ioutils._from_shm(mask_meta)

    try:
        pos_fr, _ = _get_vr_positional_neural_data(
            positions=positions,
            data_type="spike_rate",
            data=pd.DataFrame(
                fr_zarr[..., r],
                index=idx,
                columns=cols,
            ).loc[mask, unit_ids],
        )
        del positions, idx, cols
        gc.collect()

        psd = {}
        starts = pos_fr.columns.get_level_values("start").unique()

        for start in starts:
            start_df = (
                pos_fr.xs(
                    start,
                    level="start",
                    axis=1,
                ).loc[start:, ].dropna(how="all")
                .loc[landmarks[0]:, :] # crop from blackwall
            )

            # get nperseg based on trial length
            if np.all(start_df.shape[0] > p_npersegs):
                nperseg = nperseg_max
            elif np.any(start_df.shape[0] > p_npersegs):
                nperseg = p_npersegs[
                    np.where(start_df.shape[0] > p_npersegs)[0]
                ].max()
            else:
                nperseg = start_df.shape[0]

            psd[start] = get_psd(start_df, fs, nperseg)

            del start_df
            gc.collect()

        del pos_fr, starts
        gc.collect()

        psd_df = pd.concat(
            psd,
            names=["start", "frequency"],
        )
        logging.info(f"\nRepeat {r} finished.")
    finally:
        for shm in idx_shms + cols_shms + positions_shms + [mask_shm]:
            shm.close()

    return psd_df


def save_chance_psd(
    chance_data, fs, units, trial_ids, event_on_t, event_off_t,
):
    """
    Implementation of saving chance level spike data.
    """
    (idx_shms, cols_shms, positions_shms, mask_shm,
     idx_meta, cols_meta, positions_meta, mask_meta, 
     fr_zarr, _, repeats, unit_ids, n_workers) = prep_chance_data(
        chance_data,
        units,
        trial_ids,
        event_on_t,
        event_off_t,
    )
    del chance_data
    gc.collect()

    # get max possible number of frequencies per segment
    nperseg_max = max(int(fs / (1 / T_SEG)), p_npersegs.max())

    futures = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(
                _psd_chance_worker,
                fr_zarr,
                r,
                fs,
                nperseg_max,
                positions_meta,
                idx_meta,
                cols_meta,
                mask_meta,
                unit_ids,
                trial_ids,
            ) for r in range(repeats)
        ]
        # collect and concat
        results = [f.result() for f in as_completed(futures)]

    psds = pd.concat(
        results,
        axis=1,
        keys=range(repeats),
        names=["repeat", "unit", "trial"],
    )

    for shm in idx_shms + cols_shms + positions_shms:
        shm.close()
        shm.unlink()

    return psds


def notch_freq(rec, freq, bw=4.0):
    """
    Notch a frequency with narrow bandwidth.

    params
    ===
    rec: si recording object.

    freq: float or int, the target frequency in Hz of the notch filter.

    bw: float or int, bandwidth (Hz) of notch filter.
        Default: 4.0Hz.

    return
    ===
    notched: spikeinterface recording object.
    """
    notched = spre.notch_filter(
        rec,
        freq=freq,
        q=freq/bw, # quality factor
    )

    return notched


# >>> landmark responsive helpers >>>
def to_df(mean, std, zone):
    out = pd.DataFrame({"mean": mean, "std": std}).reset_index()

    # Keep only start and trial; unit is constant
    out = out.rename(
        columns={"level_0": "start", "level_1": "unit", "level_2": "trial"}
    )
    out["zone"] = zone

    # map data type
    out["start"] = out["start"].astype(str)
    out["unit"] = out["unit"].astype(str)
    out["trial"] = out["trial"].astype(str)

    return out


# Build linear contrasts for any model that uses patsy coding
def compute_contrast(fit, newdf_a, newdf_b=None):
    """
    Returns estimate and SE for:
      L'beta where L = X(newdf_a) - X(newdf_b) if newdf_b is provided,
      else L = X(newdf_a)
    fit: statsmodels results with fixed effects (OLS or MixedLM)
    newdf_a/newdf_b: small DataFrames with columns used in the formula (start,
    zone)
    """
    # For MixedLM, use fixed-effects params/cov
    if hasattr(fit, "fe_params"):
        fe_params = fit.fe_params
        cov = fit.cov_params().loc[fe_params.index, fe_params.index]
        cols = fe_params.index
    else:
        # OLS
        fe_params = fit.params
        cov = fit.cov_params()
        cols = fit.params.index

    di = fit.model.data.design_info

    Xa = build_design_matrices([di], newdf_a)[0]
    Xa = np.asarray(Xa)  # column order matches the fit
    if Xa.shape[1] != len(cols):
        # rare: ensure columns align if needed
        raise ValueError(
            "Design column mismatch; "
            "ensure newdf has the same factors and levels as the fit."
        )

    if newdf_b is not None:
        Xb = build_design_matrices([di], newdf_b)[0]
        Xb = np.asarray(Xb)
        L = (Xa - Xb).ravel()
    else:
        L = Xa.ravel()

    est = float(L @ fe_params.values)
    se = float(np.sqrt(L @ cov.values @ L))

    return est, se


# >>> single unit mixed model
def fit_per_unit_ols(df, formula):
    """
    Step 1
    Fit mean fr of pre-wall, landmark, and post-wall from each trial, each unit
    to GLM with cluster-robust SE.
    """
    # OLS with cluster-robust SE by trial
    fit = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["trial"]},
    )
    return fit


def test_diff_any(fit, starts, use_f=True):
    """
    Step 2
    Use Wald test on linear contrasts to test if jointly, all these contrasts
    are 0.
    i.e., this test if there are any difference among the mean fr comparisons.
    if wald p < alpha, then there is a significant difference, we do post-hoc to
    see where the difference come from;
    if wald p > alpha, the unit does not have different fr between landmark &
    pre-wall, and landmark & post-wall, or pre-wall & post-wall.
    """
    Ls = []
    for s in starts:
        for ref in ["pre_wall", "post_wall"]:
            row = _L_row(
                fit=fit,
                start_label=s,
                a_zone="landmark",
                b_zone=ref,
            )
            Ls.append(row)

    R = np.vstack(Ls)
    w = fit.wald_test(R, scalar=True, use_f=use_f)
    results = {
        "stat": float(w.statistic),
        "p": float(w.pvalue),
        "df_num": int(getattr(w, "df_num", R.shape[0])),
        "df_denom": float(getattr(w, "df_denom", np.nan)),
        "k": R.shape[0],
    }

    return results["p"]


def _L_row(fit, start_label, a_zone, b_zone):
    """
    Build linear contrasts of zones for a given starting position.
    """
    di = fit.model.data.design_info
    Xa = np.asarray(
        build_design_matrices(
            [di],
            pd.DataFrame({"start":[start_label], "zone":[a_zone]}),
        )[0]
    ).ravel()
    Xb = np.asarray(
        build_design_matrices(
            [di],
            pd.DataFrame({"start":[start_label], "zone":[b_zone]})
        )[0]
    ).ravel()

    return Xa - Xb


def family_comparison(fit, starts, compare_to="pre_wall", use_f=True):
    """
    Step 3
    Family level mean comparison, i.e., compare pre or post wall with landmark.
    """
    R = np.vstack(
        [_L_row(fit, s, "landmark", compare_to) for s in starts]
    )
    w = fit.wald_test(R, scalar=True, use_f=use_f)

    results = {
        "family": f"LM-{ 'Pre' if compare_to=='pre_wall' else 'Post' }",
        "stat": float(w.statistic),
        "p": float(w.pvalue),
        "df_num": int(getattr(w, "df_num", R.shape[0])),
        "df_denom": float(getattr(w, "df_denom", np.nan)),
        "n_starts": len(starts),
        "R": R
    }
    return results["p"]


def start_contrasts_ols(fit, starts, use_normal=True):
    """
    Step 4
    Post-hoc test to see where the difference in contrast come from, i.e., get
    the linear contrast for each starting positions.
    """
    params = fit.params if not hasattr(fit, "fe_params")\
            else fit.fe_params
    cov = fit.cov_params() if not hasattr(fit, "fe_params")\
            else fit.cov_params().loc[params.index, params.index]

    rows = []
    for s in sorted(starts, key=str):
        df_pre = pd.DataFrame({"start":[s], "zone":["pre_wall"]})
        df_lm  = pd.DataFrame({"start":[s], "zone":["landmark"]})
        df_post = pd.DataFrame({"start":[s], "zone":["post_wall"]})

        for label, A, B in [("lm-pre", df_lm, df_pre),
                            ("lm-post", df_lm, df_post)]:
            est, se = compute_contrast(fit, A, B)
            if not np.isfinite(se) or se <= 0:
                stat = p = np.nan
            else:
                stat = est / se
                p = float(
                    2 * (stats.norm.sf(abs(stat)) if use_normal
                   else stats.t.sf(abs(stat), df=fit.df_resid))
                )
            col_names = ["start", "contrast", "coef", "SE", "stat", "p"]
            rows.append(
                dict(zip(col_names, [s, label, est, se, stat, p]))
            )

    out = pd.DataFrame(rows)
    if not out.empty and out["p"].notna().any():
        # get Holm-adjusted p value to correct for multiple comparison
        out["p_holm"] = multipletests(
            out["p"],
            alpha=ALPHA,
            method="holm",
        )[1]

    # rename 'stat' to 'z' or 't'
    stat_label = "z" if use_normal else "t"
    out = out.rename(columns={"stat": stat_label})

    return out


def test_start_x_zone_interaction_ols(fit):
    # Wald test: all interaction terms = 0
    ix = [i for i, zone in enumerate(fit.params.index) if ":" in zone]
    if not ix:
        return np.nan
    R = np.zeros((len(ix), len(fit.params)))
    for r, i in enumerate(ix):
        R[r, i] = 1.0
    w = fit.wald_test(R, scalar=True, use_f=False)
    stat = w.statistic

    return float(w.statistic), float(w.pvalue)
# <<< single unit mixed model
# <<< landmark responsive helpers <<<


def get_landmark_responsives(pos_fr, units, ons, offs):
    """
    use int8 to encode responsiveness:
    0: not responsive
    1: positively responsive
    -1: negatively responsive
    """
    units = units.flat()

    # get all positions
    positions = pos_fr.index.to_numpy()
    # build mask for all positions
    position_mask = (positions[:, None] >= ons)\
        & (positions[:, None] < offs)

    # get pre wall and trial mask
    pre_wall = pos_fr.loc[position_mask[:, 0], :]
    trials_pre_wall = pre_wall.columns.get_level_values(
        "trial"
    ).unique()

    # get mean & std of walls and landmark
    landmark = pos_fr.loc[position_mask[:, 1], :]
    assert (landmark.columns.get_level_values("trial").unique() ==\
        trials_pre_wall).all()
    post_wall = pos_fr.loc[position_mask[:, 2], :]

    pre_wall_mean = pre_wall.mean(axis=0)
    pre_wall_std = pre_wall.std(axis=0)

    landmark_mean = landmark.mean(axis=0)
    landmark_std = landmark.std(axis=0)

    post_wall_mean = post_wall.mean(axis=0)
    post_wall_std = post_wall.std(axis=0)

    # aggregate
    agg = pd.concat([
        to_df(pre_wall_mean, pre_wall_std, "pre_wall"),
        to_df(landmark_mean, landmark_std, "landmark"),
        to_df(post_wall_mean, post_wall_std, "post_wall"),
        ],
        ignore_index=True,
    )
    agg["zone"] = pd.Categorical(
        agg["zone"],
        categories=["pre_wall", "landmark", "post_wall"],
        ordered=True,
    )

    # get all starting positions
    starts = agg.start.unique()

    # create model formula
    min_start = min(starts)
    simple_model = (
        "mean ~ C(zone, Treatment(reference='pre_wall'))"
    )
    full_model = (
        f"""mean
        ~ C(start, Treatment(reference={min_start!r}))
        * C(zone, Treatment(reference='pre_wall'))"""
    )

    lm_contrasts = {}
    responsives = pd.Series(
        np.zeros(len(units)).astype(np.int8),
        index=units,
    )
    responsives.index.name = "unit"

    # number of params and number of observations
    n_params = agg.start.nunique() * agg.zone.nunique()
    n_observs = agg.trial.nunique() * agg.zone.nunique()
    if not n_observs > n_params:
        logging.info(
            f"\n> We have {n_observs} observations but {n_params} "
            f"parameters, cannot fit to ols."
        )
        col_names = ["start", "contrast", "coef", "SE", "stat", "p", "p_holm"]
        contrast_names = ["lm_pre", "lm_post"]
        df = pd.DataFrame(
            np.full(
                (starts.size * len(contrast_names), len(col_names)),
                np.nan,
            ),
            columns=col_names,
        )
        df["start"] = np.repeat(starts, len(contrast_names))
        df["contrast"] = contrast_names * starts.size
        contrasts = pd.concat([df] * len(units), ignore_index=True)
        contrasts.index = pd.Index(np.repeat(units, len(df)), name="unit")

        return contrasts, responsives

    for unit_id in units:
        df = agg[agg["unit"] == str(unit_id)].copy()
        if df.empty:
            raise ValueError(f"No data for unit {unit_id}")

        unit_fit = fit_per_unit_ols(
            df=df,
            formula=full_model,
            #formula=simple_model,
        )
        # check contrast at each start
        unit_contrasts = start_contrasts_ols(
            fit=unit_fit,
            starts=starts,
        )
        lm_contrasts[unit_id] = unit_contrasts

        if len(starts) >= 2:
            # positive responsive
            if (unit_contrasts.coef > 0).all()\
            and (unit_contrasts.p_holm < ALPHA).all():
                responsives.loc[unit_id] = 1
            # negative responsive
            if (unit_contrasts.coef < 0).all()\
            and (unit_contrasts.p_holm < ALPHA).all():
                responsives.loc[unit_id] = -1

    contrasts = pd.concat(
        lm_contrasts,
        axis=0,
        names=["unit", "index"],
    ).droplevel("index")

    return contrasts, responsives


def filter_non_somatics(unit_ids, templates, sampling_freq):
    # NOTE: no need to worry about multi-positive-peak templates, cuz we already
    # threw them out
    from scipy.signal import find_peaks

    # True means yes somatic, False mean non somatic
    mask = np.zeros(len(templates), dtype=bool)

    # bombcell non somatic criteria
    max_repo_peak_to_trough_ratio = 0.8

    # height ratio to trough
    height_ratio_to_trough = 0.15 #0.2
    # minimum width of the peak for detection
    min_width = 0

    ## minimum width of depo peak for filtering
    #peak_width_ms = 0.1 # 0.07
    #min_depo_peak_width = int(peak_width_ms / 1000 * sampling_freq)

    # NOTE: since our data is high-pass filtered, the very transient peak before
    # could be caused by that, its width does not tell us whether it is a
    # somatic unit or not. DO NOT RELY ON PRE TROUGH PEAK TO DECIDE SOMATIC!
    # so there are two ways to do this:
    # 1. set the detection minimum to min_depo_peak_width like in
    # spikeinterface, so that we just ignore those transient peaks and consider
    # it as high-pass-filter artefact, has no weight in somatic decision;
    # 2. using a very low threshold during detection like what we do here, and
    # check the height ratio of the peak, if it exceeds our maximum, still
    # consider it as non somatic, even if it is very transient.

    # maximum depo peak to repo peak ratio
    max_depo_peak_to_repo_peak_ratio = 1.5
    # maximum depo peak to trough ratio
    max_depo_peak_to_trough_ratio = 0.5

    for t, template in enumerate(templates):
        # get absolute maximum
        template_max = np.max(np.abs(template))
        # minimum prominence of the peak
        prominence = height_ratio_to_trough * template_max
        # get trough index
        trough_idx = np.argmin(template)
        trough_height = np.abs(template)[trough_idx]

        # get positive peaks
        peak_idx, peak_properties = find_peaks(
            x=template,
            prominence=prominence,
            width=min_width,
        )

        if not len(peak_idx) <= 2:
            print(f"unit {unit_ids[t]} has more than 2 peaks")
            continue

        if not len(peak_idx) == 0:
            # maximum positive peak is not larger than 80% of trough
            if not np.abs(template[peak_idx][-1])\
                    / trough_height < max_repo_peak_to_trough_ratio:
                print(
                    f"> {unit_ids[t]} has positive peak larger than 80% trough"
                )
                continue

        if len(peak_idx) > 1:
            # if both peaks before or after trough, consider non somatic
            if (peak_idx > trough_idx).all()\
                or (peak_idx < trough_idx).all():
                print(f"> {unit_ids[t]} both peaks on one side")
                continue

            # check compare bases 
            # if pre depolarisation peak
            assert peak_properties["right_bases"][0] == trough_idx
            # if post repolarisation peak
            assert peak_properties["left_bases"][1] == trough_idx

            peak_heights = template[peak_idx]

            # check depo peak height is 
            if not peak_heights[0] / trough_height\
                    < max_depo_peak_to_trough_ratio:
                print(f"> {unit_ids[t]} depo peak is half the size of trough")
                continue

            # check height ratio of peaks, make sure the depolarisation peak is
            # NOT much bigger than repolarisation peak
            if not peak_heights[0] / peak_heights[1]\
                    < max_depo_peak_to_repo_peak_ratio:
                print(
                    f"> {unit_ids[t]} depo peak is 1.5 times larger than "
                    "the repo"
                )
                continue

        # yes somatic if no positive peaks or pass all checks
        mask[t] = True

    return mask


def prep_chance_data(
    chance_data, units, trial_ids, event_on_t, event_off_t,
):
    """
    Implementation of saving chance level spike data.
    """
    # get index and columns to reconstruct df
    spiked = chance_data["spiked"].dropna(axis=0, how="all")
    idx = spiked.index
    cols = spiked.columns
    cols_meta, cols_shms = ioutils.export_index_to_shm(cols)
    idx_meta, idx_shms = ioutils.export_multiindex_to_shm(idx)
    # get positions
    positions = chance_data["positions"].dropna(axis=1, how="all").loc[
        :, pd.IndexSlice[:, trial_ids] # only keep selected trials
    ]
    positions_meta, positions_shms = ioutils.export_df_to_shm(positions)

    # get index time bound of selected trials
    bound = pd.DataFrame(
        {"on": event_on_t, "off": event_off_t},
        index=trial_ids,
    )

    # map trial on to off, trials not in trial_ids set to NaN
    trial_ons = idx.get_level_values("trial").map(bound.on)
    trial_offs = idx.get_level_values("trial").map(bound.off)

    # full-index mask (len == len(idx))
    mask = (
        (idx.get_level_values("time") >= trial_ons)
        &
        (idx.get_level_values("time") <= trial_offs)
    )
    assert mask.shape == (len(idx),)
    # add mask to shared memory
    mask_meta, mask_shm = ioutils._to_shm(mask)

    # get fr zarr
    fr_zarr = chance_data["chance_fr"]
    count_zarr = chance_data["chance_spiked"]
    # get number of repeats
    repeats = fr_zarr.shape[-1]

    del spiked, positions, chance_data, trial_ons, trial_offs, bound
    gc.collect()

    # get unit ids
    unit_ids = np.array(units.flat(), dtype=np.int16)

    # Set up the process pool to run the worker in parallel.
    # Submit jobs for each repeat.
    n_workers = 2 ** (mp.cpu_count().bit_length() - 2)
    
    return (idx_shms, cols_shms, positions_shms, mask_shm,
     idx_meta, cols_meta, positions_meta, mask_meta, 
     fr_zarr, count_zarr, repeats, unit_ids, n_workers)


def save_chance_positional_data(
    chance_data, units, trial_ids, event_on_t, event_off_t,
):
    """
    Implementation of saving chance level spike data.
    """
    (idx_shms, cols_shms, positions_shms, mask_shm,
     idx_meta, cols_meta, positions_meta, mask_meta, 
     fr_zarr, spiked_zarr, repeats, unit_ids, n_workers) = prep_chance_data(
        chance_data,
        units,
        trial_ids,
        event_on_t,
        event_off_t,
    )
    del chance_data
    gc.collect()

    futures = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(
                _positional_fr_chance_worker,
                fr_zarr,
                spiked_zarr,
                r,
                positions_meta,
                idx_meta,
                cols_meta,
                mask_meta,
                unit_ids,
                trial_ids,
            ) for r in range(repeats)
        ]
        # collect and concat
        results = [f.result() for f in as_completed(futures)]

    for shm in idx_shms + cols_shms + positions_shms + [mask_shm]:
        shm.close()
        shm.unlink()

    # concat results
    occupancy = pd.concat(
        [result["occupancy"] for result in results],
        axis=1,
        keys=range(repeats),
        names=["repeat", "trial"],
    )
    fc = pd.concat(
        [result["pos_fc"] for result in results],
        axis=1,
        keys=range(repeats),
        names=["repeat", "start", "unit", "trial"],
    )
    fr = pd.concat(
        [result["pos_fr"] for result in results],
        axis=1,
        keys=range(repeats),
        names=["repeat", "start", "unit", "trial"],
    )
    mean_fr = _get_mean_across_repeats(fr, ["trial", "start", "unit"])
    mean_fc = _get_mean_across_repeats(fc, ["trial", "start", "unit"])
    mean_occu = _get_mean_across_repeats(occupancy, ["trial"], neural=False)

    # keep the same structure as data
    mean_fc = (
        mean_fc
        .reorder_levels(["unit", "start", "trial"], axis=1)
        .sort_index(
            axis=1,
            level=["unit", "start", "trial"],
            ascending=[True, False, True],
        )
    )
    mean_fr = (
        mean_fr
        .reorder_levels(["unit", "start", "trial"], axis=1)
        .sort_index(
            axis=1,
            level=["unit", "start", "trial"],
            ascending=[True, False, True],
        )
    )

    return {"pos_fc": mean_fc, "pos_fr": mean_fr, "occupancy": mean_occu}


def _get_mean_across_repeats(df, names, neural=True):
    trials = {}
    # group by trials
    trial_groups = df.T.groupby("trial")
    for (trial, group) in trial_groups:
        if neural:
            # group by unit, and get mean across repeats
            mean_data = group.groupby("unit").mean().T
            start = group.index.get_level_values("start")[:mean_data.shape[-1]]
            # add start level
            new_cols = pd.MultiIndex.from_arrays(
                [start, mean_data.columns],
                names=["start", "unit"],
            )
            mean_data.columns = new_cols
            trials[trial] = mean_data
        else:
            mean_data = group.mean().T
            trials[trial] = mean_data

    output = pd.concat(
        trials,
        axis=1,
        names=names,
    )

    return output


def _positional_fr_chance_worker(
    fr_zarr, spiked_zarr, r, positions_meta, idx_meta, cols_meta,
    mask_meta, unit_ids, trial_ids,
):
    """
    Worker that computes positional fr.

    params
    ===
    i: index of current repeat.

    return
    ===
    """
    # attach to shared memory and rebuild the indices
    idx, idx_shms = ioutils.import_multiindex_to_shm(idx_meta)
    cols, cols_shms = ioutils.import_index_to_shm(cols_meta)
    positions, positions_shms = ioutils.import_df_to_shm(positions_meta)
    mask, mask_shm = ioutils._from_shm(mask_meta)

    try:
        pos_data = get_vr_positional_data(
            {
                "positions": positions,
                "spiked": pd.DataFrame(
                    spiked_zarr[..., r],
                    index=idx,
                    columns=cols,
                ).loc[mask, unit_ids],
                "fr": pd.DataFrame(
                   fr_zarr[..., r],
                   index=idx,
                   columns=cols,
                ).loc[mask, unit_ids],
            }
        )
        del positions, idx, cols, mask, trial_ids, unit_ids
        gc.collect()

        logging.info(f"\nRepeat {r} finished.")
    finally:
        for shm in idx_shms + cols_shms + positions_shms + [mask_shm]:
            shm.close()

    return pos_data


def interpolate_to_grid(trials, grid_size, npz_path):
    interpolated = {}
    frs = []
    positions = {}
    timestamps = {}
    spikeds = {}
    for trial_id, group in trials["fr"].T.groupby("trial"):
        # dropna
        fr = group.dropna(axis=1, how="all").T
        pos = trials["positions"].xs(
            trial_id,
            level="trial",
            axis=1,
        ).dropna(axis=0, how="all")
        spiked = trials["spiked"].xs(
            trial_id,
            level="trial",
            axis=1,
        ).dropna(axis=0, how="all")

        frs.append(_interpolate_to_grid(fr, grid_size))
        positions[trial_id] = _interpolate_to_grid(pos, grid_size)
        spikeds[trial_id] = _interpolate_to_grid(
            spiked,
            grid_size,
            method="sum_till_idx",
        )
        # save timestamps too
        time = pos.iloc[:, 0].copy()
        time.iloc[:] = pos.index / SAMPLE_RATE
        timestamps[trial_id] = _interpolate_to_grid(time, grid_size)

    interpolated["fr"] = pd.concat(frs, axis=1).sort_index(
        axis=1,
        level=["unit", "trial"],
        ascending=[True, True],
    )
    # sort positions by trial, to be consistent with fr and spiked for arr
    pos_concat = pd.concat(positions, axis=1, names=["trial", "start"])
    # sort positions by start then trial for better view in df
    interpolated["positions"] = (
        pos_concat
        .reorder_levels(["start", "trial"], axis=1)
        .sort_index(
            axis=1,
            level=["start", "trial"],
            ascending=[False, True],
        )
    )
    interpolated["spiked"] = (
        pd.concat(spikeds, axis=1, names=["trial", "unit"])
        .reorder_levels(["unit", "trial"], axis=1)
        .sort_index(
            axis=1,
            level=["unit", "trial"],
            ascending=[True, True],
        )
    )
    interpolated["timestamps"] = pd.concat(timestamps, axis=1, names=["trial"])

    # save .npz file
    arr = {}
    trial_count = trials["fr"].columns.get_level_values("trial").nunique()

    # unit x trial x grid sample
    arr["count"] = np.reshape(
        interpolated["spiked"],
        (grid_size, trial_count, -1),
    ).astype(np.int16).T
    arr["fr"] = np.reshape(
        interpolated["fr"],
        (grid_size, trial_count, -1),
    ).T
    # trial x grid sample
    arr["pos"] = pos_concat.values.T
    arr["time"] = interpolated["timestamps"].values.T

    np.savez_compressed(npz_path, **arr)
    logging.info(f"\n> npz saved to {npz_path}.")

    return interpolated


def _interpolate_to_grid(data, grid_size, method="at_idx"):
    # get new index based on grid size
    new_t = np.linspace(
        data.index[0],
        data.index[-1],
        grid_size,
    ).round().astype(int) # make sure index is int

    # interpolate
    if method == "at_idx":
        interpolated = data.reindex(new_t).interpolate(method="index")
        interpolated.index = np.linspace(0, 1, grid_size)
    elif method == "sum_till_idx":
        # create right ward bins
        bins = pd.IntervalIndex.from_breaks(new_t, closed="right")
        interpolated = pd.DataFrame(
            np.zeros((grid_size, data.shape[1])),
            columns=data.columns,
        )
        # sum by right inclusion
        interpolated.iloc[1:, :] = data.groupby(
            pd.cut(data.index, bins, right=True), observed=False,
        ).sum()
        interpolated.iloc[0, :] = data.iloc[0, :]

    interpolated.index.name = "sample"

    return interpolated


def cut_between_bounds(df, bounds, level, ascending):
    trial_groups = df.T.groupby("trial")
    dfs = []
    for (trial, group) in trial_groups:
        trials = group.T.dropna(how="all", axis=0)
        mask = (
            (trials.index >= bounds.on.loc[trial])
            & (trials.index <= bounds.off.loc[trial]))
        masked = trials.loc[mask, :].dropna(how="all", axis=1)
        masked.reset_index(inplace=True, drop=True)
        dfs.append(masked)

    output = pd.concat(dfs, axis=1).sort_index(
        axis=1,
        level=level,
        ascending=ascending,
    )
    output.index.name = df.index.name

    return output


def whiten_psd(psd, fs, nperseg, min_cycle, n_median_filt_bins):
    """
    Normalise psd by its own background noise so that we can identify its real
    peak.
    """
    # smooth broadband to get background noise
    eps = np.finfo(float).eps
    background_noise = np.exp(
        median_filter(np.log(psd + eps), size=(n_median_filt_bins, 1))
    )

    # get freq resolvability
    fmin_resolvable = (min_cycle * fs) / nperseg
    fmax_resolvable = 1 / (2 * (1 / fs))

    mask = np.ones_like(psd.index, dtype=bool)
    mask &= (psd.index > 0)
    # NOTE: shortest wavelength (nyquist) lambda_min = 2 * (1 / fs), and longest
    # resolvable wavelength is lambda_max = 1 /fmin_resolvable.
    # the maximum ability to separate close periods is limited by
    #Q_max = nperseg * (1/fs) / lambda_0
    # freq is above the minimum resolvable freq
    mask &= (psd.index >= fmin_resolvable)
    # freq is below the maximum resolvable freq
    mask &= (psd.index <= fmax_resolvable)
    idxs = np.where(mask)[0]

    # whiten & mask
    whitened = psd.div(background_noise + eps).loc[mask, :]

    return whitened
