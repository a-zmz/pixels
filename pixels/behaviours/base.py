"""
This module provides a base class for experimental sessions that must be used as the
base for defining behaviour-specific processing.
"""

from __future__ import annotations

import functools
import json
import os
import glob
import pickle
import shutil
import tarfile
import tempfile
import re
import subprocess
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from shutil import copyfile
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.curation as sc
import spikeinterface.exporters as sexp
import spikeinterface.qualitymetrics as sqm
from scipy import interpolate
from tables import HDF5ExtError
from wavpack_numcodecs import WavPack

from pixels import ioutils
import pixels.signal_utils as signal
import pixels.pixels_utils as xut
from pixels.error import PixelsError
from pixels.constants import *

from common_utils.file_utils import load_yaml

if TYPE_CHECKING:
    from typing import Optional, Literal

BEHAVIOUR_HZ = 25000

np.random.seed(BEHAVIOUR_HZ)

# set si job_kwargs
job_kwargs = dict(
    n_jobs=0.8, # 80% core
    chunk_duration='1s',
    progress_bar=True,
)

si.set_global_job_kwargs(**job_kwargs)

# instantiate WavPack compressor
wv_compressor = WavPack(
    level=3, # high compression
    bps=None, # lossless
)

def _cacheable(method):
    """
    Methods with this decorator will have their output cached to disk so that future
    calls with the same set of arguments will simply load the result from disk. However,
    if the key word argument list contains `units` and it is not either `None` or an
    instance of `SelectedUnits` then this is disabled.
    """
    def func(*args, **kwargs):
        name = kwargs.pop("name", None)

        if "units" in kwargs:
            units = kwargs["units"]
            if not isinstance(units, SelectedUnits) or not hasattr(units, "name"):
                return method(*args, **kwargs)

        self, *as_list = list(args) + list(kwargs.values())
        if not self._use_cache:
            return method(*args, **kwargs)

        arrays = [i for i, arg in enumerate(as_list) if isinstance(arg, np.ndarray)]
        if arrays:
            if name is None:
                raise PixelsError(
                    'Cacheing methods when passing arrays requires also passing name="something"'
                )
            for i in arrays:
                as_list[i] = name

        as_list.insert(0, method.__name__)

        output = self.interim / 'cache' / ('_'.join(str(i) for i in as_list) + '.h5')
        if output.exists() and self._use_cache != "overwrite":
            # load cache
            try:
                df = ioutils.read_hdf5(output)
                print(f"> Cache loaded from {output}.")
            except HDF5ExtError:
                df = None
            except (KeyError, ValueError):
                # if key="df" is not found, then use HDFStore to list and read
                # all dfs
                # create df as a dictionary to hold all dfs
                df = {}
                with pd.HDFStore(output, "r") as store:
                    # list all keys
                    keys = store.keys()
                    # TODO apr 2 2025: for now the nested dict have keys in the
                    # format of `/imec0/positions`, this will not be the case
                    # once i flatten files at the stream level rather than
                    # session level, i.e., every pixels related cache will have
                    # stream id in their name.
                    for key in keys:
                        # remove "/" in key and split
                        key_name = key.lstrip("/").split("/")
                        if len(key_name) == 1:
                            # use the only key name as dict key
                            df[key_name[0]] = store[key]
                        elif len(key_name) == 2:
                            # stream id is the first
                            stream = key_name[0]
                            # data name is the second
                            name = "/".join(key_name[1:])
                            if stream not in df:
                                df[stream] = {}
                            df[stream][name] = store[key]
                print(f"> Cache loaded from {output}.")
        else:
            df = method(*args, **kwargs)
            output.parent.mkdir(parents=True, exist_ok=True)
            if df is None:
                output.touch()
            else:
                # allows to save multiple dfs in a dict in one hdf5 file
                if ioutils.is_nested_dict(df):
                    for stream_id, nested_dict in df.items():
                        # NOTE: we remove `.ap` in stream id cuz having `.`in
                        # the key name get problems
                        for name, values in nested_dict.items():
                            key = f"/{stream_id}/{name}"
                            ioutils.write_hdf5(
                                path=output,
                                df=values,
                                key=key,
                                mode="a",
                            )
                elif isinstance(df, dict):
                    for name, values in df.items():
                        ioutils.write_hdf5(
                            path=output,
                            df=values,
                            key=name,
                            mode="a",
                        )
                else:
                    ioutils.write_hdf5(output, df)
        return df
    return func


class SelectedUnits(list):
    name: str
    """
    This is the return type of Behaviour.select_units, which is a list in every way
    except that when represented as a string, it can return a name, if a `name`
    attribute has been set on it. This allows methods that have had `units` passed to be
    cached to file.
    """
    def __repr__(self):
        if hasattr(self, "name"):
            return self.name
        return list.__repr__(self)


class Behaviour(ABC):
    """
    This class represents a single individual recording session.

    Parameters
    ----------
    name : str
        The name of the session in the form YYMMDD_mouseID.

    data_dir : pathlib.Path
        The top level data folder, containing raw, interim and processed folders.

    metadata : dict, optional
        A dictionary of metadata for this session. This is typically taken from the
        session's JSON file.

    interim_dir : pathlib.Path
        An non-default interim folder that we can use for faster access to interim
        files, for example, instead of one in the data_dir.

    """

    SAMPLE_RATE = 2000#1000

    def __init__(self, name, data_dir, metadata=None, processed_dir=None,
                 interim_dir=None):
        self.name = name
        self.data_dir = data_dir
        self.metadata = metadata

        self.raw = self.data_dir / 'raw' / self.name

        if interim_dir is None:
            self.interim = self.data_dir / 'interim' / self.name
        else:
            self.interim = Path(interim_dir).expanduser() / self.name

        if processed_dir is None:
            self.processed = self.data_dir / 'processed' / self.name
        else:
            self.processed =  Path(processed_dir).expanduser() / self.name

        self.files = ioutils.get_data_files(self.raw, name)

        self.CatGT_dir = sorted(glob.glob(
            str(self.interim) +'/' + f'catgt_{self.name}_g[0-9]'
        ))

        self.interim.mkdir(parents=True, exist_ok=True)
        self.processed.mkdir(parents=True, exist_ok=True)

        self._action_labels = None
        self._behavioural_data = None
        self._ap_data = None
        self._spike_times_data = None
        self._lfp_data = None
        self._lag = None
        self._use_cache = True
        self._cluster_info = None
        self._good_unit_info = None
        self._probe_depths = None
        self.drop_data()

        self.ap_meta = []
        for stream_id in self.files["pixels"]:
            for meta in self.files["pixels"][stream_id]["ap_meta"]:
                self.ap_meta.append(
                    ioutils.read_meta(self.find_file(meta, copy=True))
                )

        # environmental variable PIXELS_CACHE={0,1} can be {disable,enable} cache
        self.set_cache(bool(int(os.environ.get("PIXELS_CACHE", 1))))

    def drop_data(self):
        """
        Clear attributes that store data to clear some memory.
        """
        # NOTE: number of behaviour session is independent of number of probes
        self.stream_count = len(self.files["pixels"])
        self.behaviour_count = len(self.files["behaviour"]["action_labels"])

        self._action_labels = [None] * self.behaviour_count
        self._behavioural_data = [None] * self.behaviour_count
        self._ap_data = [None] * self.stream_count
        self._spike_times_data = [None] * self.stream_count
        self._spike_rate_data = [None] * self.stream_count
        self._lfp_data = [None] * self.stream_count
        self._motion_index = [None] * self.behaviour_count
        self._cluster_info = [None] * self.stream_count
        self._probe_depths = [None] * self.stream_count
        self._load_lag()

    def set_cache(self, on: bool | Literal["overwrite"]) -> None:
        if isinstance(on, bool):
            self._use_cache = on
        else:
            assert on == "overwrite"
            self._use_cache = on

    def _load_lag(self):
        """
        Load previously-calculated lag information from a saved file if it exists,
        otherwise return Nones.
        """
        lag_file = self.processed / "lag.json"
        self._lag = [None] * self.behaviour_count
        if lag_file.exists():
            with lag_file.open() as fd:
                lag = json.load(fd)
            for rec_num, rec_lag in enumerate(lag):
                if rec_lag["lag_start"] is None:
                    self._lag[rec_num] = None
                else:
                    self._lag[rec_num] = (rec_lag["lag_start"], rec_lag["lag_end"])

    def get_probe_depth(self):
        """
        Load probe depth in um from file if it has been recorded.
        """
        for stream_num, depth in enumerate(self._probe_depths):
            # TODO jun 12 2024 skip stream 1 for now
            if stream_num > 0:
                continue
            if depth is None:
                try:
                    depth_file = self.processed / "depth.txt"
                    with depth_file.open() as fd:
                        self._probe_depths[stream_num] = [float(line) for line in
                                                          fd.readlines()][0]
                except:
                    depth_file = self.processed / self.files[stream_num]["depth_info"]
                    #self._probe_depths[stream_num] = json.load(open(depth_file, mode="r"))["clustering"]
                    self._probe_depths[stream_num] = json.load(
                            open(depth_file, mode="r"))["manipulator"]
                else:
                    msg = f": Can't load probe depth: please add it in um to\
                    \nprocessed/{self.name}/depth.txt, or save it with other depth related\
                    \ninfo in {self.processed / self.files[0]['depth_info']}."
                    raise PixelsError(msg)

            #if Path(depth_file).suffix == ".txt":
            #elif Path(depth_file).suffix == ".json":
            #    return [json.load(open(depth_file, mode="r"))["clustering"]]
        return self._probe_depths

    def find_file(self, name: str, copy: bool=True) -> Optional[Path]:
        """
        Finds the specified file, looking for it in the processed folder, interim
        folder, and then raw folder in that order. If the the file is only found in the
        raw folder, it is copied to the interim folder and uncompressed if required.

        Parameters
        ----------
        name : str or pathlib.Path
            The basename of the file to be looked for.

        copy : bool, optional
            Normally files are copied from raw to interim. If this is False, raw files
            will be decompressed in place but not copied to the interim folder.

        Returns
        -------
        pathlib.Path : the full path to the desired file in the correct folder.

        """
        processed = self.processed / name
        if processed.exists():
            return processed

        interim = self.interim / name
        if interim.exists():
            return interim

        raw = self.raw / name
        if raw.exists():
            if copy:
                print(f"    {self.name}: Copying {name} to interim")
                copyfile(raw, interim)
                return interim
            return raw

        tar = raw.with_name(raw.name + '.tar.gz')
        if tar.exists():
            if copy:
                print(f"    {self.name}: Extracting {tar.name} to interim")
                with tarfile.open(tar) as open_tar:
                    open_tar.extractall(path=self.interim)
                return interim
            print(f"    {self.name}: Extracting {tar.name}")
            with tarfile.open(tar) as open_tar:
                open_tar.extractall(path=self.raw)
            return raw

        return None

    def sync_data(self, rec_num, behavioural_data=None, sync_channel=None):
        """
        This method will calculate the lag between the behavioural data and the
        neuropixels data for each recording and save it to file and self._lag.

        behavioural_data and sync_channel will be loaded from file and downsampled if
        not provided, otherwise if provided they must already be the same sample
        frequency.

        Parameters
        ----------
        rec_num : int
            The recording number, i.e. index of self.files to get file paths.

        behavioural_data : pandas.DataFrame, optional
            The unprocessed behavioural data loaded from the TDMS file.

        sync_channel : np.ndarray, optional
            The sync channel from either the spike or LFP data.

        """
        # TODO jan 14 2025:
        # this func is not used in vr behaviour, since they are synched
        # in vd.session
        print("    Finding lag between sync channels")
        recording = self.files[rec_num]

        if behavioural_data is None:
            print("    Loading behavioural data")
            data_file = self.find_file(recording['behaviour'])
            behavioural_data = ioutils.read_tdms(data_file, groups=["NpxlSync_Signal"])
            behavioural_data = signal.resample(
                behavioural_data.values, BEHAVIOUR_HZ, self.SAMPLE_RATE
            )

        if sync_channel is None:
            print("    Loading neuropixels sync channel")
            data_file = self.find_file(recording['lfp_data'])
            num_chans = self.lfp_meta[rec_num]['nSavedChans']
            sync_channel = ioutils.read_bin(data_file, num_chans, channel=384)
            orig_rate = int(self.lfp_meta[rec_num]['imSampRate'])
            #sync_channel = sync_channel[:120 * orig_rate * 2]  # 2 mins, rec Hz, back/forward
            sync_channel = signal.resample(sync_channel, orig_rate, self.SAMPLE_RATE)

        behavioural_data = signal.binarise(behavioural_data)
        sync_channel = signal.binarise(sync_channel)

        print("    Finding lag")
        plot = self.processed / f'sync_{rec_num}.png'
        lag_start, match = signal.find_sync_lag(
            behavioural_data, sync_channel, plot=plot,
        )

        lag_end = len(behavioural_data) - (lag_start + len(sync_channel))
        self._lag[rec_num] = (lag_start, lag_end)

        if match < 95:
            print("    The sync channels did not match very well. Check the plot.")
        print(f"    Calculated lag: {(lag_start, lag_end)}")

        lag_json = []
        for lag in self._lag:
            if lag is None:
                lag_json.append(dict(lag_start=None, lag_end=None))
            else:
                lag_start, lag_end = lag
                lag_json.append(dict(lag_start=lag_start, lag_end=lag_end))
        with (self.processed / 'lag.json').open('w') as fd:
            json.dump(lag_json, fd)

    def sync_streams(self, SYNC_BIN, remap_stream_idx):
        """
        Neuropixels data streams acquired simultaneously are not synchronised, unless
        they are plugged into the same headstage, which is only the case for
        Neuropixels 2.0 probes.
        Dual-recording data acquired by Neuropixels 1.0 probes needs to be
        synchronised.
        Specifically, spike times from imecX stream need to be re-mapped to imec0
        time scale.

        For more info, see
        https://open-ephys.github.io/gui-docs/Tutorials/Data-Synchronization.html and
        https://github.com/billkarsh/TPrime.
        
        params
        ===
        SYNC_BIN: int, number of rising sync edges for calculating the scaling
        factor.
        """ 
        # TODO jan 14 2025:
        # this func does not work with the new self.files structure

        edges_list = []
        stream_ids = []
        #self.CatGT_dir = Path(self.CatGT_dir[0])
        output = self.ks_outputs[remap_stream_idx] / f'spike_times_remapped.npy'

        # do not redo the remapping if not necessary
        if output.exists():
            print(f'\n> Spike times from {self.ks_outputs[remap_stream_idx]}\
            already remapped, next session.')
            cluster_times = self.get_spike_times()[remap_stream_idx]
            remapped_cluster_times = self.get_spike_times(
                remapped=True)[remap_stream_idx]

            # get first spike time from each cluster, and their difference
            clusters_first = cluster_times.iloc[0,:]
            remapped_clusters_first = remapped_cluster_times.iloc[0,:]
            remap = remapped_clusters_first - clusters_first

            return remap

        for rec_num, recording in enumerate(self.files):
            # get file names and stuff
            # TODO jan 4 check if find_file works for catgt data
            spike_data = self.find_file(recording['CatGT_ap_data'])
            spike_meta = self.find_file(recording['CatGT_ap_meta'])
            #spike_data = self.CatGT_dir / recording['CatGT_ap_data']
            #spike_meta = self.CatGT_dir / recording['CatGT_ap_meta']
            stream_id = spike_data.as_posix()[-12:-4]
            stream_ids.append(stream_id)
            #self.gate_idx = spike_data.as_posix()[-18:-16]
            self.trigger_idx = spike_data.as_posix()[-15:-13]

            # find extracted rising sync edges, rn from CatGT
            try:
                edges_file = sorted(glob.glob(
                    rf'{self.CatGT_dir}' + f'/*{stream_id}.xd*.txt', recursive=True)
                )[0]
            except IndexError as e:
                raise PixelsError(
                    f"Can't load sync pulse rising edges. Did you run CatGT and\
                    extract edges? Full error: {e}\n"
                )
            # read sync edges, ms
            edges = np.loadtxt(edges_file)
            # pick edges by defined bin
            binned_edges = np.append(
                arr=edges[0::SYNC_BIN],
                values=edges[-1],
            )
            edges_list.append(binned_edges)

        # make list np array and calculate difference between streams to get the
        # initial difference
        edges = np.array(edges_list)
        initial_dt = np.diff(edges, axis=0).squeeze()
        # save initial diff for later plotting
        np.save(self.processed / 'sync_streams_lag.npy', initial_dt)

        # load spike times that needs to be remapped
        times = self.ks_outputs[remap_stream_idx] / f'spike_times.npy'
        try:
            times = np.load(times)
        except FileNotFoundError:
            msg = ": Can't load spike times that haven't been extracted!"
            raise PixelsError(self.name + msg)
        times = np.squeeze(times)

        # convert spike times to ms
        orig_rate = int(self.spike_meta[0]['imSampRate'])
        times_ms = times * self.SAMPLE_RATE / orig_rate

        lag = [None, 'later', 'earlier']
        print(f"""\n> {stream_ids[0]} started\r
        {abs(initial_dt[0]*1000):.2f}ms {lag[int(np.sign(initial_dt[0]))]}\r
        and finished\r
        {abs(initial_dt[-1]*1000):.2f}ms {lag[-int(np.sign(initial_dt[0]))]}\r
        than {stream_ids[1]}.""")

        # create a np array for remapped spike times from imec1
        remapped_times_ms = np.zeros(times.shape)

        # get edge difference within & between streams
        within_streams = np.diff(edges)
        # calculate scaling factor for each sync bin
        scales = within_streams[0] / within_streams[-1]

        # find out which sync period/bin each spike time belongs to, and use
        # the scale from that period/bin
        for t, time in enumerate(times_ms):
            bin_idx = np.where(time > edges[remap_stream_idx])[0][0]
            remapped_times_ms[t] = ((time - edges[remap_stream_idx][bin_idx]) *
            scales[bin_idx]) + edges[0][bin_idx]

        print(f"""\n> Remap stats {stream_ids[remap_stream_idx]} spike times:\r
        median shift {np.median(remapped_times_ms-times_ms):.2f}ms,\r
        min shift {np.min(remapped_times_ms-times_ms):.2f}ms,\r
        max shift {np.max(remapped_times_ms-times_ms):.2f}ms.""")

        # convert remappmed times back to its original sample index
        remapped_times = np.uint64(remapped_times_ms * orig_rate / self.SAMPLE_RATE)
        np.save(output, remapped_times)
        print(f'\n> Spike times remapping output saved to\n {output}.')

        # load remapped spike times of each cluster
        cluster_times = self.get_spike_times()[remap_stream_idx]
        remapped_cluster_times = self.get_spike_times(
            remapped=True)[remap_stream_idx]

        # get first spike time from each cluster, and their difference
        clusters_first = cluster_times.iloc[0,:]
        remapped_clusters_first = remapped_cluster_times.iloc[0,:]
        remap = remapped_clusters_first - clusters_first

        return remap


    def process_behaviour(self):
        """
        Process behavioural data from raw tdms and align to neuropixels data.
        """
        # NOTE jan 14 2025:
        # this func is not used by vr behaviour
        for rec_num, recording in enumerate(self.files):
            print(
                f">>>>> Processing behaviour for recording {rec_num + 1} of {len(self.files)}"
            )

            print(f"> Loading behavioural data")
            behavioural_data = ioutils.read_tdms(self.find_file(recording['behaviour']))

            # ignore any columns that have Nans; these just contain settings
            for col in behavioural_data:
                if behavioural_data[col].isnull().values.any():
                    behavioural_data.drop(col, axis=1, inplace=True)

            print(f"> Downsampling to {self.SAMPLE_RATE} Hz")
            behav_array = signal.resample(behavioural_data.values, 25000, self.SAMPLE_RATE)
            behavioural_data.iloc[:len(behav_array), :] = behav_array
            behavioural_data = behavioural_data[:len(behav_array)]

            print(f"> Syncing to Neuropixels data")
            if self._lag[rec_num] is None:
                self.sync_data(
                    rec_num,
                    behavioural_data=behavioural_data["/'NpxlSync_Signal'/'0'"].values,
                )
            lag_start, lag_end = self._lag[rec_num]
            behavioural_data = behavioural_data[max(lag_start, 0):-1-max(lag_end, 0)]
            behavioural_data.index = range(len(behavioural_data))

            print(f"> Extracting action labels")
            self._action_labels[rec_num] = self._extract_action_labels(rec_num, behavioural_data)
            output = self.processed / recording['action_labels']
            np.save(output, self._action_labels[rec_num])
            print(f">   Saved to: {output}")

            output = self.processed / recording['behaviour_processed']
            print(f"> Saving downsampled behavioural data to:")
            print(f"    {output}")
            behavioural_data.drop("/'NpxlSync_Signal'/'0'", axis=1, inplace=True)
            ioutils.write_hdf5(output, behavioural_data)
            self._behavioural_data[rec_num] = behavioural_data

        print("> Done!")

    def correct_motion(self, mc_method="dredge"):
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
        if mc_method == "ks":
            print(f"> Correct motion later with {mc_method}.")
            return None

        # get pixels streams
        streams = self.files["pixels"]

        for stream_id, stream_files in streams.items():
            output = self.interim / stream_files["motion_corrected"]
            if output.exists():
                print(f"> Motion corrected {stream_id} loaded.")
                continue

            # preprocess raw recording
            self.preprocess_raw()

            # load preprocessed rec
            rec = stream_files["preprocessed"]

            print(
                f"\n>>>>> Correcting motion for recording from {stream_id} "
                f"in total of {self.stream_count} stream(s) with {mc_method}"
            )

            mcd = xut.correct_motion(rec)

            mcd.save(
                format="zarr",
                folder=output,
                compressor=wv_compressor,
            )

        return None


    def preprocess_raw(self):
        """
        Preprocess full-band raw pixels data.

        params
        ===
        mc_method: str, motion correction method.
            Default: "dredge".
                (as of jan 2025, dredge performs better than ks motion correction.)
            "ks": let kilosort do motion correction.

        return
        ===
        preprocessed: spikeinterface recording.
        """
        # load raw recording as si recording extractor
        self.load_raw_ap()

        # get pixels streams
        streams = self.files["pixels"]

        for stream_id, stream_files in streams.items():
            # load raw si rec
            rec = stream_files["si_rec"]
            print(
                f"\n>>>>> Preprocessing data for recording from {stream_id} "
                f"in total of {self.stream_count} stream(s)"
            )

            # load brain surface depths
            surface_depths = load_yaml(
                path=self.find_file(stream_files["surface_depth"]),
            )

            # preprocess
            stream_files["preprocessed"] = xut.preprocess_raw(
                rec,
                surface_depths,
            )

        return None


    def detect_n_localise_peaks(self, loc_method="monopolar_triangulation"):
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
        self.extract_bands("ap")

        # get pixels streams
        streams = self.files["pixels"]

        for stream_id, stream_files in streams.items():
            output = self.processed / stream_files["detected_peaks"]
            if output.exists():
                print(f"> Peaks from {stream_id} already detected.")
                continue

            # get ap band
            ap_file = self.find_file(stream_files["ap_extracted"])
            rec = si.load_extractor(ap_file)

            # detect and localise peaks
            df = xut.detect_n_localise_peaks(rec)

            # write to disk
            ioutils.write_hdf5(output, df)

        return None


    def extract_bands(self, freqs=None):
        """
        extract data of ap and lfp frequency bands from the raw neural recording
        data.
        """
        if freqs == None:
            bands = freq_bands
        elif isinstance(freqs, str) and freqs in freq_bands.keys():
            bands = {freqs: freq_bands[freqs]}
        elif isinstance(freqs, dict):
            bands = freqs

        streams = self.files["pixels"]
        for stream_id, stream_files in streams.items():
            for name, freqs in bands.items():
                output = self.processed / stream_files[f"{name}_extracted"]
                if output.exists():
                    print(f"> {name} bands from {stream_id} loaded.")
                    continue
                
                # preprocess raw data
                self.preprocess_raw()

                print(
                    f">>>>> Extracting {name} bands from {self.name} "
                    f"{stream_id} in total of {self.stream_count} stream(s)"
                )

                # load preprocessed
                rec = stream_files["preprocessed"]

                # do bandpass filtering
                extracted = xut.extract_band(
                    rec,
                    freq_min=freqs[0],
                    freq_max=freqs[1],
                )

                # write to disk
                extracted.save(
                    format="zarr",
                    folder=output,
                    compressor=wv_compressor,
                )

            """
            if self._lag[rec_num] is None:
                self.sync_data(rec_num, sync_channel=data[:, -1])
            lag_start, lag_end = self._lag[rec_num]

            print(f"> Saving data to {output}")
            if lag_end < 0:
                data = data[:lag_end]
            if lag_start < 0:
                data = data[- lag_start:]
            data = pd.DataFrame(data[:, :-1])
            ioutils.write_hdf5(output, data)
            """

    def run_catgt(self, CatGT_app=None, args=None) -> None:
        """
        This func performs CatGT on copied AP data in the interim.

        params
        ====
        data_dir: path, dir to interim data and catgt output.

        catgt_app: path, dir to catgt software.

        args: str, arguments in catgt.
            default is None.
        """
        # TODO jan 14 2025:
        # this func is deprecated
        assert 0, "deprecated"
        if CatGT_app == None:
            CatGT_app = "~/CatGT3.4"
        # move cwd to catgt
        os.chdir(CatGT_app)

        # reset catgt args for current session
        session_args = None

        for f in self.files:
            # copy spike data to interim
            self.find_file(f['spike_data'])

        if (isinstance(self.CatGT_dir, list) and
             len(self.CatGT_dir) != 0 and
             len(os.listdir(self.CatGT_dir[0])) != 0):
            print(f"\nCatGT already performed on ap data of {self.name}. Next session.\n")
            return
        else:
            #TODO: finish this here so that catgt can run together with sorting
            print(f"\n> Running CatGT on ap data of {self.name}")
            #_dir = self.interim

        if args == None:
            #args = f"-no_run_fld\
            #    -g=0,9\
            #    -t=0,9\
            #    -prb=0:1\
            #    -prb_miss_ok\
            #    -ap\
            #    -lf\
            #    -apfilter=butter,12,300,9000\
            #    -lffilter=butter,12,0.5,300\
            #    -xd=2,0,384,6,350,160\
            #    -gblcar\
            #    -gfix=0.2,0.1,0.02"
            args = f"-no_run_fld\
                -g=0\
                -t=0\
                -prb=0\
                -prb_miss_ok\
                -ap\
                -xd=2,0,384,6,20,15\
                -xid=2,0,384,6,20,15"

        session_args = f"-dir={self.interim} -run={self.name} -dest={self.interim} " + args
        print(f"\ncatgt args of {self.name}: \n{session_args}")

        subprocess.run( ['./run_catgt.sh', session_args])

        # make sure CatGT_dir is set after running
        self.CatGT_dir = sorted(glob.glob(
            str(self.interim) +'/' + f'catgt_{self.name}_g[0-9]'
        ))

    def load_raw_ap(self):
        """
        Write a function to load and concatenate raw recording files for each
        stream (i.e., probe), so that data from all runs of the same probe can
        be preprocessed and sorted together.
        """
        # if multiple runs for the same probe, concatenate them
        streams = self.files["pixels"]

        for stream_id, stream_files in streams.items():
            # get paths of raw
            paths = [self.find_file(path) for path in stream_files["ap_raw"]]

            # now the value for streams dict is recording extractor
            stream_files["si_rec"] = xut.load_raw(paths, stream_id)

        return None


    def sort_spikes(self, mc_method="dredge"):
        """
        Run kilosort spike sorting on raw spike data.

        params
        ===
        mc_method: str, motion correction method.
            Default: "dredge".
                (as of jan 2025, dredge performs better than ks motion correction.)
            "ks": do motion correction with kilosort.
        """
        ks_image_path = self.interim.parent/"ks4_with_wavpack.sif"

        if not ks_image_path.exists():
            raise PixelsError("Have you craeted Singularity image for sorting?")

        # preprocess and motion correct raw
        self.correct_motion(mc_method)

        if mc_method == "ks":
            ks_mc = True
        else:
            ks_mc = False

        # set ks4 parameters
        ks4_params = {
            "do_correction": ks_mc,
            "do_CAR": False, # do not common average reference
            "save_preprocessed_copy": True, # save ks4 preprocessed data
        }

        streams = self.files["pixels"]
        for stream_num, (stream_id, stream_files) in enumerate(streams.items()):
            # check if already sorted and exported
            sa_dir = self.processed / stream_files["sorting_analyser"]
            if not sa_dir.exists():
                print(f"> {self.name} {stream_id} not sorted/exported.\n")
            else:
                print("> Already sorted and exported, next session.\n")
                continue

            # get catgt directory
            catgt_dir = self.find_file(
                stream_files["CatGT_ap_data"][stream_num]
            )

            # find spike sorting output folder
            if catgt_dir is None:
                output = sa_dir.parent
            else:
                output = self.processed / f"sorted_stream_cat_{stream_num}"

            # load rec
            if ks_mc:
                rec = stream_files["preprocessed"]
            else:
                rec_dir = self.find_file(stream_files["motion_corrected"])
                rec = si.load_extractor(rec_dir)

            # move current working directory to interim
            os.chdir(self.interim)

            # sort spikes and save sorting analyser to disk
            xut.sort_spikes(
                rec=rec,
                output=output,
                curated_sa_dir=sa_dir,
                ks_image_path=ks_image_path,
                ks4_params=ks4_params,
            )

        return None


    def extract_videos(self, force=False):
        """
        Extract behavioural videos from TDMS to avi. By default this will only run if
        the video does not aleady exist. Pass `force=True` to extract the video anyway,
        overwriting the destination file.
        """
        for recording in self.files:
            for v, video in enumerate(recording.get('camera_data', [])):
                path_out = self.interim / video.with_suffix('.avi')
                if not path_out.exists() or force:
                    meta = recording['camera_meta'][v]
                    ioutils.tdms_to_video(
                        self.find_file(video, copy=False),
                        self.find_file(meta),
                        path_out,
                    )

    def configure_motion_tracking(self, project: str) -> None:
        """
        Set up DLC project and add videos to it.
        """
        # bloated so imported when needed
        import deeplabcut  # pylint: disable=import-error

        self.extract_videos()

        working_dir = self.data_dir / "processed" / "DLC"
        matches = list(working_dir.glob(f"{project}*"))
        if matches:
            config = working_dir / matches[0] / "config.yaml"
        else:
            config = None

        videos = []
        for recording in self.files:
            for video in recording.get('camera_data', []):
                if project in video.stem:
                    videos.append(self.interim / video.with_suffix('.avi'))

        if not videos:
            print(self.name, ": No matching videos for project:", project)
            return

        if config:
            deeplabcut.add_new_videos(
                config,
                videos,
                copy_videos=False,
            )
        else:
            print(f"Config not found.")
            reply = input("Create new project? [Y/n]")
            if reply and reply[0].lower() == "n":
                raise PixelsError("A DLC project is needed for motion tracking.")

            deeplabcut.create_new_project(
                project,
                os.environ.get("USER"),
                videos,
                working_directory=working_dir,
                copy_videos=False,
            )
            raise PixelsError("Raising an exception to stop operation. Check new config.")

    def run_motion_tracking(
        self,
        project: str,
        analyse: bool = True,
        align: bool = True,
        create_labelled_video: bool = False,
        extract_outlier_frames: bool = False,
    ):
        """
        Run DeepLabCut motion tracking on behavioural videos.

        Parameters
        ==========

        project : str
            The name of the DLC project.

        create_labelled_video : bool
            Generate a labelled video from existing DLC output.

        """
        # bloated so imported when needed
        import deeplabcut  # pylint: disable=import-error

        working_dir = self.data_dir / "processed" / "DLC"
        matches = list(working_dir.glob(f"{project}*"))
        if not matches:
            raise PixelsError(f"DLC project {profile} not found.")
        config = working_dir / matches[0] / "config.yaml"
        output_dir = self.processed / f"DLC_{project}"

        videos = []

        for recording in self.files:
            for v, video in enumerate(recording.get('camera_data', [])):
                if project in video.stem:
                    avi = self.interim / video.with_suffix('.avi')
                    if not avi.exists():
                        meta = recording['camera_meta'][v]
                        ioutils.tdms_to_video(
                            self.find_file(video, copy=False),
                            self.find_file(meta),
                            avi,
                        )
                    if not avi.exists():
                        raise PixelsError(f"Path {avi} should exist but doesn't... discuss.")
                    videos.append(avi.as_posix())

        if analyse:
            if output_dir.exists():
                shutil.rmtree(output_dir.as_posix())
            deeplabcut.analyze_videos(config, videos, destfolder=output_dir)
            deeplabcut.plot_trajectories(config, videos, destfolder=output_dir)
            deeplabcut.filterpredictions(config, videos, destfolder=output_dir)

        if align:
            for video in videos:
                stem = Path(video).stem
                try:
                    result = next(output_dir.glob(f"{stem}*_filtered.h5"))
                except StopIteration:
                    raise PixelsError(f"{self.name}: DLC output not found.")

                coords = pd.read_hdf(result)
                meta_file = None
                rec_num = None

                for rec_num, recording in enumerate(self.files):
                    for meta in recording.get('camera_meta', []):
                        if stem in meta.as_posix():
                            meta_file = meta
                            break
                    if meta_file:
                        break

                assert meta_file and rec_num is not None

                metadata = ioutils.read_tdms(self.find_file(meta_file))
                aligned = self._align_dlc_coords(rec_num, metadata, coords)

                ioutils.write_hdf5(
                    self.processed / f"motion_tracking_{project}_{rec_num}.h5",
                    aligned,
                )

        if extract_outlier_frames:
            deeplabcut.extract_outlier_frames(
                config, videos, destfolder=output_dir, automatic=True,
            )

        if create_labelled_video:
            deeplabcut.create_labeled_video(
                config, videos, destfolder=output_dir, draw_skeleton=True,
            )

    def _align_dlc_coords(self, rec_num, metadata, coords):
        recording = self.files[rec_num]
        behavioural_data = ioutils.read_tdms(self.find_file(recording['behaviour']))

        # ignore any columns that have Nans; these just contain settings
        for col in behavioural_data:
            if behavioural_data[col].isnull().values.any():
                behavioural_data.drop(col, axis=1, inplace=True)

        behav_array = signal.resample(behavioural_data.values, 25000, self.SAMPLE_RATE)
        behavioural_data.iloc[:len(behav_array), :] = behav_array
        behavioural_data = behavioural_data[:len(behav_array)]

        trigger = signal.binarise(behavioural_data["/'CamFrames'/'0'"]).values
        onsets = np.where((trigger[:-1] == 1) & (trigger[1:] == 0))[0]

        timestamps = ioutils.tdms_parse_timestamps(metadata)
        assert len(timestamps) == len(coords)

        # If there are more onsets in the tdms data, just extend the motion tracking
        # data with 0s up until the end to avoid IndexErrors when trying to index into
        # shorter coordinate arrays. This shouldn't matter as it's 1-2 seconds max at
        # the end of the session where the camera stopped before the behaviour rec.
        if len(onsets) > len(coords):
            overhang = len(onsets) - len(coords)
            top = coords.shape[0]
            index = coords.index.values
            new_index = np.concatenate([
                    index,
                    np.arange(index[-1] + 1, index[-1] + overhang + 1),
            ])
            coords = coords.reindex(new_index).fillna(0)

        assert len(onsets) == len(coords)

        # The last frame sometimes gets delayed a bit, so ignoring it, are the timestamp
        # diffs fixed?
        assert len(np.unique(np.diff(onsets[:-1]))) == 1

        if self._lag[rec_num] is None:
            self.sync_data(
                rec_num,
                behavioural_data=behavioural_data["/'NpxlSync_Signal'/'0'"].values,
            )
        lag_start, lag_end = self._lag[rec_num]
        no_lag = slice(max(lag_start, 0), -1-max(lag_end, 0))

        # If this fails, we'll have to understand why and if we need to change this
        # logic.
        assert len(coords.columns.get_level_values("scorer").unique()) == 1
        scorer = coords.columns.get_level_values("scorer")[0]
        bodyparts = coords.columns.get_level_values("bodyparts").unique()
        xs = np.arange(0, len(trigger))
        likelihood_threshold = 0.05
        processed = {}
        action_labels = self.get_action_labels()[rec_num]

        for label in bodyparts:
            label_coords = coords[scorer][label]

            # Un-invert y coordinates
            label_coords.y = 480 - label_coords.y

            # Remove unlikely coordinates from fit
            bads = label_coords["likelihood"] < likelihood_threshold
            good_onsets = onsets[~bads]
            assert len(good_onsets) == len(onsets) - bads.sum()

            # Interpolate to desired sampling rate

            # B-spline poly fit
            # spl_x = interpolate.splrep(good_onsets, label_coords.x[~ bads])
            # ynew_x = interpolate.splev(xs, spl_x)
            # spl_y = interpolate.splrep(good_onsets, label_coords.y[~ bads])
            # ynew_y = interpolate.splev(xs, spl_y)

            # Linear fit
            ynew_x = interpolate.interp1d(
                good_onsets, label_coords.x[~ bads], fill_value="extrapolate",
            )(xs)
            ynew_y = interpolate.interp1d(
                good_onsets, label_coords.y[~ bads], fill_value="extrapolate",
            )(xs)

            data = np.array([ynew_x[no_lag], ynew_y[no_lag]]).T
            assert action_labels.shape == data.shape
            processed[label] = pd.DataFrame(data, columns=["x", "y"])

        df = pd.concat(processed, axis=1)
        return pd.concat({scorer: df}, axis=1, names=coords.columns.names)

    def draw_motion_index_rois(self, video_match, num_rois=1, skip=True):
        """
        Draw motion index ROIs using EasyROI. If ROIs already exist, skip.

        Parameters
        ----------
        video_match : str
            A string to match video file names. 

        num_rois : int
            The number of ROIs to draw interactively. Default: 1

        skip : bool
            Whether to skip drawing ROIs if they already exist. Default: True.

        """
        # Only needed for this method
        import cv2
        import EasyROI

        roi_helper = EasyROI.EasyROI(verbose=False)

        for i, recording in enumerate(self.files):
            for v, video in enumerate(recording.get('camera_data', [])):
                if video_match not in video.stem:
                    continue

                avi = self.interim / video.with_suffix('.avi')
                if not avi.exists():
                    meta = recording['camera_meta'][v]
                    ioutils.tdms_to_video(
                        self.find_file(video, copy=False),
                        self.find_file(meta),
                        avi,
                    )
                if not avi.exists():
                    raise PixelsError(f"Path {avi} should exist but doesn't... discuss.")

                roi_file = self.processed / (avi.with_suffix("").stem + f"-MI_ROIs_{i}.pickle")
                if skip and roi_file.exists():
                    continue

                # Load frame from video
                duration = ioutils.get_video_dimensions(avi.as_posix())[2]
                frame = ioutils.load_video_frame(avi.as_posix(), duration // 4)

                # Interactively draw ROI
                roi = roi_helper.draw_polygon(frame, num_rois)
                cv2.destroyAllWindows()  # Needed otherwise EasyROI errors

                # Save a copy of the frame with ROIs to PNG file
                copy = EasyROI.visualize_polygon(frame, roi, color=(255, 0, 0))
                plt.imsave(roi_file.with_suffix(".png"), copy, cmap='gray')

                # Save ROI to file
                with roi_file.open('wb') as fd:
                    pickle.dump(roi['roi'], fd)

    def process_motion_index(self, video_match):
        """
        Extract motion indexes from videos using already drawn ROIs.

        Parameters
        ----------
        video_match : str
            A string to match video and ROI file names. 

        """
        ses_rois = {}

        # First collect all ROIs to catch errors early
        for rec_num, recording in enumerate(self.files):
            for v, video in enumerate(recording.get('camera_data', [])):
                if video_match not in video.stem:
                    continue

                roi_file = self.processed / (video.with_suffix("").stem + f"-MI_ROIs_{rec_num}.pickle")
                if not roi_file.exists():
                    raise PixelsError(self.name + ": ROIs not drawn for motion index.")

                # Also check videos are available
                avi = self.interim / video.with_suffix('.avi')
                if not avi.exists():
                    raise PixelsError(self.name + ": AVI video not found in interim folder.")

                with roi_file.open('rb') as fd:
                    ses_rois[(rec_num, v)] = (pickle.load(fd), roi_file)

        # Then do the extraction
        for rec_num, recording in enumerate(self.files):
            for v, video in enumerate(recording.get('camera_data', [])):
                if video_match not in video.stem:
                    continue

                # Get MIs
                avi = self.interim / video.with_suffix('.avi')
                #TODO: how to load recording
                rec_rois, roi_file = ses_rois[(rec_num, v)]
                rec_mi = signal.motion_index(avi, rec_rois)

                # Align MIs to action labels
                behavioural_data = ioutils.read_tdms(self.find_file(recording['behaviour']))

                # ignore any columns that have Nans; these just contain settings
                for col in behavioural_data:
                    if behavioural_data[col].isnull().values.any():
                        behavioural_data.drop(col, axis=1, inplace=True)

                behav_array = signal.resample(behavioural_data.values, 25000, self.SAMPLE_RATE)
                behavioural_data.iloc[:len(behav_array), :] = behav_array
                behavioural_data = behavioural_data[:len(behav_array)]
                trigger = signal.binarise(behavioural_data["/'CamFrames'/'0'"]).values
                onsets = np.where((trigger[:-1] == 1) & (trigger[1:] == 0))[0]

                metadata = ioutils.read_tdms(self.find_file(recording['camera_meta'][v]))
                timestamps = ioutils.tdms_parse_timestamps(metadata)
                assert len(timestamps) == len(rec_mi)

                if len(onsets) > len(rec_mi):
                    assert False, "See _align_dlc_coords for solution here."

                assert len(onsets) == len(rec_mi)

                # The last frame sometimes gets delayed a bit, so ignoring it, are the timestamp
                # diffs fixed?
                assert len(np.unique(np.diff(onsets[:-1]))) == 1

                if self._lag[rec_num] is None:
                    self.sync_data(
                        rec_num,
                        behavioural_data=behavioural_data["/'NpxlSync_Signal'/'0'"].values,
                    )
                lag_start, lag_end = self._lag[rec_num]
                no_lag = slice(max(lag_start, 0), -1-max(lag_end, 0))

                xs = np.arange(0, len(trigger))
                ynew = interpolate.interp1d(
                    onsets, rec_mi, axis=0, fill_value="extrapolate",
                )(xs)

                result = ynew[no_lag]
                action_labels = self.get_action_labels()[rec_num]
                assert action_labels.shape[0] == result.shape[0]

                np.save(roi_file.with_suffix(".npy"), result)

    def add_motion_index_action_label(
        self, label: int, event: int, roi: int, value: int
    ) -> None:
        """
        Add motion index onset times to the action labels as an event that can be
        aligned to.

        Paramters
        ---------

        label : int
            The action following which the onset is expected to occur. A baseline level
            of motion is determined from 5s preceding these actions, and the onset time
            is searched for from this period up to event.

        event : int
            The event before which the onset is expected to occur. MI onsets are looked
            for between the times of action (above) and this event.

        roi : int
            The index in the motion index vectors of the ROI to use to found onsets.

        value : int
            The value to use to represent the onset time event in the action labels i.e.
            a value that is (or could be) part of the behaviour's `Events` enum for
            representing movement onsets.

        """
        assert False, "TODO"
        action_labels = self.get_action_labels()
        motion_indexes = self.get_motion_index_data()

        scan_duration = self.SAMPLE_RATE * 10
        half = scan_duration // 2

        # We take 200 ms before the action begins as a short baseline period for each
        # trial. The smallest standard deviation of all SDs of these baseline periods is
        # used as a threshold to identify "clean" trials (`clean_threshold` below).
        # Non-clean trials are trials where TODO
        short_pre = int(0.2 * self.SAMPLE_RATE)

        for rec_num, recording in enumerate(self.files):
            # Only recs with camera_data will have motion indexes
            if 'camera_data' in recording:

                # Get our numbers
                assert motion_indexes[rec_num] is not None
                num_rois = motion_indexes[rec_num].shape[1]
                if num_rois - 1 < roi:
                    raise PixelsError("ROI index is too high, there are only {num_rois} ROIs")
                motion_index = motion_indexes[rec_num][:, roi]

                actions = action_labels[rec_num][:, 0]
                events = action_labels[rec_num][:, 1]
                trial_starts = np.where(np.bitwise_and(actions, label))[0]

                # Cut out our trials
                trials = []
                pre_cue_SDs = []

                for start in trial_starts:
                    event_latency = np.where(np.bitwise_and(events[start:start + scan_duration], event))[0]
                    if len(event_latency) == 0:
                        raise PixelsError('Action labels probably miscalculated')
                    trial = motion_index[start:start + event_latency[0]]
                    trials.append(trial)

                    pre_cue_SDs.append(np.std(motion_index[start - short_pre:start]))

                clean_threshold = min(pre_cue_SDs)

                onsets = []

                for t in trials:
                    movements = np.where(t > clean_threshold * 10)[0]
                    if len(movements) == 0:
                        raise PixelsError("Thresholding for detecting MI onset is inadequate")
                    onsets.append(movements[0])

                assert 0


    @abstractmethod
    def _extract_action_labels(self, behavioural_data):
        """
        This method must be overriden with the derivation of action labels from
        behavioural data specific to the behavioural task.

        Parameters
        ----------
        behavioural_data : pandas.DataFrame
            A dataframe containing the behavioural DAQ data.

        Returns
        -------
        action_labels : 1D numpy.ndarray
            An array of actions of equal length to the behavioural_data.

        """

    def _get_processed_data(self, attr, key, category):
        """
        Used by the following get_X methods to load processed data.

        Parameters
        ----------
        attr : str
            The self attribute that stores the data.

        key : str
            The key for the files in each recording of self.files that contain this
            data.

        """
        saved = getattr(self, attr)

        if saved[0] is None:
            files = self.files[category]
        else:
            return saved

        if key in files:
            dirs = files[key]
            for f, file_dir in enumerate(dirs):
                file_path = self.processed / file_dir
                if file_path.exists():
                    if re.search(r'\.np[yz]$', file_path.suffix):
                        saved[f] = np.load(file_path)
                    elif file_path.suffix == '.h5':
                        saved[f] = ioutils.read_hdf5(file_path)
        else:
            msg = f"Could not find {attr[1:]} for recording {rec_num}."
            msg += f"\nFile should be at: {file_path}"
            raise PixelsError(msg)

        return saved

    def get_action_labels(self):
        """
        Returns the action labels, either from self._action_labels if they have been
        loaded already, or from file.
        """
        return self._get_processed_data("_action_labels", "action_labels",
            "behaviour")

    def get_behavioural_data(self):
        """
        Returns the downsampled behaviour channels.
        """
        return self._get_processed_data("_behavioural_data", "behaviour_processed")

    def get_motion_index_data(self, video_match):
        """
        Returns the motion indexes, either from self._motion_index if they have been
        loaded already, or from file.
        """
        if all(i is None for i in self._motion_index):
            if video_match is None:
                raise PixelsError("video_match needed to get motion index data")

            for rec_num, recording in enumerate(self.files):
                for v, video in enumerate(recording.get('camera_data', [])):
                    if video_match not in video.stem:
                        continue

                    mi_file = self.processed / (
                        video.with_suffix("").stem + f"-MI_ROIs_{rec_num}.npy"
                    )
                    if not mi_file.exists():
                        raise PixelsError(f"Can't align to motion index file that hasn't been created.")

                    self._motion_index[rec_num] = np.load(mi_file)

        return self._motion_index

    def get_motion_tracking_data(self, dlc_project: str):
        """
        Returns the DLC coordinates from self._motion_tracking if they have been loaded
        already, or from file.
        """
        key = f"motion_tracking_{dlc_project}"
        attr = f"_{key}"
        if hasattr(self, attr):
            return getattr(self, attr)

        setattr(self, attr, [None] * len(self.files))

        for rec_num, recording in enumerate(self.files):
            if "camera_data" in recording:
                recording[key] = f"{key}_{rec_num}.h5"

        return self._get_processed_data(attr, key)

    def get_spike_data(self):
        """
        Returns the processed and downsampled spike data.
        """
        return self._get_processed_data("_ap_data", "spike_processed")

    def get_lfp_data(self):
        """
        Returns the processed and downsampled LFP data.
        """
        return self._get_processed_data("_lfp_data", "lfp_processed")


    def _get_si_spike_times(self, units):
        """
        get spike times in second with spikeinterface
        """
        spike_times = self._spike_times_data

        streams = self.files["pixels"]
        for stream_num, (stream_id, stream_files) in enumerate(streams.items()):
            sa_dir = self.find_file(stream_files["sorting_analyser"])
            # load sorting analyser
            temp_sa = si.load_sorting_analyzer(sa_dir)
            # select units
            sorting = temp_sa.sorting.select_units(units)
            sa = temp_sa.select_units(units)
            sa.sorting = sorting

            times = {}
            # get spike train
            for i, unit_id in enumerate(sa.unit_ids):
                unit_times = sa.sorting.get_unit_spike_train(
                    unit_id=unit_id,
                    return_times=False,
                )
                times[unit_id] = pd.Series(unit_times)
            # concatenate units
            spike_times[stream_num] = pd.concat(
                objs=times,
                axis=1,
                names="unit",
            )
            # get sampling frequency
            fs = int(sa.sampling_frequency)
            # Convert to time into sample rate index
            spike_times[stream_num] /= fs / self.SAMPLE_RATE 

        return spike_times[0] # NOTE: only deal with one stream for now

    def get_spike_times(self, units, remapped=False, use_si=False):
        """
        Returns the sorted spike times.

        params
        ===
        remapped: bool, if using remapped (synced with imec0) spike times.
            Default: False
        """
        spike_times = self._spike_times_data

        for stream_num, stream in enumerate(range(len(spike_times))):
            if use_si:
                spike_times[stream_num] = self._get_si_spike_times(units)
            else:
                if remapped and stream_num > 0:
                    times = self.ks_outputs[stream_num] / f'spike_times_remapped.npy'
                    print(f"""\n> Found remapped spike times from\r
                    {self.ks_outputs[stream_num]}, try to load this.""")
                else:
                    times = self.ks_outputs[stream_num] / f'spike_times.npy'

                clust = self.ks_outputs[stream_num] / f'spike_clusters.npy'

                try:
                    times = np.load(times)
                    clust = np.load(clust)
                except FileNotFoundError:
                    msg = ": Can't load spike times that haven't been extracted!"
                    raise PixelsError(self.name + msg)

                times = np.squeeze(times)
                clust = np.squeeze(clust)
                by_clust = {}

                for c in np.unique(clust):
                    c_times = times[clust == c]
                    uniques, counts = np.unique(
                        c_times,
                        return_counts=True,
                    )
                    repeats = c_times[np.where(counts>1)]
                    if len(repeats>1):
                        print(f"> removed {len(repeats)} double-counted spikes from cluster {c}.")

                    by_clust[c] = pd.Series(uniques)
                spike_times[stream_num]  = pd.concat(by_clust, axis=1, names=['unit'])
                # Convert to time into sample rate index
                spike_times[stream_num] /= int(self.spike_meta[0]['imSampRate'])\
                                            / self.SAMPLE_RATE 

        return spike_times[0]

    def _get_aligned_spike_times(
        self, label, event, duration, rate=False, sigma=None, units=None
    ):
        """
        Returns spike times for each unit within a given time window around an event.
        align_trials delegates to this function, and should be used for getting aligned
        data in scripts.
        """
        action_labels = self.get_action_labels()[0]

        if units is None:
            units = self.select_units()

        #TODO: with multiple streams, spike times will be a list with multiple dfs,
        #make sure old code does not break!
        #TODO: spike times cannot be indexed by unit ids anymore
        spikes = self.get_spike_times()[units]

        if rate:
            # pad ends with 1 second extra to remove edge effects from convolution
            duration += 2

        scan_duration = self.SAMPLE_RATE * 8
        half = int((self.SAMPLE_RATE * duration) / 2)
        cursor = 0  # In sample points
        i = -1
        rec_trials = {}

        for rec_num in range(len(self.files)):
            actions = action_labels[rec_num][:, 0]
            events = action_labels[rec_num][:, 1]
            trial_starts = np.where(np.bitwise_and(actions, label))[0]

            # Account for multiple raw data files
            meta = self.spike_meta[rec_num]
            samples = int(meta["fileSizeBytes"]) / int(meta["nSavedChans"]) / 2
            assert samples.is_integer()
            milliseconds = samples / 30
            cursor_duration = cursor / 30
            rec_spikes = spikes[
                (cursor_duration <= spikes) & (spikes < (cursor_duration + milliseconds))
            ] - cursor_duration
            cursor += samples

            # Account for lag, in case the ephys recording was started before the
            # behaviour
            lag_start, _ = self._lag[rec_num]
            if lag_start < 0:
                rec_spikes = rec_spikes + lag_start

            for i, start in enumerate(trial_starts, start=i + 1):
                centre = np.where(np.bitwise_and(events[start:start + scan_duration], event))[0]
                if len(centre) == 0:
                    # See comment in align_trials as to why we just continue instead of
                    # erroring like we used to here.
                    print("No event found for an action. If this is OK, ignore this.")
                    continue
                centre = start + centre[0]

                trial = rec_spikes[centre - half < rec_spikes]
                trial = trial[trial <= centre + half]
                trial = trial - centre
                tdf = []

                for unit in trial:
                    u_times = trial[unit].values
                    u_times = u_times[~np.isnan(u_times)]
                    u_times = np.unique(u_times)  # remove double-counted spikes
                    udf = pd.DataFrame({int(unit): u_times})
                    tdf.append(udf)

                assert len(tdf) == len(units)
                if tdf:
                    tdfc = pd.concat(tdf, axis=1)
                    if rate:
                        tdfc = signal.convolve(tdfc, duration * 1000, sigma)
                    rec_trials[i] = tdfc

        if not rec_trials:
            return None

        trials = pd.concat(rec_trials, axis=1, names=["trial", "unit"])
        trials = trials.reorder_levels(["unit", "trial"], axis=1)
        trials = trials.sort_index(level=0, axis=1)

        if rate:
            # Set index to seconds and remove the padding 1 sec at each end
            points = trials.shape[0]
            start = (- duration / 2) + (duration / points)
            # Having trouble with float values
            #timepoints = np.linspace(start, duration / 2, points, dtype=np.float64)
            timepoints = list(range(round(start * 1000), int(duration * 1000 / 2) + 1))
            trials['time'] = pd.Series(timepoints, index=trials.index) / 1000
            trials = trials.set_index('time')
            trials = trials.iloc[self.SAMPLE_RATE : - self.SAMPLE_RATE]

        return trials


    def _get_aligned_trials(
        self, label, event, data, units=None, sigma=None, end_event=None,
    ):
        """
        Returns spike rate for each unit within a trial.
        align_trials delegates to this function, and should be used for getting aligned
        data in scripts.

        This function also saves binned data in the format that Alfredo wants:
        trials * units * temporal bins (100ms)

        """
        action_labels = self.get_action_labels()[0]
        streams = self.files["pixels"]
        output = {}

        if units is None:
            units = self.select_units()

        #if not pos_bin is None:
        behaviour_files = self.files["behaviour"]
        # assume only one vr session for now
        vr_dir = self.find_file(behaviour_files["vr_synched"][0])
        vr_data = ioutils.read_hdf5(vr_dir)
        # get positions
        positions = vr_data.position_in_tunnel

        #TODO: with multiple streams, spike times will be a list with multiple dfs,
        #make sure old code does not break!
        spikes = self.get_spike_times(units, use_si=True)
        # drop rows if all nans
        spikes = spikes.dropna(how="all")

        # since each session has one behaviour session, now only one action
        # label file
        actions = action_labels["outcome"]
        events = action_labels["events"]
        # get timestamps index of behaviour in self.SAMPLE_RATE hz, to convert
        # it to ms, do timestamps*1000/self.SAMPLE_RATE
        timestamps = action_labels["timestamps"]

        # select frames of wanted trial type
        trials = np.where(np.bitwise_and(actions, label))[0]
        # map starts by event
        starts = np.where(np.bitwise_and(events, event))[0]
        # map starts by end event
        ends = np.where(np.bitwise_and(events, end_event))[0]

        # only take starts from selected trials
        selected_starts = trials[np.where(np.isin(trials, starts))[0]]
        start_t = timestamps[selected_starts]
        # only take ends from selected trials
        selected_ends = trials[np.where(np.isin(trials, ends))[0]]
        end_t = timestamps[selected_ends]
        if selected_starts.size == 0:
            print(f"> No trials found with label {label} and event {event}, "
                  "output will be empty.")
            for key in streams.keys():
                output[key[:-3]] = {}
            return output

        # use original trial id as trial index
        trial_ids = vr_data.iloc[selected_starts].trial_count.unique()

        # pad ends with 1 second extra to remove edge effects from convolution
        scan_pad = self.SAMPLE_RATE
        scan_starts = start_t - scan_pad
        scan_ends = end_t + scan_pad
        scan_durations = scan_ends - scan_starts

        cursor = 0  # In sample points
        rec_trials_fr = {}
        rec_trials_spiked = {}
        trial_positions = {}

        for stream_num, (stream_id, stream_files) in enumerate(streams.items()):
            stream = stream_id[:-3]
            # allows multiple streams of recording, i.e., multiple probes
            rec_trials_fr[stream] = {}
            rec_trials_spiked[stream] = {}

            # Account for multiple raw data files
            meta = self.ap_meta[stream_num]
            samples = int(meta["fileSizeBytes"]) / int(meta["nSavedChans"]) / 2
            assert samples.is_integer()
            in_SAMPLE_RATE_scale = (samples * self.SAMPLE_RATE)\
                           / int(self.ap_meta[0]['imSampRate'])
            cursor_duration = (cursor * self.SAMPLE_RATE)\
                              / int(self.ap_meta[0]['imSampRate'])
            rec_spikes = spikes[
                (cursor_duration <= spikes)\
                & (spikes < (cursor_duration + in_SAMPLE_RATE_scale))
            ] - cursor_duration
            cursor += samples

            # Account for lag, in case the ephys recording was started before the
            # behaviour
            if not self._lag[stream_num] == None:
                lag_start, _ = self._lag[stream_num]
            else:
                lag_start = timestamps[0]

            if lag_start < 0:
                rec_spikes = rec_spikes + lag_start

            for i, start in enumerate(selected_starts):
                # select spike times of current trial
                trial_bool = (rec_spikes >= scan_starts[i])\
                            & (rec_spikes <= scan_ends[i])
                trial = rec_spikes[trial_bool]

                # get position bin ids for current trial
                trial_pos_bool = (positions.index >= start_t[i])\
                            & (positions.index < end_t[i])
                trial_pos = positions[trial_pos_bool]

                # initiate binary spike times array for current trial
                # NOTE: dtype must be float otherwise would get all 0 when
                # passing gaussian kernel
                times = np.zeros((scan_durations[i], len(units))).astype(float)
                # use pixels time as spike index
                idx = np.arange(scan_starts[i], scan_ends[i])
                # make it df, column name being unit id
                spiked = pd.DataFrame(times, index=idx, columns=units)

                # TODO mar 5 2025:
                # how to separate aligned trial times and chance, so that i can
                # use cacheable to get all conditions??????

                for unit in trial:
                    # get spike time for unit
                    u_times = trial[unit].values
                    # drop nan
                    u_times = u_times[~np.isnan(u_times)]

                    # round spike times to use it as index
                    u_spike_idx = np.round(u_times).astype(int)
                    # make sure it does not exceed scan duration
                    if (u_spike_idx >= scan_ends[i]).any():
                        beyonds = np.where(u_spike_idx >= scan_ends[i])[0]
                        u_spike_idx[beyonds] = idx[-1]
                        # make sure no double counted
                        u_spike_idx = np.unique(u_spike_idx)

                    # set spiked to 1
                    spiked.loc[u_spike_idx, unit] = 1

                # convolve spike trains into spike rates
                rates = signal.convolve_spike_trains(
                    times=spiked,
                    sigma=sigma,
                    sample_rate=self.SAMPLE_RATE,
                )
                # remove 1s padding from the start and end
                rates = rates.iloc[scan_pad: -scan_pad]
                spiked = spiked.iloc[scan_pad: -scan_pad]
                # reset index to zero at the beginning of the trial
                rates.reset_index(inplace=True, drop=True)
                rec_trials_fr[stream][trial_ids[i]] = rates
                spiked.reset_index(inplace=True, drop=True)
                rec_trials_spiked[stream][trial_ids[i]] = spiked
                trial_pos.reset_index(inplace=True, drop=True)
                trial_positions[trial_ids[i]] = trial_pos

            #if not rec_trials_fr[stream]:
            #    return None

            # TODO feb 28 2025:
            # in this func, scanning period is longer than actual trials to avoid
            # edging effect, so spiked is the whole scanning period then convolved
            # into spike rate. if however, we get the spike_bool here, it does not
            # have the scanning period buffer, the fr might have edging effect. but
            # if we take the scanning period, the number of spikes will be
            # different from the real data...
            # SOLUTION: concat as below, shuffle per column, then convolve per
            # column

            # concat trial positions
            positions = ioutils.reindex_by_longest(
                dfs=trial_positions,
                idx_names=["trial", "time"],
                level="trial",
                return_format="dataframe",
            )

            if data == "trial_times":
                # get trials vertically stacked spiked
                stacked_spiked = pd.concat(
                    rec_trials_spiked[stream],
                    axis=0,
                )
                stacked_spiked.index.names = ["trial", "time"]
                stacked_spiked.columns.names = ["unit"]

                # save index and columns to reconstruct df for shuffled data
                ioutils.save_index_to_frame(
                    df=stacked_spiked, 
                    path=self.interim / stream_files["shuffled_index"],
                )
                ioutils.save_cols_to_frame(
                    df=stacked_spiked, 
                    path=self.interim / stream_files["shuffled_columns"],
                )

                # get chance data paths
                paths = {
                    "spiked_memmap_path": self.interim /\
                        stream_files["spiked_shuffled_memmap"],
                    "fr_memmap_path": self.interim /\
                        stream_files["fr_shuffled_memmap"],
                    "spiked_df_path": self.processed / stream_files["spiked_shuffled"],
                    "fr_df_path": self.processed / stream_files["fr_shuffled"],
                }

                # save chance data
                xut.save_spike_chance(
                    **paths,
                    sigma=sigma,
                    sample_rate=self.SAMPLE_RATE,
                    spiked=stacked_spiked,
                )

                # get trials horizontally stacked spiked
                spiked = ioutils.reindex_by_longest(
                    dfs=stacked_spiked,
                    level="trial",
                    return_format="dataframe",
                )

                output[stream] = {
                    "spiked": spiked,
                    "positions": positions,
                }

            elif data == "trial_rate":
                # TODO apr 2 2025: make sure this reindex_by_longest works
                fr = ioutils.reindex_by_longest(
                    dfs=rec_trials_fr[stream],
                    level="trial",
                    idx_names=["trial", "time"],
                    col_names=["unit"],
                    return_format="dataframe",
                )

                output[stream] = {
                    "fr": fr,
                    "positions": positions,
                }

        # concat output into dataframe before cache
        #df = pd.concat(
        #    objs=output,
        #    axis=1,
        #    names=["stream"],
        #)
        return output


    def select_units(
        self, group='good', min_depth=0, max_depth=None, min_spike_width=None,
        unit_kwargs=None, max_spike_width=None, uncurated=False, name=None,
        use_si=False,
    ):
        """
        Select units based on specified criteria. The output of this can be passed to
        some other methods to apply those methods only to these units.

        Parameters
        ----------
        group : str, optional
            The group to which the units that are wanted are part of. One of: 'group',
            'mua', 'noise' or None. Default is 'good'.

        min_depth : int, optional
            (Only used when getting spike data). The minimum depth that units must be at
            to be included. Default is 0 i.e. in the brain.

        max_depth : int, optional
            (Only used when getting spike data). The maximum depth that units must be at
            to be included. Default is None i.e.  no maximum.

        min_spike_width : int, optional
            (Only used when getting spike data). The minimum median spike width that
            units must have to be included. Default is None i.e. no minimum.

        max_spike_width : int, optional
            (Only used when getting spike data). The maximum median spike width that
            units must have to be included. Default is None i.e. no maximum.

        uncurated : bool, optional
            Use uncurated units. Default: False.

        name : str, optional
            Give this selection of units a name. This allows the list of units to be
            represented as a string, which enables caching. Future calls to cacheable
            methods with a selection of units of the same name will read cached data
            from disk. It is up to the user to ensure that the actual selection of units
            is the same between uses of the same name.

        """
        if use_si:
            # NOTE: only deal with one stream for now
            stream_files = self.files["pixels"]["imec0.ap"]
            sa_dir = self.find_file(stream_files["sorting_analyser"])
            # load sorting analyser
            temp_sa = si.load_sorting_analyzer(sa_dir)
            # remove noisy units
            noisy_units = load_yaml(
                path=self.find_file(stream_files["noisy_units"]),
            )
            # remove units from sorting and reattach to sa to keep properties
            sorting = temp_sa.sorting.remove_units(remove_unit_ids=noisy_units)
            sa = temp_sa.remove_units(remove_unit_ids=noisy_units)
            sa.sorting = sorting

            # get units
            unit_ids = sa.unit_ids

            # init units class
            selected_units = SelectedUnits()
            if name is not None:
                selected_units.name = name

            if name == "all":
                selected_units.extend(unit_ids)
                return selected_units

            # get shank id for units
            shank_ids = sa.sorting.get_property("group")

            # get coordinates of channel with max. amplitude
            max_chan_coords = sa.sorting.get_property("max_chan_coords")
            # get depths
            depths = max_chan_coords[:, 1]

            if unit_kwargs:
                for shank_id, kwargs in unit_kwargs.items():
                    # get shank depths
                    min_depth = kwargs["min_depth"]
                    max_depth = kwargs["max_depth"]
                    # find units
                    in_range = unit_ids[
                        (depths >= min_depth) & (depths < max_depth) &\
                        (shank_ids == shank_id)
                    ]
                    # add to list
                    selected_units.extend(in_range)
            else:
                # if there is only one shank
                # find units
                in_range = unit_ids[
                    (depths >= min_depth) & (depths < max_depth)
                ]
                # add to list
                selected_units.extend(in_range)

            return selected_units

        else:
            cluster_info = self.get_cluster_info()
            selected_units = SelectedUnits()
            if name is not None:
                selected_units.name = name

            if min_depth is not None or max_depth is not None:
                probe_depths = self.get_probe_depth()

            if min_spike_width == 0:
                min_spike_width = None
            if min_spike_width is not None or max_spike_width is not None:
                widths = self.get_spike_widths()
            else:
                widths = None

            for stream_num, info in enumerate(cluster_info):
                # TODO jun 12 2024 skip stream 1 for now
                if stream_num > 0:
                    continue

                id_key = 'id' if 'id' in info else 'cluster_id'
                grouping = 'KSLabel' if uncurated else 'group'

                for unit in info[id_key]:
                    unit_info = info.loc[info[id_key] == unit].iloc[0].to_dict()

                    # we only want units that are in the specified group
                    if not group or unit_info[grouping] == group:

                        # and that are within the specified depth range
                        if min_depth is not None:
                            if probe_depths[stream_num] - unit_info['depth'] <= min_depth:
                                continue
                        if max_depth is not None:
                            if probe_depths[stream_num] - unit_info['depth'] > max_depth:
                                continue

                        # and that have the specified median spike widths
                        if widths is not None:
                            width = widths[widths['unit'] == unit]['median_ms']
                            assert len(width.values) == 1
                            if min_spike_width is not None:
                                if width.values[0] < min_spike_width:
                                    continue
                            if max_spike_width is not None:
                                if width.values[0] > max_spike_width:
                                    continue

                        selected_units.append(unit)

            return selected_units

    def _get_neuro_raw(self, kind):
        raw = []
        meta = getattr(self, f"{kind}_meta")
        for rec_num, recording in enumerate(self.files):
            data_file = self.find_file(recording[f'{kind}_data'], copy=False)
            orig_rate = int(meta[rec_num]['imSampRate'])
            num_chans = int(meta[rec_num]['nSavedChans'])
            factor = orig_rate / self.SAMPLE_RATE

            data = ioutils.read_bin(data_file, num_chans)

            if self._lag[rec_num] is None:
                self.sync_data(rec_num, sync_channel=data[:, -1])
            lag_start, lag_end = self._lag[rec_num]

            lag_start = int(lag_start * factor)
            lag_end = int(lag_end * factor)
            if lag_end < 0:
                data = data[:lag_end]
            if lag_start < 0:
                data = data[- lag_start:]
            raw.append(pd.DataFrame(data[:, :-1]))

        return raw, orig_rate

    def get_spike_data_raw(self):
        """
        Returns the raw spike data with lag region removed.
        """
        return self._get_neuro_raw('spike')

    def get_lfp_data_raw(self):
        """
        Returns the raw spike data with lag region removed.
        """
        return self._get_neuro_raw('lfp')

    @_cacheable
    def align_trials(
        self, label, event, units=None, data='spike_times', raw=False,
        duration=1, sigma=None, dlc_project=None, video_match=None,
        end_event=None,
    ):
        """
        Get trials aligned to an event. This finds all instances of label in the action
        labels - these are the start times of the trials. Then this finds the first
        instance of event on or after these start times of each trial. Then it cuts out
        a period around each of these events covering all units, rearranges this data
        into a MultiIndex DataFrame and returns it.

        Parameters
        ----------
        label : int
            An action label value to specify which trial types are desired.

        event : int
            An event type value to specify which event to align the trials to.

        data : str, optional
            The data type to align.

        raw : bool, optional
            Whether to get raw, unprocessed data instead of processed and downsampled
            data. Defaults to False.

        duration : int/float, optional
            The length of time in seconds desired in the output. Default is 1 second.

        sigma : int, optional
            Time in milliseconds of sigma of gaussian kernel to use when aligning firing
            rates. Default is 50 ms.

        units : list of lists of ints, optional
            The output from self.select_units, used to only apply this method to a
            selection of units.

        dlc_project : str | None
            The DLC project from which to get motion tracking coordinates, if aligning
            to motion_tracking data.

        video_match : str | None
            When aligning video or motion index data, use this fnmatch pattern to select
            videos.

        end_event : int | None
            For VR behaviour, when aligning to the whole trial, this param is
            the end event to align to.
        """
        data = data.lower()

        data_options = [
            'behavioural',  # Channels from behaviour TDMS file
            'spike',        # Raw/downsampled channels from probe (AP)
            'spike_times',  # List of spike times per unit
            'spike_rate',   # Spike rate signals from convolved spike times
            'lfp',          # Raw/downsampled channels from probe (LFP)
            'motion_index', # Motion index per ROI from the video
            'motion_tracking', # Motion tracking coordinates from DLC
            'trial_rate',   # Taking spike times from the whole duration of each
                            # trial, convolve into spike rate
            'trial_times',   # Taking spike times from the whole duration of each
                            # trial, get spike boolean
        ]
        if data not in data_options:
            raise PixelsError(f"align_trials: 'data' should be one of: {data_options}")

        if data in ("spike_times", "spike_rate"):
            print(f"Aligning {data} to trials.")
            # we let a dedicated function handle aligning spike times
            return self._get_aligned_spike_times(
                label, event, duration, rate=data == "spike_rate", sigma=sigma,
                units=units,
            )

        if "trial" in data:
            print(f"Aligning {data} of {units} units to trials.")
            return self._get_aligned_trials(
                label, event, data=data, units=units, sigma=sigma,
                end_event=end_event,
            )

        if data == "motion_tracking" and not dlc_project:
            raise PixelsError("When aligning to 'motion_tracking', dlc_project is needed.")

        action_labels = self.get_action_labels()

        if raw:
            print(f"Aligning raw {data} data to trials.")
            getter = getattr(self, f"get_{data}_data_raw", None)
            if not getter:
                raise PixelsError(f"align_trials: {data} doesn't have a 'raw' option.")
            values, SAMPLE_RATE = getter()

        else:
            print(f"Aligning {data} data to trials.")
            if dlc_project:
                values = self.get_motion_tracking_data(dlc_project)
            elif data == "motion_index":
                values = self.get_motion_index_data(video_match)
            else:
                values = getattr(self, f"get_{data}_data")()
            SAMPLE_RATE = self.SAMPLE_RATE

        if not values or values[0] is None:
            raise PixelsError(f"align_trials: Could not get {data} data.")

        trials = []
        # The logic here is that the action labels will always have a sample rate of
        # self.SAMPLE_RATE, whereas our data here may differ. 'duration' is used to scan
        # the action labels, so always give it 5 seconds to scan, then 'half' is used to
        # index data.
        scan_duration = self.SAMPLE_RATE * 10
        half = (SAMPLE_RATE * duration) // 2
        if isinstance(half, float):
            assert half.is_integer()  # In case duration is a float < 1
            half = int(half)

        for rec_num in range(len(self.files)):
            if values[rec_num] is None:
                # This means that each recording is using the same piece of data for
                # this data type, e.g. all recordings using motion indexes from a single
                # video
                break

            actions = action_labels[rec_num][:, 0]
            events = action_labels[rec_num][:, 1]
            trial_starts = np.where(np.bitwise_and(actions, label))[0]

            for start in trial_starts:
                centre = np.where(np.bitwise_and(events[start:start + scan_duration], event))[0]
                if len(centre) == 0:
                    # Previously it was assumed that if an event was not found within
                    # scan_duration after the start of the action, that something went
                    # wrong with the action labels calculation. This did not allow for
                    # some actions to have events that other of the same actions did
                    # not. Typically this is the case, but sometimes we want that
                    # flexibility. As a compromise, we can print that this has happened
                    # here to warn the user in case it is an error, while otherwise
                    # continuing.
                    #raise PixelsError('Action labels probably miscalculated')
                    print("No event found for an action. If this is OK, ignore this.")
                    continue
                centre = start + centre[0]
                centre = int(centre * SAMPLE_RATE / self.SAMPLE_RATE)
                trial = values[rec_num][centre - half + 1:centre + half + 1]

                if isinstance(trial, np.ndarray):
                    trial = pd.DataFrame(trial)
                trials.append(trial.reset_index(drop=True))

        if not trials:
            raise PixelsError("Seems the action-event combo you asked for doesn't occur")

        if data == "motion_tracking":
            ses_trials = pd.concat(
                trials,
                axis=1,
                copy=False,
                keys=range(len(trials)),
                names=["trial"] + trials[0].columns.names
            )
        else:
            ses_trials = pd.concat(
                trials, axis=1, copy=False, keys=range(len(trials)), names=["trial", "unit"]
            )
            ses_trials = ses_trials.sort_index(level=1, axis=1)
            ses_trials = ses_trials.reorder_levels(["unit", "trial"], axis=1)

        points = ses_trials.shape[0]
        start = (- duration / 2) + (duration / points)
        timepoints = np.linspace(start, duration / 2, points)
        ses_trials['time'] = pd.Series(timepoints, index=ses_trials.index)
        ses_trials = ses_trials.set_index('time')

        if data == "motion_index":
            ses_trials = ses_trials.rename_axis(columns=["ROI", "trial"])

        return ses_trials

    def align_clips(self, label, event, video_match, duration=1):
        """
        Get video clips aligned to an event. This is very similar to align_trials but is
        specific to video clips. The distinction is made for a number of reasons. Video
        clip data gets very big and so is not cached to disk. It is also not loaded
        here; this method only provides generators that can be used to consume video
        frames as numpy arrays.

        The parameters are the same as those for align_trials.
        """
        action_labels = self.get_action_labels()

        scan_duration = self.SAMPLE_RATE * 8
        half = int((self.SAMPLE_RATE * duration) / 2)
        cursor = 0  # In sample points
        i = -1
        rec_trials = []
        rec_durations = []

        for rec_num, recording in enumerate(self.files):
            for v, video in enumerate(recording.get('camera_data', [])):
                avi = video.with_suffix('.avi')
                if video_match in avi.as_posix():
                    break
            else:
                raise PixelsError(f"Failed to find a video with match {video_match}")

            path = self.find_file(avi)
            actions = action_labels[rec_num][:, 0]
            events = action_labels[rec_num][:, 1]
            trial_starts = np.where(np.bitwise_and(actions, label))[0]

            behavioural_data = ioutils.read_tdms(self.find_file(recording['behaviour']))
            assert 0
            behavioural_data = behavioural_data["/'CamFrames'/'0'"]
            behav_array = signal.resample(behavioural_data.values, 25000, self.SAMPLE_RATE)
            behavioural_data.iloc[:len(behav_array)] = np.squeeze(behav_array)
            behavioural_data = behavioural_data[:len(behav_array)]
            trigger = signal.binarise(behavioural_data).values
            onsets = np.where((trigger[:-1] == 1) & (trigger[1:] == 0))[0]

            # The last frame sometimes gets delayed a bit, so ignoring it, are the timestamp
            # diffs fixed?
            assert len(np.unique(np.diff(onsets[:-1]))) == 1

            lag_start, _ = self._lag[rec_num]
            onsets = onsets - lag_start
            # Index is the time in ms relative to start of action labels
            # Value is the frame number
            timings = pd.DataFrame(range(len(onsets)), index=onsets, columns=["Frame"])

            clips = []

            for start in trial_starts:
                centre = np.where(np.bitwise_and(events[start:start + scan_duration], event))[0]
                if len(centre) == 0:
                    print("No event found for an action. If this is OK, ignore this.")
                    continue
                centre = start + centre[0]
                frames = timings.loc[
                    (centre - half + 1 < timings.index) & (timings.index < centre + half + 1)
                ]
                rec_trials.append(
                    ioutils.load_video_frames(path, np.squeeze(frames.values))
                )
                rec_durations.append(len(frames))

        # This dataframe follows a different structure to others because the data is
        # really a 3D matrix per trial. Here the row index is trials, and each value is
        # a generator (ioutils.stream_video) that yields frames for that trial's period
        # of time. Downstream code can do what it wants with those generators. Using
        # generators means that debugging any code, both downstream code and this
        # method, is relatively painless, and consuming the frame data only loads the
        # frames that are actually needed, e.g. if you only need a few trials, this
        # method is still good for that.
        trials = pd.DataFrame(
            {
                "Generators": rec_trials,
                "Durations": rec_durations,
            },
            index=range(len(rec_trials)),
        )
        return trials

    def get_cluster_info(self):
        for stream_num, info in enumerate(self._cluster_info):
            if info is None:
                info_file = self.ks_outputs[stream_num] / 'cluster_info.tsv'
                try:
                    info = pd.read_csv(info_file, sep='\t')
                except FileNotFoundError:
                    msg = ": Can't load cluster info. Did you sort this session yet?"
                    raise PixelsError(self.name + msg)
            self._cluster_info[stream_num] = info
        return self._cluster_info

    def get_good_units_info(self):
        if self._good_unit_info is None:
            #az: good_units_info.tsv saved while running depth_profile.py
            info_file = self.interim / 'good_units_info.tsv'
            #print(f"> got good unit info at {info_file}\n")

            try:
                info = pd.read_csv(info_file, sep='\t')
            except FileNotFoundError:
                msg = ": Can't load cluster info. Did you export good unit info for this session yet?"
                raise PixelsError(self.name + msg)
            self._good_unit_info = info
        return self._good_unit_info

    @_cacheable
    def get_spike_widths(self, units=None):
        if units:
            # Always defer to getting widths for all units, so we only ever have to
            # calculate spike widths for each once.
            all_widths = self.get_spike_widths()
            return all_widths.loc[all_widths.unit.isin(units)]

        print("Calculating spike widths")
        waveforms = self.get_spike_waveforms()
        widths = []

        for unit in waveforms.columns.get_level_values('unit').unique():
            u_widths = []
            u_spikes = waveforms[unit]

            for s in u_spikes:
                spike = u_spikes[s]
                trough = np.where(spike.values == min(spike))[0][0]
                after = spike.values[trough:]
                width = np.where(after == max(after))[0][0]
                u_widths.append(width)

            widths.append((unit, np.median(u_widths)))

        df = pd.DataFrame(widths, columns=["unit", "median_ms"])
        # convert to ms from sample points
        orig_rate = int(self.spike_meta[0]['imSampRate'])
        df['median_ms'] = 1000 * df['median_ms'] / orig_rate
        return df

    @_cacheable
    def get_spike_waveforms(self, units=None, method='phy'):
        """
        Extracts waveforms of spikes.
        method: str, name of selected method.
            'phy' (default)
            'spikeinterface'
        """
        if method == 'phy':
            from phylib.io.model import load_model
            from phylib.utils.color import selected_cluster_color

            if units:
                # defer to getting waveforms for all units
                waveforms = self.get_spike_waveforms()[units]
                assert list(waveforms.columns.get_level_values("unit").unique()) == list(units)
                return waveforms

            units = self.select_units()

            #paramspy = self.processed / 'sorted_stream_0' / 'params.py'
            #TODO: with multiple streams, spike times will be a list with multiple dfs,
            #make sure put old code under loop of stream so it does not break!
            paramspy = self.ks_outputs / 'params.py'
            if not paramspy.exists():
                raise PixelsError(f"{self.name}: params.py not found")
            model = load_model(paramspy)
            rec_forms = {}

            for u, unit in enumerate(units):
                print(f"{round(100 * u / len(units), 2)}% complete")
                # get the waveforms from only the best channel
                spike_ids = model.get_cluster_spikes(unit)
                best_chan = model.get_cluster_channels(unit)[0]
                u_waveforms = model.get_waveforms(spike_ids, [best_chan])
                if u_waveforms is None:
                    raise PixelsError(f"{self.name}: unit {unit} - waveforms not read")
                rec_forms[unit] = pd.DataFrame(np.squeeze(u_waveforms).T)

            assert rec_forms

            df = pd.concat(
                rec_forms,
                axis=1,
                names=['unit', 'spike']
            )
            # convert indexes to ms
            rate = 1000 / int(self.spike_meta[0]['imSampRate'])
            df.index = df.index * rate
            return df

        #TODO: implement spikeinterface waveform extraction
        elif method == 'spikeinterface':
            ## set chunks
            #job_kwargs = dict(
            #    n_jobs=-3, # -1: num of job equals num of cores
            #    chunk_duration="1s",
            #    progress_bar=True,
            #)
            recording, _ = self.load_recording()
            #TODO: with multiple streams, spike times will be a list with multiple dfs,
            #make sure put old code under loop of stream so it does not break!
            try:
                sorting = se.read_kilosort(self.ks_outputs)
            except ValueError as e:
                raise PixelsError(
                    f"Can't load sorting object. Did you delete cluster_info.csv? Full error: {e}\n"
                )

            # check last modified time of cache, and create time of ks_output
            try:
                template_cache_mod_time = os.path.getmtime(self.interim /
                                                           'cache/templates_average.npy')
                #TODO: with multiple streams, spike times will be a list with multiple dfs,
                #make sure put old code under loop of stream so it does not break!
                ks_mod_time = os.path.getmtime(self.ks_outputs / 'cluster_info.tsv')
                assert template_cache_mod_time < ks_mod_time
                check = True # re-extract waveforms
                print("> Re-extracting waveforms since kilosort output is newer.") 
            except:
                if 'template_cache_mod_time' in locals():
                    print("> Loading existing waveforms.") 
                    check = False # load existing waveforms
                else:
                    print("> Extracting waveforms since they are not extracted.") 
                    check = True # re-extract waveforms

            """
            # for testing: get first 5 mins of the recording 
            fs = concat_rec.get_sampling_frequency()
            test = concat_rec.frame_slice(
                start_frame=0*fs,
                end_frame=300*fs,
            )
            test.annotate(is_filtered=True)
            # check all annotations
            test.get_annotation('is_filtered')
            print(test)
            """

            # extract waveforms
            waveforms = si.extract_waveforms(
                recording=recording,
                sorting=sorting,
                folder=self.interim / 'cache',
                load_if_exists=not(check), # maybe re-extracted
                max_spikes_per_unit=500, # None will extract all waveforms
                ms_before=2.0, # time before trough 
                ms_after=3.0, # time after trough 
                overwrite=check, # overwrite depends on check
                **job_kwargs,
            )
            #TODO: use cache to export the results?

            return waveforms

        else:
            raise PixelsError(f"{self.name}: waveform extraction method {method} is\
                              not implemented!")


    @_cacheable
    def get_waveform_metrics(self, units=None, window=20, upsampling_factor=10):
        """
        This func is a work-around of spikeinterface's equivalent:
        https://github.com/SpikeInterface/spikeinterface/blob/master/spikeinterface/postprocessing/template_metrics.py.
        dec 23rd 2022: motivation to write this function is that spikeinterface
        0.96.1 cannot load sorting object from `export_to_phy` output folder, i.e., i
        cannot get updated clusters/units and their waveforms, which is a huge
        problem for subsequent analyses e.g. unit type clustering.

        To learn more about waveform metrics, see
        https://github.com/AllenInstitute/ecephys_spike_sorting/tree/master/ecephys_spike_sorting/modules/mean_waveforms
        and https://journals.physiology.org/doi/full/10.1152/jn.00680.2018.

        """
        if units:
            # Always defer to getting waveform metrics for all good units, so we only
            # ever have to calculate metrics for each once.
            wave_metrics = self.get_waveform_metrics()
            return wave_metrics.loc[wave_metrics.unit.isin(units)]

        # TODO june 2nd 2023: extract amplitude, i.e., abs(trough - peak) in mV
        # make sure amplitude is in mV
        # normalise these metrics before passing to k-means
        columns = ["unit", "duration", "trough_peak_ratio", "half_width",
                   "repolarisation_slope", "recovery_slope"]
        print(f"> Calculating waveform metrics {columns[1:]}...\n")

        waveforms = self.get_spike_waveforms()
        # remove nan values
        waveforms.dropna()
        units = waveforms.columns.get_level_values('unit').unique()

        output = {}
        for i, unit in enumerate(units):
            metrics = []
            #mean_waveform = waveforms[unit].mean(axis=1)
            median_waveform = waveforms[unit].median(axis=1)
            # normalise mean waveform to remove variance caused by distance!
            norm_waveform = median_waveform / median_waveform.abs().max()
            #TODO: test! also can try clustering on normalised meann waveform
            mean_waveform = norm_waveform

            # time between trough to peak, in ms
            trough_idx = np.argmin(mean_waveform)
            peak_idx = trough_idx + np.argmax(mean_waveform.iloc[trough_idx:])
            if peak_idx == 0:
                raise PixelsError(f"> Cannot find peak in mean waveform.\n")
            if trough_idx == 0:
                raise PixelsError(f"> Cannot find trough in mean waveform.\n")
            duration = mean_waveform.index[peak_idx] - mean_waveform.index[trough_idx]
            metrics.append(duration)

            # trough to peak ratio
            trough_peak_ratio = mean_waveform.iloc[peak_idx] / mean_waveform.iloc[trough_idx]
            metrics.append(trough_peak_ratio)

            # spike half width, in ms
            half_amp = mean_waveform.iloc[trough_idx] / 2
            idx_pre_half = np.where(mean_waveform.iloc[:trough_idx] < half_amp)
            idx_post_half = np.where(mean_waveform.iloc[trough_idx:] < half_amp)
            # last occurence of mean waveform amp lower than half amp, before trough
            if len(idx_pre_half[0]) == 0:
                idx_pre_half = trough_idx - 1
                time_pre_half = mean_waveform.index[idx_pre_half]
            else:
                time_pre_half = mean_waveform.iloc[idx_pre_half[0] - 1].index[0]
            # first occurence of mean waveform amp lower than half amp, after trough
            time_post_half = mean_waveform.iloc[idx_post_half[0] + 1 +
                                                trough_idx].index[-1]
            half_width = time_post_half - time_pre_half
            metrics.append(half_width)

            # repolarisation slope
            returns = np.where(mean_waveform.iloc[trough_idx:] >= 0) + trough_idx
            if len(returns[0]) == 0:
                print(f"> The mean waveformrns never returned to baseline?\n")
                return_idx = mean_waveform.shape[0] - 1
            else:
                return_idx = returns[0][0]
                if return_idx - trough_idx < 2:
                    raise PixelsError(f"> The mean waveform returns to baseline too quickly,\
                                      \ndoes not make sense...\n")
            repo_period = mean_waveform.iloc[trough_idx:return_idx]
            repo_slope = scipy.stats.linregress(
                x=repo_period.index.values, # time in ms
                y=repo_period.values, # amp
             ).slope
            metrics.append(repo_slope)

            # recovery slope during user-defined recovery period
            recovery_end_idx = peak_idx + window
            recovery_end_idx = np.min([recovery_end_idx, mean_waveform.shape[0]])
            reco_period = mean_waveform.iloc[peak_idx:recovery_end_idx]
            reco_slope = scipy.stats.linregress(
                x=reco_period.index.values, # time in ms
                y=reco_period.values, # amp
             ).slope
            metrics.append(reco_slope)

            # save metrics in output dictionary, key is unit id
            output[unit] = metrics

        # save all template metrics as dataframe
        df = pd.DataFrame(output).T.reset_index()
        df.columns = columns
        dtype = {"unit": int}
        df = df.astype(dtype)
        # see which cols have nan
        df.isnull().sum()

        return df


    @_cacheable
    def get_aligned_spike_rate_CI(
        self, label, event,
        start=0.000, step=0.100, end=1.000,
        bl_label=None, bl_event=None, bl_start=None, bl_end=0.000,
        ss=20, CI=95, bs=10000,
        units=None, sigma=None,
    ):
        """
        Get the confidence intervals of the mean firing rates within a window aligned to
        a specified action label and event. An example would be to align firing rates to
        cue and take a 200 ms pre-cue window, mean the windows, and compute bootstrapped
        confidence intervals for those values. Optionally baseline the windowed values
        using values from another window.

        Parameters
        ----------
        label : ActionLabel int
            Action to align to, from a specific behaviour's ActionLabels class.

        event : Event int
            Event to align to, from a specific behaviour's Events class.

        start : float or np.ndarray, optional
            Time in milliseconds relative to event for the left edge of the bins.
            Alternatively, this can be an array of times in milliseconds, one per trial,
            to us per-trial start times. `step` is ignored in this case. As the
            timepoints pertain to only one session, using this via `Experiment`
            doesn't make sense.

        step : float, optional
            Time in milliseconds for the bin size.

        end : float or np.ndarray, optional
            Time in milliseconds relative to event for the right edge of the bins.
            Alternatively, this can be an array of times in milliseconds, one per trial,
            to us per-trial end times. `step` is ignored in this case.

        bl_label, bl_event, bl_start, bl_end : as above, all optional
            Equivalent to the above parameters but for baselining data. By default no
            baselining is performed.

        ss : int, optional.
            Sample size of bootstrapped samples.

        CI : int/float, optional
            Confidence interval size. Default is 95: this returns the 2.5%, 50% and
            97.5% bootstrap sample percentiles.

        bs : int, optional
            Number of bootstrapped samples. Default is 10000.

        units : list of lists of ints, optional
            The output from self.select_units, used to only apply this method to a
            selection of units.

        sigma : int, optional
            Time in milliseconds of sigma of gaussian kernel to use. Default is 50 ms.

        """
        if bl_start is not None:
            bl_start = round(bl_start, 3)
        bl_end = round(bl_end, 3)

        # Set timepoints to 3 decimal places (ms) to make things easier
        if isinstance(start, float):
            start = round(start, 3)
        if isinstance(end, float):
            end = round(end, 3)

        if isinstance(end, np.ndarray) or isinstance(start, np.ndarray):
            # We only use step and bin data when we aren't passing in arrays of
            # timepoints.
            step = None
        else:
            step = round(step, 3)

        max_start = abs(start) if isinstance(start, float) else np.abs(start).max()
        max_end = abs(end) if isinstance(end, float) else np.abs(end).max()
        duration = round(2 * max(max_start, max_end) + 0.002, 3)

        # Get firing rates
        responses = self.align_trials(
            label, event, 'spike_rate', duration=duration, sigma=sigma, units=units
        )
        if responses is None:
            return None
        series = responses.index.values

        if step is None:
            trials = responses.columns.get_level_values("trial").unique()
            if isinstance(start, float):
                assert series[0] <= start <= series[-1]
                start = np.full(trials.shape, start)
            elif isinstance(end, float):
                assert series[0] <= end <= series[-1]
                end = np.full(trials.shape, end)

            responses = responses.swaplevel(axis=1)
            trial_responses = []
            for trial, t_start, t_end in zip(trials, start, end):
                if not (t_start < t_end):
                    print(
                        f"Warning: trial {trial} skipped in CI calculation due to bad timepoints"
                    )
                    continue
                trial_responses.append(
                    responses[trial].loc[t_start:t_end].mean()
                )
                assert not responses[trial].loc[t_start:t_end].mean().isna().any()

            averages = pd.concat(trial_responses, axis=1, keys=trials)
            averages = averages.melt(ignore_index=False).rename(dict(value=0), axis=1)
            averages = averages.set_index("trial", append=True).sort_index()

        else:
            assert series[0] <= start < end <= series[-1] + 0.001

            bins = round((end - start) / step, 10)  # Round in case of floating point recursive
            assert bins.is_integer()
            bin_responses = []

            for i in range(int(bins)):
                bin_start = start + i * step
                bin_end = bin_start + step
                bin_responses.append(responses.loc[bin_start : bin_end].mean())

            averages = pd.concat(bin_responses, axis=1)

        # Optionally baseline the firing rates
        if bl_start is not None and bl_end is not None:
            duration = 2 * max(abs(bl_start), abs(bl_end))

            if bl_label is None:
                bl_label = label
            if bl_event is None:
                bl_event = event

            baselines = self.align_trials(
                bl_label, bl_event, 'spike_rate', duration=duration, sigma=sigma, units=units
            )
            series = baselines.index.values
            assert series[0] <= (bl_start + 0.001) < bl_end <= series[-1]
            baseline = baselines.loc[bl_start : bl_end].mean()
            for i in averages:
                averages[i] = averages[i] - baseline

        # Calculate the confidence intervals for each unit and bin
        lower = (100 - CI) / 2
        upper = 100 - lower
        percentiles = [lower, 50, upper]
        cis = []

        rec_cis = []
        for unit in units:
            u_resps = averages.loc[unit]
            samples = np.array([
                [np.random.choice(u_resps[i], size=ss) for b in range(bs)]
                for i in u_resps
            ])
            medians = np.median(samples, axis=2)
            results = np.percentile(medians, percentiles, axis=1)
            rec_cis.append(pd.DataFrame(results))

        if rec_cis:
            rec_df = pd.concat(rec_cis, axis=1, keys=units)
        else:
            rec_df = pd.DataFrame(
                {rec_num: np.nan},
                index=range(3),
                columns=pd.MultiIndex.from_arrays([[-1], [-1]], names=['unit', 'bin'])
            )
        cis.append(rec_df)

        df = pd.concat(cis, axis=1, names=['unit', 'bin'])
        df.set_index(pd.Index(percentiles, name="percentile"), inplace=True)
        return df


    @_cacheable
    def get_positional_rate(
        self, label, event, end_event=None, sigma=None, units=None,
    ):
        """
        Get positional firing rate of selected units in vr, and spatial
        occupancy of each position.
        """
        # get constants from vd
        from vision_in_darkness.constants import TUNNEL_RESET, ZONE_END

        # TODO dec 18 2024:
        # rearrange vr specific funcs to vr module
        # put pixels specific funcs in pixels module

        # NOTE: order of args matters for loading the cache!
        # always put units first, cuz it is like that in
        # experiemnt.align_trials, otherwise the same cache cannot be loaded

        # get aligned firing rates and positions
        trials = self.align_trials(
            units=units, # NOTE: ALWAYS the first arg
            data="trial_rate", # NOTE: ALWAYS the second arg
            label=label,
            event=event,
            sigma=sigma,
            end_event=end_event,
        )
        fr = trials["fr"]
        positions = trials["positions"]

        # get unit_ids
        unit_ids = fr.columns.get_level_values("unit").unique()

        # create position indices
        indices = np.arange(0, TUNNEL_RESET+2)
        # create occupancy array for trials
        occupancy = np.full(
            (TUNNEL_RESET+2, positions.shape[1]),
            np.nan,
        )
        # create array for positional firing rate
        pos_fr = {}

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
            trial_fr = fr.xs(
                key=trial,
                axis=1,
                level="trial",
            ).dropna(how="all").copy()

            # get all indices before post reset
            no_post_reset = trial_fr.index.intersection(trial_pos.index)
            # remove post reset rows
            trial_fr = trial_fr.loc[no_post_reset]
            trial_pos = trial_pos.loc[no_post_reset]

            # put trial positions in trial fr df
            trial_fr["position"] = trial_pos.values
            # group values by position and get mean
            mean_fr = trial_fr.groupby("position")[unit_ids].mean()
            # reindex into full tunnel length
            pos_fr[trial] = mean_fr.reindex(indices)
            # get trial occupancy
            pos_count = trial_fr.groupby("position").size()
            occupancy[pos_count.index.values, t] = pos_count.values

        # concatenate dfs
        pos_fr = pd.concat(pos_fr, axis=1, names=["trial", "unit"])
        # convert to df
        occupancy = pd.DataFrame(
            data=occupancy,
            index=indices,
            columns=positions.columns,
        )

        # add another level of starting position
        # Get the starting index for each trial (column)
        starts = occupancy.apply(lambda col: col.first_valid_index())
        # Group trials by their starting index
        trial_level = pos_fr.columns.get_level_values("trial")
        unit_level = pos_fr.columns.get_level_values("unit")
        # map start level
        start_level = trial_level.map(starts)
        # define new columns
        new_cols = pd.MultiIndex.from_arrays(
            [start_level, unit_level, trial_level],
            names=["start", "unit", "trial"],
        )
        pos_fr.columns = new_cols
        # sort by unit
        pos_fr = pos_fr.sort_index(level="unit", axis=1)

        return {"pos_fr": pos_fr, "occupancy": occupancy}

    
    def bin_aligned_trials(
        self, label, event, units=None, sigma=None, end_event=None,
        time_bin=None, pos_bin=None,
    ):
        """
        Returns spike rate for each unit within a trial.
        align_trials delegates to this function, and should be used for getting aligned
        data in scripts.

        This function also saves binned data in the format that Andrew wants:
        trials * units * temporal bins (ms)

        time_bin: str | None
            For VR behaviour, size of temporal bin for spike rate data.

        pos_bin: int | None
            For VR behaviour, size of positional bin for position data.

        """
        # TODO mar 31 2025:
        # use cached get_aligned_trials to bin so that we do not need to
        # duplicate code

        bin_frs = {}
        bin_counts = {}
        bin_counts_chance = {}

        # get aligned spiked and positions
        spiked = self.align_trials(
            units=units, # NOTE: ALWAYS the first arg
            data="trial_times", # NOTE: ALWAYS the second arg
            label=label,
            event=event,
            sigma=sigma,
            end_event=end_event,
        )
        fr = self.align_trials(
            units=units, # NOTE: ALWAYS the first arg
            data="trial_rate", # NOTE: ALWAYS the second arg
            label=label,
            event=event,
            sigma=sigma,
            end_event=end_event,
        )

        streams = self.files["pixels"]
        for stream_num, (stream_id, stream_files) in enumerate(streams.items()):
            stream = stream_id[:-3]
            # define output path for binned spike rate
            output_fr_path = self.interim/\
                f'cache/{self.name}_{stream}_{label}_{units}_{time_bin}_spike_rate.npz'
            output_count_path = self.interim/\
                f'cache/{self.name}_{stream}_{label}_{units}_{time_bin}_spike_count.npz'
            if output_count_path.exists():
                continue

            key = f"{stream_id[:-3]}/"
            print(f"\n> Binning trials from {stream_id}.")

            # get stream spiked
            stream_spiked = spiked[stream]["spiked"]
            if stream_spiked.size == 0:
                print(f"\n> No units found in {units}, continue.")
                return None

            # get stream positions
            positions = spiked[stream]["positions"]
            # get stream firing rates
            stream_fr = fr[stream]["fr"]

            # TODO apr 11 2025:
            # bin chance while bin data
            #spiked_chance_path = self.processed / stream_files["spiked_shuffled"]
            #spiked_chance = ioutils.read_hdf5(spiked_chance_path, "spiked")
            #bin_counts_chance[stream_id] = {}

            bin_frs[stream_id] = {}
            bin_counts[stream_id] = {}
            for trial in positions.columns.unique():
                counts = stream_spiked.xs(trial, level="trial", axis=1).dropna()
                rates = stream_fr.xs(trial, level="trial", axis=1).dropna()
                trial_pos = positions[trial].dropna()

                # get bin spike count
                bin_counts[stream_id][trial] = xut.bin_vr_trial(
                    data=counts,
                    positions=trial_pos,
                    sample_rate=self.SAMPLE_RATE,
                    time_bin=time_bin,
                    pos_bin=pos_bin,
                    bin_method="sum",
                )
                # get bin firing rates
                bin_frs[stream_id][trial] = xut.bin_vr_trial(
                    data=rates,
                    positions=trial_pos,
                    sample_rate=self.SAMPLE_RATE,
                    time_bin=time_bin,
                    pos_bin=pos_bin,
                    bin_method="mean",
                )

            # stack df values into np array
            # reshape into trials x units x bins
            bin_count_arr = ioutils.reindex_by_longest(bin_counts[stream_id]).T
            bin_fr_arr = ioutils.reindex_by_longest(bin_frs[stream_id]).T

            # save bin_fr and bin_count, for alfredo & andrew
            # use label as array key name
            fr_to_save = {
                "fr": bin_fr_arr[:, :-2, :],
                "pos": bin_fr_arr[:, -2:, :],
            }
            np.savez_compressed(output_fr_path, **fr_to_save)
            print(f"> Output saved at {output_fr_path}.")

            count_to_save = {
                "count": bin_count_arr[:, :-2, :],
                "pos": bin_count_arr[:, -2:, :],
            }
            np.savez_compressed(output_count_path, **count_to_save)
            print(f"> Output saved at {output_count_path}.")

        return None

    def get_chance_data(self, time_bin, pos_bin):

        streams = self.files["pixels"]
        for stream_num, (stream_id, stream_files) in enumerate(streams.items()):

            paths = {
                "spiked_memmap_path": self.interim /\
                    stream_files["spiked_shuffled_memmap"],
                "fr_memmap_path": self.interim /\
                    stream_files["fr_shuffled_memmap"],
                "memmap_shape_path": self.interim /\
                    stream_files["shuffled_shape"],
                "idx_path": self.interim / stream_files["shuffled_index"],
                "col_path": self.interim /\
                    stream_files["shuffled_columns"],
            }

            # TODO apr 3 2025: how the fuck to get positions here????
            # TEMP: get it manually...
            # light
            pos_path = self.interim /\
                    "cache/align_trials_all_trial_times_725_1_100_512.h5"
            # dark
            #pos_path = self.interim /\
            #        "cache/align_trials_all_trial_times_1322_1_100_512.h5"

            with pd.HDFStore(pos_path, "r") as store:
                # list all keys
                keys = store.keys()
                # create df as a dictionary to hold all dfs
                df = {}
                # TODO apr 2 2025: for now the nested dict have keys in the
                # format of `/imec0.ap/positions`, this will not be the case
                # once i flatten files at the stream level rather than
                # session level, i.e., every pixels related cache will have
                # stream id in their name.
                for key in keys:
                    # read current df
                    data = store[key]
                    # remove "/" in key
                    key_name = key.lstrip("/")
                    # use key name as dict key
                    df[key_name] = data
            positions = df[f"{stream_id[:-3]}/positions"]

            xut.get_spike_chance(
                sample_rate=self.SAMPLE_RATE,
                positions=positions,
                time_bin=time_bin,
                pos_bin=pos_bin,
                **paths,
            )
            assert 0
            #spiked_chance = ioutils.read_hdf5(spiked_chance_path, key="spiked")

        return None
