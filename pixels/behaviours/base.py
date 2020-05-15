"""
This module provides a base class for experimental sessions that must be used as the
base for defining behaviour-specific processing.
"""


import datetime
import json
import os
import tarfile
import time
from abc import ABC, abstractmethod
from shutil import copyfile

import numpy as np
import pandas as pd
import scipy.signal

from pixels import ioutils
from pixels import signal
from pixels.error import PixelsError


BEHAVIOUR_HZ = 25000


class Behaviour(ABC):

    sample_rate = 1000

    def __init__(self, name, data_dir, metadata=None):
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

        """
        self.name = name
        self.data_dir = data_dir
        self.metadata = metadata

        self.raw = self.data_dir / 'raw' / self.name
        self.interim = self.data_dir / 'interim' / self.name
        self.processed = self.data_dir / 'processed' / self.name
        self.interim.mkdir(parents=True, exist_ok=True)
        (self.processed / 'cache').mkdir(parents=True, exist_ok=True)
        self.files = ioutils.get_data_files(self.raw, name)

        self._action_labels = None
        self._behavioural_data = None
        self._spike_data = None
        self._lfp_data = None
        self._lag = None
        self.drop_data()

        self.spike_meta = [
            ioutils.read_meta(self.find_file(f['spike_meta'])) for f in self.files
        ]
        self.lfp_meta = [
            ioutils.read_meta(self.find_file(f['lfp_meta'])) for f in self.files
        ]

        self.trial_duration = 6  # number of seconds in which to extract trials

    def drop_data(self):
        """
        Clear attributes that store data to clear some memory.
        """
        self._action_labels = [None] * len(self.files)
        self._behavioural_data = [None] * len(self.files)
        self._spike_data = [None] * len(self.files)
        self._lfp_data = [None] * len(self.files)
        self._lag = [None] * len(self.files)

    def find_file(self, name):
        """
        Finds the specified file, looking for it in the processed folder, interim
        folder, and then raw folder in that order. If the the file is only found in the
        raw folder, it is copied to the interim folder and uncompressed if required.

        Parameters
        ----------
        name : str or pathlib.Path
            The basename of the file to be looked for.

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
            print(f"    Copying {name} to interim")
            copyfile(raw, interim)
            return interim

        tar = raw.with_name(raw.name + '.tar.gz')
        if tar.exists():
            print(f"    Extracting {tar.name} to interim")
            with tarfile.open(tar) as open_tar:
                open_tar.extractall(path=self.interim)
            return interim

    def process_behaviour(self):
        """
        Process behavioural data from raw tdms and align to neuropixels data.
        """
        for rec_num, recording in enumerate(self.files):
            print(f">>>>> Processing behaviour for recording {rec_num + 1} of {len(self.files)}")

            print(f"> Loading behavioural data")
            behavioural_data = ioutils.read_tdms(self.find_file(recording['behaviour']))

            print(f"> Downsampling to {self.sample_rate} Hz")
            behav_array = signal.resample(behavioural_data, 25000, self.sample_rate)
            behavioural_data.iloc[:len(behav_array), :] = behav_array
            behavioural_data = behavioural_data[:len(behav_array)]
            del behav_array

            if self._lag[rec_num] is not None:
                lag_start, lag_end = self._lag[rec_num]
            else:
                lag_start, lag_end = self.sync_data(rec_num, behavioural_data=behavioural_data)

            print(f"> Extracting action labels")
            behavioural_data = behavioural_data[max(lag_start, 0):-1-max(lag_end, 0)]
            behavioural_data.index = range(len(behavioural_data))
            self._action_labels[rec_num] = self._extract_action_labels(behavioural_data)

            output = self.processed / recording['action_labels']
            print(f"> Saving action labels to:")
            print(f"    {output}")
            np.save(output, self._action_labels[rec_num])

            output = self.processed / recording['behaviour_processed']
            print(f"> Saving downsampled behavioural data to:")
            print(f"    {output}")
            behavioural_data.drop("/'NpxlSync_Signal'/'0'", axis=1, inplace=True)
            ioutils.write_hdf5(output, behavioural_data)
            self._behavioural_data[rec_num] = behavioural_data

        print("> Done!")

    def sync_data(self, rec_num, behavioural_data=None, sync_channel=None):
        """
        This method will calculate and return the lag between the behavioural data and
        the neuropixels data for each recording.

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

        Returns
        -------
        lag_start : int
            The number of sample points that the behavioural data has extra at the
            start of the recording.

        lag_end : int
            The same as above but at the end.

        """
        print("> Finding lag between sync channels")
        recording = self.files[rec_num]

        if behavioural_data is None:
            print("    Loading behavioural data")
            behavioural_data = ioutils.read_tdms(
                self.find_file(recording['behaviour']), groups=["NpxlSync_Signal"]
            )
            behavioural_data = signal.resample(
                behavioural_data.values, BEHAVIOUR_HZ, self.sample_rate
            )

        if sync_channel is None:
            print("    Loading neuropixels sync channel")
            sync_channel = ioutils.read_bin(
                self.find_file(recording['lfp_data']),
                self.lfp_meta[rec_num]['nSavedChans'],
                channel=384,
            )
            original_samp_rate = int(self.lfp_meta[rec_num]['imSampRate'])
            sync_channel = sync_channel[:120 * original_samp_rate * 2]  # 2 mins, rec Hz, back/forward
            sync_channel = signal.resample(
                sync_channel, original_samp_rate, self.sample_rate
            )

        behavioural_data = signal.binarise(behavioural_data)
        sync_channel = signal.binarise(sync_channel).squeeze()

        print("    Finding lag")
        plot_path = self.processed / recording['lfp_data']
        plot_path = plot_path.with_name(plot_path.stem + '_sync.png')
        lag_start, match = signal.find_sync_lag(
            behavioural_data, sync_channel, length=120000, plot=plot_path,
        )
        if match < 95:
            print("    The sync channels did not match very well. Check the plot.")

        lag_end = len(behavioural_data) - (lag_start + len(sync_channel))
        self._lag[rec_num] = (lag_start, lag_end)
        return lag_start, lag_end

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

    def _get_processed_data(self, attr, key):
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
            for rec_num, recording in enumerate(self.files):
                file_path = self.processed / recording[key]
                if file_path.exists():
                    if file_path.suffix == '.npy':
                        saved[rec_num] = np.load(file_path)
                    elif file_path.suffix == '.h5':
                        saved[rec_num] = ioutils.read_hdf5(file_path)
                else:
                    print(f"Could not find {attr[1:]} for recording {rec_num}.")
                    saved[rec_num] = None
        return saved

    def get_action_labels(self):
        """
        Returns the action labels, either from self._action_labels if they have been
        loaded already, or from file.
        """
        return self._get_processed_data("_action_labels", "action_labels")

    def get_behavioural_data(self):
        """
        Returns the downsampled behaviour channels.
        """
        return self._get_processed_data("_behavioural_data", "behaviour_processed")

    def get_spike_data(self):
        """
        Returns the processed and downsampled spike data.
        """
        return self._get_processed_data("_spike_data", "spike_processed")

    def get_lfp_data(self):
        """
        Returns the processed and downsampled LFP data.
        """
        return self._get_processed_data("_lfp_data", "lfp_processed")

    def process_spikes(self):
        """
        Process the spike data from the raw neural recording data.

        Spike data is processed one channel at a time to not overload memory.
        """
        for rec_num, recording in enumerate(self.files):
            print(f">>>>> Processing spike data for recording {rec_num + 1} of {len(self.files)}")

            orig_rate = self.spike_meta[rec_num]['imSampRate']
            data_file = self.find_file(recording['spike_data'])
            num_chans = int(self.spike_meta[rec_num]['nSavedChans'])

            print("> Mapping spike data")
            data = ioutils.read_bin(data_file, num_chans)

            print(f"> Downsampling to {self.sample_rate} Hz")
            data = signal.resample(data, orig_rate, self.sample_rate)

            if self._lag[rec_num] is not None:
                lag_start, lag_end = self._lag[rec_num]
            else:
                lag_start, lag_end = self.sync_data(rec_num, sync_channel=data[:, -1])

            print(f"> Saving data to {output}")
            output = self.processed / recording['spike_processed']
            raise Exception
            data = data[max(-lag_start, 0):-max(-lag_end, 0)]
            data = pd.DataFrame(data[:, :-1])
            ioutils.write_hdf5(output, data)

    def process_lfp(self):
        """
        Process the LFP data from the raw neural recording data.
        """
        for rec_num, recording in enumerate(self.files):
            print(f">>>>> Processing LFP for recording {rec_num + 1} of {len(self.files)}")

            print("> Mapping LFP data")
            lfp_data = ioutils.read_bin(
                self.find_file(recording['lfp_data']),
                self.lfp_meta[rec_num]['nSavedChans'],
            )

            print(f"> Downsampling to {self.sample_rate} Hz")
            lfp_data = signal.resample(
                lfp_data, self.lfp_meta[rec_num]['imSampRate'], self.sample_rate
            )

            if self._lag[rec_num] is not None:
                lag_start, lag_end = self._lag[rec_num]
            else:
                lag_start, lag_end = self.sync_data(rec_num, sync_channel=lfp_data[:, -1])

            output = self.processed / recording['lfp_processed']
            print(f"> Saving data to {output}")
            lfp_data = lfp_data[max(-lag_start, 0):-1-max(-lag_end, 0)]
            lfp_data = pd.DataFrame(lfp_data[:, :-1])
            ioutils.write_hdf5(output, lfp_data)

    def extract_spikes(self):
        """
        Extract the spikes from raw spike data.
        """

    def extract_videos(self):
        """
        Extract behavioural videos from TDMS to avi.
        """
        for rec_num, recording in enumerate(self.files):
            path = self.find_file(recording['camera_data'])
            path_avi = path.with_suffix('.avi')
            if not path_avi.exists():
                df = ioutils.read_tdms(recording['camera_data'])
                ioutils.save_df_as_avi(df, path_avi)

    def process_motion_tracking(self):
        """
        Process motion tracking data either from raw camera data, or from
        previously-generated deeplabcut coordinate data.
        """

    def align_trials(self, label, event, data):
        """
        Get trials aligned to an event. This finds all instances of label in the action
        labels - these are the start times of the trials. Then this finds the first
        instance of event on or after these start times of each trial. Then this cuts
        out an 8 second period around each of these events covering all cells,
        rearranges this data into a MultiIndex DataFrame and returns it.

        Parameters
        ----------
        label : int
            An action label value to specify which trial types are desired.

        event : int
            An event type value to specify which event to align the trials to.

        data : str
            One of 'behaviour', 'spikes' or 'lfp'.

        """
        print(f"Aligning {data} data to trials.")
        action_labels = self.get_action_labels()

        if data == 'behaviour':
            data = self.get_behavioural_data()
        elif data == 'spikes':
            data = self.get_spike_data()
        elif data == 'lfp':
            data = self.get_lfp_data()
        else:
            raise PixelsError(f"data parameter should be 'behaviour', 'spikes' or 'lfp'")

        if not data or data[0] is None:
            raise PixelsError(f"Data does not appear to have been processed yet.")

        trials = []
        for rec_num in range(len(self.files)):
            actions = action_labels[rec_num][:, 0]
            events = action_labels[rec_num][:, 1]
            trial_starts = np.where((actions == label))[0]
            duration = self.sample_rate * self.trial_duration
            half = duration // 2

            for start in trial_starts:
                centre = start + np.where(events[start:start + duration] == event)[0][0]
                trial = data[rec_num][centre - half + 1:centre + half + 1]
                trials.append(trial.reset_index(drop=True))

        trials = pd.concat(trials, axis=1, copy=False, keys=range(len(trials)), names=["trial", "unit"])
        trials.sort_index(level=1, axis=1, inplace=True)
        trials = trials.reorder_levels(["unit", "trial"], axis=1)
        return trials
