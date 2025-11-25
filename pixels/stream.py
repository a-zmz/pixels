# annotations not evaluated at runtime
from __future__ import annotations

import gc
from shutil import copyfile

import numpy as np
import pandas as pd

import spikeinterface as si

from pixels import ioutils
from pixels import pixels_utils as xut
import pixels.signal_utils as signal
from pixels.configs import *
from pixels.constants import *
from pixels.decorators import cacheable
from pixels.error import PixelsError

from common_utils import file_utils

class Stream:
    def __init__(
        self,
        stream_id,
        stream_num,
        files,
        session,
    ):
        self.stream_id = stream_id
        self.stream_num = stream_num
        self.probe_id = stream_id[:-3]
        self.files = files

        self.session = session
        self.behaviour_files = session.files["behaviour"]
        self.SAMPLE_RATE = session.SAMPLE_RATE
        self.raw = session.raw
        self.interim = session.interim
        self.cache = self.interim / "cache/"
        self.additional_cache = session.additional_interim / "cache/"
        self.processed = session.processed
        self.histology = session.histology

        self._use_cache = True

    def __repr__(self):
        return f"<Stream id = {self.stream_id}>"


    def load_raw_ap(self, copy=False):
        paths = [
            self.session.find_file(path, copy=copy)
            for path in self.files["ap_raw"]
        ]
        self.files["si_rec"] = xut.load_raw(paths, self.stream_id)

        return self.files["si_rec"]


    def _map_trials(self, label, event, end_event=None):
        # get synched pixels stream with vr and action labels
        synched_vr, action_labels = self.get_synched_vr()

        # get action and event label file
        outcomes = action_labels["outcome"]
        events = action_labels["events"]
        # get timestamps index of behaviour in self.SAMPLE_RATE hz, to
        # convert it to ms, do timestamps*1000/self.SAMPLE_RATE
        timestamps = action_labels["timestamps"]

        # select frames of wanted trial type
        trials = np.flatnonzero(outcomes & label)
        # map starts by event
        starts = np.flatnonzero(events & event)
        # map starts by end event
        ends = np.flatnonzero(events & end_event)

        if ("dark" in label.name) and\
            any(name in event.name for name in ["landmark", "wall"]):
            # get dark onset and offset edges
            dark_on = ((events & event.dark_on) != 0).astype(np.int8)
            dark_off = ((events & event.dark_off) != 0).astype(np.int8)
            # make in dark boolean
            in_dark = np.cumsum(dark_on - dark_off) > 0
            # get starts only when also in dark
            starts = np.flatnonzero(((events & event) != 0) & in_dark)

        # only take starts and ends from selected trials
        selected_starts = trials[np.isin(trials, starts)]
        selected_ends = trials[np.isin(trials, ends)]

        # make sure trials have both starts and ends, some trials ended before
        # the end_event, and some dark trials are not in dark at start event 
        start_ids = synched_vr.iloc[selected_starts].trial_count.unique()
        end_ids = synched_vr.iloc[selected_ends].trial_count.unique()
        common_ids = np.intersect1d(start_ids, end_ids)
        if len(start_ids) != len(end_ids):
            selected_starts = selected_starts[np.isin(start_ids, common_ids)]
            selected_ends = selected_ends[np.isin(end_ids, common_ids)]

        # get timestamps
        start_t = timestamps[selected_starts]
        end_t = timestamps[selected_ends]

        # use original trial ids as trial index
        trial_ids = pd.Index(common_ids, dtype=np.int16)

        return trials, events, selected_starts, start_t, end_t, trial_ids


    #@cacheable(cache_format="zarr")
    def _get_vr_positions(self, label, event, end_event):
        logging.info(
            f"\n> Getting {self.session.name} {self.stream_id} positions."
        )

        # map trials
        (trials, events, selected_starts,
         start_t, end_t, trial_ids) = self._map_trials(
            label,
            event,
            end_event,
        )

        if selected_starts.size == 0:
            logging.info(f"\n> No trials found with label {label} and event "
                         f"{event.name}, output will be empty.")
            return None

        # get synched vr
        synched_vr, _ = self.get_synched_vr()

        # get positions of all trials
        all_pos = synched_vr.position_in_tunnel
        all_pos_val = all_pos.to_numpy()
        all_pos_idx = all_pos.index.to_numpy()

        # find start and end position index
        trial_start_t = np.searchsorted(all_pos_idx, start_t, side="left")
        trial_end_t = np.searchsorted(all_pos_idx, end_t, side="right")
        trials_positions = [
            pd.Series(all_pos_val[s:e])
            for s, e in zip(trial_start_t, trial_end_t)
        ]
        positions = pd.concat(trials_positions, axis=1)

        # map actual starting locations
        if not "trial_start" in event.name:
            all_start_idx = np.flatnonzero(events & event.trial_start)
            trial_start_idx = trials[np.isin(trials, all_start_idx)]
            # make sure to only get the included trials' starting positions,
            # i.e., the one with start and end event, not all trials in the
            # label by aligning trial ids
            all_ids = synched_vr.trial_count.iloc[trial_start_idx].values
            start_idx = trial_start_idx[np.isin(all_ids, trial_ids)]
        else:
            start_idx = selected_starts.copy()

        # get start positions
        start_pos = all_pos_val[start_idx].astype(int)
        # create multiindex with starts
        cols_with_starts = pd.MultiIndex.from_arrays(
            [start_pos, trial_ids],
            names=("start", "trial"),
        )

        # add level with start positions
        positions.columns = cols_with_starts
        positions = positions.sort_index(axis=1, ascending=[False, True])
        positions.index.name = "time"

        return positions


    #@cacheable(cache_format="zarr")
    def _get_vr_spikes(self, units, label, event, sigma, end_event):
        logging.info(
            f"\n> Getting {self.session.name} {self.stream_id} spikes."
        )

        # map trials timestamps and index
        (_, _, selected_starts,
         start_t, end_t, trial_ids) = self._map_trials(
            label,
            event,
            end_event,
        )

        if selected_starts.size == 0:
            logging.info(f"\n> No trials found with label {label} and event "
                         f"{event.name}, output will be empty.")
            return None

        # get spike times
        spikes = self.get_spike_times(units)
        units = units[self.stream_id]

        # pad ends with 1 second extra to remove edge effects from
        # convolution
        scan_pad = self.SAMPLE_RATE
        scan_starts = start_t - scan_pad
        scan_ends = end_t + scan_pad + 1
        scan_durations = scan_ends - scan_starts

        cursor = 0
        raw_rec = self.load_raw_ap()
        samples = raw_rec.get_total_samples()
        # Account for multiple raw data files
        in_SAMPLE_RATE_scale = (samples * self.SAMPLE_RATE)\
                / raw_rec.sampling_frequency
        cursor_duration = (cursor * self.SAMPLE_RATE)\
                / raw_rec.sampling_frequency
        rec_spikes = spikes[
            (cursor_duration <= spikes)\
            & (spikes < (cursor_duration + in_SAMPLE_RATE_scale))
        ] - cursor_duration
        cursor += samples

        trials_spiked = {}
        trials_fr = {}
        for i, start in enumerate(selected_starts):
            # select spike times of current trial
            trial_bool = (rec_spikes >= scan_starts[i])\
                    & (rec_spikes <= scan_ends[i])
            trial = rec_spikes[trial_bool]

            # initiate binary spike times array for current trial
            # NOTE: dtype must be float otherwise would get all 0 when passing
            # gaussian kernel
            times = np.zeros((scan_durations[i], len(units)), dtype=np.float32)
            # use pixels time as spike index
            idx = np.arange(scan_starts[i], scan_ends[i])

            for j, unit in enumerate(trial):
                # get spike time for unit
                u_times = trial[unit].values
                # drop nan
                u_times = u_times[~np.isnan(u_times)]
                # round spike times to use it as index
                u_spike_idx = np.round(u_times).astype(int) - scan_starts[i]
                # make sure it does not exceed scan duration
                u_spike_idx = u_spike_idx[
                    (u_spike_idx >= 0) & (u_spike_idx < scan_durations[i])
                ]
                if u_spike_idx.size:
                    # set spiked to 1
                    times[np.unique(u_spike_idx), j] = 1

            # make it df, column name being unit id
            spiked = pd.DataFrame(times, index=idx, columns=units)

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
            trials_fr[trial_ids[i]] = rates
            spiked.reset_index(inplace=True, drop=True)
            trials_spiked[trial_ids[i]] = spiked

        # get trials vertically stacked spiked
        stacked_spiked = pd.concat(
            trials_spiked,
            axis=0,
        )
        stacked_spiked.index.names = ["trial", "time"]
        stacked_spiked.columns.names = ["unit"]
        
        output = {}
        # get trials horizontally stacked spiked
        spiked = ioutils.reindex_by_longest(
            dfs=stacked_spiked,
            level="trial",
            return_format="dataframe",
        )
        fr = ioutils.reindex_by_longest(
            dfs=trials_fr,
            level="trial",
            idx_names=["trial", "time"],
            col_names=["unit"],
            return_format="dataframe",
        )

        output["spiked"] = spiked
        output["fr"] = fr

        return stacked_spiked, output


    # TODO nov 24 2025: consider to cache this in zarr cuz it's quite big
    @cacheable#(cache_format="zarr")
    def align_trials(self, units, data, label, event, sigma, end_event):
        """
        Align pixels data to behaviour trials.

        params
        ===
        units : dictionary of lists of ints
            The output from self.select_units, used to only apply this method to a
            selection of units.

        data : str, optional
            The data type to align.

        label : int
            An action label value to specify which trial types are desired.

        event : int
            An event type value to specify which event to align the trials to.

        sigma : int, optional
            Time in milliseconds of sigma of gaussian kernel to use when
            aligning firing rates.

        end_event : int | None
            For VR behaviour, when aligning to the whole trial, this param is
            the end event to align to.

        return
        ===
        df, output from individual functions according to data type.
        """

        if "spike_trial" in data:
            logging.info(
                f"\n> Aligning spike times and spike rate of {units} units to "
                f"<{label.name}> trials, from {event.name} to {end_event.name}."
            )
            return self._get_aligned_trials(
                label, event, units=units, sigma=sigma, end_event=end_event,
            )
        elif "spike_event" in data:
            logging.info(
                f"\n> Aligning spike times and spike rate of {units} units to "
                f"{event.name} event in <{label.name}> trials."
            )
            return self._get_aligned_events(
                label, event, units=units, sigma=sigma,
            )
        else:
            raise NotImplementedError(
                "> Other types of alignment are not implemented."
            )


    def _get_aligned_trials(
        self, label, event, units=None, sigma=None, end_event=None,
    ):
        # get positions
        positions = self._get_vr_positions(label, event, end_event)

        # get spikes and firing rate
        _, output = self._get_vr_spikes(
            units,
            label,
            event,
            sigma,
            end_event,
        )
        output["positions"] = positions

        return output


    def _get_aligned_events(self, label, event, units=None, sigma=None):
        # TODO oct 17 2025:
        # use _get_vr_spikes??
        # get spikes and firing rate
        _, output = self._get_vr_spikes(
            units,
            label,
            event,
            sigma,
            end_event,
        )

        # get synched pixels stream with vr and action labels
        synched_vr, action_labels = self.get_synched_vr()

        # get positions of all trials
        all_pos = synched_vr.position_in_tunnel

        # get spike times
        spikes = self.get_spike_times(units)
        # now get array unit ids
        units = units[self.stream_id]

        # get action and event label file
        outcomes = action_labels["outcome"]
        events = action_labels["events"]
        # get timestamps index of behaviour in self.SAMPLE_RATE hz, to
        # convert it to ms, do timestamps*1000/self.SAMPLE_RATE
        timestamps = action_labels["timestamps"]

        # select frames of wanted trial type
        trials = np.where(np.bitwise_and(outcomes, label))[0]
        # map starts by event
        starts = np.where(np.bitwise_and(events, event))[0]

        # only take starts from selected trials
        selected_starts = trials[np.where(np.isin(trials, starts))[0]]
        start_t = timestamps[selected_starts]

        if selected_starts.size == 0:
            logging.info(f"\n> No trials found with label {label} and event "
                         f"{event.name}, output will be empty.")
            return None

        # use original trial id as trial index
        trial_ids = pd.Index(
            synched_vr.iloc[selected_starts].trial_count.unique()
        )

        # TODO aug 1 2025:
        # lick happens more than once in a trial, thus i here does not
        # correspond to trial index, fit it
        # check if event happens more than once in each trial
        if start_t.size > trial_ids.size:
            trial_counts = synched_vr.loc[start_t, "trial_count"]

        # map actual starting locations
        if not "trial_start" in event.name:
            all_start_idx = np.where(
                np.bitwise_and(events, event.trial_start)
            )[0]
            start_idx = trials[np.where(
                np.isin(trials, all_start_idx)
            )[0]]
        else:
            start_idx = selected_starts.copy()

        start_pos = synched_vr.position_in_tunnel.iloc[
            start_idx
        ].values.astype(int)

        # map starting position with trial
        start_trial_maps = dict(zip(trial_ids, start_pos))

        # pad ends with 1 second extra to remove edge effects from convolution,
        # during of event is 2s (pre + post)
        duration = 1
        pad_duration = 1
        scan_pad = self.SAMPLE_RATE
        one_side_frames = scan_pad * (duration + pad_duration)
        scan_starts = start_t - one_side_frames
        scan_ends = start_t + one_side_frames + 1
        scan_duration = one_side_frames * 2 + 1
        relative_idx = np.linspace(
            -(duration+pad_duration),
            (duration+pad_duration),
            scan_duration,
        )

        cursor = 0
        raw_rec = self.load_raw_ap()
        samples = raw_rec.get_total_samples()
        # Account for multiple raw data files
        in_SAMPLE_RATE_scale = (samples * self.SAMPLE_RATE)\
                / raw_rec.sampling_frequency
        cursor_duration = (cursor * self.SAMPLE_RATE)\
                / raw_rec.sampling_frequency
        rec_spikes = spikes[
            (cursor_duration <= spikes)\
            & (spikes < (cursor_duration + in_SAMPLE_RATE_scale))
        ] - cursor_duration
        cursor += samples

        output = {}
        trials_fr = {}
        trials_spiked = {}
        trials_positions = {}
        for i, start in enumerate(selected_starts):
            assert 0
            # TODO sep 16 2025:
            # make sure it does not fail for event that happens more than once,
            # e.g., licks
            # select spike times of event in current trial
            trial_bool = (rec_spikes >= scan_starts[i])\
                    & (rec_spikes <= scan_ends[i])
            trial = rec_spikes[trial_bool]

            # initiate binary spike times array for current trial
            # NOTE: dtype must be float otherwise would get all 0 when passing
            # gaussian kernel
            times = np.zeros((scan_duration, len(units))).astype(float)
            # use pixels time as spike index
            idx = np.arange(scan_starts[i], scan_ends[i])
            # make it df, column name being unit id
            spiked = pd.DataFrame(times, index=idx, columns=units)

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

            # set spiked index to relative index
            spiked.index = relative_idx
            # convolve spike trains into spike rates
            rates = signal.convolve_spike_trains(
                times=spiked,
                sigma=sigma,
                sample_rate=self.SAMPLE_RATE,
            )

            # remove 1s padding from the start and end
            rates = rates.iloc[scan_pad: -scan_pad]
            spiked = spiked.iloc[scan_pad: -scan_pad]

            trials_fr[trial_ids[i]] = rates
            trials_spiked[trial_ids[i]] = spiked

        # get trials vertically stacked spiked
        stacked_spiked = pd.concat(
            trials_spiked,
            axis=0,
        )
        stacked_spiked.index.names = ["trial", "time"]
        stacked_spiked.columns.names = ["unit"]

        # get trials horizontally stacked spiked
        spiked = ioutils.reindex_by_longest(
            dfs=stacked_spiked,
            level="trial",
            return_format="dataframe",
        )
        trial_cols = spiked.columns.get_level_values("trial")
        start_cols = trial_cols.map(start_trial_maps)
        unit_cols = spiked.columns.get_level_values("unit")
        new_cols = pd.MultiIndex.from_arrays(
            [unit_cols, start_cols, trial_cols],
            names=["unit", "start", "trial"]
        )
        spiked.columns = new_cols
        spiked = spiked.sort_index(
            axis=1,
            level=["unit", "start", "trial"],
            ascending=[True, False, True],
        )

        fr = ioutils.reindex_by_longest(
            dfs=trials_fr,
            level="trial",
            idx_names=["trial", "time"],
            col_names=["unit"],
            return_format="dataframe",
        )
        fr.columns = new_cols
        fr = fr.loc[:, spiked.columns]

        output["spiked"] = spiked
        output["fr"] = fr

        return output


    def get_spike_times(self, units):
        units = units[self.stream_id]

        # find sorting analyser, use merged if there is one
        merged_sa_path = self.session.find_file(
            self.files["merged_sorting_analyser"]
        )
        if merged_sa_path:
            sa_path = merged_sa_path
        else:
            sa_path = self.session.find_file(self.files["sorting_analyser"])
        # load sorting analyser
        temp_sa = si.load_sorting_analyzer(sa_path)
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
        spike_times = pd.concat(
            objs=times,
            axis=1,
            names="unit",
        )
        # get sampling frequency
        fs = int(sa.sampling_frequency)
        # Convert to time into sample rate index
        spike_times /= fs / self.SAMPLE_RATE

        return spike_times


    def sync_vr(self, vr_session):
        # get action labels & synched vr path
        action_labels = self.session.get_action_labels()[self.stream_num]
        synched_vr_path = self.session.find_file(
            self.behaviour_files["vr_synched"][self.stream_num],
        )
        if action_labels and synched_vr_path:
            logging.info(f"\n> {self.stream_id} from {self.session.name} is "
                         "already synched with vr.")
        else:
            self._sync_vr(vr_session)

        return None


    def _sync_vr(self, vr_session):
        # get synchronised vr path
        synched_vr_path = vr_session.cache_dir + "synched/" +\
                    vr_session.name + "_vr_synched.h5"

        try:
            synched_vr = file_utils.read_hdf5(synched_vr_path)
            logging.info("\n> synchronised vr loaded")
        except:
            # get spike data
            spike_data = self.session.find_file(
                name=self.files["ap_raw"][self.stream_num],
                copy=True,
            )

            # get sync pulses
            sync_map = ioutils.read_bin(spike_data, 385, 384)
            syncs = signal.binarise(sync_map)

            # >>>> resample pixels sync pulse to sample rate >>>>
            # get ap data sampling rate
            spike_samp_rate = int(self.session.ap_meta[0]['imSampRate'])
            # downsample pixels sync pulse
            downsampled = signal.decimate(
                array=syncs,
                from_hz=spike_samp_rate,
                to_hz=self.SAMPLE_RATE,
            )
            # binarise to avoid non integers
            pixels_syncs = signal.binarise(downsampled)
            # <<<< resample pixels sync pulse to 1kHz <<<<

            # TODO apr 11 2025:
            # for 20250723 VDCN09, sync pulses are weird. check number of syncs
            # from 145s onwards, see if it matches with vr frames from 100s
            # onwards

            # get the rise and fall edges in pixels sync
            pixels_edges = np.where(np.diff(pixels_syncs) != 0)[0] + 1
            # double check if the pulse from arduino initiation is also
            # included, if so there will be two long pulses before vr frames
            first_pulses = np.diff(pixels_edges)[:4]
            if (first_pulses > 1000).all():
                logging.info("\n> There are two long pulses before vr frames, "
                                "remove both.")
                remove = 4
            else:
                remove = 2
            pixels_vr_edges = pixels_edges[remove:]
            # convert value into their index to calculate all timestamps
            pixels_idx = np.arange(pixels_syncs.shape[0])

            synched_vr = vr_session.sync_streams(
                self.SAMPLE_RATE,
                pixels_vr_edges,
                pixels_idx,
            )

        synched_vr_file = self.behaviour_files["vr_synched"][self.stream_num]
        try:
            assert self.session.find_file(synched_vr_file)
        except:
            file_utils.write_hdf5(
                self.processed / synched_vr_file,
                synched_vr,
            )

        # get action label dir
        action_labels_path = self.processed /\
            self.behaviour_files["action_labels"][self.stream_num]

        # extract and save action labels
        action_labels = self.session._extract_action_labels(
            vr_session,
            synched_vr,
        )
        labels_dict = action_labels._asdict()
        np.savez_compressed(
            action_labels_path,
            **labels_dict,
        )
        logging.info(f"\n> Action labels saved to: {action_labels_path}.")

        if hasattr(self.session, "backup"):
            # copy to backup if backup setup
            copyfile(
                self.processed / synched_vr_file,
                self.session.backup / synched_vr_file,
            )
            copyfile(
                action_labels_path,
                self.session.backup / action_labels_path.name,
            )
            logging.info(
                "\n> Syched vr and action labels copied to: "
                f"{self.session.backup}."
            )

        return None


    def get_synched_vr(self):
        """
        Get synchronised vr data and action labels.
        """
        action_labels = self.session.get_action_labels()[self.stream_num]

        synched_vr_path = self.session.find_file(
            self.behaviour_files["vr_synched"][self.stream_num],
        )
        synched_vr = file_utils.read_hdf5(synched_vr_path)

        return synched_vr, action_labels

    
    @cacheable
    def get_binned_trials(
        self, label, event, units=None, sigma=None, end_event=None,
        time_bin=None, pos_bin=None
    ):
        # define output path for binned spike rate
        file_name = f"{self.session.name}_{units}_{label.name}_"\
            f"{time_bin}_{pos_bin}cm_{self.stream_id}.npz"
        output_path = self.cache / file_name

        binned = self._bin_aligned_trials(
            label=label,
            event=event,
            units=units,
            sigma=sigma,
            end_event=end_event,
            time_bin=time_bin,
            pos_bin=pos_bin,
            output_path=output_path,
        )

        if hasattr(self.session, "backup"):
            # copy to backup if backup setup
            copyfile(
                output_path,
                self.session.backup / file_name,
            )

        return binned


    def _bin_aligned_trials(
        self, label, event, units, sigma, end_event, time_bin, pos_bin,
        output_path,
    ):
        # get aligned trials
        trials = self.align_trials(
            units=units, # NOTE: ALWAYS the first arg
            data="spike_trial", # NOTE: ALWAYS the second arg
            label=label,
            event=event,
            sigma=sigma,
            end_event=end_event, # NOTE: ALWAYS the last arg
        )

        if trials is None:
            logging.info(f"\n> No trials found with label {label.name} and "
                         f"event {event.name}, output will be empty.")
            return None

        logging.info(
            f"\n> Binning <{label.name}> trials from {self.stream_id} "
            f"in {units}."
        )

        # get fr, spiked, positions
        fr = trials["fr"]
        spiked = trials["spiked"]
        positions = trials["positions"]

        bin_arr = {}
        binned_count = {}
        binned_fr = {}

        trial_ids = positions.columns.get_level_values("trial").unique()
        for trial in trial_ids:
            counts = spiked.xs(trial, level="trial", axis=1).dropna()
            rates = fr.xs(trial, level="trial", axis=1).dropna()
            trial_pos = positions.xs(trial, level="trial", axis=1).dropna()

            # get bin spike count
            binned_count[trial] = xut.bin_vr_trial(
                data=counts,
                positions=trial_pos,
                sample_rate=self.SAMPLE_RATE,
                time_bin=time_bin,
                pos_bin=pos_bin,
                bin_method="sum",
            )
            # get bin firing rates
            binned_fr[trial] = xut.bin_vr_trial(
                data=rates,
                positions=trial_pos,
                sample_rate=self.SAMPLE_RATE,
                time_bin=time_bin,
                pos_bin=pos_bin,
                bin_method="mean",
            )

        # stack df values into np array
        # reshape into trials x units x bins
        count_arr = ioutils.reindex_by_longest(binned_count).T
        fr_arr = ioutils.reindex_by_longest(binned_fr).T

        # save bin_fr and bin_count, for andrew
        # use label as array key name
        bin_arr["count"] = count_arr[:, :-2, :]
        bin_arr["fr"] = fr_arr[:, :-2, :]
        bin_arr["pos"] = count_arr[:, -2:, :]

        np.savez_compressed(output_path, **bin_arr)
        logging.info(f"\n> Output saved at {output_path}.")

        # extract binned data in df format
        bin_fc, bin_pos = self._extract_binned_data(
            binned_count,
            positions.columns,
        )
        bin_fr, _ = self._extract_binned_data(
            binned_fr,
            positions.columns,
        )

        # convert it to binned positional data
        pos_data = xut.get_vr_positional_data(
            {
                "positions": bin_pos.bin_pos,
                "fr": bin_fr,
                "spiked": bin_fc,
            },
        )

        return pos_data


    def _extract_binned_data(self, binned_data, pos_cols):
        """
        """
        df = ioutils.reindex_by_longest(
            dfs=binned_data,
            idx_names=["trial", "time_bin"],
            col_names=["unit"],
            return_format="dataframe",
        )
        data = df.drop(
            labels=["positions", "bin_pos"],
            axis=1,
            level="unit",
        )
        pos = df.filter(
            like="pos",
            axis=1,
        )
        pos.columns.names = ["pos_type", "trial"]

        # convert columns to df
        pos_col_df = pos.columns.to_frame(index=False)
        start_trial_df = pos_cols.to_frame(index=False)
        # merge columns
        merged_cols = pd.merge(pos_col_df, start_trial_df, on="trial")

        # create new columns
        new_cols = pd.MultiIndex.from_frame(
            merged_cols[["pos_type", "start", "trial"]],
            names=["pos_type", "start", "trial"],
        )
        pos.columns = new_cols

        return data, pos


    #@cacheable
    def get_positional_data(
        self, label, event, end_event=None, sigma=None, units=None,
        normalised=False,
    ):
        """
        Get positional firing rate of selected units in vr, and spatial
        occupancy of each position.
        """
        # NOTE: order of args matters for loading the cache!
        # always put units first, cuz it is like that in
        # experiemnt.align_trials, otherwise the same cache cannot be loaded

        # get aligned firing rates and positions
        trials = self.align_trials(
            units=units, # NOTE: ALWAYS the first arg
            data="spike_trial", # NOTE: ALWAYS the second arg
            label=label,
            event=event,
            sigma=sigma,
            end_event=end_event, # NOTE: ALWAYS the last arg
        )

        if normalised:
            grays = self.align_trials(
                units=units, # NOTE: ALWAYS the first arg
                data="spike_trial", # NOTE: ALWAYS the second arg
                label=getattr(label, label.name.split("_")[-1]),
                event=event.gray_on,
                sigma=sigma,
                end_event=end_event.gray_off, # NOTE: ALWAYS the last arg
            )

            # NOTE july 24 2025: if get gray mu & sigma per trial we got z score
            # of very quiet & stable in gray units >9e+15... thus, we normalise
            # average all trials for each unit, rather than per trial

            # NOTE: 500ms of 500Hz sine wave sound at each trial start, 2000ms
            # of gray, so only take the second 1000ms in gray to get mean and
            # std

            # only select trials exists in aligned trials
            baseline = grays["fr"].iloc[
                self.SAMPLE_RATE: self.SAMPLE_RATE * 2
            ].loc[:, trials["fr"].columns].T.groupby("unit").mean().T

            mu = baseline.mean()
            centered = trials["fr"].sub(mu, axis=1, level="unit")
            std = baseline.std()
            z_fr = centered.div(std, axis=1, level="unit")
            trials["fr"] = z_fr

            del grays, baseline, mu, std, z_fr
            gc.collect()

        # get positional spike rate, spike count, and occupancy
        positional_data = xut.get_vr_positional_data(trials)

        return positional_data


    def preprocess_raw(self):
        # load raw ap
        raw_rec = self.load_raw_ap(copy=True)

        # load brain surface depths
        depth_info = file_utils.load_yaml(
            path=self.histology / self.files["depth_info"],
        )
        surface_depths = depth_info["raw_signal_depths"][self.stream_id]

        # find faulty channels to remove
        faulty_channels = file_utils.load_yaml(
            path=self.processed / self.files["faulty_channels"],
        )

        # preprocess
        self.files["preprocessed"] = xut.preprocess_raw(
            raw_rec,
            surface_depths,
            faulty_channels,
        )

        return None


    def extract_bands(self, freqs, preprocess=True):
        if preprocess:
            self.preprocess_raw()
            rec = self.files["preprocessed"]
        else:
            rec = self.load_raw_ap(copy=True)

        if freqs == None:
            bands = freq_bands
        elif isinstance(freqs, str) and freqs in freq_bands.keys():
            bands = {freqs: freq_bands[freqs]}
        elif isinstance(freqs, dict):
            bands = freqs

        for name, freqs in bands.items():
            logging.info(
                f"\n> Extracting {name} bands from {self.stream_id}."
            )
            # do bandpass filtering
            extracted = xut.extract_band(
                rec,
                freq_min=freqs[0],
                freq_max=freqs[1],
            )

            logging.info(
                f"\n> Common average referencing {name} band."
            )
            self.files[f"{name}_extracted"] = xut.CAR(extracted)

        return None


    def correct_ap_motion(self, output, mc_method):
        # get ap band
        self.extract_bands("ap")
        ap_rec = self.files["ap_extracted"]

        # correct ap motion
        xut.correct_ap_motion(ap_rec, output, mc_method)
        # load motion corrected ap cuz otherwise it is not an si extractor and
        # it will not pass json serialisation test
        self.files["ap_motion_corrected"] = si.load(output)

        return None


    def correct_lfp_motion(self):
        raise NotImplementedError("> Not implemented.")


    def whiten_ap(self):
        # get motion corrected ap
        mcd = self.files["ap_motion_corrected"]

        # whiten
        self.files["ap_whitened"] = xut.whiten(mcd)

        return None


    def sort_spikes(self, ks_mc, ks4_params, ks_image_path, output, sa_dir):
        """
        Sort spikes of stream.

        params
        ===
        ks_mc: bool, whether using kilosort 4 innate motion correction.

        ks4_params: dict, kilosort 4 parameters.

        output: path, directory to save sorting output.

        sa_dir: path, directory to save sorting analyser.

        return
        ===
        None
        """
        # use only preprocessed if use ks motion correction
        if ks_mc:
            self.preprocess_raw()
            rec = self.files["preprocessed"]
            sa_rec = self.files["ap_motion_corrected"]
        else:
            # XXX: as of may 2025, whiten ap band and feed to ks reduce units!
            #rec = self.files["ap_whitened"]
            # use non-whitened recording for ks4 and sorting analyser
            sa_rec = rec = self.files["ap_motion_corrected"]

        # sort spikes and save sorting analyser to disk
        xut.sort_spikes(
            rec=rec,
            sa_rec=sa_rec,
            output=output,
            curated_sa_dir=sa_dir,
            ks_image_path=ks_image_path,
            ks4_params=ks4_params,
        )

        return None


    def get_spatial_psd(
        self, label, event, end_event=None, sigma=None, units=None,
        crop_from=None, use_binned=False, time_bin=None, pos_bin=None,
    ):
        """
        Get spatial power spectral density of selected units.
        """
        # NOTE: jun 19 2025
        # potentially we could use aligned trials directly for psd estimation,
        # with trial position as x, fr as y, and use lomb-scargle method

        # NOTE: order of args matters for loading the cache!
        # always put units first, cuz it is like that in
        # experiemnt.align_trials, otherwise the same cache cannot be loaded

        # get aligned firing rates and positions
        if not use_binned:
            trials = self.get_positional_data(
                units=units, # NOTE: ALWAYS the first arg
                label=label,
                event=event,
                sigma=sigma,
                end_event=end_event, # NOTE: ALWAYS the last arg
            )
            crop_from = crop_from
        else:
            trials = self.get_binned_trials(
                units=units, # NOTE: ALWAYS the first arg
                label=label,
                event=event,
                sigma=sigma,
                end_event=end_event,
                time_bin=time_bin,
                pos_bin=pos_bin,
            )
            crop_from = crop_from // pos_bin + 1

        # get positional fr
        pos_fr = trials["pos_fr"]

        starts = pos_fr.columns.get_level_values("start").unique()
        psds = {}
        for s, start in enumerate(starts):
            data = pos_fr.xs(start, level="start", axis=1)
            # crop if needed
            cropped = data.loc[crop_from:, :]

            # get power spectral density
            psds[start] = xut.get_psd(cropped)

        psd_df = pd.concat(
            psds,
            names=["start","frequency"],
        )
        # NOTE: all trials will appear in all starts, but their values will be
        # all nan in other starts, so remember to dropna(axis=1)!

        return psd_df


    @cacheable(
        cache_format="zarr",
        cache_dir=lambda self: self.additional_cache,
    )
    def get_spike_chance(self, units, label, event, sigma, end_event,
        # reserved kwargs injected by decorator when cache_format='zarr'
        _zarr_out: Path | str | None = None,
    ):
        # get positions
        positions = self._get_vr_positions(label, event, end_event)

        # get spikes
        stacked_spikes, _ = self._get_vr_spikes(
            units,
            label,
            event,
            sigma,
            end_event,
        )

        logging.info(
            f"\n> Getting {self.session.name} {self.stream_id} spike chance."
        )

        xut.save_spike_chance_zarr(
            zarr_path=_zarr_out,
            spiked=stacked_spikes,
            sigma=sigma,
            sample_rate=self.SAMPLE_RATE,
            repeats=REPEATS,
            positions=positions,
            meta=dict(
                label=str(label),
                event=str(event),
                end_event=str(end_event),
                units_name=getattr(units, "name", None),
            ),
        )
        del stacked_spikes, positions
        gc.collect()
        
        return None


    @cacheable(cache_format="zarr")
    def get_binned_chance(
        self, units, label, event, sigma, end_event, time_bin, pos_bin,
    ):
        # NOTE: we only need the condition name
        label_name = label.name.split("_")[-1]

        # get array name and path 
        name_parts = [self.session.name, self.probe_id, label.name, units.name,
        "shuffled", time_bin, f"{pos_bin}cm.npz"]
        file_name = "_".join(p for p in name_parts)
        arr_path = self.processed / file_name

        # get chance data
        chance_data = self.get_spike_chance(
            units=self.session.select_units(name="all"),
            label=getattr(label, label_name),
            event=event.trial_start,
            sigma=sigma,
            end_event=end_event.trial_end,
        )
        # bin chance data
        binned_chance = xut.bin_spike_chance(
            chance_data,
            self.SAMPLE_RATE,
            time_bin,
            pos_bin,
            arr_path,
        )

        if hasattr(self.session, "backup"):
            # copy to backup if backup setup
            copyfile(
                arr_path,
                self.session.backup / file_name,
            )

        return binned_chance


    @cacheable
    def get_chance_positional_psd(self, units, label, event, sigma, end_event):
        # NOTE: we only use completed trials for this analysis
        label_name = label.name.split("_")[-1]
        # get timestamps and trial ids of all trials of current label
        _, _, _, trial_start_t, _, all_trial_ids = self._map_trials(
            label,
            event.trial_start,
            end_event.trial_end,
        )

        # get trial ids of target events
        _, _, _, start_t, end_t, trial_ids = self._map_trials(
            label,
            event,
            end_event,
        )

        # get chance data
        chance_data = self.get_spike_chance(
            units=self.session.select_units(name="all"),
            label=getattr(label, label_name),
            event=event.trial_start,
            sigma=sigma,
            end_event=end_event.trial_end,
        )

        # get trial start and end timestamps
        target_trial_start_t = trial_start_t[all_trial_ids.isin(trial_ids)]
        # get trial start and end time in reference of each trial
        event_on_t = start_t - target_trial_start_t
        event_off_t = end_t - target_trial_start_t

        logging.info("\n> getting chance psd")
        psds = xut.save_chance_psd(
            chance_data,
            self.SAMPLE_RATE,
            units,
            trial_ids,
            event_on_t,
            event_off_t,
        )

        return psds


    @cacheable(cache_format="zarr")
    def get_landmark_responsives(self, units, label, sigma):

        from vision_in_darkness.constants import landmarks, mid_walls
        from pixels.behaviours.virtual_reality import Events

        wall_events = np.array([
            value for key, value in Events.__dict__.items()
            if not (key.startswith("__") or key.startswith("_")
                or callable(value)
                or isinstance(value, (classmethod, staticmethod)))
                and "wall" in key],
            dtype=object,
        ).reshape(-1, 2)

        # get start and end events, excluding the last landmark
        start_events = wall_events[:, 0][:-1]
        end_events = wall_events[:, 1][1:]

        # get on & off of landmark and walls and stack them
        landmark_names = np.arange(1, len(start_events)+1)
        lms = landmarks[1:-2].reshape((-1, 2))
        walls = mid_walls.reshape((-1, 2))
        pre_walls = walls[:-1, :]
        post_walls = walls[1:, :]

        # group pre wall, landmarks, and post wall together
        chunks = np.stack([pre_walls, lms, post_walls], axis=1)

        ons = chunks[..., 0]
        offs = chunks[..., 1]

        def _get_results(fr_fn):
            resps = {}
            contrasts = {}
            for l, lm in enumerate(landmark_names):
                fr = fr_fn(
                    units=units,
                    label=label,
                    event=start_events[l],
                    sigma=sigma,
                    end_event=end_events[l],
                )["pos_fr"]
                contrasts[lm], resps[lm] = xut.get_landmark_responsives(
                    pos_fr=fr,
                    units=units,
                    ons=ons[l, :],
                    offs=offs[l, :],
                )

            responsives = pd.concat(resps, axis=1, names=["landmark"])
            contrasts = pd.concat(
                contrasts,
                axis=0,
                names=["landmark", "unit"],
            )
            contrasts.columns.name = "metrics"
            # if not already int
            contrasts.start = contrasts.start.astype(int)
            # so that row index is unique
            contrasts = contrasts.set_index(
                ["start", "contrast"],
                append=True,
            )
            return responsives, contrasts

        # landmark responsives in data
        responsives, contrasts = _get_results(self.get_positional_data)
        # landmark responsives in chance
        chance_responsives, chance_contrasts = _get_results(
            self.get_chance_positional_data
        )

        return {"contrasts": contrasts,
                "responsives": responsives,
                "chance_contrasts": chance_contrasts,
                "chance_responsives": chance_responsives}


    @cacheable
    def get_chance_positional_data(
        self, units, label, event, sigma, end_event, #normalised,
    ):
        # NOTE: we only use completed trials for this analysis
        label_name = label.name.split("_")[-1]
        # get timestamps and trial ids of all trials of current label
        _, _, _, trial_start_t, _, all_trial_ids = self._map_trials(
            label,
            event.trial_start,
            end_event.trial_end,
        )

        # get trial ids of target events
        _, _, _, start_t, end_t, trial_ids = self._map_trials(
            label,
            event,
            end_event,
        )

        # get chance data
        chance_data = self.get_spike_chance(
            units=self.session.select_units(name="all"),
            label=getattr(label, label_name),
            event=event.trial_start,
            sigma=sigma,
            end_event=end_event.trial_end,
        )

        # get trial start and end timestamps
        target_trial_start_t = trial_start_t[all_trial_ids.isin(trial_ids)]
        # get trial start and end time in reference of each trial
        event_on_t = start_t - target_trial_start_t
        event_off_t = end_t - target_trial_start_t

        pos_data = xut.save_chance_positional_data(
            chance_data,
            self.SAMPLE_RATE,
            units,
            trial_ids,
            event_on_t,
            event_off_t,
        )

        return pos_data

    
    @cacheable(cache_format="zarr")
    def align_to_fixed_grid(
        self, label, event, units=None, sigma=None, end_event=None,
        grid_size=100,
    ):

        # define output path for binned spike rate
        npz_name = f"{self.session.name}_{units}_{label.name}_"\
            f"{grid_size}_samples_{self.stream_id}.npz"
        npz_path = self.cache / npz_name

        aligned = self._align_to_fixed_grid(
            label=label,
            event=event,
            units=units,
            sigma=sigma,
            end_event=end_event,
            grid_size=grid_size,
            npz_path=npz_path,
        )

        if hasattr(self.session, "backup"):
            # copy to backup if backup setup
            copyfile(
                npz_path,
                self.session.backup / npz_name,
            )

        return aligned


    def _align_to_relative_positions(
        self, label, event, units, sigma, end_event, grid_size, npz_path,
    ):
        # create grid index
        grid_idx = np.linspace(0, 1, grid_size)

        trials = self.align_trials(
            units=units, # NOTE: ALWAYS the first arg
            data="spike_trial", # NOTE: ALWAYS the second arg
            label=label,
            event=event,
            sigma=sigma,
            end_event=end_event, # NOTE: ALWAYS the last arg
        )
        # NOTE nov 19 2025
        # we might need to exclude some very short trials, but for now, we use
        # all lengths

        # interpolate to grid
        interpolated = xut.interpolate_to_grid(trials, grid_size, npz_path)

        return interpolated
