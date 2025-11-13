"""
This module provides an Experiment class which serves as the main interface to process
data and run subsequent analyses.
"""


from itertools import islice
from operator import attrgetter, itemgetter
from pathlib import Path

import pandas as pd

from pixels import ioutils
from pixels.error import PixelsError
from pixels.configs import *


class Experiment:
    """
    This represents an experiment and can be used to process or analyse data for a
    group of mice.

    Parameters
    ----------
    mouse_ids : list of strs
        List of IDs of the mice to be included in this experiment.

    behaviour : class
        Class definition subclassing from pixels.behaviours.Behaviour.

    data_dir : str
        Path to the top-level folder containing data for these mice. This folder
        should contain these folders: raw, interim, processed

    meta_dir : str
        Path to the folder containing training metadata JSON files.

    interim_dir : str
        Path to the folder to use for interim files. If not passed, this will default to
        a folder inside the data_dir called 'interim'.

    session_date_fmt : str
        A format string used to parse the date from the name of session folders. By
        default this is "%y%m%d" which will capture YYMMDD formats.

    """
    def __init__(
        self,
        mouse_ids,
        behaviour,
        data_dir,
        meta_dir=None,
        interim_dir=None,
        processed_dir=None,
        hist_dir=None,
        additional_interim_dir=None,
        session_date_fmt="%y%m%d",
        of_date=None,
    ):
        if not isinstance(mouse_ids, (list, tuple, set)):
            mouse_ids = [mouse_ids]

        self.behaviour = behaviour
        self.mouse_ids = mouse_ids

        self.data_dir = Path(data_dir).expanduser()
        if not self.data_dir.exists():
            raise PixelsError(f"Directory not found: {data_dir}")

        if meta_dir:
            self.meta_dir = Path(meta_dir).expanduser()
            if not self.meta_dir.exists():
                raise PixelsError(f"Directory not found: {meta_dir}")
        else:
            self.meta_dir = None

        if hist_dir:
            self.hist_dir = Path(hist_dir).expanduser()

        self.sessions = []
        sessions = ioutils.get_sessions(
            mouse_ids,
            self.data_dir,
            self.meta_dir,
            session_date_fmt,
            of_date,
        )

        for name, metadata in sessions.items():
            assert len(set(s['data_dir'] for s in metadata)) == 1,\
            "All JSON items with same day must use same data folder."
            self.sessions.append(
                behaviour(
                    name,
                    metadata=[s['metadata'] for s in metadata],
                    data_dir=metadata[0]['data_dir'],
                    interim_dir=interim_dir,
                    processed_dir=processed_dir,
                    hist_dir=hist_dir,
                    additional_interim_dir=additional_interim_dir,
                )
            )

        self.sessions.sort(key=attrgetter("name"))

    def __getitem__(self, index):
        """
        Allow indexing directly of sessions with myexp[X].
        """
        return self.sessions[index]

    def __len__(self):
        """
        Length of experiment is the number of sessions.
        """
        return len(self.sessions)

    def __repr__(self):
        rep = "Experiment with sessions:"
        for session in self.sessions:
            rep += "\n\t" + session.name
        return rep

    def set_cache(self, on):
        for session in self.sessions:
            session.set_cache(on)

    def extract_ap(self):
        """
        Process the ap band from the raw neural recording data for all sessions.
        """
        for i, session in enumerate(self.sessions):
            print(">>>>> Processing ap band for session {} ({} / {})"
                   .format(session.name, i + 1, len(self.sessions)))
            session.extract_ap()

    def sort_spikes(self, mc_method="dredge"):
        """ Extract the spikes from raw spike data for all sessions.  """
        for i, session in enumerate(self.sessions):
            logging.info(
                "\n>>>>> Sorting spikes for session "
                f"{session.name} ({i + 1} / {len(self.sessions)})"
            )
            session.sort_spikes(mc_method=mc_method)

    def assess_noise(self):
        """
        Assess the noise for the raw AP data.
        """
        for i, session in enumerate(self.sessions):
            print(">>>>> Assessing noise for session {} ({} / {})"
                   .format(session.name, i + 1, len(self.sessions)))
            session.assess_noise()

    def extract_bands(self):
        """
        Extract ap & lfp data from the raw neural recording data for all
        sessions.
        """
        for i, session in enumerate(self.sessions):
            print(">>>>> Extracting ap & lfp data for session {} ({} / {})"
                   .format(session.name, i + 1, len(self.sessions)))
            session.extract_bands()

    def process_behaviour(self):
        """
        Process behavioural data from raw tdms files for all sessions.
        """
        for i, session in enumerate(self.sessions):
            print(">>>>> Processing behaviour for session {} ({} / {})"
                   .format(session.name, i + 1, len(self.sessions)))
            session.process_behaviour()

    def extract_videos(self, force=False):
        """
        Extract videos from TDMS in the raw folder to avi files in the interim folder.
        """
        for i, session in enumerate(self.sessions):
            print(">>>>> Extracting videos for session {} ({} / {})"
                   .format(session.name, i + 1, len(self.sessions)))
            session.extract_videos(force=force)

    def configure_motion_tracking(self, project):
        """
        Process motion tracking data either from raw camera data, or from
        previously-generated deeplabcut coordinate data, for all sessions.
        """
        for i, session in enumerate(self.sessions):
            print(">>>>> Configuring motion tracking for session {} ({} / {})"
                   .format(session.name, i + 1, len(self.sessions)))
            session.configure_motion_tracking(project)

    def run_motion_tracking(self, *args, **kwargs):
        """
        Run motion tracking on camera data for all sessions.
        """
        for i, session in enumerate(self.sessions):
            print(">>>>> Running motion tracking for session {} ({} / {})"
                   .format(session.name, i + 1, len(self.sessions)))
            session.run_motion_tracking(*args, **kwargs)

    def draw_motion_index_rois(self, video_match, num_rois=1, skip=True):
        """
        Draw motion index ROIs using EasyROI. If ROIs already exist, skip, unless skip
        is False.
        """
        for i, session in enumerate(self.sessions):
            print(">>>>> Drawing motion index ROIs for session {} ({} / {})"
                   .format(session.name, i + 1, len(self.sessions)))
            session.draw_motion_index_rois(video_match, num_rois=num_rois, skip=skip)

    def process_motion_index(self, video_match, num_rois=1, skip=True):
        """
        Extract motion indexes from videos for all sessions.
        """
        for session in self.sessions:
            session.draw_motion_index_rois(video_match, num_rois=num_rois, skip=skip)

        for i, session in enumerate(self.sessions):
            print(">>>>> Processing motion index for session {} ({} / {})"
                   .format(session.name, i + 1, len(self.sessions)))
            session.process_motion_index(video_match)

    def select_units(self, *args, **kwargs):
        """
        Select units based on specified criteria. The output of this can be passed to
        some other methods to apply those methods only to these units.
        """
        units = {}

        for i, session in enumerate(self.sessions):
            name = session.name
            selected = session.select_units(*args, **kwargs)
            if len(selected) == 0:
                logging.warning(
                    f"\n> {name} does not have units in {selected}, "
                    "skip."
                )
            else:
                units[name] = selected

        return units

    def align_trials(self, *args, units=None, **kwargs):
        """
        Get trials aligned to an event. Check behaviours.base.Behaviour.align_trials for
        usage information.
        """
        trials = {}
        for i, session in enumerate(self.sessions):
            name = session.name
            result = None
            if units:
                if units[name]:
                    result = session.align_trials(
                        *args,
                        units=units[name],
                        **kwargs,
                    )
            else:
                result = session.align_trials(*args, **kwargs)
            if result is not None:
                trials[name] = result

        if "motion_tracking" in args:
            df = pd.concat(
                trials.values(), axis=1, copy=False,
                keys=trials.keys(),
                names=["session", "trial", "scorer", "bodyparts", "coords"]
            )

        if "trial_rate" in kwargs.values():
            level_names = ["session", "stream", "unit", "trial"]
            fr = ioutils.get_aligned_data_across_sessions(
                trials=trials,
                key="fr",
                level_names=level_names,
            )
            spiked = ioutils.get_aligned_data_across_sessions(
                trials=trials,
                key="spiked",
                level_names=level_names,
            )
            positions = ioutils.get_aligned_data_across_sessions(
                trials=trials,
                key="positions",
                level_names=["session", "stream", "start", "trial"],
            )
            df = {
                "fr": fr,
                "spiked": spiked,
                "positions": positions,
            }
        else:
            df = pd.concat(
                trials.values(), axis=1, copy=False,
                keys=trials.keys(),
                names=["session"] + list(trials.values())[0].columns.names,
            )

        return df

    def align_clips(self, label, event, video_match, duration=1):
        trials = []
        for session in self.sessions:
            trials.append(session.align_clips(label, event, video_match, duration))

        df = pd.concat(trials, axis=1, copy=False, names=["Session"], keys=range(len(trials)))
        return df

    def get_cluster_info(self):
        """
        Get some basic high-level information for each cluster. This is mostly just the
        information seen in the table in phy.
        """
        return [s.get_cluster_info() for s in self.sessions]

    def get_good_units_info(self):
        """
        Get some basic high-level information for each good unit. This is mostly just the
        information seen in the table in phy, plus their region.
        """
        info = {}

        for m, mouse in enumerate(self.mouse_ids):
            info[mouse] = {}
            mouse_sessions = []
            for session in self.sessions:
                if mouse in session.name:
                    mouse_sessions.append(session)

            for i, session in enumerate(mouse_sessions):
                info[mouse][i] = session.get_good_units_info()

            long_df = pd.concat(
                info[mouse],
                axis=0,
                names=["session", "unit_idx"],
            )
            info[mouse] = long_df

        mouse_info = info

        info_pooled = pd.concat(
            info, axis=0, copy=False,
            keys=info.keys(),
            names=["mouse", "session", "unit_idx"],
        )

        return mouse_info, info_pooled

    def get_spike_widths(self, units=None):
        """
        Get the widths of spikes for units matching the specified criteria.
        """
        widths = {}

        for i, session in enumerate(self.sessions):
            if units:
                if units[i]:
                    widths[i] = session.get_spike_widths(units=units[i])
            else:
                widths[i] = session.get_spike_widths()

        df = pd.concat(
            widths.values(), axis=1, copy=False,
            keys=widths.keys(),
            names=["session"]
        )
        return df

    def get_spike_waveforms(self, units=None):
        """
        Get the waveforms of spikes for units matching the specified criteria.
        """
        waveforms = {}

        for i, session in enumerate(self.sessions):
            if units:
                if units[i]:
                    waveforms[i] = session.get_spike_waveforms(units=units[i])
            else:
                waveforms[i] = session.get_spike_waveforms()
        assert 0
        df.add_prefix(i)

        #TODO: get concat waveforms for each mouse
        df = pd.concat(
            waveforms.values(), axis=1, copy=False,
            keys=waveforms.keys(),
            names=["session"]
        )
        return df

    def get_waveform_metrics(self, units=None):
        """
        Get waveform metrics of mean waveform for units matching the specified
        criteria; separated by mouse.
        """
        waveform_metrics = {}

        for m, mouse in enumerate(self.mouse_ids):
            mouse_sessions = []
            waveform_metrics[mouse] = {}
            for session in self.sessions:
                if mouse in session.name:
                    mouse_sessions.append(session)

            for i, session in enumerate(mouse_sessions):
                if units:
                    if units[i]:
                        waveform_metrics[mouse][i] = session.get_waveform_metrics(units=units[i])
                else:
                    waveform_metrics[mouse][i] = session.get_waveform_metrics()

            long_df = pd.concat(
                waveform_metrics[mouse],
                axis=0,
                names=["session", "unit_idx"],
            )
            # drop nan rows
            long_df.dropna(inplace=True)
            waveform_metrics[mouse] = long_df

        mouse_waveform_metrics = waveform_metrics

        waveform_metrics_pooled = pd.concat(
            waveform_metrics, axis=0, copy=False,
            keys=waveform_metrics.keys(),
            names=["mouse", "session", "unit_idx"],
        )

        return mouse_waveform_metrics, waveform_metrics_pooled

    def get_spike_times(self, units):
        """
        Get spike times of each units, separated by mouse.
        """
        spike_times = {}

        for m, mouse in enumerate(self.mouse_ids):
            mouse_sessions = []
            spike_times[mouse] = {}
            for session in self.sessions:
                if mouse in session.name:
                    mouse_sessions.append(session)

            for i, session in enumerate(mouse_sessions):
                spike_times[mouse][i] = session._get_spike_times()[units[i]]
                #spike_times[mouse][i] = spike_times[mouse][i].add_prefix(f'{i}_')

            df = pd.concat(
                spike_times[mouse],
                axis=1,
                names=["session", "unit"],
            )
            spike_times[mouse] = df

        mouse_spike_times = spike_times

        """
        spike_times_pooled = pd.concat(
            mouse_spike_times, axis=1, copy=False,
            keys=mouse_spike_times.keys(),
            names=["mouse", "session", "unit"],
        )
        """

        return mouse_spike_times


    def get_aligned_spike_rate_CI(self, *args, units=None, **kwargs):
        """
        Get the confidence intervals of the mean firing rates within a window aligned to
        a specified action label and event.
        """
        CIs = []

        for i, session in enumerate(self.sessions):
            if units:
                ses_cis = session.get_aligned_spike_rate_CI(*args, units=units[i], **kwargs)
            else:
                ses_cis = session.get_aligned_spike_rate_CI(*args, **kwargs)
            CIs.append(ses_cis)

        df = pd.concat(
            CIs, axis=1, copy=False,
            keys=range(len(CIs)),
            names=["session"]
        )
        return df

    def get_session_by_name(self, name: str):
        for session in self.sessions:
            if session.name == name:
                return session
        raise PixelsError


    def get_positional_data(self, *args, units=None, **kwargs):
        """
        Get positional firing rate for aligned vr trials.
        Check behaviours.base.Behaviour.get_positional_data for usage
        information.
        """
        trials = {}
        for i, session in enumerate(self.sessions):
            name = session.name
            result = None
            if units:
                if units[name]:
                    result = session.get_positional_data(
                        *args,
                        units=units[name],
                        **kwargs,
                    )
            else:
                result = session.get_positional_data(*args, **kwargs)
            if result is not None:
                trials[name] = result

        level_names = ["session", "stream", "start", "unit", "trial"]
        pos_fr = ioutils.get_aligned_data_across_sessions(
            trials=trials,
            key="pos_fr",
            level_names=level_names,
        )
        pos_fc = ioutils.get_aligned_data_across_sessions(
            trials=trials,
            key="pos_fc",
            level_names=level_names,
        )
        occupancies = ioutils.get_aligned_data_across_sessions(
            trials=trials,
            key="occupancy",
            level_names=["session", "stream", "trial"],
        )
        df = {
            "pos_fr": pos_fr,
            "pos_fc": pos_fc,
            "occupancy": occupancies,
        }

        return df


    def get_binned_trials(self, *args, units=None, **kwargs):
        """
        Get binned firing rate and spike count for aligned vr trials.
        Check behaviours.base.Behaviour.get_binned_trials for usage information.
        """
        # TODO jun 21 2025:
        # can we combine this func with get_positional_data since they are
        # basically the same, we just need to add `use_binned` bool in the arg
        session_names = [session.name for session in self.sessions]
        trials = {}
        if not units is None:
            for name in units.keys():
                session = self.sessions[session_names.index(name)]
                trials[name] = session.get_binned_trials(
                    *args,
                    units=units[name],
                    **kwargs,
                )
        else:
            for i, session in enumerate(self.sessions):
                name = session.name
                result = session.get_binned_trials(*args, **kwargs)
                if not result is None:
                    trials[name] = result

        level_names = ["session", "stream", "start", "unit", "trial"]
        bin_fr = ioutils.get_aligned_data_across_sessions(
            trials=trials,
            key="pos_fr",
            level_names=level_names,
        )
        bin_fc = ioutils.get_aligned_data_across_sessions(
            trials=trials,
            key="pos_fc",
            level_names=level_names,
        )
        bin_occupancies = ioutils.get_aligned_data_across_sessions(
            trials=trials,
            key="occupancy",
            level_names=["session", "stream", "trial"],
        )
        df = {
            "bin_fr": bin_fr,
            "bin_fc": bin_fc,
            "bin_occupancy": bin_occupancies,
        }

        return df


    def sync_vr(self, vr):
        """
        Synchronise virtual reality data of a mouse (or mice) with pixels
        streams.
        """
        trials = {}
        for i, session in enumerate(self.sessions):
            # vr is a vision-in-darkness mouse object
            vr_session = vr.sessions[i]
            assert session.name.split("_")[0] in vr_session.name

            session.sync_vr(vr_session)

        return None


    def get_spike_chance(self, *args, **kwargs):
        chance = {}
        for i, session in enumerate(self.sessions):
            name = session.name
            chance[name] = session.get_spike_chance(*args, **kwargs)

        return chance


    def get_binned_chance(self, *args, **kwargs):
        """
        Get binned chance firing rate and spike count for aligned vr trials.
        Check behaviours.base.Behaviour.get_binned_chance for usage information.
        """
        binned = {}
        for i, session in enumerate(self.sessions):
            name = session.name
            binned[name] = session.get_binned_chance(*args, **kwargs)

        return binned
