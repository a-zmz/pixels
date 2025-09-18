"""
This module provides reach task specific operations.
"""

# NOTE: for event alignment, we align to the first timepoint when the event
# starts, and the last timepoint before the event ends, i.e., think of an event
# as a train of 0s and 1s, we align to the first 1s and the last 1s of a given
# event, except for licks since it could only be on or off per frame.

from enum import IntFlag, auto
from typing import NamedTuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from vision_in_darkness.base import Outcomes, Worlds, Conditions

from pixels import PixelsError
from pixels.behaviours import Behaviour
from pixels.configs import *
from pixels.constants import SAMPLE_RATE, V1_SPIKE_LATENCY


class TrialTypes(IntFlag):
    """
    These cover all possible trial types.

    To align trials to more than one action type they can be bitwise OR'd i.e.
    `miss_light | miss_dark` will match all miss trials.

    Trial types can NOT be added on top of each other, they should be mutually
    exclusive.
    """
    # TODO jun 7 2024 does the name "action" make sense?

    # TODO jul 4 2024 only label trial type at the first frame of the trial to
    # make it easier for alignment???
    # triggered vr trials
    NONE = 0
    miss_light = auto()#1 << 0 # 1
    miss_dark = auto()#1 << 1 # 2
    triggered_light = auto()#1 << 2 # 4
    triggered_dark = auto()#1 << 3 # 8
    punished_light = auto()#1 << 4 # 16
    punished_dark = auto()#1 << 5 # 32

    # given reward
    default_light = auto()#1 << 6 # 64
    auto_light = auto()#1 << 7 # 128
    auto_dark = auto()#1 << 8 # 256
    reinf_light = auto()#1 << 9 # 512
    reinf_dark = auto()#1 << 10 # 1024

    # combos
    miss = miss_light | miss_dark
    triggered = triggered_light | triggered_dark
    punished = punished_light | punished_dark

    # trial type combos
    light = miss_light | triggered_light | punished_light | default_light\
            | auto_light | reinf_light
    dark = miss_dark | triggered_dark | punished_dark | auto_dark | reinf_dark
    rewarded_light = triggered_light | default_light | auto_light | reinf_light
    rewarded_dark = triggered_dark | auto_dark | reinf_dark
    given_light = default_light | auto_light | reinf_light
    given_dark = auto_dark | reinf_dark
    # no punishment, i.e., no missing data
    completed_light = miss_light | triggered_light | default_light\
            | auto_light | reinf_light
    completed_dark = miss_dark | triggered_dark | auto_dark | reinf_dark


class Events(IntFlag):
    """
    Defines events that could happen during vr sessions.

    Events can be added on top of each other.
    """
    NONE = 0
    # vr events
    trial_start = auto() # 1
    gray_on = auto() # 2
    gray_off = auto() # 4
    light_on = auto() # 8
    light_off = auto() # 16
    dark_on = auto() # 32
    dark_off = auto() # 64
    punish_on = auto() # 128
    punish_off = auto() # 256
    trial_end = auto() # 512

    # positional events
    pre_dark_end = auto()# 50 cm

    # TODO jun 4 2025:
    # how to mark wall?
    wall = auto()# in between landmarks

    # black wall
    landmark0_on = auto()# wherever trial starts, before 60cm
    landmark0_off = auto()# 60 cm

    landmark1_on = auto()# 110 cm
    landmark1_off = auto()# 130 cm

    landmark2_on = auto()# 190 cm
    landmark2_off = auto()# 210 cm

    landmark3_on = auto()# 270 cm
    landmark3_off = auto()# 290 cm

    landmark4_on = auto()# 350 cm
    landmark4_off = auto()# 370 cm

    landmark5_on = auto()# 430 cm
    landmark5_off = auto()# 450 cm

    reward_zone_on = auto()# 460 cm
    reward_zone_off = auto()# 495 cm

    # sensors
    valve_open = auto()
    valve_closed = auto()
    licked = auto()
    #run_start = auto()
    #run_stop = auto()

    # temporal events in dark only
    dark_luminance_off = auto()# 500 ms


class LabeledEvents(NamedTuple):
    """Return type: timestamps + bitfields for outcome & events."""
    timestamps: np.ndarray # shape (N,)
    outcome: np.ndarray # shape (N,) dtype uint32
    events: np.ndarray # shape (N,) dtype uint32


class WorldMasks(NamedTuple):
    in_gray: pd.Series
    in_dark: pd.Series
    in_white: pd.Series
    in_light: pd.Series
    in_tunnel: pd.Series


class ConditionMasks(NamedTuple):
    light_trials: pd.Series
    dark_trials: pd.Series


class VR(Behaviour):
    """Behaviour subclass to extract action labels & events from vr_data."""
    def _get_world_masks(self, df: pd.DataFrame) -> WorldMasks:
        # define in gray
        in_gray = (df.world_index == Worlds.GRAY)
        # define in dark
        in_dark = df.world_index.isin(
            {w.value for w in Worlds if w.is_dark}
        )
        # define in white
        in_white = (df.world_index == Worlds.WHITE)
        # define in light
        in_light = (df.world_index == Worlds.TUNNEL)
        # define in tunnel
        in_tunnel = (~in_gray & ~in_white)

        return WorldMasks(
            in_gray = in_gray,
            in_dark = in_dark,
            in_white = in_white,
            in_light = in_light,
            in_tunnel = in_tunnel,
        )


    def _get_condition_masks(self, df: pd.DataFrame) -> ConditionMasks:
        return ConditionMasks(
            light_trials = (df.trial_type == Conditions.LIGHT),
            dark_trials = (df.trial_type == Conditions.DARK),
        )


    def _extract_action_labels(
        self,
        session,
        data: pd.DataFrame
    ) -> LabeledEvents:
        """
        Go over every frame in data and assign:
         - `events_arr[i]` := bitmask of Events that occur at frame i
         - `outcomes_arr[i]` := the trial‐outcome (one and only one TrialType)
           at i
        """
        N = len(data)
        events_arr = np.zeros(N, dtype=np.uint32)
        outcomes_arr = np.zeros(N, dtype=np.uint32)

        # world index based events
        world_index_based_events = self._world_event_indices(data)
        for event, idx in world_index_based_events.items():
            mask = self._get_index(data, idx.to_numpy())
            self._stamp_mask(events_arr, mask, event)

        # get dark onset times for pre_dark_len labels
        dark_on_t = world_index_based_events[Events.dark_on]

        # positional events (pre‐dark end, landmarks, reward‐zone)
        pos_based_events = self._position_event_indices(
            session,
            data,
            dark_on_t,
        )
        for event, idx in pos_based_events.items():
            mask = self._get_index(data, idx.to_numpy())
            self._stamp_mask(events_arr, mask, event)

        # sensors: lick rising‐edge
        self._stamp_rising(events_arr, data.lick_detect.values, Events.licked)

        # map trial outcomes
        outcome_map = self._build_outcome_map()
        for trial_id, group in data.groupby("trial_count"):
            trial_t = group.index.values
            flag, valve_events = self._compute_outcome_flag(
                trial_id,
                group,
                outcome_map,
                session,
            )

            # save outcomes
            idx = self._get_index(data, trial_t)
            outcomes_arr[idx] = flag

            if len(valve_events) > 0:
                for v_event, v_idx in valve_events.items():
                    valve_mask = self._get_index(data, v_idx)
                    self._stamp_mask(events_arr, valve_mask, v_event)

        # return typed arrays
        return LabeledEvents(
            timestamps = data.index.values,
            outcome = outcomes_arr,
            events = events_arr,
        )


    # -------------------------------------------------------------------------
    #  Core stamping helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _stamp_mask(array: np.ndarray, mask: np.ndarray, flag: IntFlag):
        """
        Bitwise‐OR `flag` into `storage` at every True in `mask`.
        """
        np.bitwise_or.at(array, mask, flag)

    @staticmethod
    def _stamp_rising(array: np.ndarray, signal: np.ndarray, flag: IntFlag):
        """
        Find rising‐edge frames in a 0/1 `signal` array (diff == +1)
        and stamp `flag` at those indices.
        """
        # extract edges
        edges = np.flatnonzero(np.diff(signal, prepend=0) == 1)
        np.bitwise_or.at(array, edges, flag)


    # -------------------------------------------------------------------------
    #  Build world‐based event masks
    # -------------------------------------------------------------------------

    def _world_event_indices(
            self,
            df: pd.DataFrame
        ) -> dict[Events, pd.Series]:
        """
        Build (event_flag, boolean_mask) for gray, white (punish),
        light tunnel, and dark tunnels, but *per trial*,
        so each trial contributes its own on/off.
        """
        masks: dict[Events, pd.Series] = {}
        N = len(df)

        world_masks = self._get_world_masks(df)

        specs = [
            # (which‐world‐test, on‐event, off‐event)
            (world_masks.in_gray, Events.gray_on, Events.gray_off),
            (world_masks.in_white, Events.punish_on, Events.punish_off),
            (world_masks.in_dark, Events.dark_on, Events.dark_off),
        ]

        for bool_mask, event_on, event_off in specs:
            # compute the per‐trial first‐index
            trials = df[bool_mask].groupby("trial_count")
            masks[event_on] = trials.apply(self._first_index)
            # compute the per‐trial last‐index
            masks[event_off] = trials.apply(self._last_index)


        # >>>> trial ends >>>>
        gray_on_t = masks[Events.gray_on]

        # for non punished trials, right before gray on is when trial ends, plus
        # the last frame of the session
        trial_ends_t = gray_on_t.copy()
        trial_ends_t.iloc[:-1] = (gray_on_t[1:] - 1).to_numpy()
        trial_ends_t.iloc[-1] = df.index[-1]

        # trial ends right before punishment starts
        punish_on_t = masks[Events.punish_on]
        trial_ends_t.loc[punish_on_t.index] = punish_on_t - 1

        masks[Events.trial_end] = trial_ends_t
        # <<<< trial ends <<<<

        # >>> handle light on separately >>>
        # build a run id that increments whenever in_light flips
        run_id = world_masks.in_light.ne(
            world_masks.in_light.shift(fill_value=False)
        ).cumsum()

        # restrict to just the True‐runs
        light_runs = df[world_masks.in_light].copy()
        light_runs['run_id'] = run_id[world_masks.in_light]

        # for each (trial, run_id) get the on‐times and off‐times
        firsts = (
            light_runs
            .groupby(["trial_count", "run_id"])
            .apply(self._first_index)
        )
        light_ons = firsts.droplevel("run_id")
        lasts = (
            light_runs
            .groupby(["trial_count", "run_id"])
            .apply(self._last_index)
        )
        light_offs = lasts.droplevel("run_id")

        masks[Events.light_on] = light_ons
        masks[Events.light_off] = light_offs
        # <<< handle light on separately <<<

        return masks


    # -------------------------------------------------------------------------
    #  Build positional event masks (landmarks, pre_dark_end, reward zone)
    # -------------------------------------------------------------------------

    def _position_event_indices(
        self,
        session,
        df: pd.DataFrame,
        dark_on_t,
    ) -> dict[Events, pd.Series]:
        masks: dict[Events, pd.Series] = {}

        in_tunnel = self._get_world_masks(df).in_tunnel

        def _first_post_mark(group_df, check_marks):
            if isinstance(check_marks, pd.Series):
                group_id = group_df.name
                mark = check_marks.loc[group_id]
            elif isinstance(check_marks, int):
                mark = check_marks

            # mask and pick the first index
            mask = (group_df["position_in_tunnel"] >= mark)
            if not mask.any():
                return None
            return group_df.index[mask].min()

        # >>> distance travelled before dark onset per trial >>>
        # NOTE: this applies to light trials too to keep data symetrical
        # AL remove pre_dark_len + 10cm in all his data
        in_tunnel_trials = df[in_tunnel].groupby("trial_count")
        # get start of each trial
        trial_starts = in_tunnel_trials.apply(self._first_index)
        masks[Events.trial_start] = trial_starts

        # NOTE: dark trials should in theory have EQUAL index pre_dark_end_t
        # and dark_on, BUT! after interpolation, some dark trials have their
        # dark onset earlier than expected, those are corrected to the first
        # expected position, they will have the same index as pre_dark_end_t.
        # others will not since their world_index change later than expected.
        # SO! to keep it consistent, for dark trials, pre_dark_end will be the
        # SAME frame as dark onsets.

        # get starting positions of all trials
        start_pos = in_tunnel_trials["position_in_tunnel"].first()

        # get light trials
        lights = self._get_condition_masks(df).light_trials
        light_trials  = df[in_tunnel & lights].groupby("trial_count")
        # get starting positions of light trials plus pre dark length
        light_pre_dark_len = light_trials["position_in_tunnel"].first()\
                            + session.pre_dark_len
        light_pre_dark_end_t = light_trials.apply(
            lambda df: _first_post_mark(df, light_pre_dark_len)
        ).dropna().astype(int)

        # concat dark and light trials
        pre_dark_end_t = pd.concat([dark_on_t, light_pre_dark_end_t])
        masks[Events.pre_dark_end] = pre_dark_end_t
        # >>> distance travelled before dark onset per trial >>>

        # >>> end of luminance change after dark onset >>>
        n_frames = int(V1_SPIKE_LATENCY * SAMPLE_RATE / 1000)
        dark_luminance_off_t = dark_on_t + n_frames
        masks[Events.dark_luminance_off] = dark_luminance_off_t
        # <<< end of luminance change after dark onset <<<

        # >>> landmark 0 black wall >>>
        black_off = session.landmarks[0]

        starts_before_black = (start_pos < black_off)
        early_ids = start_pos[starts_before_black].index
        early_trials = df[
            in_tunnel & df.trial_count.isin(early_ids)
        ].groupby("trial_count")

        # first frame of black wall
        landmark0_on = early_trials.apply(
            self._first_index
        )
        masks[Events.landmark0_on] = landmark0_on

        # last frame of black wall
        landmark0_off = early_trials.apply(
            lambda df: _first_post_mark(df, black_off)
        )
        masks[Events.landmark0_off] = landmark0_off
        # <<< landmark 0 black wall <<<

        # >>> landmarks 1 to 5 >>>
        landmarks = session.landmarks[1:]

        for l, landmark in enumerate(landmarks):
            if l % 2 != 0:
                continue

            landmark_idx = l // 2 + 1

            # even idx on, odd idx off
            on_landmark = landmark
            off_landmark = landmarks[l + 1]

            in_landmark = (
                (df.position_in_tunnel >= on_landmark) &
                (df.position_in_tunnel <= off_landmark)
            )

            landmark_on = df[in_landmark].groupby("trial_count").apply(
                self._first_index
            )
            landmark_off = df[in_landmark].groupby("trial_count").apply(
                self._last_index
            )

            masks[getattr(Events, f"landmark{landmark_idx}_on")] = landmark_on
            masks[getattr(Events, f"landmark{landmark_idx}_off")] = landmark_off
        # <<< landmarks 1 to 5 <<<

        # >>> reward zone >>>
        in_zone = (
            df.position_in_tunnel >= session.reward_zone_start
        ) & (
            df.position_in_tunnel <= session.reward_zone_end
        )
        in_zone_trials = df[in_zone].groupby("trial_count")

        # first frame in reward zone
        zone_on_t = in_zone_trials.apply(self._first_index)
        masks[Events.reward_zone_on] = zone_on_t

        # last frame in reward zone
        zone_off_t = in_zone_trials.apply(self._last_index)
        masks[Events.reward_zone_off] = zone_off_t
        # <<< reward zone <<<

        return masks


    # -------------------------------------------------------------------------
    #  Run‐start / run‐end utilities for boolean masks
    # -------------------------------------------------------------------------

    def _first_index(self, group: pd.DataFrame) -> int:
        idx = group.index
        early_idx = idx[:len(idx) // 2]

        # double check the last index is the first time reaching that point
        idx_discontinued = (np.diff(early_idx) > 1)
        #if np.any(idx_discontinued):
        #    print("discontinued")
        #    assert 0
        #    last_disc = np.where(idx_discontinued)[0][-1]
        #    return group.iloc[last_disc:].index.min()
        #else:
        return group.index.min()


    def _last_index(self, group: pd.DataFrame) -> int:
        # only check the second half in case the discontinuity happened at the
        # beginning
        idx = group.index
        late_idx = idx[-len(idx)//4:]

        # double check the last index is the first time reaching that point
        idx_discontinued = (np.diff(late_idx) > 1)
        if np.any(idx_discontinued):
            logging.warning("\n> index discontinued.")
            print(group)
            first_disc = np.where(idx_discontinued)[0][0]
            return late_idx[first_disc]
        else:
            return idx.max()


    def _get_index(self, df: pd.DataFrame, index) -> int:
        return df.index.get_indexer(index)

    # -------------------------------------------------------------------------
    #  Outcome mapping
    # -------------------------------------------------------------------------

    def _build_outcome_map(self) -> dict:
        m = {
            (Outcomes.ABORTED_LIGHT, Conditions.LIGHT): TrialTypes.miss_light,
            (Outcomes.ABORTED_DARK, Conditions.DARK): TrialTypes.miss_dark,

            (Outcomes.NONE, Conditions.LIGHT): TrialTypes.punished_light,
            (Outcomes.NONE, Conditions.DARK): TrialTypes.punished_dark,

            (Outcomes.DEFAULT, Conditions.LIGHT): TrialTypes.default_light,

            (Outcomes.AUTO_LIGHT, Conditions.LIGHT): TrialTypes.auto_light,
            (Outcomes.AUTO_DARK, Conditions.DARK): TrialTypes.auto_dark,

            (Outcomes.REINF_LIGHT, Conditions.LIGHT): TrialTypes.reinf_light,
            (Outcomes.REINF_DARK, Conditions.DARK): TrialTypes.reinf_dark,

            (Outcomes.TRIGGERED, Conditions.LIGHT): TrialTypes.triggered_light,
            (Outcomes.TRIGGERED, Conditions.DARK): TrialTypes.triggered_dark,
        }
        return m

    def _compute_outcome_flag(
        self,
        trial_id: int,
        trial_df: pd.DataFrame,
        outcome_map: dict,
        session,
    ) -> TrialTypes:

        valve_events = {}

        # get non-zero reward types
        reward_not_none = (trial_df.reward_type != Outcomes.NONE)
        reward_typed = trial_df[reward_not_none]

        # get trial type
        trial_type = int(trial_df.trial_type.iloc[0])

        # get punished
        punished = (trial_df.world_index == Worlds.WHITE)

        if (reward_typed.size == 0) & (not np.any(punished)):
                # >>>> unfinished trial >>>>
                # double check it is the last trial
                assert (trial_df.position_in_tunnel.max()\
                        < session.tunnel_reset)
                logging.info(f"\n> trial {trial_id} is unfinished when session "
                      "ends, so there is no outcome.")
                return TrialTypes.NONE, valve_events
                # <<<< unfinished trial <<<<
        elif (reward_typed.size == 0) & np.any(punished):
            logging.info(f"\n> trial {trial_id} is punished.")
            # get reward type zero for punished
            reward_type = int(trial_df.reward_type.unique())
        else:
            # get non-zero reward type in current trial
            reward_type = int(reward_typed.reward_type.unique())

            if reward_type > Outcomes.NONE:
                # >>>> non aborted, valve events >>>>
                # if not aborted, map valve open & closed
                # map valve open
                valve_open_t = reward_typed.index[0]
                valve_events[Events.valve_open] = [valve_open_t]
                # map valve closed
                valve_closed_t = reward_typed.index[-1]
                valve_events[Events.valve_closed] = [valve_closed_t]
                # <<<< non aborted, valve events <<<<

        # build key for outcome_map
        key = (reward_type, trial_type)

        try:
            return outcome_map[key], valve_events
        except KeyError:
            raise PixelsError(f"No mapping for outcome {key}")
