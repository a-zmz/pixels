"""
This module provides reach task specific operations.
"""

# NOTE: for event alignment, we align to the first timepoint when the event
# starts, and the last timepoint before the event ends, i.e., think of an event
# as a train of 0s and 1s, we align to the first 1s and the last 1s of a given
# event, except for licks since it could only be on or off per frame.

from enum import IntFlag, auto

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from vision_in_darkness.base import Outcomes, Worlds, Conditions

from pixels import PixelsError
from pixels.behaviours import Behaviour
from pixels.configs import *


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
    # vr events
    trial_start = auto()#1 << 0 # 1
    gray_on = auto()#1 << 1 # 2
    gray_off = auto()#1 << 2 # 4
    light_on = auto()#1 << 3 # 8
    light_off = auto()#1 << 4 # 16
    dark_on = auto()#1 << 5 # 32
    dark_off = auto()#1 << 6 # 64
    punish_on = auto()#1 << 7 # 128
    punish_off = auto()#1 << 8 # 256
    trial_end = auto()#1 << 9 # 512

    # positional events
    pre_dark_end = auto()#1 << 10 # 50 cm

    wall = auto()#1 << 12 # in between landmarks

    black_off = auto()#1 << 11 # 0 - 60 cm

    landmark1_on = auto()#1 << 13 # 110 cm
    landmark1_off = auto()#1 << 19 # 130 cm

    landmark2_on = auto()#1 << 14 # 190 cm
    landmark2_off = auto()#1 << 20 # 210 cm

    landmark3_on = auto()#1 << 15 # 270 cm
    landmark3_off = auto()#1 << 21 # 290 cm

    landmark4_on = auto()#1 << 16 # 350 cm
    landmark4_off = auto()#1 << 22 # 370 cm

    landmark5_on = auto()#1 << 17 # 430 cm
    landmark5_off = auto()#1 << 23 # 450 cm

    reward_zone_on = auto()#1 << 18 # 460 cm
    reward_zone_off = auto()#1 << 24 # 495 cm

    # sensors
    valve_open = auto()#1 << 25 # 524288
    valve_closed = auto()#1 << 26 # 1048576
    licked = auto()#1 << 27 # 134217728
    #run_start = 1 << 28
    #run_stop = 1 << 29


# map trial outcome
_outcome_map = {
    Outcomes.ABORTED_DARK: "miss_dark",
    Outcomes.ABORTED_LIGHT: "miss_light",
    Outcomes.TRIGGERED: "triggered",
    Outcomes.AUTO_LIGHT: "auto_light",
    Outcomes.DEFAULT: "default_light",
    Outcomes.REINF_LIGHT: "reinf_light",
    Outcomes.AUTO_DARK: "auto_dark",
    Outcomes.REINF_DARK: "reinf_dark",
}

# function to look up trial type
trial_type_lookup = {v: k for k, v in vars(Conditions).items()}

class VR(Behaviour):

    def _extract_action_labels(self, vr, vr_data):
        # create action label array for actions & events
        action_labels = np.zeros((vr_data.shape[0], 2), dtype=np.uint32)

        # >>>> definitions >>>>
        # define in gray
        in_gray = (vr_data.world_index == Worlds.GRAY)
        # define in dark
        in_dark = (vr_data.world_index == Worlds.DARK_5)\
                | (vr_data.world_index == Worlds.DARK_2_5)\
                | (vr_data.world_index == Worlds.DARK_FULL)
        # define in white
        in_white = (vr_data.world_index == Worlds.WHITE)
        # define in tunnel
        in_tunnel = ~in_gray & ~in_white
        # define in light
        in_light = (vr_data.world_index == Worlds.TUNNEL)
        # define light & dark trials
        trial_light = (vr_data.trial_type == Conditions.LIGHT)
        trial_dark = (vr_data.trial_type == Conditions.DARK)
        # <<<< definitions <<<<

        logging.info("\n>> Mapping vr event times...")

        # >>>> gray >>>>
        # get timestamps of gray
        gray_idx = vr_data.world_index[in_gray].index
        # get first grays
        grays = np.where(gray_idx.diff() != 1)[0]

        # find time for first frame of gray
        gray_on_t = gray_idx[grays]
        # find their index in vr data
        gray_on = vr_data.index.get_indexer(gray_on_t)
        action_labels[gray_on, 1] += Events.gray_on

        # find time for last frame of gray
        gray_off_t = np.append(gray_idx[grays[1:] - 1], gray_idx[-1])
        # find their index in vr data
        gray_off = vr_data.index.get_indexer(gray_off_t)
        action_labels[gray_off, 1] += Events.gray_off
        # <<<< gray <<<<

        # >>>> punishment >>>>
        # get timestamps of punishment 
        punish_idx = vr_data[in_white].index
        # get first punishment
        punishes = np.where(punish_idx.diff() != 1)[0]

        # find time for first frame of punishment
        punish_on_t = punish_idx[punishes]
        # find their index in vr data
        punish_on = vr_data.index.get_indexer(punish_on_t)
        action_labels[punish_on, 1] += Events.punish_on

        # find time for last frame of punish
        punish_off_t = np.append(punish_idx[punishes[1:] - 1], punish_idx[-1])
        # find their index in vr data
        punish_off = vr_data.index.get_indexer(punish_off_t)
        action_labels[punish_off, 1] += Events.punish_off
        # <<<< punishment <<<<

        # >>>> trial ends >>>>
        # trial ends right before punishment starts
        action_labels[punish_on-1, 1] += Events.trial_end

        # for non punished trials, right before gray on is when trial ends, and
        # the last frame of the session
        pre_gray_on_idx = np.append(gray_on[1:] - 1, vr_data.shape[0] - 1)
        pre_gray_on = vr_data.iloc[pre_gray_on_idx]
        # drop punish_off times
        no_punished_t = pre_gray_on.drop(punish_off_t).index
        # get index of trial ends in non punished trials
        no_punished_idx = vr_data.index.get_indexer(no_punished_t)
        action_labels[no_punished_idx, 1] += Events.trial_end
        # <<<< trial ends <<<<

        # >>>> light >>>>
        # get index of data in light tunnel
        light_idx = vr_data[in_light].index
        # get where light turns on
        lights = np.where(light_idx.diff() != 1)[0]
        # get timepoint of when light turns on
        light_on_t = light_idx[lights]
        # get index of when light turns on
        light_on = vr_data.index.get_indexer(light_on_t)
        action_labels[light_on, 1] += Events.light_on

        # get interval of possible starting position
        start_interval = int(vr.meta_item('rand_start_int'))

        # find starting position in all light_on
        trial_starts = light_on[np.where(
            vr_data.iloc[light_on].position_in_tunnel % start_interval == 0
        )[0]]
        # label trial starts
        action_labels[trial_starts, 1] += Events.trial_start

        if not trial_starts.size == vr_data[in_tunnel].trial_count.max():
            raise PixelsError(f"Number of trials does not equal to "
                    "{vr_data.trial_count.max()}.")
        # NOTE: if trial starts at 0, the first position_in_tunnel value will
        # NOT be nan

        # last frame of light
        light_off_t = np.append(light_idx[lights[1:] - 1], light_idx[-1])
        light_off = vr_data.index.get_indexer(light_off_t)
        action_labels[light_off, 1] += Events.light_off
        # <<<< light <<<<

        # NOTE: dark trials should in theory have EQUAL index pre_dark_end_idx
        # and dark_on, BUT! after interpolation, some dark trials have their
        # dark onset earlier than expected, those are corrected to the first
        # expected position, they will have the same index as pre_dark_end_idx.
        # others will not since their world_index change later than expected.

        # NOTE: if dark trial is aborted, light tunnel only turns off once; but
        # if it is a reward is dispensed, light tunnel turns off twice

        # NOTE: number of dark_on does not match with number of dark trials
        # caused by triggering punishment before dark

        # >>>> dark >>>>
        # get index in dark
        dark_idx = vr_data[in_dark].index
        darks = np.where(dark_idx.diff() != 1)[0]

        # first frame of dark
        dark_on_t = dark_idx[darks]
        dark_on = vr_data.index.get_indexer(dark_on_t)
        action_labels[dark_on, 1] += Events.dark_on

        # last frame of dark
        dark_off_t = np.append(dark_idx[darks[1:] - 1], dark_idx[-1])
        dark_off = vr_data.index.get_indexer(dark_off_t)
        action_labels[dark_off, 1] += Events.dark_off
        # <<<< dark <<<<

        # >>>> licks >>>>
        lick_onsets = np.diff(vr_data.lick_detect, prepend=0)
        licked_idx = np.where(lick_onsets == 1)[0]
        action_labels[licked_idx, 1] += Events.licked
        # <<<< licks <<<<

        # TODO jun 27 2024 positional events and valve events needs mapping

        # >>>> Event: end of pre dark length >>>>
        # NOTE: AL remove pre_dark_len + 10cm of his data
        # get starting positions of all trials
        start_pos = vr_data[in_tunnel].groupby(
            "trial_count"
        )["position_in_tunnel"].first()

        # plus pre dark
        end_pre_dark = start_pos + vr.pre_dark_len

        def _first_post_pre_dark(df):
            trial = df.name
            pre_dark_end = end_pre_dark.loc[trial]
            # mask and pick the first index
            mask = df['position_in_tunnel'] >= pre_dark_end
            if not mask.any():
                return None
            return df.index[mask].min()

        pre_dark_end_t = vr_data[in_tunnel].groupby("trial_count").apply(
            _first_post_pre_dark
        )
        pre_dark_end_idx = vr_data.index.get_indexer(
            pre_dark_end_t.dropna().astype(int)
        )
        action_labels[pre_dark_end_idx, 1] += Events.pre_dark_end
        # <<<< Event: end of pre dark length <<<<

        # >>>> Event: reward zone >>>>
        assert 0
        # TODO jun 2 2025:
        # do positional event mapping
        vr_data.position_in_tunnel
        # <<<< Event: reward zone <<<<

        logging.info("\n>> Mapping vr action times...")

        # get non-zero reward types
        reward_not_none = (vr_data.reward_type != Outcomes.NONE)

        for t, trial in enumerate(vr_data.trial_count.unique()):
            # get current trial
            of_trial = (vr_data.trial_count == trial)
            # get index of current trial
            trial_idx = np.where(of_trial)[0]
            # get start index of current trial
            start_idx = trial_idx[np.isin(trial_idx, trial_starts)]

            # find where is non-zero reward type in current trial
            reward_typed = vr_data[of_trial & reward_not_none]
            # get trial type of current trial
            trial_type = int(vr_data[of_trial].trial_type.unique())
            # get name of trial type in string
            trial_type_str = trial_type_lookup.get(trial_type).lower()

            # >>>> map reward types >>>>

            # >>>> punished >>>>
            if (reward_typed.size == 0)\
                & (vr_data[of_trial & in_white].size != 0):
                # punished outcome
                outcome = f"punished_{trial_type_str}"
                outcomes_arr[trial_idx] = getattr(TrialTypes, outcome)
                # or only mark the beginning of the trial?
                #outcomes_arr[start_idx] = getattr(TrialTypes, outcome)

                #action_labels[start_idx, 0] = getattr(TrialTypes, outcome)
            # <<<< punished <<<<

            elif (reward_typed.size == 0)\
                & (vr_data[of_trial & in_white].size == 0):
                # >>>> unfinished trial >>>>
                # double check it is the last trial
                assert (trial == vr_data.trial_count.unique().max())
                assert (vr_data[of_trial].position_in_tunnel.max()\
                        < vr.tunnel_reset)
                logging.info(f"\n> trial {trial} is unfinished when session "
                      "ends, so there is no outcome.")
                # <<<< unfinished trial <<<<
            else:
                # >>>> non punished >>>>
                # get non-zero reward type in current trial
                reward_type = int(reward_typed.reward_type.unique())
                # double check reward_type is in outcome map
                assert (reward_type in _outcome_map)

                """ triggered """
                # catch triggered trials and separate trial types
                if reward_type == Outcomes.TRIGGERED:
                    outcome = f"{_outcome_map[reward_type]}_{trial_type_str}"
                else:
                    """ given & aborted """
                    outcome = _outcome_map[reward_type]
                # label outcome
                outcomes_arr[trial_idx] = getattr(TrialTypes, outcome)
                # or only mark the beginning of the trial?
                #outcomes_arr[start_idx] = getattr(TrialTypes, outcome)
                # <<<< non punished <<<<

                # >>>> non aborted, valve only >>>>
                # if not aborted, map valve open & closed
                if reward_type > Outcomes.NONE:
                    # map valve open
                    valve_open_idx = vr_data.index.get_indexer([reward_typed.index[0]])
                    action_labels[valve_open_idx, 1] += Events.valve_open
                    # map valve closed
                    valve_closed_idx = vr_data.index.get_indexer(
                        [reward_typed.index[-1]]
                    )
                    action_labels[valve_closed_idx, 1] += Events.valve_closed
                # <<<< non aborted, valve only <<<<

            # <<<< map reward types <<<<

        # put pixels timestamps in the third column
        action_labels = np.column_stack((action_labels, vr_data.index.values))

        return action_labels


    def _check_action_labels(self, vr_data, action_labels, plot=True):

        # TODO jun 9 2024 make this work, save the plot
        if plot:
            plt.clf()
            _, axes = plt.subplots(4, 1, sharex=False, sharey=False)
            axes[0].plot(back_sensor_signal)
            if "/'Back_Sensor'/'0'" in behavioural_data:
                axes[1].plot(behavioural_data["/'Back_Sensor'/'0'"].values)
            else:
                axes[1].plot(behavioural_data["/'ReachCue_LEDs'/'0'"].values)
            axes[2].plot(action_labels[:, 0])
            axes[3].plot(action_labels[:, 1])
            plt.plot(action_labels[:, 1])
            plt.show()

        return action_labels

'''
from enum import IntFlag, auto
from typing    import NamedTuple, List, Tuple

import numpy  as np
import pandas as pd

from vision_in_darkness.base import Outcomes, Worlds, Conditions
from pixels.behaviours       import Behaviour
from pixels                   import PixelsError


class Events(IntFlag):
    """Bit-flags for everything that can happen to the animal in VR."""
    NONE            = 0
    trial_start     = auto()
    gray_on         = auto()
    gray_off        = auto()
    light_on        = auto()
    light_off       = auto()
    dark_on         = auto()
    dark_off        = auto()
    punish_on       = auto()
    punish_off      = auto()
    pre_dark_end    = auto()
    reward_zone     = auto()
    valve_open      = auto()
    valve_closed    = auto()
    lick            = auto()


class ActionLabels(IntFlag):
    """Mutually exclusive trial‐outcome labels, plus helpful combos."""
    NONE              = 0
    miss_light        = auto()
    miss_dark         = auto()
    triggered_light   = auto()
    triggered_dark    = auto()
    punished_light    = auto()
    punished_dark     = auto()
    default_light     = auto()
    auto_light        = auto()
    auto_dark         = auto()
    reinf_light       = auto()
    reinf_dark        = auto()

    # handy OR-combos
    miss        = miss_light      | miss_dark
    triggered   = triggered_light | triggered_dark
    punished    = punished_light  | punished_dark

    light       = (miss_light    | triggered_light | punished_light
                   | default_light | auto_light      | reinf_light)
    dark        = (miss_dark     | triggered_dark  | punished_dark
                   | auto_dark     | reinf_dark)
    rewarded_light = (triggered_light | default_light
                      | auto_light   | reinf_light)
    rewarded_dark  = (triggered_dark  | auto_dark
                      | reinf_dark)
    given_light    = default_light    | auto_light | reinf_light
    given_dark     = auto_dark        | reinf_dark
    completed_light = miss_light     | triggered_light | default_light \
                      | auto_light    | reinf_light
    completed_dark  = miss_dark      | triggered_dark  | auto_dark \
                      | reinf_dark


class LabeledEvents(NamedTuple):
    """Structured return from _extract_action_labels."""
    timestamps: np.ndarray       # shape (N,)
    outcome:    np.ndarray       # shape (N,) of ActionLabels
    events:     np.ndarray       # shape (N,) of Events


class VR(Behaviour):
    """Behaviour subclass that extracts events & action labels from vr_data."""

    def _extract_action_labels(
        self,
        vr_data: pd.DataFrame
    ) -> LabeledEvents:
        """
        Go over every frame in vr_data and assign:
         - `events[i]` := bitmask of Events that occur at frame i
         - `outcome[i]` := the trial‐outcome (one and only one ActionLabel) at i
        """
        N = len(vr_data)
        events  = np.zeros(N, dtype=np.uint64)
        outcome = np.zeros(N, dtype=np.uint64)

        # 1) stamp world‐based events (gray, light, dark, punish) via run masks
        for evt, mask in self._world_event_masks(vr_data):
            self._stamp_mask(events, mask, evt)

        # 2) stamp positional events (pre‐dark end, reward‐zone)
        for evt, mask in self._position_event_masks(vr_data):
            self._stamp_mask(events, mask, evt)

        # 3) stamp sensors: lick, valve open/closed
        self._stamp_rising(events, vr_data.lick_detect.values, Events.lick)
        # (if you have valve signals, do same for them)

        # 4) map each trial’s outcome into the outcome array
        outcome_map = self._build_outcome_map()
        for trial_id, group in vr_data.groupby("trial_count"):
            idxs = group.index.values
            flag = self._compute_outcome_flag(group, outcome_map)
            outcome[idxs] = flag

            # stamp trial_start at the first frame of each triggered trial:
            if flag in (ActionLabels.triggered_light, ActionLabels.triggered_dark):
                first_idx = idxs[0]
                events[first_idx] |= Events.trial_start

        # 5) return timestamps + two bit‐masked channels
        return LabeledEvents(
            timestamps = vr_data.index.values,
            outcome    = outcome.astype(np.uint32),
            events     = events.astype(np.uint32)
        )


    # -------------------------------------------------------------------------
    #  Helpers to apply a flag wherever mask is true or on rising edges
    # -------------------------------------------------------------------------

    @staticmethod
    def _stamp_mask(
        storage: np.ndarray,
        mask:    np.ndarray,
        flag:    IntFlag
    ) -> None:
        """Bitwise‐OR `flag` into `storage` at every True in `mask`."""
        storage[mask] |= flag

    @staticmethod
    def _stamp_rising(
        storage: np.ndarray,
        signal:  np.ndarray,
        flag:    IntFlag
    ) -> None:
        """
        Find rising‐edge frames in a 0/1 `signal` array (diff == +1)
        and stamp `flag` at those indices.
        """
        edges = np.flatnonzero(np.diff(signal, prepend=0) == 1)
        storage[edges] |= flag


    # -------------------------------------------------------------------------
    #  Build lists of (EventFlag, boolean_mask) for world & positional events
    # -------------------------------------------------------------------------

    def _world_event_masks(
        self,
        df: pd.DataFrame
    ) -> List[Tuple[Events, np.ndarray]]:
        w = Worlds
        return [
            # gray: enters in GRAY, leaves when GRAY ends
            (Events.gray_on,  self._first_in_run(df.world_index == w.GRAY)),
            (Events.gray_off, self._last_in_run (df.world_index == w.GRAY)),

            # white (“punish” region)
            (Events.punish_on,  self._first_in_run(df.world_index == w.WHITE)),
            (Events.punish_off, self._last_in_run (df.world_index == w.WHITE)),

            # light tunnel
            (Events.light_on,   self._first_in_run(df.world_index == w.TUNNEL)),
            (Events.light_off,  self._last_in_run (df.world_index == w.TUNNEL)),

            # dark tunnels (could be multiple dark worlds)
            (Events.dark_on,    self._first_in_run(df.world_index.isin(w.DARKS))),
            (Events.dark_off,   self._last_in_run (df.world_index.isin(w.DARKS))),
        ]

    def _position_event_masks(
        self,
        df: pd.DataFrame
    ) -> List[Tuple[Events, np.ndarray]]:
        # pre‐dark end: when position ≥ (start_pos + pre_dark_len)
        start_pos = df[df.world_index==Worlds.TUNNEL] \
                    .groupby("trial_count")["position_in_tunnel"] \
                    .first()
        end_vals  = start_pos + self.pre_dark_len
        # mask per‐trial then merge
        pre_dark_mask = np.zeros(len(df), bool)
        for trial, thresh in end_vals.items():
            idx = df.trial_count==trial
            pre_dark_mask[idx] |= (df.position_in_tunnel[idx] >= thresh)

        # reward zone: any frame at or beyond rz_start
        rz_mask = df.position_in_tunnel >= self.rz_start

        return [
            (Events.pre_dark_end, pre_dark_mask),
            (Events.reward_zone,  rz_mask),
        ]


    # -------------------------------------------------------------------------
    #  Utilities for detecting run‐start and run‐end of a boolean mask
    # -------------------------------------------------------------------------

    @staticmethod
    def _first_in_run(mask: np.ndarray) -> np.ndarray:
        """
        True exactly at the first True of each contiguous run in `mask`.
        """
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return np.zeros_like(mask)
        # a run starts where current True differs from previous True
        starts = np.concatenate([[True],
                                 mask[idx[1:]] != mask[idx[:-1]]])
        first_idx = idx[starts]
        out = np.zeros_like(mask)
        out[first_idx] = True
        return out

    @staticmethod
    def _last_in_run(mask: np.ndarray) -> np.ndarray:
        """
        True exactly at the last True of each contiguous run in `mask`.
        """
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return np.zeros_like(mask)
        ends = np.concatenate([mask[idx[:-1]] != mask[idx[1:]], [True]])
        last_idx = idx[ends]
        out = np.zeros_like(mask)
        out[last_idx] = True
        return out


    # -------------------------------------------------------------------------
    #  Outcome‐mapping machinery
    # -------------------------------------------------------------------------

    def _build_outcome_map(self) -> dict:
        """
        Returns a dict mapping (Outcomes, Conditions) → ActionLabels.
        """
        m = {
            (Outcomes.ABORTED_DARK,   Conditions.DARK):  ActionLabels.miss_dark,
            (Outcomes.ABORTED_LIGHT,  Conditions.LIGHT): ActionLabels.miss_light,
            (Outcomes.PUNISHED,       Conditions.LIGHT): ActionLabels.punished_light,
            (Outcomes.PUNISHED,       Conditions.DARK):  ActionLabels.punished_dark,
            (Outcomes.AUTO_LIGHT,     Conditions.LIGHT): ActionLabels.auto_light,
            (Outcomes.DEFAULT,        Conditions.LIGHT): ActionLabels.default_light,
            (Outcomes.REINF_LIGHT,    Conditions.LIGHT): ActionLabels.reinf_light,
            (Outcomes.AUTO_DARK,      Conditions.DARK):  ActionLabels.auto_dark,
            (Outcomes.REINF_DARK,     Conditions.DARK):  ActionLabels.reinf_dark,

            # triggered must include light vs dark
            (Outcomes.TRIGGERED,      Conditions.LIGHT): ActionLabels.triggered_light,
            (Outcomes.TRIGGERED,      Conditions.DARK):  ActionLabels.triggered_dark,
        }
        return m

    def _compute_outcome_flag(
        self,
        trial_df:   pd.DataFrame,
        outcome_map: dict
    ) -> ActionLabels:
        """
        Given one trial's DataFrame, look at its reward_type & trial_type
        and return the matching ActionLabels member.
        """
        rts = trial_df.reward_type.unique()
        if rts.size == 0:
            # no reward_type → unfinished or last‐trial abort
            return ActionLabels.NONE

        rt = Outcomes(int(rts[0]))
        cond = Conditions(int(trial_df.trial_type.iloc[0]))
        key  = (rt, cond)
        try:
            return outcome_map[key]
        except KeyError:
            raise PixelsError(f"No outcome mapping for {key}")
'''
