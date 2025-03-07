"""
This module provides reach task specific operations.
"""

# NOTE: for event alignment, we align to the first timepoint when the event
# starts, and the last timepoint before the event ends, i.e., think of an event
# as a train of 0s and 1s, we align to the first 1s and the last 1s of a given
# event, except for licks since it could only be on or off per frame.

from __future__ import annotations

import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from vision_in_darkness.base import Outcomes, Worlds, Conditions

from pixels import Experiment, PixelsError
import pixels.signal_utils as signal
from pixels import ioutils
from pixels.behaviours import Behaviour

from common_utils import file_utils


class ActionLabels:
    """
    These actions cover all possible trial types.

    To align trials to more than one action type they can be bitwise OR'd i.e.
    `miss_light | miss_dark` will match all miss trials.

    Actions can NOT be added on top of each other, they should be mutually
    exclusive.
    """
    # TODO jun 7 2024 does the name "action" make sense?

    # TODO jul 4 2024 only label trial type at the first frame of the trial to
    # make it easier for alignment???
    # triggered vr trials
    miss_light = 1 << 0 # 1
    miss_dark = 1 << 1 # 2
    triggered_light = 1 << 2 # 4
    triggered_dark = 1 << 3 # 8
    punished_light = 1 << 4 # 16
    punished_dark = 1 << 5 # 32

    # given reward
    default_light = 1 << 6 # 64
    auto_light = 1 << 7 # 128
    auto_dark = 1 << 8 # 256
    reinf_light = 1 << 9 # 512
    reinf_dark = 1 << 10 # 1024

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


class Events:
    """
    Defines events that could happen during vr sessions.

    Events can be added on top of each other.
    """
    # vr events
    trial_start = 1 << 0 # 1
    gray_on = 1 << 1 # 2
    gray_off = 1 << 2 # 4
    light_on = 1 << 3 # 8
    light_off = 1 << 4 # 16
    dark_on = 1 << 5 # 32
    dark_off = 1 << 6 # 64
    punish_on = 1 << 7 # 128
    punish_off = 1 << 8 # 256
    trial_end = 1 << 9 # 512

    # positional events
    black = 1 << 10 # 0 - 60 cm
    wall = 1 << 11 # in between landmarks
    landmark1 = 1 << 12 # 110 - 130 cm
    landmark2 = 1 << 13 # 190 - 210 cm
    landmark3 = 1 << 14 # 270 - 290 cm
    landmark4 = 1 << 15 # 350 - 370 cm
    landmark5 = 1 << 16 # 430 - 450 cm
    reward_zone = 1 << 17 # 460 - 495 cm

    # sensors
    valve_open = 1 << 18 # 262144
    valve_closed = 1 << 19 # 524288
    licked = 1 << 20 # 1048576
    #run_start = 1 << 12
    #run_stop = 1 << 13


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
        action_labels = np.zeros((vr_data.shape[0], 2), dtype=np.int32)

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

        print(">> Mapping vr event times...")

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

        if not trial_starts.size == vr_data.trial_count.max():
            raise PixelsError(f"Number of trials does not equal to\
                    \n{vr_data.trial_count.max()}.")
        # NOTE: if trial starts at 0, the first position_in_tunnel value will
        # NOT be nan

        # last frame of light
        light_off_t = np.append(light_idx[lights[1:] - 1], light_idx[-1])
        light_off = vr_data.index.get_indexer(light_off_t)
        action_labels[light_off, 1] += Events.light_off
        # <<<< light <<<<

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

        print(">> Mapping vr action times...")

        # >>>> map reward types >>>>
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

            # >>>> punished >>>>
            if (reward_typed.size == 0)\
                & (vr_data[of_trial & in_white].size != 0):
                # punished outcome
                outcome = f"punished_{trial_type_str}"
                action_labels[trial_idx, 0] = getattr(ActionLabels, outcome)
                #action_labels[start_idx, 0] = getattr(ActionLabels, outcome)
            # <<<< punished <<<<

            elif (reward_typed.size == 0)\
                & (vr_data[of_trial & in_white].size == 0):
                # >>>> unfinished trial >>>>
                # double check it is the last trial
                assert (trial == vr_data.trial_count.unique().max())
                assert (vr_data[of_trial].position_in_tunnel.max()\
                        < vr.tunnel_reset)
                print(f"> trial {trial} is unfinished when session ends, so "
                      "there is no outcome.")
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
                action_labels[trial_idx, 0] = getattr(ActionLabels, outcome)
                #action_labels[start_idx, 0] = getattr(ActionLabels, outcome)
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
    # TODO sep 30 2024:
    # refactored code from chatgpt
    # needs testing!
    def _assign_event_label(self, action_labels, event_times, event_type, column=1):
        """
        Helper function to assign event labels to action_labels array.
        """
        event_indices = event_times.index
        event_on_idx = np.where(event_indices.diff() != 1)[0]

        # Find first and last timepoints for events
        event_on_t = event_indices[event_on_idx]
        event_off_t = np.append(event_indices[event_on_idx[1:] - 1], event_indices[-1])

        event_on = event_times.index.get_indexer(event_on_t)
        event_off = event_times.index.get_indexer(event_off_t)

        action_labels[event_on, column] += event_type['on']
        action_labels[event_off, column] += event_type['off']

        return action_labels

    def _map_trial_events(self, action_labels, vr_data, vr):
        """
        Maps different trial events like gray, light, dark, and punishments.
        """
        # Define event mappings for gray, light, dark, punishments
        event_mappings = {
            'gray': {'on': Events.gray_on, 'off': Events.gray_off,
                'condition': vr_data.world_index == Worlds.GRAY},
            'light': {'on': Events.light_on, 'off': Events.light_off,
                'condition': vr_data.world_index == Worlds.TUNNEL},
            'dark': {'on': Events.dark_on, 'off': Events.dark_off,
                'condition': (vr_data.world_index == Worlds.DARK_5)\
                    | (vr_data.world_index == Worlds.DARK_2_5)\
                    | (vr_data.world_index == Worlds.DARK_FULL)},
            'punish': {'on': Events.punish_on, 'off': Events.punish_off,
                'condition': vr_data.world_index == Worlds.WHITE},
        }

        for event_name, event_type in event_mappings.items():
            event_times = vr_data[event_type['condition']]
            action_labels = self._assign_event_label(action_labels, event_times, event_type)

        return action_labels

    def _assign_trial_outcomes(self, action_labels, vr_data, vr):
        """
        Assign outcomes for each trial, including rewards and punishments.
        """
        for t, trial in enumerate(vr_data.trial_count.unique()):
            # Extract trial-specific information
            of_trial = (vr_data.trial_count == trial)
            trial_idx = np.where(of_trial)[0]

            reward_not_none = (vr_data.reward_type != Outcomes.NONE)
            reward_typed = vr_data[of_trial & reward_not_none]
            trial_type = int(vr_data[of_trial].trial_type.unique())
            trial_type_str = trial_type_lookup.get(trial_type).lower()

            if reward_typed.size == 0\
                and vr_data[of_trial\
                    & (vr_data.world_index == Worlds.WHITE)].size != 0:
                # Handle punishment case
                outcome = f"punished_{trial_type_str}"
            else:
                reward_type = int(reward_typed.reward_type.unique())
                outcome = _outcome_map.get(reward_type, "unknown")

                if reward_type == Outcomes.TRIGGERED:
                    outcome = f"{outcome}_{trial_type_str}"

            action_labels[trial_idx, 0] = getattr(ActionLabels, outcome, 0)

            if reward_type > Outcomes.NONE:
                valve_open_idx = vr_data.index.get_indexer([reward_typed.index[0]])
                valve_closed_idx = vr_data.index.get_indexer([reward_typed.index[-1]])
                action_labels[valve_open_idx, 1] += Events.valve_open
                action_labels[valve_closed_idx, 1] += Events.valve_closed

        return action_labels

    def _extract_action_labels(self, vr, vr_data):
        """
        Extract action labels from VR data and assign events and outcomes.
        """
        action_labels = np.zeros((vr_data.shape[0], 2), dtype=np.int32)

        # Map events
        action_labels = self._map_trial_events(action_labels, vr_data, vr)

        # Assign trial outcomes
        action_labels = self._assign_trial_outcomes(action_labels, vr_data, vr)

        # Add timestamps to action labels
        action_labels = np.column_stack((action_labels, vr_data.index.values))

        return action_labels
    '''
