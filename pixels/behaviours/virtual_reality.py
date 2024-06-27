"""
This module provides reach task specific operations.
"""

# NOTE: for event alignment, we align to the first timepoint when the event
# starts, and the last timepoint before the event ends, i.e., think of an event
# as a train of 0s and 1s, we align to the first 1s and the last 1s of a given
# event.

from __future__ import annotations

import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from vision_in_darkness.session import Outcome, World, Trial_Type

from pixels import Experiment, PixelsError
from pixels import signal, ioutils
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

    # triggered vr trials
    miss_light = 1 << 0
    miss_dark = 1 << 1
    triggered_light = 1 << 2
    triggered_dark = 1 << 3
    punished_light = 1 << 4
    punished_dark = 1 << 5

    # given reward
    default_light = 1 << 6
    auto_light = 1 << 7
    auto_dark = 1 << 8
    reinf_light = 1 << 9
    reinf_dark = 1 << 10

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
    gray_on = 1 << 1
    gray_off = 1 << 2
    light_on = 1 << 3
    light_off = 1 << 4
    dark_on = 1 << 5
    dark_off = 1 << 6
    punish_on = 1 << 7
    punish_off = 1 << 8
    session_end = 1 << 9
    # NOTE if use this event to mark trial ending, begin of the first trial
    # needs to be excluded
    trial_end = gray_on | punish_on

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
    valve_open = 1 << 18
    valve_closed = 1 << 19
    licked = 1 << 20
    #run_start = 1 << 12
    #run_stop = 1 << 13


# convert the trial data into Actions and Events
_action_map = {
    Outcome.ABORTED_DARK: "miss_dark",
    Outcome.ABORTED_LIGHT: "miss_light",
    Outcome.NONE: "miss",
    Outcome.TRIGGERED: "triggered",
    Outcome.AUTO_LIGHT: "auto_light",
    Outcome.DEFAULT: "default_light",
    Outcome.REINF_LIGHT: "reinf_light",
    Outcome.AUTO_DARK: "auto_dark",
    Outcome.REINF_DARK: "reinf_dark",
}



class VR(Behaviour):

    def _extract_action_labels(self, vr, vr_data):
        # create action label array for actions & events
        action_labels = np.zeros((vr_data.shape[0], 2), dtype=np.int32)

        # make sure position is not nan
        no_nan = (~vr_data.position_in_tunnel.isna())
        # define in gray
        in_gray = (vr_data.world_index == World.GRAY)
        # define in dark
        in_dark = (vr_data.world_index == World.DARK_5)\
                | (vr_data.world_index == World.DARK_2_5)\
                | (vr_data.world_index == World.DARK_FULL)\
                & no_nan
        # define in white
        in_white = (vr_data.world_index == World.WHITE)
        # define in tunnel
        in_tunnel = ~in_gray & ~in_white & no_nan
        # define in light
        in_light = (vr_data.world_index == World.TUNNEL)\
                & no_nan
        # define light & dark trials
        trial_light = (vr_data.trial_type == Trial_Type.LIGHT)
        trial_dark = (vr_data.trial_type == Trial_Type.DARK)

        print(">> Mapping vr event times...")

        # >>>> gray >>>>
        # get gray_on times, i.e., trial starts
        gray_idx = vr_data.world_index[in_gray].index
        # grays
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

        # >>>> white >>>>
        # get punish_on times
        punish_idx = vr_data.world_index[in_white].index
        # punishes
        punishes = np.where(punish_idx.diff() != 1)[0]

        # find time for first frame of punish
        punish_on_t = punish_idx[punishes]
        # find their index in vr data
        punish_on = vr_data.index.get_indexer(punish_on_t)
        action_labels[punish_on, 1] += Events.punish_on

        # find time for last frame of punish
        punish_off_t = np.append(punish_idx[punishes[1:] - 1], punish_idx[-1])
        # find their index in vr data
        punish_off = vr_data.index.get_indexer(punish_off_t)
        action_labels[punish_off, 1] += Events.punish_off
        # <<<< white <<<<

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

        # >>>> session ends >>>>
        session_end = vr_data.shape[0]
        action_labels[session_end, 1] += Events.session_end
        # <<<< session ends <<<<

        # >>>> licks >>>>
        licked_idx = np.where(vr_data.lick_count == 1)[0]
        action_labels[licked_idx, 1] += Events.licked
        # <<<< licks <<<<

        # TODO jun 27 positional events and valve events needs mapping
        assert 0

        print(">> Mapping vr action times...")
        # map trial types
        light_trials = vr_data[vr_data.trial_type == 0]
        dark_trials = vr_data[vr_data.trial_type == 1]

        # map pre-reward zone
        pre_zone = (vr_data.position_in_tunnel < vr.reward_zone_start)
        # get in reward zone index
        pre_zone_idx = vr_data[pre_zone].index
        # get reward type while in reward zone
        pre_reward_type = vr_data.reward_type.loc[pre_zone_idx]

        # default reward light trials
        default_light = pre_reward_type.index[pre_reward_type == Outcome.DEFAULT]
        default_light_id = light_trials.trial_count.loc[default_light].unique()
        for i in default_light_id:
            default_light_idx = np.where(vr_data.trial_count == i)[0]
            action_labels[default_light_idx, 0] = ActionLabels.default_light

        # punished
        punished_idx = vr_data.index[in_white]
        # punished light
        punished_light = light_trials.reindex(punished_idx).dropna()
        punished_light_id = punished_light.trial_count.unique()
        for i in punished_light_id:
            punished_light_idx = np.where(vr_data.trial_count == i)[0]
            action_labels[punished_light_idx, 0] = ActionLabels.punished_light

        # punished dark
        punished_dark = dark_trials.reindex(punished_idx).dropna()
        punished_dark_id = punished_dark.trial_count.unique()
        for i in punished_dark_id:
            punished_dark_idx = np.where(vr_data.trial_count == i)[0]
            action_labels[punished_dark_idx, 0] = ActionLabels.punished_dark

        # map reward zone
        in_zone = (vr_data.position_in_tunnel >= vr.reward_zone_start)\
            & (vr_data.position_in_tunnel <= vr.reward_zone_end)
        # get in reward zone index
        in_zone_idx = vr_data[in_zone].index
        # get reward type while in reward zone
        reward_type = vr_data.reward_type.loc[in_zone_idx]

        # triggered
        triggered_idx = reward_type.index[reward_type == Outcome.TRIGGERED]
        # triggered light trials
        triggered_light = light_trials.reindex(triggered_idx).dropna()
        triggered_light_id = triggered_light.trial_count.unique()
        for i in triggered_light_id:
            trig_light_idx = np.where(vr_data.trial_count == i)[0]
            action_labels[trig_light_idx, 0] = ActionLabels.triggered_light

        # automatically rewarded light trials
        auto_light = reward_type.index[reward_type == Outcome.AUTO_LIGHT]
        auto_light_id = light_trials.trial_count.loc[auto_light].unique()
        for i in auto_light_id:
            auto_light_idx = np.where(vr_data.trial_count == i)[0]
            action_labels[auto_light_idx, 0] = ActionLabels.auto_light

        # reinforcement reward light trials
        reinf_light = reward_type.index[reward_type == Outcome.REINF_LIGHT]
        reinf_light_id = light_trials.trial_count.loc[reinf_light].unique()
        for i in reinf_light_id:
            reinf_light_idx = np.where(vr_data.trial_count == i)[0]
            action_labels[reinf_light_idx, 0] = ActionLabels.reinf_light

        # triggered dark trials
        triggered_dark = dark_trials.reindex(triggered_idx).dropna()
        triggered_dark_id = triggered_dark.trial_count.unique()
        for i in triggered_dark_id:
            trig_dark_idx = np.where(vr_data.trial_count == i)[0]
            action_labels[trig_dark_idx, 0] = ActionLabels.triggered_dark

        # automatically rewarded dark trials
        auto_dark = reward_type.index[reward_type == Outcome.AUTO_DARK]
        auto_dark_id = dark_trials.trial_count.loc[auto_dark].unique()
        for i in auto_dark_id:
            auto_dark_idx = np.where(vr_data.trial_count == i)[0]
            action_labels[auto_dark_idx, 0] = ActionLabels.auto_dark

        # reinforcement reward dark trials
        reinf_dark = reward_type.index[reward_type == Outcome.REINF_DARK]
        reinf_dark_id = dark_trials.trial_count.loc[reinf_dark].unique()
        for i in reinf_dark_id:
            reinf_dark_idx = np.where(vr_data.trial_count == i)[0]
            action_labels[reinf_dark_idx, 0] = ActionLabels.reinf_dark

        # after reward zone before trial resets
        pass_zone = (vr_data.position_in_tunnel > vr.reward_zone_end)\
            & (vr_data.position_in_tunnel <= vr.tunnel_length)
        # get passed reward zone index
        pass_zone_idx = vr_data[pass_zone].index
        end_reward_type = vr_data.reward_type.loc[pass_zone_idx]

        # missed light trials
        miss_light = end_reward_type[end_reward_type ==
                                 Outcome.ABORTED_LIGHT].index.values
        miss_light_id = vr_data.trial_count.loc[miss_light].unique()
        for i in miss_light_id:
            miss_light_idx = np.where(vr_data.trial_count == i)[0]
            action_labels[miss_light_idx, 0] = ActionLabels.miss_light

        # missed dark trials
        miss_dark = end_reward_type[end_reward_type ==
                                Outcome.ABORTED_DARK].index.values
        miss_dark_id = vr_data.trial_count.loc[miss_dark].unique()
        for i in miss_dark_id:
            miss_dark_idx = np.where(vr_data.trial_count == i)[0]
            action_labels[miss_dark_idx, 0] = ActionLabels.miss_dark

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
