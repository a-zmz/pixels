"""
This module provides reach task specific operations.
"""

from __future__ import annotations

import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from vision_in_darkness.session import Outcome, World

from pixels import Experiment, PixelsError
from pixels import signal, ioutils
from pixels.behaviours import Behaviour

from common_utils import file_utils

SAMPLE_RATE = 1000

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
    light = miss_light | triggered_light | punished_light | default_light |
        auto_light | reinf_light
    dark = miss_dark | triggered_dark | punished_dark | auto_dark | reinf_dark
    rewarded_light = triggered_light | default_light | auto_light | reinf_light
    rewarded_dark = triggered_dark | auto_dark | reinf_dark
    given_light = default_light | auto_light | reinf_light
    given_dark =  auto_dark | reinf_dark


class Events:
    """
    Defines events that could happen during vr sessions.

    Events can be added on top of each other.
    """
    # vr events
    gray_on = 1 << 0
    gray_off = 1 << 1
    tunnel_on = 1 << 2
    tunnel_off = 1 << 3
    dark_on = 1 << 4
    dark_off = 1 << 5
    punish_on = 1 << 6
    punish_off = 1 << 7
    session_end = 1 << 8

    # positional events
    black = 1 << 9 # 0 - 60 cm
    wall = 1 << 10 # in between landmarks
    landmark1 = 1 << 11 # 110 - 130 cm
    landmark2 = 1 << 12 # 190 - 210 cm
    landmark3 = 1 << 13 # 270 - 290 cm
    landmark4 = 1 << 14 # 350 - 370 cm
    landmark5 = 1 << 15 # 430 - 450 cm
    reward_zone = 1 << 16 # 460 - 495 cm

    # sensors
    valve_open = 1 << 17
    valve_closed = 1 << 18
    licked = 1 << 19
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

    def _extract_action_labels(self, vr_data):

        # TEMPORARY load synced vr data
        cache_dir = "/home/amz/interim/behaviour_cache/temp100/trials/"
        vr_data = file_utils.load_pickle(
            cache_dir + "20231130_az_WDAN07_upsampled.pickle"
            #self.cache_dir + 'trials/' + self.name + '_upsampled.pickle'
        )

        # create action label array for actions & events
        action_label = np.zeros((vr_data.shape[0], 2), dtype=np.int32)

        print(">> Mapping vr event times...")

        # get gray_on times, i.e., trial starts
        gray_idx = vr_data.world_index[vr_data.world_index == World.GRAY].index
        # grays
        grays = np.where(gray_idx.diff() != 1)[0]

        # first frame of gray
        gray_on = gray_idx[grays].values
        action_labels[gray_on, 1] += Events.gray_on

        # last frame of gray
        gray_off = np.append(gray_idx[grays[1:] - 1], gray_idx[-1])
        action_labels[gray_off, 1] += Events.gray_off

        # get tunnel_on times, i.e., tunnel starts
        tunnel_idx = vr_data.world_index[vr_data.world_index == World.TUNNEL].index
        # tunnels
        tunnels = np.where(tunnel_idx.diff() != 1)[0]

        # tunnel starts is first frame after gray
        tunnel_on = gray_off + 1
        action_labels[tunnel_on, 1] += Events.tunnel_on

        # get usual tunnel_off times
        tunnel_off = np.append(tunnel_idx[tunnels[1:] - 1], tunnel_idx[-1])
        action_labels[tunnel_off, 1] += Events.tunnel_off

        # get dark on
        # define in dark
        in_dark = (vr_data.world_index == World.DARK_5)\
                |(vr_data.world_index == World.DARK_2_5)\
                |(vr_data.world_index == World.DARK_FULL)
        # get index in dark
        in_dark_idx = vr_data[in_dark].index
        # number of dark_on does not match with number of dark trials caused by
        # triggering punishment before dark
        # darks
        darks = np.where(in_dark_idx.diff() != 1)[0]

        # first frame of dark
        dark_on = in_dark_idx[darks].values
        action_labels[dark_on, 1] += Events.dark_on

        # last frame of dark
        dark_off = np.append(in_dark_idx[darks[1:] - 1], in_dark_idx[-1])
        action_labels[dark_off, 1] += Events.dark_off

        # map licks
        licked_idx = vr_data.lick_count[vr_data.lick_count == 1].index.values
        action_labels[licked_idx, 1] += Events.licked


        print(">> Mapping vr action times...")
        # map reward zone
        in_zone = (vr_data.position_in_tunnel >= self.reward_zone_start)\
            & (vr_data.position_in_tunnel <= self.reward_zone_end)
        # get in reward zone index
        in_zone_idx = vr_data[in_zone].index
        # get reward type while in reward zone
        reward_type = vr_data.reward_type.loc[in_zone_idx]

        # after reward zone before trial resets
        pass_zone = (vr_data.position_in_tunnel > self.reward_zone_end)\
            & (vr_data.position_in_tunnel <= self.tunnel_length)
        # get passed reward zone index
        pass_zone_idx = vr_data[pass_zone].index
        end_reward_type = vr_data.reward_type.loc[pass_zone_idx]

        # missed light trials
        miss_light = end_reward_type[end_reward_type ==
                                 Outcome.ABORTED_LIGHT].index.values
        # missed dark trials
        miss_dark = end_reward_type[end_reward_type ==
                                Outcome.ABORTED_DARK].index.values

        # TODO jun 7 2024 action labels not mapped


        return action_labels


    def _old_extract_action_labels(self, vr_data, action_labels, plot=False):

        for i, trial in enumerate(self.metadata[rec_num]["trials"]):
            side = _side_map[trial["spout"]]
            outcome = trial["outcome"]
            if outcome in _action_map:
                action = _action_map[trial["outcome"]]
                action_labels[led_onsets[i], 0] += getattr(ActionLabels, f"{action}_{side}")

        if plot:
            plt.clf()
            _, axes = plt.subplots(4, 1, sharex=True, sharey=True)
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
