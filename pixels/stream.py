import numpy as np
import pandas as pd

import spikeinterface as si

from pixels import ioutils
from pixels import pixels_utils as xut
import pixels.signal_utils as signal
from pixels.configs import *
from pixels.decorators import cacheable

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

    def __repr__(self):
        return f"<Stream id = {self.stream_id}>"


    def sync_vr(self, vr):
        # get action labels
        action_labels = self.session.get_action_labels()[self.stream_num]
        if action_labels:
            logging.info(f"\n> {self.stream_id} from {self.session.name} is "
                         "already synched with vr, continue.")
        else:
            _sync_vr(self, vr)

        return None


    def _sync_vr(self, vr):
        # get spike data
        spike_data = self.session.find_file(
            name=self.files["ap_raw"][self.stream_num],
            copy=True,
        )

        # get vr data
        vr_session = vr.sessions[0]
        synched_vr_path = vr_session.cache_dir + "synched/" +\
                    vr_session.name + "_vr_synched.h5"

        try:
            synched_vr = file_utils.read_hdf5(synched_vr_path)
            logging.info("> synchronised vr loaded")
        except:
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

            synched_vr = vr.sync_streams(
                self.SAMPLE_RATE,
                pixels_vr_edges,
                pixels_idx,
            )[vr_session.name]

        # save to pixels processed dir
        file_utils.write_hdf5(
            self.session.processed /\
                self.behaviour_files['vr_synched'][self.stream_num],
            synched_vr,
        )

        # get action label dir
        action_labels_path = self.session.processed /\
            self.behaviour_files["action_labels"][self.stream_num]

        # extract and save action labels
        action_labels = self.session._extract_action_labels(
            vr_session,
            synched_vr,
        )
        np.savez_compressed(
            action_labels_path,
            outcome=action_labels[:, 0],
            events=action_labels[:, 1],
            timestamps=action_labels[:, 2],
        )
        logging.info(f"> Action labels saved to: {action_labels_path}.")

        return None


    def save_spike_chance(self, spiked, sigma):
        # TODO apr 21 2025:
        # do we put this func here or in stream.py??
        # save index and columns to reconstruct df for shuffled data
        assert 0
        ioutils.save_index_to_frame(
            df=spiked,
            path=shuffled_idx_path,
        )
        ioutils.save_cols_to_frame(
            df=spiked,
            path=shuffled_col_path,
        )

        # get chance data paths
        paths = {
            "spiked_memmap_path": self.interim /\
                stream_files["spiked_shuffled_memmap"],
             "fr_memmap_path": self.interim / stream_files["fr_shuffled_memmap"],
        }

        # save chance data
        xut.save_spike_chance(
            **paths,
            sigma=sigma,
            sample_rate=self.SAMPLE_RATE,
            spiked=spiked,
        )

        return None
