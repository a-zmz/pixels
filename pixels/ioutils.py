"""
This module contains helper functions for reading and writing files.
"""


import datetime
import glob
import json
import os
from pathlib import Path
from tempfile import gettempdir

import cv2
import ffmpeg
import numpy as np
import pandas as pd
from nptdms import TdmsFile

from pixels.error import PixelsError
from pixels.configs import *


def get_data_files(data_dir, session_name):
    """
    Get the file names of raw data for a session.

    Parameters
    ----------
    data_dir : str
        The directory containing the data.

    session_name : str
        The name of the session for which to get file names.

    Returns
    -------
    A nested dicts, where each dict corresponds to one session. Data is
    separated to two main categories: pixels and behaviour.

    In `pixels`, data is separated by their stream id, this is to allow:
        - easy concatenation of recordings files from the same probe, i.e.,
          stream id,
        - different numbers of pixels recordings and behaviour recordings,

    {session_name:{
        "pixels":{
            "imec0":{
                "ap_raw": [PosixPath("name.bin")],
                "ap_meta": [PosixPath("name.meta")],
                "preprocessed": spikeinterface recording obj,
                "ap_extracted": spikeinterface recording obj,
                "ap_whitened": spikeinterface recording obj,
                "lfp_extracted": spikeinterface recording obj,
                "surface_depth": PosixPath("name.yaml"),
                "sorting_analyser": PosixPath("name.zarr"),
            },
            "imecN":{
            },
        },
        "behaviour":{
            "vr": PosixPath("name.h5"),
            "action_labels": PosixPath("name.npz"),
        },
    }
    """
    if session_name != data_dir.stem:
        data_dir = list(data_dir.glob(f"{session_name}*"))[0]

    files = {}

    ap_raw = sorted(glob.glob(f"{data_dir}/{session_name}_g[0-9]_t0.imec[0-9].ap.bin*"))
    ap_meta = sorted(glob.glob(f"{data_dir}/{session_name}_g[0-9]_t0.imec[0-9].ap.meta*"))

    if not ap_raw:
        raise PixelsError(f"{session_name}: could not find raw AP data file.")
    if not ap_meta:
        raise PixelsError(f"{session_name}: could not find raw AP metadata file.")

    pupil_raw = sorted(glob.glob(f"{data_dir}/behaviour/pupil_cam/*.avi*"))

    behaviour = {
        "vr_synched": [],
        "action_labels": [],
        "pupil_raw": pupil_raw,
    }

    pixels = {}
    for r, rec in enumerate(ap_raw):
        stream_id = rec[-12:-4]
        probe_id = stream_id[:-3]
        # separate recordings by their stream ids
        if stream_id not in pixels:
            pixels[stream_id] = {
                "ap_raw": [], # there could be mutliple, thus list
                "ap_meta": [],
                "si_rec": None, # there could be only one, thus None
                "preprocessed": None,
                "ap_extracted": None,
                "ap_whitened": None,
                "lfp_extracted": None,
                "CatGT_ap_data": [],
                "CatGT_ap_meta": [],
            }

        base_name = original_name(rec)
        pixels[stream_id]["ap_raw"].append(base_name)
        pixels[stream_id]["ap_meta"].append(original_name(ap_meta[r]))

        behaviour["vr_synched"].append(base_name.with_name(
            f"{session_name}_{probe_id}_vr_synched.h5"
        ))
        behaviour["action_labels"].append(base_name.with_name(
            f"action_labels_{probe_id}.npz"
        ))

        # >>> spikeinterface cache >>>
        # extracted & motion corrected ap stream, 300Hz+
        pixels[stream_id]["ap_motion_corrected"] = base_name.with_name(
            f"{base_name.stem}.mcd.zarr"
        )
        # extracted & motion corrected lfp stream, 500Hz-
        pixels[stream_id]["lfp_motion_corrected"] = base_name.with_name(
            f"{base_name.stem[:-3]}.lf.mcd.zarr"
        )
        pixels[stream_id]["detected_peaks"] = base_name.with_name(
            f"{base_name.stem}_detected_peaks.h5"
        )
        pixels[stream_id]["sorting_analyser"] = base_name.parent/\
            f"sorted_stream_{probe_id[-1]}/curated_sa.zarr"
        # <<< spikeinterface cache <<<

        # depth info of probe
        pixels[stream_id]["surface_depth"] = base_name.with_name(
                f"{session_name}_{probe_id}_surface_depth.yaml"
        )
        pixels[stream_id]["clustered_channels"] = base_name.with_name(
            f"{session_name}_{stream_id}_channel_clustering_results.h5"
        )

        # TODO mar 5 2025:
        # maybe do NOT put shuffled data in here, cuz there will be different
        # trial conditions, better to cache them???

        # shuffled response for each unit, in light & dark conditions, to get
        # the chance
        # memmaps for temporary storage
        pixels[stream_id]["spiked_shuffled_memmap"] = base_name.with_name(
            f"{session_name}_{probe_id}_spiked_shuffled.bin"
        )
        pixels[stream_id]["fr_shuffled_memmap"] = base_name.with_name(
                f"{session_name}_{probe_id}_fr_shuffled.bin"
        )
        pixels[stream_id]["shuffled_shape"] = base_name.with_name(
            f"{session_name}_{probe_id}_shuffled_shape.json"
        )
        pixels[stream_id]["shuffled_index"] = base_name.with_name(
            f"{session_name}_{probe_id}_shuffled_index.h5"
        )
        pixels[stream_id]["shuffled_columns"] = base_name.with_name(
            f"{session_name}_{probe_id}_shuffled_columns.h5"
        )
        # .h5 files
        pixels[stream_id]["spiked_shuffled"] = base_name.with_name(
            f"{session_name}_{probe_id}_spiked_shuffled.h5"
        )
        pixels[stream_id]["fr_shuffled"] = base_name.with_name(
            f"{session_name}_{probe_id}_fr_shuffled.h5"
        )

        # noise in curated units
        pixels[stream_id]["noisy_units"] = base_name.with_name(
                f"{session_name}_{probe_id}_noisy_units.yaml"
        )

        # old catgt data
        pixels[stream_id]["CatGT_ap_data"].append(
            str(base_name).replace("t0", "tcat")
        )
        pixels[stream_id]["CatGT_ap_meta"].append(
            str(base_name).replace("t0", "tcat")
        )

        # histology
        mouse_id = session_name.split("_")[-1]
        pixels[stream_id]["depth_info"] = base_name.with_name(
            f"{mouse_id}_depth_info.yaml"
        )

        # identified faulty channels
        pixels[stream_id]["faulty_channels"] = base_name.with_name(
            f"{session_name}_{probe_id}_faulty_channels.yaml"
        )

        #pixels[stream_id]["spike_rate_processed"] = base_name.with_name(
        #    f"spike_rate_{stream_id}.h5"
        #)

    if pupil_raw:
        behaviour["pupil_processed"] = []
        behaviour["motion_index"] = []
        behaviour["motion_tracking"] = []
        for r, rec in enumerate(pupil_raw):
            behaviour["pupil_processed"].append(base_name.with_name(
                session_name + "_pupil_processed.h5"
            ))
            behaviour["motion_index"] = base_name.with_name(
                session_name + "_motion_index.npz"
            )
            behaviour["motion_tracking"] = base_name.with_name(
                session_name + "_motion_tracking.h5"
            )

    files = {
        "pixels": pixels,
        "behaviour": behaviour,
    }

    return files


def original_name(path):
    """
    Get the original name of a file, uncompressed, as a pathlib.Path.
    """
    name = os.path.basename(path)
    if name.endswith(".tar.gz"):
        name = name[:-7]
    return Path(name)


def read_meta(path):
    """
    Read metadata from a .meta file.

    Parameters
    ----------
    path : pathlib.Path
        Path to the meta file to be read.

    Returns
    -------
    dict : A dictionary containing the metadata from the specified file.

    """
    metadata = {}
    for entry in path.read_text().split("\n"):
        if entry:
            key, value = entry.split("=")
            metadata[key] = value
    return metadata


def read_bin(path, num_chans, channel=None):
    """
    Read data from a bin file.

    Parameters
    ----------
    path : str
        Path to the bin file to be read.

    num_chans : int
        The number of channels of data present in the file.

    channel : int or slice, optional
        The channel to read. If None (default), all channels are read.

    Returns
    -------
    numpy.memmap array : A 2D memory-mapped array containing containing the binary
        file"s data.

    """
    if not isinstance(num_chans, int):
        num_chans = int(num_chans)

    mapping = np.memmap(path, np.int16, mode="r").reshape((-1, num_chans))

    if channel is not None:
        mapping = mapping[:, channel]

    return mapping


def read_tdms(path, groups=None):
    """
    Read data from a TDMS file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the TDMS file to be read.

    groups : list of strs (optional)
        Names of groups stored inside the TDMS file that should be loaded. By default,
        all groups are loaded, so specifying the groups you want explicitly can avoid
        loading the entire file from disk.

    Returns
    -------
    pandas.DataFrame : A dataframe containing the data from the TDMS file.

    """
    with TdmsFile.read(path, memmap_dir=gettempdir()) as tdms_file:
        if groups is None:
            df = tdms_file.as_dataframe()
        else:
            # TODO: Use TdmsFile.open instead of read, and only load desired groups
            data = []
            for group in groups:
                channel = tdms_file[group].channels()[0]
                group_data = tdms_file[group].as_dataframe()
                group_data = group_data.rename(columns={channel.name: channel.path})
                data.append(group_data)
            df = pd.concat(data, axis=1)
    return df


def save_ndarray_as_video(video, path, frame_rate, dims=None):
    """
    Save a numpy.ndarray as video file.

    Parameters
    ----------
    video : numpy.ndarray, or generator
        Video data to save to file. It"s dimensions should be (duration, height, width)
        and data should be of uint8 type. The file extension determines the resultant
        file type. Alternatively, this can be a generator that yields frames of this
        description, in which case "dims" must also be passed.

    path : string / pathlib.Path object
        File to which the video will be saved.

    frame_rate : int
        The frame rate of the output video.

    dims : (int, int)
        (height, width) of video. This is only needed if "video" is a generator that
        yields frames, as then the shape cannot be taken from it directly.

    """
    if isinstance(video, np.ndarray):
        _, height, width = video.shape
    else:
        height, width = dims

    path = Path(path)

    process = (
        ffmpeg
        .input("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{width}x{height}", r=frame_rate)
        .output(path.as_posix(), pix_fmt="yuv420p", r=frame_rate, crf=0, vcodec="libx264")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for frame in video:
        if not isinstance(frame, list):
            # We can accept a 3D array as a list of 3 2D arrays, or just one 2D array
            frame = [frame, frame, frame]
        process.stdin.write(
            np.stack(frame, axis=2)
            .astype(np.uint8)
            .tobytes()
        )

    process.stdin.close()
    process.wait()
    if not path.exists():
        raise PixelsError(f"Video creation failed: {path}")


def read_hdf5(path, key="df"):
    """
    Read a dataframe from a h5 file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the h5 file to read.

    Returns
    -------
    pandas.DataFrame : The dataframe stored within the hdf5 file under the name "df".

    """
    df = pd.read_hdf(
        path_or_buf=path,
        key=key,
    )
    return df


def write_hdf5(path, df, key="df", mode="w", format="fixed"):
    """
    Write a dataframe to an h5 file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the h5 file to write to.

    df : pd.DataFrame
        Dataframe to save to h5.

    key : str
        identifier for the group in the store.
        Default: "df".

    mode : str
        mode to open file.
        Default: "w" write.
        Options:
            "a": append, if file does not exists it is created.
            "r+": similar to "a" but file must exists.
    """
    df.to_hdf(
        path_or_buf=path,
        key=key,
        mode=mode,
        format=format,
        complevel=9,
        #complib="bzip2", # slower but higher compression ratio
        complib="blosc:lz4hc",
    )
    
    print("HDF5 saved to", path)

    return


def get_sessions(mouse_ids, data_dir, meta_dir, session_date_fmt, of_date=None):
    """
    Get a list of recording sessions for the specified mice, excluding those whose
    metadata contain "'exclude' = True".

    Parameters
    ----------
    mouse_ids : list of strs
        List of mouse IDs.

    data_dir : str
        The path to the folder containing data for all sessions. This is searched for
        available sessions.

    meta_dir : str or None
        If not None, the path to the folder containing training metadata JSON files. If
        None, no metadata is collected.

    session_date_fmt : str
        String format used to extract the date from folder names.

    Returns
    -------
    list of dicts : Dictionaries containing the values that can be used to create new
        Behaviour subclass instances.

    """
    if not isinstance(mouse_ids, (list, tuple, set)):
        mouse_ids = [mouse_ids]
    sessions = {}
    raw_dir = data_dir / "raw"

    for mouse in mouse_ids:
        mouse_sessions = sorted(list(raw_dir.glob(f"*{mouse}")))

        if not mouse_sessions:
            print(f"Found no sessions for: {mouse}")
            continue

        # allows different session date formats
        session_dates = sorted([
            datetime.datetime.strptime(s.stem.split("_")[0], session_date_fmt)
            for s in mouse_sessions
        ])

        if of_date is not None:
            if isinstance(of_date, str):
                date_list = [of_date]
            else:
                date_list = of_date

            date_sessions = []
            for date in date_list:
                date_struct = datetime.datetime.strptime(date, session_date_fmt)
                date_sessions.append(mouse_sessions[session_dates.index(date_struct)])
                logging.info(
                    f"\n> Getting one session from {mouse} on "
                    f"{datetime.datetime.strftime(date_struct, '%Y %B %d')}."
                )
            mouse_sessions = date_sessions

        if not meta_dir:
            # Do not collect metadata
            for session in mouse_sessions:
                name = session.stem
                if name not in sessions:
                    sessions[name] = []
                sessions[name].append(dict(
                    metadata=None,
                    data_dir=data_dir,
                ))
            continue
        else:
            meta_file = meta_dir / (mouse + ".json")
            with meta_file.open() as fd:
                mouse_meta = json.load(fd)

            if len(session_dates) != len(set(session_dates)):
                raise PixelsError(f"{mouse}: Data folder dates must be unique.")

            included_sessions = set()
            for i, session in enumerate(mouse_meta):
                try:
                    meta_date = datetime.datetime.strptime(session["date"], session_date_fmt)
                except TypeError:
                    raise PixelsError(f"{mouse} session #{i}: 'date' not found in JSON.")

                for index, ses_date in enumerate(session_dates):
                    if ses_date == meta_date and not session.get("exclude", False):
                        name = mouse_sessions[index].stem
                        if name not in sessions:
                            sessions[name] = []
                        sessions[name].append(dict(
                            metadata=session,
                            data_dir=data_dir,
                        ))
                        included_sessions.add(name)

            if included_sessions:
                print(f"{mouse} has {len(included_sessions)} sessions:", ", ".join(included_sessions))
            else:
                print(f"No session dates match between folders and metadata for: {mouse}")

    return sessions


def tdms_parse_timestamps(metadata):
    """Extract timestamps from video metadata."""
    ts_high = np.uint64(metadata["/'keys'/'IMAQdxTimestampHigh'"])
    ts_low = np.uint64(metadata["/'keys'/'IMAQdxTimestampLow'"])
    stamps = ts_low + np.left_shift(ts_high, 32)
    return stamps / 1000000


def _parse_tdms_metadata(meta_path):
    meta = read_tdms(meta_path)

    stamps = tdms_parse_timestamps(meta)
    rate = round(np.median(np.diff(stamps)))
    print(f"    Frame rate is {rate} ms per frame, {1000/rate} fps")

    # Indexes of the dropped frames
    if "/'frames'/'ind_skipped'" in meta:
        # We add one here to account for 1-based indexing
        # (I think. Compare with where actual_heights == 0)
        skipped = meta["/'frames'/'ind_skipped'"].dropna().size
        print(f"    Warning: video has skipped {skipped} frames.")
    else:
        skipped = 0

    actual_heights = meta["/'keys'/'IMAQdxActualHeight'"]
    height = int(actual_heights.max())  # Largest height is presumably the real one
    # The number of points with heights==0 should match skipped
    remainder = skipped - actual_heights[actual_heights != height].size
    duration = actual_heights.size - remainder
    fps = 1000 / rate

    return fps, height, duration


def load_tdms_video(path, meta_path, frame=None):
    """
    Calculate the 3 dimensions for a given video from TDMS metadata and reshape the
    video to these dimensions.

    Parameters
    ----------
    path : pathlib.Path
        File path to TDMS video file.

    meta_path : pathlib.Path
        File path to TDMS file containing metadata about the video.

    frame : int, optional
        Read this one single frame rather than them all.

    """
    fps, height, duration = _parse_tdms_metadata(meta_path)

    if frame is None:
        video = read_tdms(path)
        width = int(video.size / (duration * height))
        return video.values.reshape(duration, height, width), fps

    with TdmsFile.open(path) as tdms_file:
        group = tdms_file.groups()[0]
        channel = group.channels()[0]
        width = int(len(channel) / (duration * height))
        length = width * height
        start = frame * length
        video = channel[start : start + length]
        return video.reshape(height, width), fps


def tdms_to_video(tdms_path, meta_path, output_path):
    """
    Convert a TDMS video to a video file. This streams data from TDMS to the saved video
    in a way that never loads all data into memory, so works well on huge videos.

    Parameters
    ----------
    tdms_path : pathlib.Path
        File path to TDMS video file.

    meta_path : pathlib.Path
        File path to TDMS file containing metadata about the video.

    output_path : pathlib.Path
        Save the video to this file. The video format used is taken from the file
        extension of this path.

    """
    fps, height, duration = _parse_tdms_metadata(meta_path)

    tdms_file = TdmsFile.open(tdms_path)
    group = tdms_file.groups()[0]
    channel = group.channels()[0]

    if height == 480:
        # Normally we get duration from _parse_tdms_metadata, but on the occasion where
        # the metadata file has not been saved for whatever reason - which has happened
        # at least one time - if we know the height is 480 we can assume the width is
        # 640 and calculate the duration from the video's size itself
        width = 640
        duration = len(channel) // (height * width)
    else:
        width = int(len(channel) / (duration * height))
    step = width * height

    process = (
        ffmpeg
        .input("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{width}x{height}", r=fps)
        .output(output_path.as_posix(), pix_fmt="yuv420p", r=fps, crf=20, vcodec="libx264")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for frame in range(duration):
        pixels = channel[frame * step: (frame + 1) * step].reshape((width, height))
        process.stdin.write(
            np.stack([pixels, pixels, pixels], axis=2)
            .astype(np.uint8)
            .tobytes()
        )

    process.stdin.close()
    process.wait()
    tdms_file.close()


def load_video_frame(path, frame):
    """
    Load a frame from a video into a numpy array.

    Parameters
    ----------
    path : str
        File path to a video file.

    frame : int
        0-based index of frame to load.

    """
    video = cv2.VideoCapture(path)

    retval = video.set(cv2.CAP_PROP_POS_FRAMES, frame)
    assert retval  # Check it worked fine

    retval, frame = video.read()
    assert retval  # Check it worked fine

    return frame


def load_video_frames(path, frames):
    """
    Load a consecutive sequence of frames from a video into a numpy array.

    Parameters
    ----------
    path : str
        File path to a video file.

    frame : Sequence
        Array/list/etc of 0-based indices of frames to load. This function only
        considers the first value and the length, it doesn't check the actual values of
        the remaining elements.

    """
    if not isinstance(path, str):
        path = path.as_posix()

    video = cv2.VideoCapture(path)

    retval = video.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    assert retval  # Check it worked fine

    return stream_video(video, length=len(frames))


def get_video_dimensions(path):
    """
    Get a tuple of (width, height, frames) for a video.

    Parameters
    ----------
    path : str
        File path to a video file.

    """
    video = cv2.VideoCapture(path)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    return width, height, frames


def get_video_fps(path):
    """
    Get the frame rate of a video.

    Parameters
    ----------
    path : str
        File path to a video file.

    """
    video = cv2.VideoCapture(path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    return fps


def stream_video(video, length=None):
    """
    Iterate over a video's frames.

    Parameters
    ----------
    path : str or cv2.VideoCapture
        File path to a video file to open, or already open VideoCapture instance.

    length : int
        Positive integer representing the number of frames to load.

    """
    if not isinstance(video, cv2.VideoCapture):
        if isinstance(video, Path):
            video = video.as_posix()
        video = cv2.VideoCapture(video)

    if length is not None:
        assert length > 0

    while True:
        _, pixels = video.read()
        if pixels is None:
            break

        yield pixels[:, :, 0]  # TODO: should be 1 channel

        if length is not None:
            length -= 1
            if length == 0:
                break

def reindex_by_longest(dfs, idx_names=None, col_names=None, level=0, sort=True,
                       return_format="array"):
    """
    params
    ===
    dfs: dict, dictionary with pandas dataframe as values.

    return_format: str, format of return value.
        "array": stacked np array.
        "dataframe": concatenated pandas dataframe.

    names: str or list of str, names for levels of concatenated dataframe.

    return
    ===
    np.array or pd.DataFrame.
    """
    if return_format == "array":
        # align all trials by index
        indices = list(set().union(
            *[df.index for df in dfs.values()])
        )
        # reindex by the longest
        reidx_dfs = {key: df.reindex(index=indices)
            for key, df in dfs.items()}
        # stack df values into np array
        # NOTE: this create multidimensional data, different from if return
        # format is df!
        output = np.stack(
            [df.values for df in reidx_dfs.values()],
            axis=-1,
        )

    elif return_format == "dataframe":
        if isinstance(dfs, dict):
            # stack dfs vertically
            stacked_df = pd.concat(dfs, axis=0)
            # set index name
            if idx_names:
                stacked_df.index.names = idx_names
            if col_names:
                stacked_df.columns.names = col_names
        elif isinstance(dfs, pd.DataFrame):
            stacked_df = dfs

        # unstack df at level
        output = stacked_df.unstack(level=level, sort=sort)
        del stacked_df

    return output

def is_nested_dict(d):
    """
    Returns True if at least one value in dictionary d is a dict.
    """
    return any(isinstance(v, dict) for v in d.values())


def save_index_to_frame(df, path):
    idx_df = df.index.to_frame(index=False)
    write_hdf5(
        path=path,
        df=idx_df,
        key="multiindex",
    )
    # NOTE: to reconstruct:
    # recs_idx = pd.MultiIndex.from_frame(idx_df)


def save_cols_to_frame(df, path):
    col_df = df.columns.to_frame(index=False)
    write_hdf5(
        path=path,
        df=col_df,
        key="cols",
    )
    # NOTE: to reconstruct:
    # df.columns = col_df.values


def get_aligned_data_across_sessions(trials, key, level_names):
    """
    Get aligned trials across sessions.

    params
    ===
    trials: nested dict, aligned trials from multiple sessions.
        keys: session_name -> stream_id -> "fr", "positions", "spiked"

    key: str, type of data to get.
        "fr": firing rate, longest trial time x (unit x trial)
        "spiked": spiked boolean, longest trial time x (unit x trial)
        "positions": trial positions, longest trial time x trial

    return
    ===
    df: pandas df, concatenated key data across sessions.
    """
    per_session = {}
    for s_name, s_data in trials.items():
        key_data = {}
        for stream_id, stream_data in s_data.items():
            key_data[stream_id] = stream_data[key]

        # concat at stream level
        per_session[s_name] = pd.concat(
            key_data,
            axis=1,
            names=level_names[1:],
        )

    # concat at session level
    df = pd.concat(
        per_session,
        axis=1,
        names=level_names,
    )

    # swap stream and session so that stream is the most outer level
    output = df.swaplevel("session", "stream", axis=1)

    return output
