"""
This module provides functions that operate on signal data.
"""


import time
from pathlib import Path

import multiprocessing as mp
from joblib import Parallel, delayed

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal
from scipy.signal.windows import gaussian
from scipy.ndimage import gaussian_filter1d, convolve1d

from pixels import ioutils, PixelsError


def decimate(array, from_hz, to_hz, ftype="fir"):
    """
    Downsample the signal after applying an anti-aliasing filter.
    Downsampling factor MUST be an integer, if not, call `resample`.

    Params
    ===
    array : ndarray, Series or similar
        The data to be resampled.

    from_hz : int or float
        The starting frequency of the data.

    sample_rate : int or float, optional
        The resulting sample rate.

    ftype: str or dlit instance
        low pass filter type.
        Default: fir (finite impulse response) filter
    """
    if from_hz % to_hz == 0:
        factor = from_hz // to_hz
        output = scipy.signal.decimate(
            x=array,
            q=factor,
            ftype=ftype,
        )
    else:
        output = resample(array, from_hz, to_hz)

    return output


def resample(array, from_hz, to_hz, poly=True, padtype=None):
    """
    Resample an array from one sampling rate to another.

    Parameters
    ----------
    array : ndarray, Series or similar
        The data to be resampled.

    from_hz : int or float
        The starting frequency of the data.

    sample_rate : int or float, optional
        The resulting sample rate.

    poly : bool, choose the resample function.
        If True, use scipy.signal.resample_poly; if False, use scipy.signal.resample.
        Default is True.
        lfp downsampling only works if using scipy.signal.resample.

    Returns
    -------
    np.ndarray : Array of the same width of the input array, but altered height.

    """
    from_hz = float(from_hz)
    to_hz = float(to_hz)

    if from_hz == to_hz:
        return array

    if from_hz > to_hz:
        up = 1
        factor = from_hz / to_hz
        down = factor
        while not down.is_integer():
            down += factor
            up += 1

    elif from_hz < to_hz:
        factor = to_hz / from_hz
        up = factor
        down = 1
        while not up.is_integer():
            up += factor
            down += 1

    new_data = []
    if array.ndim == 1:
        cols = 1
        array = array.reshape((-1, cols))
    else:
        cols = array.shape[1]

    # resample_poly preallocates an entire new array of float64 values, so to prevent
    # MemoryErrors we will run it with 5GB chunks that cover a subset of channels.
    if isinstance(array, pd.DataFrame):
        size_bytes = array.iloc[0, 0].dtype.itemsize * array.size
    elif isinstance(array, np.array):
        size_bytes = array[..., 0].dtype.itemsize * array.size
    chunks = int(np.ceil(size_bytes / 5368709120))
    chunk_size = int(np.ceil(cols / chunks))

    # get index & chunk data
    #chunk_indices = [(i, min(i + chunk_size, cols)) for i in range(0, cols, chunk_size)]
    #chunks_data = [array[:, start:end] for start, end in chunk_indices]
    # get number of processes/jobs
    #n_processes = mp.cpu_count() - 2
    # initiate a mp pool
    #pool = mp.Pool(n_processes)
    ## does resample for each chunk
    #results = pool.starmap(
    #    _resample_chunk, 
    #    [(chunk, up, down, poly, padtype) for chunk in chunks_data],
    #)

    ## stop adding task to pool
    #pool.close()
    ## wait till all tasks in pool completed
    #pool.join()
    #results = Parallel(n_jobs=-1)(
    #    delayed(_resample_chunk)(chunk, up, down, poly, padtype) for chunk in chunks_data
    #)
    #print(">> mapped chunk data to pool...")

    if chunks > 1:
        print(f"    0%", end="\r")
    current = 0
    for _ in range(chunks):
        chunk_data = array[:, current:min(current + chunk_size, cols)]
        if poly:
            # matt's old poly func
            result = scipy.signal.resample_poly(
                chunk_data, up, down, axis=0, padtype=padtype or 'minimum',
            )
        else:
            # get number of samples given the new sample rate
            samp_num=int(np.ceil(
                array.shape[0]
                * (up / down))
            )
            result = scipy.signal.resample(
                chunk_data,
                samp_num,
                axis=0,
            )
        new_data.append(result)
        current += chunk_size
        print(f"    {100 * current / cols:.1f}%", end="\r")
    #new_data = np.concatenate(results, axis=1).squeeze()
    
    return  np.concatenate(new_data, axis=1).squeeze()#.astype(np.int16)
    #return  new_data


def _resample_chunk(chunk_data, up, down, poly, padtype):
    if poly:
        result = scipy.signal.resample_poly(
            chunk_data, up, down, axis=0, padtype=padtype or 'minimum',
        )
    else:
        samp_num = int(np.ceil(
            chunk_data.shape[0] * (up / down)
        ))
        result = scipy.signal.resample(
            chunk_data, samp_num, axis=0
        )
    return result

def binarise(data):
    """
    This normalises an array to between 0 and 1 and then makes all values below 0.5
    equal to 0 and all values above 0.5 to 1. The array is returned as np.int8 to save
    some memory when using large datasets.

    Parameters
    ----------
    data : numpy.ndarray or pandas.DataFrame
        If the data is a dataframe then each column will individually be binarised.

    """
    if isinstance(data, pd.DataFrame):
        for column in data.columns:
            data[column] = _binarise_real(data[column])
    else:
        data = _binarise_real(data)

    return data


def _binarise_real(data):
    data = data - min(data)
    data = data / max(data)
    return (data > 0.5).astype(np.int8)


def find_sync_lag(array1, array2, plot=False):
    """
    Find the lag between two arrays where they have the greatest number of the same
    values. This functions assumes that the lag is less than 300,000 points.

    Parameters
    ----------
    array1 : array, Series or similar
        The first array. A positive result indicates that THIS ARRAY HAS LEADING DATA
        NOT PRESENT IN THE SECOND ARRAY. e.g. if lag == 5 then array2 starts on the 5th
        index of array1.

    array2 : array, Series or similar
        The array to look for in the first.

    plot : string, optional
        False (default), or a path specifying where to save a png of the best match.  If
        it already exists, it will be suffixed with the time.

    Returns
    -------
    int : The lag between the starts of the two arrays. A positive number indicates that
        the first array begins earlier than the second.

    float : The percentage of values that were identical between the two arrays when
        aligned with the calculated lag, for the length compared.

    """
    length = min(len(array1), len(array2)) // 2
    length = min(length, 300000)

    array1 = array1.squeeze()
    array2 = array2.squeeze()

    sync_pos = []
    for i in range(length):
        # finds how many values are the same in array1 as in array2 till given length
        matches = np.count_nonzero(array1[i:i + length] == array2[:length])
        # append the percentage of match given length
        sync_pos.append(100 * matches / length)
    # take the highest percentage during checks as the match
    match_pos = max(sync_pos)
    # find index where lag started in array1
    lag_pos = sync_pos.index(match_pos)

    sync_neg = []
    for i in range(length):
        # finds how many values are the same in array2 as in array1 till given length
        matches = np.count_nonzero(array2[i:i + length] == array1[:length])
        # append the percentage of match given length
        sync_neg.append(100 * matches / length)
    match_neg = max(sync_neg)
    lag_neg = sync_neg.index(match_neg)

    if match_pos > match_neg:
        lag = lag_pos
        match = match_pos
    else:
        lag = - lag_neg
        match = match_neg

    if plot:
        plot = Path(plot)
        if plot.exists():
            plot = plot.with_name(plot.stem + '_' + time.strftime('%y%m%d-%H%M%S') + '.png')
        fig, axes = plt.subplots(nrows=2, ncols=1)
        plot_length = min(length, 30000)
        if lag >= 0:
            axes[0].plot(array1[lag:lag + plot_length])
            axes[1].plot(array2[:plot_length])
        else:
            axes[0].plot(array1[:plot_length])
            axes[1].plot(array2[-lag:-lag + plot_length])
        fig.savefig(plot)
        print(f"    Sync plot saved to:\n    {plot}")

    return lag, match


def median_subtraction(data, axis=0):
    """
    Perform a median subtraction on some data.

    Parameters
    ----------
    data : numpy.ndarray
        The data to perform the subtraction on.

    axis : int
        The axis from which to get the median for subtraction.

    """
    return data - np.median(data, axis=axis, keepdims=True)


def _convolve_worker(shm_kernal, shm_times, sample_rate):
    # TODO sep 22 2025:
    # CONTINUE HERE!
    # attach to shared memory
    shm = shared_memory.SharedMemory(name=shm_kernal)

    convolved = convolve1d(
        input=times.values,
        weights=n_kernel,
        output=np.float32,
        mode="nearest",
        axis=0,
    ) * sample_rate # rescale it to second

    return None


def convolve_spike_trains(times, sigma=100, size=10, sample_rate=1000):
    """
    Convolve spike times data with 1D gaussian kernel to get spike rate.

    Parameters
    -------
    times : pandas.DataFrame, time x units
        Spike bool of units at each time point of a trial.
        Dtype needs to be float, otherwise convolved results will be all 0.

    sigma : float/int, optional
        Time in milliseconds of sigma of gaussian kernel to use.
        Default: 100 ms.

    size : float/int, optional
        Number of sigma for gaussian kernel to cover, i.e., size of the kernel
        Default: 10.

    """
    # get kernel size in ms
    kernel_size = int(sigma * size)
    # get gaussian kernel
    kernel = gaussian(kernel_size, std=sigma)
    # normalise kernel to ensure that the total area under the Gaussian is 1
    n_kernel = kernel / np.sum(kernel)

    # TODO sept 19 2025:
    # implement multiprocessing?
    if isinstance(times, pd.DataFrame):
        # convolve with gaussian
        convolved = convolve1d(
            input=times.values,
            weights=n_kernel,
            output=np.float32,
            mode='nearest',
            axis=0,
        ) * sample_rate # rescale it to second

        output = pd.DataFrame(
            convolved,
            columns=times.columns,
            index=times.index,
        )

    elif isinstance(times, np.ndarray):
        # convolve with gaussian
        output = convolve1d(
            input=times,
            weights=n_kernel,
            output=np.float32,
            mode='nearest',
            axis=0,
        ) * sample_rate # rescale it to second

    return output


def convolve(times, duration, sigma=None):
    """
    Create a continuous signal from a set of spike times in milliseconds and convolve
    into a smooth firing rate signal.

    Parameters
    -------
    times : pandas.DataFrame
        Spike times in milliseconds to use to generate signal. Each column should
        correspond to one unit's spike times.

    duration : int
        Number of milliseconds of final signal.

    sigma : float/int, optional
        Time in milliseconds of sigma of gaussian kernel to use. Default is 50 ms.

    """
    if sigma is None:
        sigma = 50

    # turn into array of 0s and 1s
    times_arr = np.zeros((duration.astype(int), len(times.columns)))
    for i, unit in enumerate(times):
        u_times = times[unit] + duration / 2
        u_times = u_times[~np.isnan(u_times)].astype(int)
        try:
            times_arr[u_times, i] = 1
        except IndexError:
            # sometimes the conversion to np.int can make spikes round to the index just
            # outside of the range
            u_times.values[-1] = u_times.values[-1] - 1

    # convolve and re-scale so units are per second
    convolved = gaussian_filter1d(times_arr, sigma, axis=0) * 1000
    df = pd.DataFrame(convolved, columns=times.columns)

    return df


def motion_index(video, rois):
    """
    Calculating motion indexes from a video for a set of ROIs.

    Parameters
    -------
    video : str
        Path to a video.

    rois : dict, as saved by Behaviour.draw_motion_index_rois
        Regions of interest used to mask video when calculating MIs.

    """
    if not isinstance(video, str):
        video = video.as_posix()
    width, height, duration = ioutils.get_video_dimensions(video)
    mi = np.zeros((duration, len(rois)))

    # Create roi masks
    # height and width are in this order due to how frames are usually saved
    masks = np.zeros((height, width, len(rois)), dtype=np.uint8)

    for i, roi in enumerate(sorted(rois)):
        polygon = np.array(rois[roi]['vertices'], dtype=np.int32)
        mask = np.zeros((height, width, 1), dtype=np.uint8)
        # this complains when passed a view into another array for some reason
        cv2.fillConvexPoly(mask, polygon, (1,))
        np.copyto(masks[:, :, i], np.squeeze(mask))

    # Calculate motion indexes
    prev_frame = np.zeros((height, width, 1), dtype=np.uint8)
    masked = np.zeros(masks.shape, dtype=np.uint8)

    for i, frame in enumerate(ioutils.stream_video(video)):
        masked = masks[:, :] * frame[:, :, None] - prev_frame
        mi[i, :] = (masked * masked).sum(axis=0).sum(axis=0)

    # Normalise
    mi = mi / mi.max(axis=0)

    return mi


def freq_notch(x, fs, w0, axis=0, bw=4.0):
    """
    Use a notch filter that is a band-stop filter with a narrow bandwidth
    between 48 to 52Hz.
    It rejects a narrow frequency band and leaves the rest of the spectrum
    little changed.

    params
    ===
    x: array like, data to filter.

    fs: float or int, sampling frequency of x.

    w0: float or int, target frequency to notch (Hz).

    axis: int, axis of x to apply filter.

    bw: float or int, bandwidth of notch filter.

    return
    ===
    notched: array like, notch filtered x.
    """
    # convert to float
    x = x.astype(np.float32, copy=False)
    # set quality factor
    Q = w0 / bw
    # get numerator b & denominator a of IIR filter
    b, a = scipy.signal.iirnotch(w0, Q, fs=fs)
    # apply digital filter forward and backward
    notched = scipy.signal.filtfilt(b, a, x, axis=axis)

    return notched
