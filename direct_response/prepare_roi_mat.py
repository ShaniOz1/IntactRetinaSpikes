import numpy as np


def get_recorded_pulses(stim_ind, data, sample_rate, win_size):
    """win size in sec"""
    """result is 3D numpy array: pulse num x channel x segment data"""
    win_size = int(win_size/1000 * sample_rate)
    result = []
    if len(data.shape) == 1:
        data = np.vstack([data, data])
    for idx in stim_ind:
        start = int(max(0, idx - win_size * (1/3)))
        end = int(min(data.shape[1], idx + win_size * (2/3)))
        result.append(data[:, start:end])
    result = np.array(result)
    return result
