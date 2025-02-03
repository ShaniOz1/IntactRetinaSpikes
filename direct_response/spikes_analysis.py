import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
from scipy.signal import find_peaks
from scipy.stats import kurtosis
from datetime import datetime


def get_spikes(obj):
    stimulation_ch_index = next((i for i, s in enumerate(obj.recording_channels) if str(obj.stimulation_channels) in s), -1)
    spikes_dict = {}
    for channel_ind in range(0, len(obj.recording_channels)):
        if channel_ind != stimulation_ch_index:
            current_channel_mat = obj.signals_mat_3d[:, channel_ind, :]
            pulse_ind_to_remove = []
            for pulse_ind in range(0, obj.signals_mat_3d.shape[0]):
                pulse = obj.signals_mat_3d[pulse_ind, channel_ind, :]
                if not is_spike(pulse, obj.sample_rate):
                    pulse_ind_to_remove.append(pulse_ind)
            current_channel_mat = np.delete(current_channel_mat, pulse_ind_to_remove, axis=0)
            if current_channel_mat.shape[0] > 10:
                spikes_dict[obj.recording_channels[channel_ind]] = current_channel_mat

    # Iterate over channels and remove mirror-like artifact
    ch_to_delete = []
    for key, value in spikes_dict.items():
        if is_mirror_artifact(value):
            ch_to_delete.append(key)
    for key in ch_to_delete:
        del spikes_dict[key]


    # import matplotlib.pyplot as plt
    # for ch in spikes_dict.values():
    #     plt.figure()
    #     for p in ch:
    #         plt.plot(p, 'grey')

    return spikes_dict


def is_spike(sig, fs):
    try:
        # Hard threshold
        hard_threshold_res = not(all(x > -50 for x in sig))

        # Spike width
        if hard_threshold_res is False:
            width_res = False
        else:
            spike_signal = -sig
            peaks, _ = scipy.signal.find_peaks(spike_signal)
            peak_properties = scipy.signal.peak_widths(spike_signal, peaks)
            largest_peak_index = peaks[np.argmax(peak_properties[1])]
            left_ind = scipy.signal.peak_widths(spike_signal, [largest_peak_index], rel_height=0.5)[2]
            right_ind = scipy.signal.peak_widths(spike_signal, [largest_peak_index], rel_height=0.5)[3]
            width_sec = 1/(fs/1000) * (right_ind - left_ind)
            width_res = width_sec > 0.4

            # Number of peaks
            peaks, _ = scipy.signal.find_peaks(spike_signal, height=0.7*spike_signal[largest_peak_index])
            peaks_num_res = False if len(peaks) > 2 else True

        return True if (hard_threshold_res and width_res and peaks_num_res) else False

    except Exception as e:
        print(e)


def is_mirror_artifact(sig_mat):
    # max_values = np.max(sig_mat, axis=0)
    # max_peaks, _ = find_peaks(max_values, height=0)
    # sorted_peaks_indices_max = np.argsort(max_values[max_peaks])[::-1][:5]
    # highest_peaks_indices_max = max_peaks[sorted_peaks_indices_max]
    #
    # # max_values_diff = np.diff(max_values)
    # min_values = -np.min(sig_mat, axis=0)
    # min_peaks, _ = find_peaks(min_values, height=0)
    # sorted_peaks_indices_min = np.argsort(min_values[min_peaks])[::-1][:5]
    # highest_peaks_indices_min = max_peaks[sorted_peaks_indices_min]
    # intersection = np.intersect1d(highest_peaks_indices_max, highest_peaks_indices_min)

    # min_values_diff = np.diff(min_values)
    # # diff = abs(np.mean(max_values_diff - min_values_diff))
    # mean_values = np.mean(sig_mat, axis=0)
    # distance = abs(np.sum((max_values - mean_values) + (min_values - mean_values)))
    std = np.sum(np.std(sig_mat, axis=0))
    std_m = np.mean(np.std(sig_mat, axis=0))

    # envelope_max_kurtosis = kurtosis(max_values)


    #
    # widths = np.arange(10, 35, 5)
    # detected_spikes = {}
    # for pulse in sig_mat:
    #     for width in widths:
    #         smoothed_signal = gaussian_filter1d(pulse, sigma=10)
    #         peak_indices = np.where(np.diff(np.sign(np.diff(smoothed_signal))) < 0)[0] + 1
    #         detected_spikes[width] = peak_indices

    # import matplotlib.pyplot as plt
    # for r in range(0, sig_mat.shape[0]):
    #     plt.plot(sig_mat[r, :])
    # plt.title(f'{envelope_max_kurtosis}')
    # plt.savefig(f'kur{datetime.now().strftime("%H-%M-%S.%f")}.png')

    return True if std_m < 10 else False



def find_best_width_for_spike(pulse):
    widths = np.arange(5, 30, 5)
    best_width = None
    max_similarity = -np.inf

    for width in widths:
        smoothed_signal = gaussian_filter1d(pulse, sigma=width)
        similarity = pearsonr(pulse, smoothed_signal)[0]
        if similarity > max_similarity:
            max_similarity = similarity
            best_width = width
    return best_width