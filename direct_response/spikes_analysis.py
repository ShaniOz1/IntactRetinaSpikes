import numpy as np
import scipy


def get_spikes(obj):
    stimulation_ch_index = next((i for i, s in enumerate(obj.recording_channels) if str(obj.stimulation_channels) in s), -1)
    spikes_dict = {}
    for channel_ind in range(0, len(obj.recording_channels)):
        if channel_ind != stimulation_ch_index:
            current_channel_mat = obj.signals_mat_3d[:, channel_ind, :]
            pulse_ind_to_remove = []
            for pulse_ind in range(0, obj.signals_mat_3d.shape[0]):
                pulse = obj.signals_mat_3d[pulse_ind, channel_ind, :]
                if not is_spike(pulse):
                    pulse_ind_to_remove.append(pulse_ind)
            current_channel_mat = np.delete(current_channel_mat, pulse_ind_to_remove, axis=0)
            if current_channel_mat.shape[0] > 10:
                spikes_dict[obj.recording_channels[channel_ind]] = current_channel_mat

    import matplotlib.pyplot as plt
    for ch in spikes_dict.values():
        plt.figure()
        for p in ch:
            plt.plot(p)
    return spikes_dict


def is_spike(sig):
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
        width_sec = 1/25 * (right_ind - left_ind)
        width_res = width_sec > 0.4

        # Number of peaks
        peaks, _ = scipy.signal.find_peaks(spike_signal, height=0.7*spike_signal[largest_peak_index])
        peaks_num_res = False if len(peaks) > 2 else True

    return True if (hard_threshold_res and width_res and peaks_num_res) else False
