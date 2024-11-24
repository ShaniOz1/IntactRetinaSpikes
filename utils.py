import numpy as np
import scipy
import matplotlib.pyplot as plt


def indirect_response_indices(obj):
    blanking_win_msec = 15
    blanking_win_samples = int((blanking_win_msec / 1000) * obj.sample_rate)

    # first, detect threshold:
    copy_data = np.copy(obj.recording_data)
    copy_data = copy_data[:, 0: len(np.copy(obj.recording_data[0, :])) - int(obj.sample_rate)]
    indices = sorted(obj.stimulation_indexes)
    mask = np.ones(copy_data.shape[1], dtype=bool)
    for index in indices:
        start = int(max(0, index - 0.005 * obj.sample_rate))
        end = int(min(copy_data.shape[1], index + blanking_win_samples + 1))
        mask[start:end] = False
    dropped_data = copy_data[:, mask]

    blanked_data = np.copy(obj.recording_data)
    blanked_data = blanked_data[:, 0: len(np.copy(obj.recording_data[0, :])) - int(obj.sample_rate)]
    blanked_data[:, ~mask] = 0
    obj.blanked_data = blanked_data

    std = np.std(dropped_data, axis=1)
    threshold = 4 * std

    spikes_indices = []
    for ch_ind in range(len(obj.recording_channels)):
        channel_data = blanked_data[ch_ind, :]
        peaks, _ = scipy.signal.find_peaks(-channel_data, height=threshold[ch_ind], distance=obj.sample_rate/1000)
        spikes_indices.append(peaks)

    # for ch_ind in range(25):
        # plt.plot(obj.blanked_filtered_arrays[ch_ind, :] + ch_ind, color='k', linewidth=0.3)
        # plt.plot(blanked_data[ch_ind, :] + 100*ch_ind, color='k', linewidth=0.3, alpha=0.3)
        # plt.scatter(spikes_indices[ch_ind], np.zeros_like(spikes_indices[ch_ind]) + 100*ch_ind, color='r')
        # plt.scatter(spikes_indices[ch_ind], np.zeros_like(spikes_indices[ch_ind]) + 3000, color='r')

    max_len = max(len(arr) for arr in spikes_indices)
    padded_list = [np.pad(arr.astype(float), (0, max_len - len(arr)), constant_values=np.nan) for arr in spikes_indices]
    result_matrix = np.array(padded_list)

    # np.savetxt(f'{obj.output_folder_name}/indirect_spikes_indices_mat{obj.file_name}.csv', result_matrix,
    #            delimiter=',')

    return spikes_indices
