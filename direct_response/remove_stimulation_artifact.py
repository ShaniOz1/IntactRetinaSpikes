import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def ica_based_method(pulses):
    """result is 3D numpy array: pulse num x channel x segment data"""
    artifacts_mat = []
    signals_mat = []

    for channel_ind in range(0, pulses.shape[1]):
        # print(f'Processing channel {channel_ind}')
        data = pulses[:, channel_ind, :].T
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data = (data - data_mean) / data_std

        # Preform ICA
        num_comp = 6
        ica = FastICA(n_components=num_comp, random_state=42, max_iter=100)
        ica.fit(data)
        # print(f'Number of iterations taken for convergence: {ica.n_iter_}')
        components = ica.transform(data)

        # For each component, calculate correlation to the template
        pp_list = []
        real_max_list = []
        for ind in range(0, num_comp):
            # fig=plt.figure()
            # plt.plot(components[:, ind] - 10 * ind)
            plt.plot((np.arange(0, len(components[:, 0])) / 20)-5, components[:, ind]-10*ind)
            largest_peak_index = np.argmax(np.abs(components[:, ind]))
            pp_list.append(largest_peak_index)

            comp_temp = components.copy()
            arr = np.arange(0, 6)
            arr = arr[~np.isin(arr, ind)]
            comp_temp[:, arr] = 0
            restored_comp = ica.inverse_transform(comp_temp) * data_std + data_mean
            real_max_list.append(abs(np.max(restored_comp)))

        # Divide components to artifact-related and signal-related
        pp_thresh = 160  # 1.4 msec gap
        art_related_comp = np.where((np.array(real_max_list) > 750) | (np.array(pp_list) < pp_thresh))
        signal_related_comp = np.setdiff1d(np.arange(0, num_comp), art_related_comp)

        signal_comp = components.copy()
        signal_comp[:, art_related_comp] = 0
        restored_signal = ica.inverse_transform(signal_comp)
        restored_signal = restored_signal * data_std + data_mean
        signals_mat.append(restored_signal)

        art_comp = components.copy()
        art_comp[:, signal_related_comp] = 0
        restored_artifact = ica.inverse_transform(art_comp)
        restored_artifact = restored_artifact * data_std + data_mean
        artifacts_mat.append(restored_artifact)

    signals_mat_3d = np.stack(signals_mat, axis=1).transpose(2, 1, 0)
    artifacts_mat_3d = np.stack(artifacts_mat, axis=1).transpose(2, 1, 0)

    ####################
    # plt.figure()
    # for p in np.arange(0, len(signals_mat_3d[:, 0, 0])):
    #     plt.plot(signals_mat_3d[p, 0, :], color='grey')

    # plt.figure()
    # for p in np.arange(0, len(artifacts_mat_3d[:, 0, 0])):
    #     plt.plot(artifacts_mat_3d[p, 0, :], color='k')

    return signals_mat_3d, artifacts_mat_3d

