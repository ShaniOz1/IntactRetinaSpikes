import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os


def plot_direct_spikes(obj):
    fig, axs = plt.subplots(3, 2, figsize=(10, 12))
    response_channels = [int(key[-3:]) for key in obj.spikes_dict.keys()]
    plot_prob_schem(axs[0, 0], [obj.stimulation_channels], response_channels)

    for (title, signals), ax in zip(obj.spikes_dict.items(), axs.flatten()[1:]):
        for row in signals:
            ax.plot((np.arange(0, signals.shape[1]) - 125) / 25, row, color='lightgrey', alpha=0.8)
        ax.plot((np.arange(0, signals.shape[1]) - 125) / 25, np.mean(signals, axis=0), 'k')

        ax.set_title(f'Ch - {title}')
        ax.set_xlabel('Time after stimulation [msec]')
        ax.set_ylabel('Amplitude [uV]')
        ax.set_ylim([-500, 500])
    plt.suptitle(f'{obj.file_name[:-3]}')
    plt.tight_layout()
    plt.savefig(rf'{obj.output_folder}/Direct_response.png')
    plt.close()






def plot_prob_schem(ax, red_indices, blue_indices, grey_indices=None):
    # Numbers for the inner and outer circles
    inner_numbers = [13, 2, 12, 3, 11, 4, 10, 5, 9, 6, 8, 7, 99, 99, 99, 99, 15, 0, 14, 1]
    outer_numbers = [25, 22, 26, 21, 27, 20, 28, 19, 29, 18, 30, 17, 31, 16, 99, 99, 99, 99, 24,23]

    # Plot concentric circles
    circle1 = plt.Circle((0, 0), 1, color='grey', alpha=0.3, fill=False, linewidth=20)
    circle2 = plt.Circle((0, 0), 2, color='grey', alpha=0.3, fill=False, linewidth=20)
    ax.add_artist(circle1)
    ax.add_artist(circle2)

    # Calculate points on the circles
    angles_inner = np.linspace(0, 2*np.pi, len(inner_numbers), endpoint=False)
    x1_inner = np.cos(angles_inner)
    y1_inner = np.sin(angles_inner)

    angles_outer = np.linspace(0, 2*np.pi, len(outer_numbers), endpoint=False)
    x1_outer = 2 * np.cos(angles_outer)
    y1_outer = 2 * np.sin(angles_outer)

    # Plot numbers on the inner circle
    for i, (px, py) in enumerate(zip(x1_inner, y1_inner)):
        if inner_numbers[i] == 99:
            continue
        if inner_numbers[i] in red_indices:
            color = 'red'
            label = 'Red'
        elif inner_numbers[i] in blue_indices:
            color = 'dodgerblue'
            label = 'dodgerblue'
        else:
            color = 'black'
            label = None
        ax.text(px, py, str(inner_numbers[i]), color=color, fontsize=8, ha='center', va='center')

    # Plot numbers on the outer circle
    for i, (px, py) in enumerate(zip(x1_outer, y1_outer)):
        if outer_numbers[i] == 99:
            continue
        if outer_numbers[i] in red_indices:
            color = 'red'
        elif outer_numbers[i] in blue_indices:
            color = 'dodgerblue'
        else:
            color = 'black'
        ax.text(px, py, str(outer_numbers[i]), color=color, fontsize=10, ha='center', va='center')


    # Set equal aspect ratio and limits
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)

    # Create custom legend
    red_patch = mpatches.Patch(color='red', label='Stimulation')
    blue_patch = mpatches.Patch(color='dodgerblue', label='Direct response')
    ax.legend(handles=[red_patch, blue_patch], loc='upper left', fontsize='x-small', ncol=2, bbox_to_anchor=(0, 1.15))

    # Remove x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])



def plot_spikes_amps_vs_time(obj):
    # mat = obj.signals_mat_3d[:, 22, :]
    # mat = obj.signals_mat_3d[:, 7, :]

    fig = plt.figure(figsize=(15, 4))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 1])

    # First row: Single plot spanning the entire width
    ax1 = fig.add_subplot(gs[0, 0:1])

    ch = 25
    mat = obj.signals_mat_3d[:, ch, :]
    min_vals = []
    for row in mat:
        min_vals.append(np.min(row))
        # plt.plot((np.arange(0, len(row))-125)/25, row/1000, color='grey', alpha=0.1)
        ax1.plot((np.arange(0, len(row))-125)/25, row/1000, color='k', linewidth=0.1)
    ax1.set_xlim(0, 10)
    avg = np.mean(mat, axis=0)
    ax1.plot((np.arange(0, len(row))-125)/25, avg/1000, color='k', linewidth=2)
    plt.ylim(-0.8, 0.4)
    # ax1.set_ylim(-0.95, 0.7)
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.set_xlabel('Time [s]', fontsize=14)
    ax1.set_ylabel('Amplitude [mV]', fontsize=14)

    ax2 = fig.add_subplot(gs[0, 1:4])

    # plt.figure(figsize=(8, 3))
    ax2.scatter(np.arange(0, len(min_vals)), np.array(min_vals) / 1000, color='k', marker='.', s=30)
    ax2.set_xlabel('# Pulse', fontsize=14)
    # ax2.set_ylabel('Amplitude [mV]')
    plt.ylim(-0.7, 0)
    # ax2.set_ylim(-0.95, 0)
    ax2.set_xlim(0, 100)
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)


    arr = np.abs(np.array(min_vals) / 1000)
    # plt.scatter(np.arange(0, len(min_vals)), arr, color='k', marker='.', s=30)

    # Extract the first and last 10 points
    first_10 = arr[:10]
    last_10 = arr[-10:]

    # Calculate averages
    avg_first_10 = np.mean(first_10)
    avg_last_10 = np.mean(last_10)

    # Calculate the decrease
    decrease = round(100 * (avg_first_10 - avg_last_10), 2)
    plt.title(f'decrease={decrease}_{obj.file_name}_ch{ch}', fontsize=8)
    plt.tight_layout()

    # window_size = 15
    # moving_avg = np.convolve(min_vals, np.ones(window_size) / window_size, mode='valid')
    # x_moving_avg = np.arange(len(moving_avg)) + window_size // 2
    # plt.plot(x_moving_avg, moving_avg / 1000, color='k', label=f'Moving Avg (win={window_size})')
    #




def plot_indirect_response(obj, spikes_indices):

    # 320us 40usdelay 5uA 10hz_100pulses_230528_131155
    # obj.recording_data = obj.recording_data[:, obj.sample_rate * 33:]
    # obj.stimulation_indexes = obj.stimulation_indexes[100:] - obj.sample_rate * 33
    # spikes_indices = [arr - obj.sample_rate * 33 for arr in spikes_indices]
    # spikes_indices = [arr[arr >= 0] for arr in spikes_indices]

    # 320us 40usdelay 1uA 10hz_100pulses_230528_131053
    # obj.recording_data = obj.recording_data[:, :obj.sample_rate * 15]
    # obj.stimulation_indexes = obj.stimulation_indexes[:100]
    # spikes_indices = [arr[arr <= 293721] for arr in spikes_indices]

    # softC_iv_2023_3_120us_100 times_10hz_20uA___230524_144755.rhs
    # obj.recording_data = obj.recording_data[:, :obj.sample_rate * 15]
    # obj.stimulation_indexes = obj.stimulation_indexes[:100]
    # spikes_indices = [arr[arr <= 366586] for arr in spikes_indices]

    # 320us 40usdelay 5uA 1hz_10pulses_230528_131609.rhs
    # obj.recording_data = obj.recording_data[:, :obj.sample_rate * 14]
    # obj.stimulation_indexes = obj.stimulation_indexes[:10]
    # spikes_indices = [arr[arr <= 295564] for arr in spikes_indices]
    #
    for i, row in enumerate(obj.recording_data):
        color = 'r' if i == 15 else 'k'
        plt.plot(row+300*i, color=color, linewidth=0.2)


    times = np.arange(0, len(obj.recording_data[0, :])) / obj.sample_rate
    selected_ch_ind = 15
    # Create a figure and GridSpec layout
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(4, 4, figure=fig, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1, 1, 1])

    # First row: Single plot spanning the entire width
    ax1 = fig.add_subplot(gs[0, :])  # This spans all columns in the first row

    # Highlight the row for the selected electrode
    # selected_ch_ind = [(i, s) for i, s in enumerate(obj.channel_labels) if selected_electrode in s][0][0]
    ax1.fill_between(times, 0.5 + selected_ch_ind, 1 + selected_ch_ind, color='lightgrey', alpha=0.5)
    # First subplot: raster plot of all channels
    for ch, ind_list in enumerate(spikes_indices):
        for ind in ind_list:
            ax1.vlines(times[ind], ymin=0.5 + ch, ymax=1 + ch, color='black', linewidth=0.7)
    ax1.scatter(times[obj.stimulation_indexes], len(obj.recording_channels) + np.zeros_like(obj.stimulation_indexes), color='#800020', marker='v', s=25)
    ax1.set_ylabel('# electrode')
    ax1.set_title('Ruster Plot (all electrodes)')
    ax1.set_xlabel('Time [s]')
    ax1.set_xlim(0, np.max(times[obj.stimulation_indexes]) + 1)


    # Second row: Single plot spanning the entire width
    ax2 = fig.add_subplot(gs[1, :])  # This spans all columns in the second row
    times = np.arange(0, len(np.copy(obj.recording_data[0, :]))) / obj.sample_rate

    blanking_win_msec = 15
    blanking_win_samples = int((blanking_win_msec / 1000) * obj.sample_rate)

    indices = sorted(obj.stimulation_indexes)
    array_len = len(times)
    mask = np.ones(array_len, dtype=bool)
    for index in indices:
        start = int(max(0, index - 0.005 * obj.sample_rate))
        end = int(min(array_len, index + blanking_win_samples + 1))
        mask[start:end] = False

    blanked_data = np.copy(obj.recording_data[selected_ch_ind, :])
    blanked_data[~mask] = 0

    ax2.plot(times, blanked_data, color='k', linewidth=0.05)

    # Highlight the spike times for the selected electrode
    lim =200
    # for ind in spikes_indices[selected_ch_ind]:
    #     spike_time = times[ind]
    #     ax2.fill_betweenx(y=[-lim, lim], x1=spike_time - 0.00001, x2=spike_time + 0.00001, color='lightblue',
    #                          alpha=0.2)

    ax2.scatter(times[obj.stimulation_indexes], np.zeros_like(obj.stimulation_indexes) + lim - 100, color='#800020', marker='v', s=20)
    ax2.set_ylim(-lim, lim)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Amplitude [uV]')
    ax2.set_xlim(0, np.max(times[obj.stimulation_indexes]) + 1)
    ax2.set_title(f'Blanked ({blanking_win_msec} ms) Selected Electrode: {selected_ch_ind}')

    # Calculate features:
    stim_indices = obj.stimulation_indexes
    spike_indices = spikes_indices[selected_ch_ind]

    # Initialize arrays for the results
    spike_count = []
    inter_spike_interval = []
    latency = []
    peak_firing_rate = []

    # Loop through each stimulus index and calculate the metrics
    for i in range(len(stim_indices)):
        stim_start = stim_indices[i]

        # Find the spike indices between this stimulus and the next stimulus
        if i < len(stim_indices) - 1:
            stim_end = stim_indices[i + 1]
        else:
            stim_end = stim_start + obj.sample_rate  # For the last stimulus, consider all remaining spikes

        # Get the spike indices between this stimulus and the next one
        spikes_in_window = spike_indices[(spike_indices > stim_start) & (spike_indices < stim_end)]

        # 1. Spike Count
        spike_count.append(len(spikes_in_window))

        # 2. Inter-Spike Interval (ISI): Calculate average difference between consecutive spikes
        if len(spikes_in_window) > 1:
            isi = np.diff(spikes_in_window) / obj.sample_rate * 1000
            inter_spike_interval.append(np.mean(isi))
        else:
            inter_spike_interval.append(np.nan)  # If no spikes, set to NaN

        # 3. Latency: The time difference between the stimulus and the first spike after it
        if len(spikes_in_window) > 0:
            latency.append((spikes_in_window[0] - stim_start) / obj.sample_rate * 1000)
        else:
            latency.append(np.nan)  # If no spike, set to NaN

        # 4. Peak Firing Rate: 1 / minimum ISI between consecutive spikes
        if len(spikes_in_window) > 1:
            isi_diff = np.diff(spikes_in_window) / obj.sample_rate * 1000 # switch to ms
            peak_firing_rate.append(1 / np.min(isi_diff))  # Inverse of the smallest ISI
        else:
            peak_firing_rate.append(np.nan)  # If only one spike or no spikes, set to NaN

    # Convert lists to numpy arrays
    spike_count = np.array(spike_count)
    inter_spike_interval = np.array(inter_spike_interval)
    latency = np.array(latency)
    peak_firing_rate = np.array(peak_firing_rate)

    ax3 = fig.add_subplot(gs[2, 0:2])  # Left subplot, spans the first 2 columns
    ax3.scatter(np.arange(len(spike_count)), spike_count, color='k', marker='.', s=30)  # Spike Count scatter plot
    ax3.set_ylabel("Spike Count")
    ax3.set_xlabel('# pulses')

    ax4 = fig.add_subplot(gs[2, 2:4])  # Right subplot, spans the last 2 columns
    ax4.scatter(np.arange(len(inter_spike_interval)), inter_spike_interval, color='k', marker='.', s=30)  # ISI scatter plot
    ax4.set_ylabel("Inter-Spike Interval [ms]")
    ax4.set_xlabel('# pulses')

    ax5 = fig.add_subplot(gs[3, 0:2])  # Left subplot, spans the first 2 columns
    ax5.scatter(np.arange(len(latency)), latency, color='k', marker='.', s=30)  # Latency scatter plot
    ax5.set_ylabel("Latency [ms]")
    ax5.set_xlabel('# pulses')

    ax6 = fig.add_subplot(gs[3, 2:4])  # Right subplot, spans the last 2 columns
    ax6.scatter(np.arange(len(peak_firing_rate)), peak_firing_rate, color='k', marker='.', s=30) # Peak Firing Rate scatter plot
    ax6.set_ylabel("Peak Firing Rate [1/ms]")
    ax6.set_xlabel('# pulses')

    plt.suptitle(obj.file_name[:-4])
    plt.tight_layout()
    plt.savefig(rf'{obj.output_folder}/Indirect_{obj.file_name[:-4]}_{selected_ch_ind}.png')
    plt.close()


def plot_overlay_pulses(obj_list, average=True):
    if not average:
        obj = obj_list[0]
        output_folder = r'C:\Users\YHLab\PycharmProjects\IntactRetinaSpikes\impedance'
        ylim = 7000
        fig, axes = plt.subplots(4, 4, figsize=(10, 15))
        axes = axes.flatten()
        title = f'Stimulating ch {obj.stimulation_channels}'
        data = obj.pulses
        ch_names = obj.recording_channels
        sample_rate = obj.sample_rate

        for i, ax in enumerate(axes):
            if i < data.shape[1]:
                for j in range(data.shape[0]):
                    x_axis = -3 + (np.arange(0, len(data[j, i, :]))) / sample_rate * 1000
                    ax.plot(x_axis, data[j, i, :], color='grey', linewidth=0.5, alpha=0.3)
                    if i == obj.stimulation_channels:
                        ax.set_title(f"Channel {ch_names[i]}", color='red')
                    else:
                        ax.set_title(f"Channel {ch_names[i]}")
                    if ylim is not None:
                        ax.set_ylim(-ylim, ylim)

                # Only show x-ticks and x-labels on the last row
                if i < 12:  # Exclude last row
                    ax.set_xticks([])
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel('Time (msec)')  # Show x-label only on the last row

                # Only show y-ticks and y-labels on the first column
                if i % 4 != 0:  # Exclude first column
                    ax.set_yticks([])
                    ax.set_yticklabels([])
            else:
                ax.axis('off')

        # plt.suptitle(title, y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(rf'{output_folder}/{title}.png')
        plt.close()

    else:
        output_folder = r'C:\Users\YHLab\PycharmProjects\IntactRetinaSpikes\impedance'
        ylim = 7000
        fig, axes = plt.subplots(4, 4, figsize=(10, 15))
        axes = axes.flatten()
        colors = ['#5f8fb7', '#6d9d74', '#9c5a61']
        title = f'stimulating ch 2'
        category = []
        for index, color, obj in zip(range(3), colors, obj_list):
            data = obj.pulses
            ch_names = obj.recording_channels
            sample_rate = obj.sample_rate
            category.append(obj.parent_folder)
            average_data = np.mean(data, axis=0)
            for i, ax in enumerate(axes):
                if i < data.shape[1]:
                    x_axis = -3 +(np.arange(0, len(data[0, i, :]))) / sample_rate * 1000
                    ax.plot(x_axis, average_data[i, :], color=color, linewidth=0.7, alpha=1)
                    ax.set_title(f"Channel {ch_names[i]}")
                    ax.set_xlabel('Time (msec)')
                    if ylim is not None:
                        ax.set_ylim(-ylim, ylim)

                    # Add the estimated impedance
                    current_a = 5 / 1e6
                    max_v = np.max(abs(average_data[i, :]) / 1e6)
                    z = (max_v / current_a) / 1e3

                    ax.text(
                        0.95, 0.05+index*0.06,  # Position in normalized axis coordinates
                        f"{z:.2f}",  # Text to display
                        color=color,
                        fontsize=7,
                        transform=ax.transAxes,  # Use axis coordinates
                        verticalalignment='bottom',
                        horizontalalignment='right')
                else:
                    ax.axis('off')

        # Add a single-row legend below the title
        legend_handles = [
            plt.Line2D([0], [0], color=color, lw=2, label=parent)
            for color, parent in zip(colors, category)
        ]
        plt.figlegend(handles=legend_handles, loc='upper center', ncol=len(obj_list), frameon=False)
        plt.suptitle(title, y=0.95)  # Adjust title to make room for legend
        plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust layout to fit legend and title
        plt.savefig(rf'{output_folder}/{title}.png')
        plt.close()



###################################################

    # times = np.arange(0, len(obj.recording_data[0, :])) / obj.sample_rate
    # selected_ch_ind = 15
    # # Create a figure and GridSpec layout
    # fig = plt.figure(figsize=(18, 12))
    # gs = GridSpec(4, 4, figure=fig, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1, 1, 1])
    #
    # # First row: Single plot spanning the entire width
    # ax1 = fig.add_subplot(gs[0, :])  # This spans all columns in the first row
    #
    # # Highlight the row for the selected electrode
    # # selected_ch_ind = [(i, s) for i, s in enumerate(obj.channel_labels) if selected_electrode in s][0][0]
    # # ax1.fill_between(times, 0.5 + selected_ch_ind, 1 + selected_ch_ind, color='lightgrey', alpha=0.5)
    # # First subplot: raster plot of all channels
    # for ch, ind_list in enumerate(spikes_indices):
    #     for ind in ind_list:
    #         ax1.vlines(times[ind], ymin=0.5 + ch, ymax=1 + ch, color='black', linewidth=0.7)
    # ax1.scatter(times[obj.stimulation_indexes], len(obj.recording_channels) + np.zeros_like(obj.stimulation_indexes),
    #             color='#800020', marker='v', s=25)
    # ax1.set_ylabel('# electrode')
    # ax1.set_title('Ruster Plot (all electrodes)')
    # ax1.set_xlabel('Time [s]')
    # ax1.set_xlim(0, np.max(times[obj.stimulation_indexes]) + 1)
    #
    # # Second row: Single plot spanning the entire width
    # ax2 = fig.add_subplot(gs[1, 0:2])  # This spans all columns in the second row
    # times = (np.arange(0, len(np.copy(obj.recording_data[0, :]))) / obj.sample_rate) - 1.41136
    #
    # blanking_win_msec = 15
    # blanking_win_samples = int((blanking_win_msec / 1000) * obj.sample_rate)
    #
    # indices = sorted(obj.stimulation_indexes)
    # array_len = len(times)
    # mask = np.ones(array_len, dtype=bool)
    # for index in indices:
    #     start = int(max(0, index - 0.005 * obj.sample_rate))
    #     end = int(min(array_len, index + blanking_win_samples + 1))
    #     mask[start:end] = False
    #
    # blanked_data = np.copy(obj.recording_data[15, :])
    # blanked_data[~mask] = 0
    #
    # ax2.plot(times, blanked_data, color='k', linewidth=0.2)
    #
    # # Highlight the spike times for the selected electrode
    # lim = 200
    # ax2.scatter(times[obj.stimulation_indexes], np.zeros_like(obj.stimulation_indexes) + lim - 100, color='#800020',
    #             marker='v', s=20)
    # ax2.set_ylim(-lim, lim)
    # ax2.set_xlabel('Time [s]')
    # ax2.set_ylabel('Amplitude [uV]')
    # ax2.set_xlim(-0.1, 0.6)
    # # ax2.set_title(f'Blanked ({blanking_win_msec} ms) Selected Electrode: {selected_ch_ind}')
    #
    # # Second row: Single plot spanning the entire width
    # ax3 = fig.add_subplot(gs[1, 2:4])  # This spans all columns in the second row
    # # times = np.arange(0, len(np.copy(obj.recording_data[0, :]))) / obj.sample_rate
    #
    # blanking_win_msec = 15
    # blanking_win_samples = int((blanking_win_msec / 1000) * obj.sample_rate)
    #
    # indices = sorted(obj.stimulation_indexes)
    # array_len = len(times)
    # mask = np.ones(array_len, dtype=bool)
    # for index in indices:
    #     start = int(max(0, index - 0.005 * obj.sample_rate))
    #     end = int(min(array_len, index + blanking_win_samples + 1))
    #     mask[start:end] = False
    #
    # blanked_data = np.copy(obj.recording_data[19, :])
    # blanked_data[~mask] = 0
    #
    # ax3.plot(times, blanked_data, color='k', linewidth=0.2)
    #
    # # Highlight the spike times for the selected electrode
    # lim = 200
    # ax3.scatter(times[obj.stimulation_indexes], np.zeros_like(obj.stimulation_indexes) + lim - 100, color='#800020',
    #             marker='v', s=20)
    # ax3.set_ylim(-lim, lim)
    # ax3.set_xlabel('Time [s]')
    # ax3.set_ylabel('Amplitude [uV]')
    # ax3.set_xlim(-0.1, 0.6)
    # # ax3.set_title(f'Blanked ({blanking_win_msec} ms) Selected Electrode: {selected_ch_ind}')
    #
    #
    # plt.tight_layout()