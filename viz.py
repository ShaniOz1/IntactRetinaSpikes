import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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
    plt.savefig(rf'{obj.output_folder}_Direct_response.png')
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

