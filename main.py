from DataObj import DataObj
import os
from scipy import signal
import glob
from direct_response import prepare_roi_mat, remove_stimulation_artifact, spikes_analysis
import viz

single_file_running = False
multiple_files_running = True
multiple_files_comparison = False

Apply_filter = True
Impedance_check = True
Direct_response = True
Non_direct_response = False
Spontaneous_activity = False

if single_file_running:
    file = (
        r'C:\Users\Asus\PycharmProjects\MasterNotebook\data\2024_01_21\Stimulation Ch 20\e18 320us biphasic '
        r'1hz_3uA__240121_152738.rhs')

if multiple_files_running:
    parent_directory = r'C:\Users\Asus\PycharmProjects\MasterNotebook\data\2024_01_25\Stimulation Ch 20'
    pattern = os.path.join(parent_directory, '*.rhs*')
    matching_files = glob.glob(pattern)


def main(file_path, apply_filter, impedance_check, direct_response, non_direct_response,
         spontaneous_activity):
    obj = DataObj(file_path, reduce_faulty_electrodes=impedance_check)

    if apply_filter:
        sos = signal.butter(2, [300, 3000], btype='bandpass', fs=int(obj.sample_rate), output='sos')
        obj.recording_data = signal.sosfiltfilt(sos, obj.recording_data)

    if direct_response:
        obj.pulses = prepare_roi_mat.get_recorded_pulses(stim_ind=obj.stimulation_indexes,
                                                         data=obj.recording_data, sample_rate=obj.sample_rate,
                                                         win_size=15)

        obj.signals_mat_3d, obj.artifacts_mat_3d = remove_stimulation_artifact.ica_based_method(obj.pulses)
        obj.spikes_dict = spikes_analysis.get_spikes(obj)
        viz.plot_direct_spikes(obj)


    print('Done')
    return obj


if single_file_running:
    main(file, Apply_filter, Impedance_check, Direct_response, Non_direct_response, Spontaneous_activity)

if multiple_files_running:
    for file in matching_files:
        print(f'Analyzing {file}...')
        main(file, Apply_filter, Impedance_check, Direct_response, Non_direct_response, Spontaneous_activity)
