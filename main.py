from DataObj import DataObj
import os
from scipy import signal
import glob
from direct_response import prepare_roi_mat, remove_stimulation_artifact, spikes_analysis
import viz
import utils

single_file_running = False
multiple_files_running = True
multiple_files_comparison = False

Apply_filter = True
Impedance_check = True
Direct_response = False
Indirect_response = True
Spontaneous_activity = False

if single_file_running:
    file = (
        # r'C:\Users\Asus\PycharmProjects\MasterNotebook\data\2024_01_21\Stimulation Ch 20\e18 320us biphasic 1hz_2uA_240121_152227.rhs')
        r'C:\Shani\Intact\old\2023.05.24\retina 1\softC_iv_2023_3_320us_40us_200 times_1hz_10uA_230524_094035\softC_iv_2023_3_320us_40us_200 times_1hz_10uA_230524_094235.rhs')

if multiple_files_running:
    parent_directory = r'C:\Shani\Intact\old'
    pattern = os.path.join(parent_directory, '**', '*.rhs*')
    matching_files = glob.glob(pattern, recursive=True)


def main(file_path, apply_filter, impedance_check, direct_response, non_direct_response,
         spontaneous_activity):
    try:
        obj = DataObj(file_path, reduce_faulty_electrodes=impedance_check)
        # TODO:
        # obj.stimulation_indexes = obj.stimulation_indexes[200:300]

        if apply_filter:
            sos = signal.butter(2, [300, 3000], btype='bandpass', fs=int(obj.sample_rate), output='sos')
            obj.recording_data = signal.sosfiltfilt(sos, obj.recording_data)

        if direct_response:
            obj.pulses = prepare_roi_mat.get_recorded_pulses(stim_ind=obj.stimulation_indexes,
                                                             data=obj.recording_data, sample_rate=obj.sample_rate,
                                                             win_size=15)

            obj.signals_mat_3d, obj.artifacts_mat_3d = remove_stimulation_artifact.ica_based_method(obj.pulses)
            # viz.plot_spikes_amps_vs_time(obj)
            obj.spikes_dict = spikes_analysis.get_spikes(obj)
            viz.plot_direct_spikes(obj)

        if Indirect_response:
            indirect_response_spikes = utils.indirect_response_indices(obj)
            viz.plot_indirect_response(obj, indirect_response_spikes)

        print('Done')
        return obj
    except Exception as e:
        print(f' {file_path} ERROR: {e}')


if single_file_running:
    main(file, Apply_filter, Impedance_check, Direct_response, Indirect_response, Spontaneous_activity)

if multiple_files_running:
    for file in matching_files:
        if os.path.basename(file) == 'softC_iv_2023_3_120us_10 times_1hz_20uA__230524_144600.rhs':
            print(f'Analyzing {file}...')
            main(file, Apply_filter, Impedance_check, Direct_response, Indirect_response, Spontaneous_activity)
