from DataObj import DataObj
import os
from scipy import signal
import glob
from direct_response import prepare_roi_mat, remove_stimulation_artifact, spikes_analysis
import viz
import utils
import shutil

single_file_running = False
multiple_files_running = True
multiple_files_comparison = False

Apply_filter = True
Impedance_check = False
Direct_response = True
Indirect_response = False
Spontaneous_activity = False

if single_file_running:
    file = (
        # r'C:\Users\Asus\PycharmProjects\MasterNotebook\data\2024_01_21\Stimulation Ch 20\e18 320us biphasic 1hz_2uA_240121_152227.rhs')
        # r'C:\Shani\Intact\old\2023.05.24\retina 1\softC_iv_2023_3_320us_40us_200 times_1hz_10uA_230524_094035\softC_iv_2023_3_320us_40us_200 times_1hz_10uA_230524_094235.rhs')
        r'C:\Shani\SoftC prob\16Ch prob experiments\2025.01.08 E14\Retina2\Recordings on Retina\Ch2_300us_50us_15uA_1Hz_20pulse_250108_155301\Ch2_300us_50us_15uA_1Hz_20pulse_250108_155301.rhs')

if multiple_files_running:
    parent_directory = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.02.02 E14'
    pattern = os.path.join(parent_directory, '**', '*.rhs*')
    matching_files = glob.glob(pattern, recursive=True)


def main(file_path, apply_filter, impedance_check, direct_response, non_direct_response,
         spontaneous_activity):
    try:
        obj = DataObj(file_path, reduce_faulty_electrodes=impedance_check)

        if apply_filter:
            sos = signal.butter(2, [300, 3000], btype='bandpass', fs=int(obj.sample_rate), output='sos')
            obj.recording_data = signal.sosfiltfilt(sos, obj.recording_data)

        if direct_response:
            obj.pulses = prepare_roi_mat.get_recorded_pulses(stim_ind=obj.stimulation_indexes,
                                                             data=obj.recording_data, sample_rate=obj.sample_rate,
                                                             win_size=15)  # win size should be 3x

            obj.signals_mat_3d, obj.artifacts_mat_3d = remove_stimulation_artifact.ica_based_method(obj.pulses)
            obj.spikes_dict = spikes_analysis.get_spikes(obj)
            viz.plot_direct_spikes(obj)
            viz.plot_artifacts_vs_signals(obj)
            # viz.plot_spikes_amps_vs_time(obj)

        if Indirect_response:
            indirect_response_spikes = utils.indirect_response_indices(obj)
            viz.plot_indirect_response(obj, indirect_response_spikes)

        # If no response found, remove folder
        if not os.listdir(obj.output_folder):  # os.listdir() returns an empty list if the folder is empty
            os.rmdir(obj.output_folder)  # Remove the empty directory

        return obj
    except Exception as e:
        print(f' {file_path} ERROR: {e}')


if single_file_running:
    main(file, Apply_filter, Impedance_check, Direct_response, Indirect_response, Spontaneous_activity)

if multiple_files_running:
    for file in matching_files:
        # if os.path.basename(file) == '320us 40usdelay 10uA 10hz_100pulses_230528_120802.rhs':
        # if os.path.basename(file) == 'Ch4_300us_50us_5uA_1Hz_250202_114004.rhs':
        if 'spont' not in os.path.basename(file).lower():
            print(f'Analyzing {file}...')
            main(file, Apply_filter, Impedance_check, Direct_response, Indirect_response, Spontaneous_activity)
