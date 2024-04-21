import os
import pyintan.pyintan as pyintan
import numpy as np
import scipy
from load_intan_rhd_format import read_data
import pandas as pd
import glob
from datetime import datetime

class DataObj:
    def __init__(self, path, reduce_faulty_electrodes=None):
        self.path = path
        self.output_folder = self.create_output_folder(path)
        # self.output_sub_folders = self.create_output_sub_folders(self.output_folder)

        if path[-3:] == 'rhs':
            file = pyintan.File(path)
            self.start_time = file.datetime
            self.file_name = file.fname
            self.sample_rate = int(file.sample_rate)
            self.recording_data = file.analog_signals[0].signal
            self.recording_channels = file.analog_signals[0].channel_names[0]
            try:
                self.stimulation_data = file.stimulation[0].signal
                self.stimulation_channels = file.stimulation[0].channels
                self.stimulation_indexes = self.get_stimulation_indexes(self.stimulation_data, path[-3:])
                self.stimulation_trials = self.get_stimulation_trials(self.stimulation_data, path[-3:])

            except Exception:
                print('No stimulation identified')
                self.stimulation_data = None
                self.stimulation_channels = None
                self.stimulation_indexes = None

        if path[-3:] == 'rhd':
            file = read_data(path)
            self.start_time = pd.to_datetime('00:00:00', format='%H:%M:%S').time()
            self.file_name = os.path.basename(path)[:-4]
            self.sample_rate = file["frequency_parameters"]['amplifier_sample_rate']
            self.recording_data = file['amplifier_data']
            self.recording_channels = np.arange(10).astype(str)
            self.stimulation_data = file["board_adc_data"][0, :]
            self.stimulation_channels = file['board_adc_channels'][0]['native_order']
            self.stimulation_indexes = self.get_stimulation_indexes(self.stimulation_data, path[-3:])

        if path[-3:] == 'txt':
            self.start_time = None
            self.file_name = os.path.basename(path)
            self.sample_rate = self.read_txt_sample_rate()
            self.recording_data = self.read_txt_sig()
            self.recording_channels = self.read_txt_ch_names()
            try:
                self.stimulation_data = None
                self.stimulation_channels = None
                self.stimulation_indexes = None
            except Exception as e:
                print('No stimulation identified')
                self.stimulation_data = None
                self.stimulation_channels = None
                self.stimulation_indexes = None

        if reduce_faulty_electrodes:
            impedance_thresh = 1000*300  # I
            try:
                parent_directory = os.path.dirname(os.path.dirname(path))
                if 'TTX' in self.file_name:
                    pattern = os.path.join(parent_directory, '*ttx*.csv')
                    matching_files = glob.glob(pattern)
                    impedance_csv = pd.read_csv(matching_files[0])
                else:
                    pattern = os.path.join(parent_directory, '*start*.csv')
                    matching_files = glob.glob(pattern)
                    impedance_csv = pd.read_csv(matching_files[0])

                impedance_vals = impedance_csv['Impedance Magnitude at 1000 Hz (ohms)']
                bad_impedance_ch_ind = np.where(impedance_vals < impedance_thresh)
                self.recording_data = np.delete(self.recording_data, bad_impedance_ch_ind, axis=0)
                self.recording_channels = np.delete(self.recording_channels, bad_impedance_ch_ind, axis=0)

            except Exception:
                print('No impedance csv found. All electrodes are being used')

    @staticmethod
    def get_stimulation_indexes(stim, file_type):
        if file_type == 'rhs':
            peaks, _ = scipy.signal.find_peaks(stim, height=0.9, distance=30)
            crossings = np.where(np.diff(stim < -0.01))[0]
            crossings = crossings + 1
            crossings = crossings[::2]
            return crossings
        if file_type == 'rhd':
            crossings = np.where(np.diff(np.sign(stim - 0.9)))[0]
            sample_interval = 4
            filtered_crossings = crossings[::sample_interval]
            return filtered_crossings

    @staticmethod
    def get_stimulation_trials(stim, file_type):
        if file_type == 'rhs':
            peaks, _ = scipy.signal.find_peaks(stim, height=0.9, distance=30)
            peaks_diff = np.diff(peaks)
            most_common_val = np.argmax(np.bincount(peaks_diff))
            indices_greater_than_3x = np.where(peaks_diff > 3 * most_common_val)[0]
            trails_indices = peaks[indices_greater_than_3x]
            return trails_indices
        if file_type == 'rhd':
            print('not supported yet')

    @staticmethod
    def create_output_folder(path):
        output_folder = path.split(os.path.sep)[-1][:-4]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        now_output_folder = os.path.join(output_folder, now_str)
        if not os.path.exists(now_output_folder):
            os.makedirs(now_output_folder)
        return now_output_folder

    @staticmethod
    def create_output_sub_folders(path):
        sub_folders_list = ['a. Overlay pulses and artifacts- all channels',
                            'b. Direct response clusters per channel',
                            'c. Direct response over artifact - all channels']
        sub_folders_paths = []
        for new_sub_folder in sub_folders_list:
            sub_path = os.path.join(path, new_sub_folder)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)
                sub_folders_paths.append(sub_path)
        return sub_folders_paths



    def keep_specific_channels(self, channels_to_keep):
        if channels_to_keep is not None:
            indices_dict = {num: [idx for idx, name in enumerate(self.recording_channels) if str(num) in name] for num in
                       channels_to_keep}
            indices = np.concatenate(list(indices_dict.values()))
            indices_to_delete = np.setdiff1d(np.arange(len(self.recording_channels)), indices)

            self.recording_data = np.delete(self.recording_data, indices_to_delete, axis=0)
            self.recording_channels = np.delete(self.recording_channels, indices_to_delete, axis=0)

    def read_txt_sig(self):
        data_matrix = []
        with open(self.path, 'r') as file:
            lines = file.readlines()
        for line in lines[4:]:  # was 4
            values = line.strip().split('\t')
            values = [float(value) for value in values]
            data_matrix.append(values)
        result = np.transpose(np.array(data_matrix))
        return result


    def read_txt_sample_rate(self):
        data_matrix = []
        with open(self.path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            if 'Sample interval' in line:
                substr = line.split('Sample interval')[1]
                interval = float([float(s) for s in substr.split() if s.replace('.', '', 1).isdigit()][0])
                break
        return int(1/(interval/1000))

    def read_txt_ch_names(self):
        data_matrix = []
        with open(self.path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            if 'Number of channels = ' in line:
                substr = line.split('Number of channels = ')[1]
                ch_num = int([float(s) for s in substr.split() if s.replace('.', '', 1).isdigit()][0])
                break
            if 'Number of data channels:' in line:
                substr = line.split('Number of data channels:')[1]
                ch_num = int([float(s) for s in substr.split() if s.replace('.', '', 1).isdigit()][0])
                break

        return [f'Ch-{i}' for i in range(ch_num)]
