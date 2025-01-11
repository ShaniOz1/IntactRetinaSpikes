import numpy as np
import pyedflib
import scipy
import pyintan.pyintan as pyintan
from scipy import signal
import os

folder_path = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.01.08 E14\Retina1\Ch10_300us_50us_1uA_1Hz_50pulse_250108_134747'
for filename in os.listdir(folder_path):
    if filename[-3:] == 'rhs':
        try:
            file_path = os.path.join(folder_path, filename)
            file = pyintan.File(file_path)
            analog_signals = file.analog_signals[0]
            sig = analog_signals.signal
            try:
                stimulation = file.stimulation[0].signal
                data = np.vstack((sig, stimulation))
            except Exception:
                data = sig
            # Define the parameters
            num_channels = len(data)
            sample_frequency = int(file.sample_rate)
            duration = len(data[0]) / sample_frequency

            edf_file = f"{file_path[0:-4]}.edf"
            pyedflib.highlevel.write_edf_quick(edf_file, data, sample_frequency)
            print("EDF file created successfully.")
        except Exception as e:
            print(f'Files {filename} could not convert: {e}')