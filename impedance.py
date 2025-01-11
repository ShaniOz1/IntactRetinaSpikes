from DataObj import DataObj
import os
from scipy import signal
import glob
from direct_response import prepare_roi_mat, remove_stimulation_artifact, spikes_analysis
import viz
import utils
from pathlib import Path

# local_path = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.01.08 E14\Retina2'
local_path = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.01.08 E14\Retina3'

pattern = os.path.join(local_path, '**', '*.rhs*')
matching_files = glob.glob(pattern, recursive=True)
objects = []

for file in matching_files:
    if 'Ch10' in file:
        obj = DataObj(file, create_output_folder=False)
        # sos = signal.butter(2, [300, 3000], btype='bandpass', fs=int(obj.sample_rate), output='sos')
        # obj.recording_data = signal.sosfiltfilt(sos, obj.recording_data)

        obj.parent_folder = next((p.name for p in Path(file).parents if 'Recordings' in p.name), None)
        obj.pulses = prepare_roi_mat.get_recorded_pulses(stim_ind=obj.stimulation_indexes,
                                                         data=obj.recording_data, sample_rate=obj.sample_rate,
                                                         win_size=9)
        objects.append(obj)


    # viz.plot_overlay_pulses([obj])
viz.plot_overlay_pulses(objects)

print('done')


