import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TDSpy functions
from TDSpy.feature_extraction import sn_getEEGBandPower 
from TDSpy.feature_extraction import sn_getVariance 
from TDSpy import sn_TDS as TDS
from TDSpy.tools import sn_plotTDS as plot_TDS
from TDSpy.tools.edf_reader import read_all_EDF_channels

# Preparation: Please download these files:
# https://physionet.org/content/ucddb/1.0.0/ucddb002.rec
# https://physionet.org/content/ucddb/1.0.0/ucddb002_stage.txt

def main():
    sleep_stage = 0 # wake
    dur = 16000 # seconds

    # Read EDF
    data_dict, data_dict_sampling_rate = read_all_EDF_channels("ucddb002.rec", startrecord=0, endrecord=dur)

    # Read sleep stages
    stages = pd.read_csv("ucddb002_stage.txt", header=None)
    stages = stages.to_numpy()

    # Read only single stage
    stage_idx = np.where(stages == sleep_stage)[0]
    stage_idx = stage_idx[stage_idx < dur/30]

    # Process single ECG channel
    ecg_signal = nk.ecg_process(data_dict['ECG'], sampling_rate=data_dict_sampling_rate['ECG'], method='neurokit')
    hr_signal_resampled = nk.signal_resample(ecg_signal[0]['ECG_Rate'], sampling_rate=data_dict_sampling_rate['ECG'], desired_sampling_rate=1, method="interpolation")

    # Process single RESP channel
    rsp_rate = nk.rsp_rate(data_dict['Flow'], sampling_rate=8, method="trough")
    rsp_signal_resampled = nk.signal_resample(rsp_rate, sampling_rate=data_dict_sampling_rate['Flow'], desired_sampling_rate=1, method="interpolation")

    # Process single EMG channel
    emg_signal = data_dict['EMG']
    emg_var = sn_getVariance(emg_signal, sf=data_dict_sampling_rate['EMG'])

    # Process single EOG channel
    eog_signal = data_dict['Lefteye']
    eog_var = sn_getVariance(eog_signal, sf=data_dict_sampling_rate['Lefteye'])

    # Process single EEG channel
    eeg_signal = data_dict['C3A2']
    fpb, _ = sn_getEEGBandPower(eeg_signal, sf=data_dict_sampling_rate['C3A2'], bandlimits=np.array([[0.5, 4, 8, 12, 16], [3.5, 7.5, 11.5, 15.5, 19.5]]))

    # Compute TDS
    data_dict = {'HR': hr_signal_resampled, 'Resp': rsp_signal_resampled, 'Chin': emg_var, 'Eye': eog_var, 'Delta':  fpb[:,0], 'Theta':  fpb[:,1], 'Alpha':  fpb[:,2], 'Sigma':  fpb[:,3], 'Beta':  fpb[:,4]}
    tds, combination, stages = sn_TDS(data_dict=data_dict)

    # Plot result
    nk.signal_plot(pd.DataFrame(data_dict), subplots=True)
    plt.show()

    # Limit to sleep stage
    tds = tds[:,stage_idx[:-1]]

    # Matrix plot
    plot_TDS(tds, combination)

if __name__ == '__main__':
    main()
