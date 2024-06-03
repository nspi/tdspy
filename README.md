# TDSpy

An open-source implementation of Time Selay Stability (TDS) in Python.

This is a copy. Main repository of our group: https://gitlab.gwdg.de/medinfpub/biosignal-processing-group/tdspy

## Name
TDSpy

## Description
TDS is an established tool for analyzing the interaction between physiological systems in the human organism based on analysis of physiological time series (e.g. ECG, EEG) and has been proposed by Bashan et al. (https://doi.org/10.1038/ncomms1705) in 2012. Since then, it has been used by different groups on different datasets and problems. This led to research groups implementing their own algorithms in different programming languages which poses the risk of differences between implementations and parameters, leading to a lack of reproducibility. Hence, we propose this implementation to work towards reproducible research.

## Project Structure

- TDSpy
    - feature_extraction: Extract features from biosignals
        - sn_getBreathingRate.py
        - sn_getEEGBandPower.py
        - sn_getEventRate.py
        - sn_getQRS.py
        - sn_getVariance.py
    - signal_processing: Some signal processing tools
        - nld_movingAverage.py
        - nld_movingMedian.py
        - sn_getExtrema.py
    - tools
        - edf_reader.py: read single EDF file or all EDF files in a directory
        - signal_generator.py: generate signals for unit tests    
    - nld_spectrogram.py: compute short-time fourier transform
    - sn_TDS.py: Main function: reads an EDF file and returns TDS value
    - sn_getCrossCorrelation.py: computes cross correlation between two signals (used by sn_TDS.py)
    - sn_getStability.py: computes stable sequences in lag series (used by sn_TDS.py)



## Installation
Download available at [PyPI](https://pypi.org/project/TDSpy/)

`pip install -i TDSpy`


## Usage
```python
import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TDSpy functions
from TDSpy.sn_TDS import sn_TDS_no_feature_extraction as TDS
from TDSpy.feature_extraction.sn_getEEGBandPower import sn_getEEGBandPower 
from TDSpy.feature_extraction.sn_getVariance import sn_getVariance 
from TDSpy.tools.sn_plotTDS import plot_TDS
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
    tds, combination, stages = TDS(data_dict=data_dict)

    # Plot result
    nk.signal_plot(pd.DataFrame(data_dict), subplots=True)
    plt.show()

    # Limit to sleep stage
    tds = tds[:,stage_idx[:-1]]

    # Matrix plot
    plot_TDS(tds, combination)

if __name__ == '__main__':
    main()
```

## Support
Sebastian Schmale

## Authors and acknowledgment
Tabea Steinbrinker, tabea.steinbrinker@med.uni-goettingen.de  
Dagmar Krefting  
Ronny P. Bartsch  
Jan W. Kantelhardt  
Nicolai Spicher  

## License
MIT License

## Project status
v1.0 - first implementation ready.
