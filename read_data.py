import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import neurokit as nk
import seaborn as sns
import pandas as pd

def load_data(path, subject):
    """Given path and subject, load the data of the subject"""
    os.chdir(path)
    os.chdir(subject)
    with open(subject + '.pkl', 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    return data

class read_data_of_one_subject:
    """Read data from WESAD dataset"""
    def __init__(self, path, subject):
        self.keys = ['label', 'subject', 'signal']
        self.signal_keys = ['wrist', 'chest']
        self.chest_sensor_keys = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
        self.wrist_sensor_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
        os.chdir(path)
        os.chdir(subject)
        with open(subject + '.pkl', 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        self.data = data

    def get_labels(self):
        return self.data[self.keys[0]]

    def get_wrist_data(self):
        """"""
        #label = self.data[self.keys[0]]
        assert subject == self.data[self.keys[1]]
        signal = self.data[self.keys[2]]
        wrist_data = signal[self.signal_keys[0]]
        #wrist_ACC = wrist_data[self.wrist_sensor_keys[0]]
        #wrist_ECG = wrist_data[self.wrist_sensor_keys[1]]
        return wrist_data

    def get_chest_data(self):
        """"""
        signal = self.data[self.keys[2]]
        chest_data = signal[self.signal_keys[1]]
        return chest_data

if __name__ == '__main__':
    data_set_path = "/media/jac/New Volume/Datasets/WESAD"
    subject = 'S5'
    #print(label.size)  # 4545100
    #print(wrist_ACC.size)  # 623328
    #print(wrist_ECG.size)  # 415552
    #data = load_data(data_set_path,subject)
    obj_data = {}
    obj_data[subject] = read_data_of_one_subject(data_set_path, subject)
    #print(obj_data[subject].data)
    #print(obj_data[subject].get_label())
    wrist_data_dict = obj_data[subject].get_wrist_data()
    wrist_dict_length = {key: len(value) for key, value in wrist_data_dict.items()}
    #print(wrist_dict_length)
    #print(wrist_data_dict['EDA'])

    chest_data_dict = obj_data[subject].get_chest_data()
    chest_dict_length = {key: len(value) for key, value in chest_data_dict.items()}
    print(chest_dict_length)
    #for one in chest_data_dict.keys():
    #    print(one,":")
    #    print(chest_data_dict[one])

    chest_data = np.concatenate((chest_data_dict['ACC'], chest_data_dict['ECG'], chest_data_dict['EDA'],
                                 chest_data_dict['EMG'], chest_data_dict['Resp'], chest_data_dict['Temp']), axis=1)
    #print(chest_data_dict['ECG'])

    # Get labels
    labels = obj_data[subject].get_labels()
    #print(labels.shape)

    # 'ACC' : 3, 'ECG' 1: , 'EDA' : 1, 'EMG': 1, 'RESP': 1, 'Temp': 1  ===> Total dimensions : 8
    # No. of Labels ==> 8 ; 0 = not defined / transient, 1 = baseline, 2 = stress, 3 = amusement,
    # 4 = meditation, 5/6/7 = should be ignored in this dataset

    baseline = np.asarray([idx for idx,val in enumerate(labels) if val == 1])
    #print(baseline)

    print("Baseline:", chest_data_dict['ECG'][baseline].shape)

    start = 10000
    end = 100000000

    """
    ecg, eda = chest_data_dict['ECG'], chest_data_dict['EDA']
    x = [i for i in range(len(baseline))]
    plt.plot(x, [ecg[i] for i in baseline])
    plt.show()
    
    stress = np.asarray([idx for idx, val in enumerate(labels) if val == 2])
    print(stress.shape)
    
    x = [i for i in range(len(stress))]
    plt.plot(x, [ecg[i] for i in stress])
    plt.show()
    """
    # Process the signals
    #ecg_data = chest_data_dict["ECG"][baseline].flatten()
    #ecg = nk.ecg_process(ecg=ecg_data, sampling_rate=700)

    """
    print("HRV", ecg['ECG']['HRV'])

    print("CC", ecg['ECG']['Cardiac_Cycles'])

    print(".............")
    print(ecg['df']['ECG_Signal_Quality'])

    """

    # Plot the processed dataframe, normalizing all variables for viewing purpose

    bio = nk.bio_process(ecg=chest_data_dict["ECG"][baseline].flatten(), rsp=chest_data_dict['Resp'][baseline].flatten()
                         , eda=chest_data_dict["EDA"][baseline].flatten(), sampling_rate=700)
    #nk.z_score(bio["df"]).plot()

    print(bio["ECG"].keys())
    print(bio["EDA"].keys())
    print(bio["RSP"].keys())

    #ECG
    print(bio["ECG"]["HRV"])
    print(bio["ECG"]["R_Peaks"])

    #EDA
    print(bio["EDA"]["SCR_Peaks_Amplitudes"])
    print(bio["EDA"]["SCR_Onsets"])


    #RSP
    print(bio["RSP"]["Cycles_Onsets"])
    print(bio["RSP"]["Cycles_Length"])

    print("Read data file")
    #Flow: Read data for all subjects -> Extract features (Preprocessing) -> Train the model

