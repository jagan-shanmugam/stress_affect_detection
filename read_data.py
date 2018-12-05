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

class read_data_one_subject:
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
        #assert subject == self.data[self.keys[1]]
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

def extract_mean_std_features(ecg_data, label=0, block = 700):
    #print (len(ecg_data))
    i = 0
    mean_features = np.empty(int(len(ecg_data)/block), dtype=np.float64)
    std_features = np.empty(int(len(ecg_data)/block), dtype=np.float64)
    max_features = np.empty(int(len(ecg_data)/block), dtype=np.float64)
    min_features = np.empty(int(len(ecg_data)/block), dtype=np.float64)

    idx = 0
    while i < len(ecg_data):
        temp = ecg_data[i:i+block]
        #print(len(temp))
        if idx < int(len(ecg_data)/block):
            mean_features[idx] = np.mean(temp)
            std_features[idx] = np.std(temp)
            min_features[idx] = np.amin(temp)
            max_features[idx] = np.amax(temp)
        i += 700
        idx += 1
    #print(len(mean_features), len(std_features))
    #print(mean_features, std_features)
    features = {'mean':mean_features, 'std':std_features, 'min':min_features, 'max':max_features}

    one_set = np.column_stack((mean_features, std_features, min_features, max_features))
    return one_set

def extract_one(chest_data_dict, idx, l_condition=0):
    ecg_data = chest_data_dict["ECG"][idx].flatten()
    ecg_features = extract_mean_std_features(ecg_data, label=l_condition)
    #print(ecg_features.shape)

    eda_data = chest_data_dict["EDA"][idx].flatten()
    eda_features = extract_mean_std_features(eda_data, label=l_condition)
    #print(eda_features.shape)

    emg_data = chest_data_dict["EMG"][idx].flatten()
    emg_features = extract_mean_std_features(emg_data, label=l_condition)
    #print(emg_features.shape)

    temp_data = chest_data_dict["Temp"][idx].flatten()
    temp_features = extract_mean_std_features(temp_data, label=l_condition)
    #print(temp_features.shape)

    baseline_data = np.hstack((eda_features, temp_features, ecg_features, emg_features))
    #print(len(baseline_data))
    label_array = np.full(len(baseline_data), l_condition)
    #print(label_array.shape)
    #print(baseline_data.shape)
    baseline_data = np.column_stack((baseline_data, label_array))
    #print(baseline_data.shape)
    return baseline_data

def recur_print(ecg):
    while ecg is dict:
        print(ecg.keys())
        for k in ecg.keys():
            recur_print(ecg[k])

def execute():
    data_set_path = "/media/jac/New Volume/Datasets/WESAD"
    file_path = "ecg.txt"
    subject = 'S3'
    obj_data = {}
    labels = {}
    all_data = {}
    subs = [2, 3, 4, 5, 6]
    for i in subs:
        subject = 'S' + str(i)
        print("Reading data", subject)
        obj_data[subject] = read_data_one_subject(data_set_path, subject)
        labels[subject] = obj_data[subject].get_labels()

        wrist_data_dict = obj_data[subject].get_wrist_data()
        wrist_dict_length = {key: len(value) for key, value in wrist_data_dict.items()}

        chest_data_dict = obj_data[subject].get_chest_data()
        chest_dict_length = {key: len(value) for key, value in chest_data_dict.items()}
        print(chest_dict_length)
        chest_data = np.concatenate((chest_data_dict['ACC'], chest_data_dict['ECG'], chest_data_dict['EDA'],
                                     chest_data_dict['EMG'], chest_data_dict['Resp'], chest_data_dict['Temp']), axis=1)
        # Get labels


        # 'ACC' : 3, 'ECG' 1: , 'EDA' : 1, 'EMG': 1, 'RESP': 1, 'Temp': 1  ===> Total dimensions : 8
        # No. of Labels ==> 8 ; 0 = not defined / transient, 1 = baseline, 2 = stress, 3 = amusement,
        # 4 = meditation, 5/6/7 = should be ignored in this dataset

        # Do for each subject
        baseline = np.asarray([idx for idx, val in enumerate(labels[subject]) if val == 1])
        # print("Baseline:", chest_data_dict['ECG'][baseline].shape)
        # print(baseline.shape)

        stress = np.asarray([idx for idx, val in enumerate(labels[subject]) if val == 2])
        # print(stress.shape)

        amusement = np.asarray([idx for idx, val in enumerate(labels[subject]) if val == 3])
        # print(amusement.shape)

        baseline_data = extract_one(chest_data_dict, baseline, l_condition=1)
        stress_data = extract_one(chest_data_dict, stress, l_condition=2)
        amusement_data = extract_one(chest_data_dict, amusement, l_condition=3)

        full_data = np.vstack((baseline_data, stress_data, amusement_data))
        print("One subject data", full_data.shape)
        all_data[subject] = full_data

    i = 0
    for k, v in all_data.items():
        if i == 0:
            data = all_data[k]
            i += 1
        print(all_data[k].shape)
        data = np.vstack((data, all_data[k]))

    print(data.shape)
    return data

if __name__ == '__main__':
    execute()
    """
    ecg, eda = chest_data_dict['ECG'], chest_data_dict['EDA']
    x = [i for i in range(len(baseline))]
    for one in baseline:
        x = [i for i in range(99)]
        plt.plot(x, ecg[one:100])
        break
    """
    #x = [i for i in range(10000)]
    #plt.plot(x, chest_data_dict['ECG'][:10000])
    #plt.show()

    # BASELINE

    #                                    [ecg_features[k] for k in ecg_features.keys()])

    #ecg = nk.ecg_process(ecg=ecg_data, rsp=chest_data_dict['Resp'][baseline].flatten(), sampling_rate=700)
    #print(os.getcwd())

    """
    #recur_print
    print(type(ecg))
    print(ecg.keys())
    for k in ecg.keys():
        print(k)
        for i in ecg[k].keys():
            print(i)
    

    resp = nk.eda_process(eda=chest_data_dict['EDA'][baseline].flatten(), sampling_rate=700)
    resp = nk.rsp_process(chest_data_dict['Resp'][baseline].flatten(), sampling_rate=700)
    for k in resp.keys():
        print(k)
        for i in resp[k].keys():
            print(i)
    
    # For baseline, compute mean, std, for each 700 samples. (1 second values)

    #file_path = os.getcwd()
    with open(file_path, "w") as file:
        #file.write(str(ecg['df']))
        file.write(str(ecg['ECG']['HRV']['RR_Intervals']))
        file.write("...")
        file.write(str(ecg['RSP']))
        #file.write("RESP................")
        #file.write(str(resp['RSP']))
        #file.write(str(resp['df']))
        #print(type(ecg['ECG']['HRV']['RR_Intervals']))

        #file.write(str(ecg['ECG']['Cardiac_Cycles']))
        #print(type(ecg['ECG']['Cardiac_Cycles']))

        #file.write(ecg['ECG']['Cardiac_Cycles'].to_csv())

    # Plot the processed dataframe, normalizing all variables for viewing purpose
    """
    """
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
    """
    print("Read data file")
    #Flow: Read data for all subjects -> Extract features (Preprocessing) -> Train the model

