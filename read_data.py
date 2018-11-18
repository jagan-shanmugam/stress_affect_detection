import os
import pickle
data_set_path = "/media/jac/New Volume/Datasets/WESAD"
subject = 'S3'


def load_data(self, path, subject):
    """Given path and subject, load the data of the subject"""
    os.chdir(path)
    os.chdir(subject)
    with open(subject + '.pkl', 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    return data

class read_data:
    """Read data from WESAD dataset"""
    def __init__(self, data):
        self.keys = ['label', 'subject', 'signal']
        self.signal_keys = ['wrist', 'chest']
        self.chest_sensor_keys = ['ACC', 'ECG', 'EDA', 'EMG', 'RESP', 'TEMP']
        self.wrist_sensor_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
        self.data = data

    def get_label(self):
        return self.data[self.keys[0]]

    def get_wrist_data(self):
        label = self.data[self.keys[0]]
        assert subject == self.data[self.keys[1]]
        signal = self.data[self.keys[2]]
        wrist_data = signal[self.signal_keys[0]]
        chest_data = signal[self.signal_keys[1]]
        wrist_ACC = wrist_data[self.wrist_sensor_keys[0]]
        wrist_ECG = wrist_data[self.wrist_sensor_keys[1]]

if __name__ == '__main__':
    #print(label.size)  # 4545100
    #print(wrist_ACC.size)  # 623328
    #print(wrist_ECG.size)  # 415552
    print("Test")
    #Flow: Read data for all subjects -> Extract features (Preprocessing) -> Train the model