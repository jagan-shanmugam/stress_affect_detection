import os
import pickle
os.chdir("..")
os.chdir("..")
os.chdir("..")
os.chdir("..")
os.chdir("media/jac/New Volume/Datasets/WESAD")
print(os.listdir())

subject = 'S3'
keys = ['label', 'subject', 'signal']
signal_keys = ['wrist', 'chest']
chest_sensor_keys = ['ACC', 'ECG', 'EDA', 'EMG', 'RESP', 'TEMP']
wrist_sensor_keys = ['ACC', 'BVP', 'EDA', 'TEMP']

os.chdir(subject)

with open(subject+'.pkl', 'rb') as file:
    data = pickle.load(file, encoding='latin1')

label = data[keys[0]]
assert subject == data[keys[1]]
signal = data[keys[2]]
wrist_data = signal[signal_keys[0]]
chest_data = signal[signal_keys[1]]
wrist_ACC = wrist_data[wrist_sensor_keys[0]]
wrist_ECG = wrist_data[wrist_sensor_keys[1]]

print(label.size) #4545100
print(wrist_ACC.size) #623328
print(wrist_ECG.size) #415552
