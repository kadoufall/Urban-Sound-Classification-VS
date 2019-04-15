import os

import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

sound_classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling',
                 'gun_shot', 'jackhammer', 'siren', 'street_music']


def get_durations(row):
    id = row[0]
    print(id)
    wav_file = os.path.join(data_path, 'Train', str(id) + '.wav')
    time_series, sampling_rate = librosa.load(wav_file, res_type='kaiser_fast')
    duration = librosa.get_duration(y=time_series, sr=sampling_rate)
    return duration


def data_explore(data):
    print(data.count())
    print(data['Class'].value_counts())
    print(data['duration'].value_counts())


data_path = 'data'
data = pd.read_csv(os.path.join(data_path, 'train.csv'))
# data['duration'] = data.apply(get_durations, axis=1)
# data_explore(data)

wav_air_conditioner = os.path.join(data_path, 'Train', '22.wav')
wav_car_horn = os.path.join(data_path, 'Train', '48.wav')
wav_children_playing = os.path.join(data_path, 'Train', '6.wav')
wav_dog_bark = os.path.join(data_path, 'Train', '4.wav')
wav_drilling = os.path.join(data_path, 'Train', '2.wav')
wav_engine_idling = os.path.join(data_path, 'Train', '17.wav')
wav_gun_shot = os.path.join(data_path, 'Train', '12.wav')
wav_jackhammer = os.path.join(data_path, 'Train', '33.wav')
wav_siren = os.path.join(data_path, 'Train', '3.wav')
wav_street_music = os.path.join(data_path, 'Train', '10.wav')

wav_files = [wav_air_conditioner, wav_car_horn, wav_children_playing, wav_dog_bark, wav_drilling,
             wav_engine_idling, wav_gun_shot, wav_jackhammer, wav_siren, wav_street_music]


def waveplot(wav_file, class_name):
    time_series, sampling_rate = librosa.load(wav_file)

    plt.figure(figsize=(14, 5))
    plt.title('Amplitude envelope - ' + class_name)
    librosa.display.waveplot(time_series, sr=sampling_rate)
    plt.tight_layout()
    plt.show()


# for i in range(len(sound_classes)):
#     waveplot(wav_files[i], sound_classes[i])


def specshow(wav_file, class_name):
    time_series, sampling_rate = librosa.load(wav_file)
    X = librosa.stft(time_series)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    plt.title('Amplitude envelope - Hz - ' + class_name)
    librosa.display.specshow(Xdb, sr=sampling_rate, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.tight_layout()
    plt.show()


# for i in range(len(sound_classes)):
#     specshow(wav_files[i], sound_classes[i])


def specshow_log(wav_file, class_name):
    time_series, sampling_rate = librosa.load(wav_file)
    X = librosa.stft(time_series)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    plt.title('Amplitude envelope - Log - ' + class_name)
    librosa.display.specshow(Xdb, sr=sampling_rate, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.tight_layout()
    plt.show()


for i in range(len(sound_classes)):
    specshow_log(wav_files[i], sound_classes[i])
