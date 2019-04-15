import os

import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, scale

data_path = 'data'
data = pd.read_csv(os.path.join(data_path, 'train.csv'))

sound_classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling',
                 'gun_shot', 'jackhammer', 'siren', 'street_music']
le = LabelEncoder()
le.fit(sound_classes)
data['label'] = le.transform(data['Class'])


def parse_wav(data):
    all_zrc_m = np.empty((0, 1))
    all_cent_m = np.empty((0, 1))
    all_mfcc_m = np.empty((0, 20))
    all_chroma_m = np.empty((0, 12))
    all_tonnetz_m = np.empty((0, 6))

    for i, row in data.iterrows():
        id = row[0]
        print(id)

        wav_file = os.path.join(data_path, 'Train', str(id) + '.wav')
        time_series, sampling_rate = librosa.load(wav_file, res_type='kaiser_fast')

        zcr = librosa.feature.zero_crossing_rate(time_series)
        cent = librosa.feature.spectral_centroid(y=time_series, sr=sampling_rate)
        mfccs = librosa.feature.mfcc(y=time_series, sr=sampling_rate, n_mfcc=20)
        chroma = librosa.feature.chroma_stft(y=time_series, sr=sampling_rate)
        tonnetz = librosa.feature.tonnetz(y=time_series, sr=sampling_rate)

        zrc_m = np.mean(zcr, axis=1).T
        cent_m = np.mean(cent, axis=1).T
        mfccs_m = np.mean(mfccs, axis=1).T
        chroma_m = np.mean(chroma, axis=1).T
        tonnetz_m = np.mean(tonnetz, axis=1).T

        mfccs_m = scale(mfccs_m)
        chroma_m = scale(chroma_m)
        tonnetz_m = scale(tonnetz_m)

        all_zrc_m = np.vstack((all_zrc_m, [zrc_m]))
        all_cent_m = np.vstack((all_cent_m, [cent_m]))
        all_mfcc_m = np.vstack((all_mfcc_m, [mfccs_m]))
        all_chroma_m = np.vstack((all_chroma_m, [chroma_m]))
        all_tonnetz_m = np.vstack((all_tonnetz_m, [tonnetz_m]))

    return all_zrc_m, all_cent_m, all_mfcc_m, all_chroma_m, all_tonnetz_m


all_zrc_m, all_cent_m, all_mfcc_m, all_chroma_m, all_tonnetz_m = parse_wav(data)
all_zrc_m = scale(all_zrc_m)
all_cent_m = scale(all_cent_m)

features = np.hstack([all_zrc_m, all_cent_m, all_mfcc_m, all_chroma_m, all_tonnetz_m])
y = np.array(data['label'].tolist())

np.savez('npz/feature', features=features, y=y)
