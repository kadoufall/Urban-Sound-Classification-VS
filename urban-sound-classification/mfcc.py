import os

import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, scale

data_path = 'data'
data = pd.read_csv(os.path.join(data_path, 'train.csv'))

print(data.shape)

sound_classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling',
                 'gun_shot', 'jackhammer', 'siren', 'street_music']
le = LabelEncoder()
le.fit(sound_classes)
data['label'] = le.transform(data['Class'])


def parse_wav(data):
    n_mfcc = 40
    all_mfcc = np.empty((0, n_mfcc, 173))
    all_mfcc_m = np.empty((0, n_mfcc))
    all_mfcc_scale = np.empty((0, n_mfcc, 173))

    for i, row in data.iterrows():
        id = row[0]
        print(id)

        wav_file = os.path.join(data_path, 'Train', str(id) + '.wav')
        time_series, sampling_rate = librosa.load(wav_file, res_type='kaiser_fast')

        mfcc = librosa.feature.mfcc(y=time_series, sr=sampling_rate, n_mfcc=n_mfcc)
        mfcc_m = np.mean(mfcc, axis=1).T

        if mfcc.shape[1] < 173:
            padding = np.zeros((n_mfcc, 173 - mfcc.shape[1]))
            mfcc = np.concatenate([mfcc, padding], axis=1)

        all_mfcc = np.vstack((all_mfcc, [mfcc]))
        all_mfcc_m = np.vstack((all_mfcc_m, [mfcc_m]))

        mfcc_scale = scale(mfcc)
        all_mfcc_scale = np.vstack((all_mfcc_scale, [mfcc_scale]))

    return all_mfcc, all_mfcc_m, all_mfcc_scale


all_mfcc, all_mfcc_m, all_mfcc_scale = parse_wav(data)
print(all_mfcc.shape, all_mfcc_m.shape, all_mfcc_scale.shape)

y = np.array(data['label'].tolist())

np.savez('npz/mfcc_scale', all_mfcc=all_mfcc, all_mfcc_m=all_mfcc_m, y=y, all_mfcc_scale=all_mfcc_scale)
