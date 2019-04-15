import os

import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder, scale

data_path = 'data'
data = pd.read_csv(os.path.join(data_path, 'train.csv'))

sound_classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling',
                 'gun_shot', 'jackhammer', 'siren', 'street_music']
le = LabelEncoder()
le.fit(sound_classes)
data['label'] = le.transform(data['Class'])


def parse_wav(data):
    all_chroma = np.empty((0, 12, 173))
    all_chroma_m = np.empty((0, 12))
    all_chroma_scale = np.empty((0, 12, 173))

    for i, row in data.iterrows():
        id = row[0]
        print(id)

        wav_file = os.path.join(data_path, 'Train', str(id) + '.wav')
        time_series, sampling_rate = librosa.load(wav_file, res_type='kaiser_fast')

        chroma = librosa.feature.chroma_stft(y=time_series, sr=sampling_rate)
        chroma_m = np.mean(chroma, axis=1).T

        if chroma.shape[1] < 173:
            padding = np.zeros((12, 173 - chroma.shape[1]))
            chroma = np.concatenate([chroma, padding], axis=1)

        all_chroma = np.vstack((all_chroma, [chroma]))
        all_chroma_m = np.vstack((all_chroma_m, [chroma_m]))

        chroma_scale = scale(chroma)
        all_chroma_scale = np.vstack((all_chroma_scale, [chroma_scale]))

    return all_chroma, all_chroma_m, all_chroma_scale


all_chroma, all_chroma_m, all_chroma_scale = parse_wav(data)
print(all_chroma.shape, all_chroma_m.shape, all_chroma_scale.shape)

y = np.array(data['label'].tolist())

np.savez('npz/chroma_scale', all_chroma=all_chroma, all_chroma_m=all_chroma_m, y=y,
         all_chroma_scale=all_chroma_scale)
