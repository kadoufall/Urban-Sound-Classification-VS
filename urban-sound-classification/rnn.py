import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

tf.enable_eager_execution()


def gru(x_train, y_train, x_test, y_test):
    learning_rate = 0.01
    batch_size = 300

    n_timesteps = len(x_train[0])
    n_feature = len(x_train[0][0])

    inputs = tf.keras.Input(shape=(n_timesteps, n_feature))

    x = tf.keras.layers.CuDNNGRU(50)(inputs)
    predictions = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)

    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1000, validation_data=(x_test, y_test))
    return history


def lstm(x_train, y_train, x_test, y_test):
    learning_rate = 0.01
    batch_size = 300

    n_timesteps = len(x_train[0])
    n_feature = len(x_train[0][0])

    inputs = tf.keras.Input(shape=(n_timesteps, n_feature))

    x = tf.keras.layers.CuDNNLSTM(50)(inputs)
    predictions = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)

    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1000, validation_data=(x_test, y_test))
    return history


def get_mfcc():
    data = np.load('npz/mfcc_scale.npz')

    all_mfcc = data['all_mfcc']
    all_mfcc_scale = data['all_mfcc_scale']

    x = all_mfcc_scale
    y = data['y']

    return x, y


def get_chroma():
    data = np.load('npz/chroma_scale.npz')

    all_chroma = data['all_chroma']
    all_chroma_scale = data['all_chroma_scale']

    x = all_chroma_scale
    y = data['y']

    return x, y


def show(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def train(feature='mfcc', net='gru'):
    if feature == 'mfcc':
        x, y = get_mfcc()
    elif feature == 'chroma':
        x, y = get_chroma()
    else:
        x, y = get_mfcc()

    x = x.transpose((0, 2, 1))
    y = tf.keras.utils.to_categorical(y, num_classes=10)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    if net == 'gru':
        history = gru(x_train, y_train, x_test, y_test)
    else:
        history = lstm(x_train, y_train, x_test, y_test)
    show(history)


train('mfcc', 'gru')
