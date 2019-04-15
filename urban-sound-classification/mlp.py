import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

tf.enable_eager_execution()


def mlp(x_train, y_train, x_test, y_test):
    learning_rate = 0.01
    batch_size = 200

    n_input = len(x_train[0])
    n_hidden_1 = 50
    n_hidden_2 = 50
    n_classes = 10

    inputs = tf.keras.Input(shape=(n_input,))

    print(inputs.shape)

    x = tf.keras.layers.Dense(n_hidden_1, activation='relu')(inputs)
    x = tf.keras.layers.Dense(n_hidden_2, activation='relu')(x)
    predictions = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)

    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1000, validation_data=(x_test, y_test))
    return history


def get_mfcc():
    data = np.load('npz/mfcc.npz')

    all_mfcc = data['all_mfcc']
    all_mfcc_m = data['all_mfcc_m']

    x = all_mfcc_m
    y = data['y']

    y = tf.keras.utils.to_categorical(y, num_classes=10)

    return x, y


def get_chroma():
    data = np.load('npz/chroma.npz')

    all_chroma = data['all_chroma']
    all_chroma_m = data['all_chroma_m']

    x = all_chroma_m
    y = data['y']

    y = tf.keras.utils.to_categorical(y, num_classes=10)

    return x, y


def get_features():
    data = np.load('npz/feature.npz')

    x = data['features']
    y = data['y']

    y = tf.keras.utils.to_categorical(y, num_classes=10)

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


def train(feature='mfcc'):
    if feature == 'mfcc':
        x, y = get_mfcc()
    elif feature == 'chroma':
        x, y = get_chroma()
    elif feature == 'features':
        x, y = get_features()
    else:
        x, y = get_mfcc()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    history = mlp(x_train, y_train, x_test, y_test)
    show(history)


train('mfcc')
