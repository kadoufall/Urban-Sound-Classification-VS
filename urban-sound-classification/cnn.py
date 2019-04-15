import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

tf.enable_eager_execution()


def cnn(x_train, y_train, x_test, y_test):
    learning_rate = 0.001
    batch_size = 100

    inputs = tf.keras.Input(shape=(len(x_train[0]), len(x_train[0][0]), 1))

    x = tf.keras.layers.Conv2D(32, kernel_size=3)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

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


def train(feature='mfcc'):
    if feature == 'mfcc':
        x, y = get_mfcc()
    elif feature == 'chroma':
        x, y = get_chroma()
    else:
        x, y = get_mfcc()

    x = np.reshape(x, (len(x), len(x[0]), len(x[0][0]), 1))
    y = tf.keras.utils.to_categorical(y, num_classes=10)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    history = cnn(x_train, y_train, x_test, y_test)
    show(history)


train('chroma')
