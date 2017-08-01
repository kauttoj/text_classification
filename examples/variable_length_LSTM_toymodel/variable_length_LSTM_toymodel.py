# https://stackoverflow.com/questions/38189070/how-do-i-create-a-variable-length-input-lstm-in-keras

import numpy as np
np.random.seed(1)

import tensorflow as tf
tf.set_random_seed(1)

from keras import models
from keras.layers import Dense, Masking, LSTM

import matplotlib.pyplot as plt


def stateful_model():
    hidden_units = 256

    model = models.Sequential()
    model.add(LSTM(hidden_units, batch_input_shape=(1, 1, 1), return_sequences=False, stateful=True))
    model.add(Dense(1, activation='relu', name='output'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    return model


def train_rnn(x_train, y_train, max_len, mask):
    epochs = 10
    batch_size = 200

    vec_dims = 1
    hidden_units = 256
    in_shape = (max_len, vec_dims)

    model = models.Sequential()

    model.add(Masking(mask, name="in_layer", input_shape=in_shape,))
    model.add(LSTM(hidden_units, return_sequences=False))
    model.add(Dense(1, activation='relu', name='output'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_split=0.05)

    return model


def gen_train_sig_cls_pair(t_stops, num_examples, mask):
    x = []
    y = []
    max_t = int(np.max(t_stops))

    for t_stop in t_stops:
        one_indices = np.random.choice(a=num_examples, size=num_examples // 2, replace=False)

        sig = np.zeros((num_examples, max_t), dtype=np.int8)
        sig[one_indices, 0] = 1
        sig[:, t_stop:] = mask
        x.append(sig)

        cls = np.zeros(num_examples, dtype=np.bool)
        cls[one_indices] = 1
        y.append(cls)

    return np.concatenate(x, axis=0), np.concatenate(y, axis=0)


def gen_test_sig_cls_pair(t_stops, num_examples):
    x = []
    y = []

    for t_stop in t_stops:
        one_indices = np.random.choice(a=num_examples, size=num_examples // 2, replace=False)

        sig = np.zeros((num_examples, t_stop), dtype=np.bool)
        sig[one_indices, 0] = 1
        x.extend(list(sig))

        cls = np.zeros((num_examples, t_stop), dtype=np.bool)
        cls[one_indices] = 1
        y.extend(list(cls))

    return x, y


if __name__ == '__main__':
    noise_mag = 0.01
    mask_val = -10
    signal_lengths = (10, 15, 20)

    x_in, y_in = gen_train_sig_cls_pair(signal_lengths, 10, mask_val)

    mod = train_rnn(x_in[:, :, None], y_in, int(np.max(signal_lengths)), mask_val)

    testing_dat, expected = gen_test_sig_cls_pair(signal_lengths, 3)

    state_mod = stateful_model()
    state_mod.set_weights(mod.get_weights())

    res = []
    for s_i in range(len(testing_dat)):
        seq_in = list(testing_dat[s_i])
        seq_len = len(seq_in)

        for t_i in range(seq_len):
            res.extend(state_mod.predict(np.array([[[seq_in[t_i]]]])))

        state_mod.reset_states()

    fig, axes = plt.subplots(2)
    axes[0].plot(np.concatenate(testing_dat), label="input")

    axes[1].plot(res, "ro", label="result", alpha=0.2)
    axes[1].plot(np.concatenate(expected, axis=0), "bo", label="expected", alpha=0.2)
    axes[1].legend(bbox_to_anchor=(1.1, 1))

    plt.show()