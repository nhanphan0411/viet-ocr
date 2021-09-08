import tensorflow as tf
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Embedding, Dense, Dropout, Flatten, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Bidirectional, LSTM, Reshape
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, MaxPool2D, Dropout, Lambda
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow.keras.backend as K
import matplotlib as plt
import os
import numpy as np

def ctc_loss_lambda_func(y_true, y_pred):
    """Function for computing the CTC loss"""

    input_length = tf.ones(BATCH_SIZE) * MAX_LABEL_LENGTH
    input_length = tf.expand_dims(input_length, axis=-1)
    label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype="int64")

    loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    loss = tf.reduce_mean(loss)

    return loss

def plot_stats(training_stats, val_stats, x_label='Training Steps', stats='loss'):
    stats, x_label = stats.title(), x_label.title()
    legend_loc = 'upper right' if stats=='loss' else 'lower right'
    training_steps = len(training_stats)
    test_steps = len(val_stats)

    plt.figure()
    plt.ylabel(stats)
    plt.xlabel(x_label)
    plt.plot(training_stats, label='Training ' + stats)
    plt.plot(np.linspace(0, training_steps, test_steps), val_stats, label='Validation ' + stats)
    plt.ylim([0,max(plt.ylim())])
    plt.legend(loc=legend_loc)
    plt.show()

def get_callbacks(checkpoint='weights_1.hdf5', monitor="val_loss", verbose=0):
    """Setup the list of callbacks for the model"""
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=True,
            verbose=verbose),
        EarlyStopping(
            monitor=monitor,
            min_delta=1e-8,
            patience=15,
            restore_best_weights=True,
            verbose=verbose),
        ReduceLROnPlateau(
            monitor=monitor,
            min_delta=1e-8,
            factor=0.2,
            patience=10,
            verbose=verbose)
    ]

    return callbacks

def build_model(input_size, d_model, learning_rate=1e-3):    
    inputs = Input(shape=(input_size))
 
    x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(inputs)

    # Block 1 
    x = Conv2D(64, (3,3), padding='same')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=3)(x)
    x = Activation('relu')(x)

    # Block 2 
    x = Conv2D(128, (3,3), padding='same')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=3)(x)
    x = Activation('relu')(x)

    # Block 3 
    x = Conv2D(256, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x_1 = x

    x = Conv2D(256, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_1])
    x = Activation('relu')(x)

    # Block 4 
    x = Conv2D(512, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x_2 = x

    x = Conv2D(512, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_2])
    x = Activation('relu')(x)

    # Block 5
    x = Conv2D(1024, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(3, 1))(x)
    x = Activation('relu')(x)

    x = MaxPool2D(pool_size=(3, 1))(x)

    squeezed = Lambda(lambda x: K.squeeze(x, 1))(x)

    blstm_1 = Bidirectional(LSTM(512, return_sequences=True, dropout = 0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(512, return_sequences=True, dropout = 0.2))(blstm_1)

    outputs = Dense(units = VOCAB_SIZE+1, activation = 'softmax')(blstm_2)

    model = Model(inputs, outputs)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=ctc_loss_lambda_func)

    return model 
