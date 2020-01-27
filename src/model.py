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

    if len(y_true.shape) > 2:
        y_true = tf.squeeze(y_true)

    input_length = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)
    input_length = tf.math.reduce_sum(input_length, axis=-1, keepdims=True)
    label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype="int64")

    loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    loss = tf.reduce_mean(loss)

    return loss

class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []
        self.batch_val_losses = []
        self.batch_val_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        # reset_metrics: the metrics returned will be only for this batch. 
        # If False, the metrics will be statefully accumulated across batches.
        self.model.reset_metrics()
  
    def on_test_batch_end(self, batch, logs=None):
        self.batch_val_losses.append(logs['loss'])
        self.batch_val_acc.append(logs['acc'])
        # reset_metrics: the metrics returned will be only for this batch. 
        # If False, the metrics will be statefully accumulated across batches.
        self.model.reset_metrics()

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
        TensorBoard(
            log_dir='./logs',
            histogram_freq=10,
            profile_batch=0,
            write_graph=True,
            write_images=False,
            update_freq="epoch"),
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
    inputs = Input(shape=input_size)
    (64, 1024, 1)
    conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
    (32, 512, 64)
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
    
    (32, 512, 128)
    conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
    (16, 256, 128)
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

    (16, 256, 256)
    conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)
    (8, 256, 256)
    pool_3 = MaxPool2D(pool_size=(2, 1))(conv_3)
    batch_norm_3 = BatchNormalization()(pool_3)

    conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(batch_norm_3)
    batch_norm_5 = BatchNormalization()(conv_4)

    (8, 256, 512)
    conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)

    conv_7 = Conv2D(512, (2,2), activation = 'relu', padding='same')(batch_norm_6)
    (4, 128, 512)
    pool_7 = MaxPool2D(pool_size=(2, 2))(conv_7)

    conv_8 = Conv2D(512, (2,2), activation = 'relu', padding='same')(pool_7)
    (2, 128, 512)
    pool_8 = MaxPool2D(pool_size=(2, 1))(conv_8)
    (1, 128, 512)
    pool_9 = MaxPool2D(pool_size=(2, 1))(pool_8)
    
    # # to remove the first dimension of one: (1, 31, 512) to (31, 512) 
    squeezed = Lambda(lambda x: K.squeeze(x, 1))(pool_9)
    
    # # # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(blstm_1)

    # # this is our softmax character proprobility with timesteps 
    outputs = Dense(units = d_model, activation = 'softmax')(blstm_2)

    # model to be used at test time
    model = Model(inputs, outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=ctc_loss_lambda_func)

    return model 