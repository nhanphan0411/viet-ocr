#coding=utf-8
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K

import h5py
import numpy as np
import unicodedata
import json
import os
import pathlib
import cv2
import datetime
import argparse
import itertools

import preprocess as prep
import model as mb
import evaluation

charset_base = "φ #'()+,-./0123456789:ABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstuvwxyzĂÂÊÔƠƯăâêôơưÁẮẤÉẾÓỐỚÚỨÍÝáắấéếóốớúứíýÀẰẦÈỀÒỒỜÙỪÌỲàằầèềòồờùừìỳẢẲẨẺỂỎỔỞỦỬỈỶảẳẩẻểỏổởủửỉỷÃẴẪẼỄÕỖỠŨỮĨỸãẵẫẽễõỗỡũữĩỹẠẶẬẸỆỌỘỢỤỰỊỴạặậẹệọộợụựịỵĐđ"
vocab_size = len(charset_base)
MAX_LABEL_LENGTH = 277
INPUT_SIZE = (128, 2048, 1)
PAD_TK = "φ"

BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

# ------------- LOAD LABELS FOR TRAINING -------------
def get_label(type_):
    labels = json.load(open(os.path.join('..', 'data', '{}.json'.format(type_))))
    return labels

def text_to_labels(text):
    return np.asarray(list(map(lambda x: charset_base.index(x), text)), dtype=np.uint8)

def labels_to_text(labels):
    return ''.join(list(map(lambda x: charset_base[x] if x < len(charset_base) else "", labels)))

# ------------- LOAD IMAGES FOR TRAINING -------------
def load_image_to_tensor(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image

def load_and_preprocess_from_path_label(path, label):
    return load_image_to_tensor(path), label

# ------------- PREPARE TF DATASET -------------
def prepare_for_training(dataset, cache=True, shuffle_buffer_size=100, augment=False):
    if cache:
        if isinstance(cache, str):
            dataset = dataset.cache(cache)
        else: 
            dataset = dataset.cache()
    
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    
    if augment == True:
        dataset.map(prep.augmentation, num_parallel_calls=AUTOTUNE)
    
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset

def prepare_for_testing(dataset):
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset

def build_dataset(type_, cache=False, augment=False, training=True):
    """Load and parse dataset.
       Args: type_ : string to dictate 'train' or 'test' session
             cache: boolean to indicate logging. 
             trainning: boolean to indicate training mode
             augment: boolean to indicate augment option. 
    """
    # Load dataset and pad labels
    DATA_FOLDER = os.path.join('..', 'data', type_)
    dataset = tf.data.Dataset.list_files(os.path.join(DATA_FOLDER, '*'))
    labels = get_label(type_) 
    all_image_paths = [str(item) for item in pathlib.Path(DATA_FOLDER).glob('*') if item.name in labels]
    labels = [labels[pathlib.Path(path).name] for path in all_image_paths]
    all_image_labels = [text_to_labels(label) for label in labels]
    all_image_labels = pad_sequences(all_image_labels, maxlen=MAX_LABEL_LENGTH, padding='post')

    n_samples = len(all_image_labels)
    steps_per_epoch = tf.math.ceil(n_samples/BATCH_SIZE)

    # Creat a dataset from images paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
    
    # Parse and preprocess observations in parallel
    dataset = dataset.map(load_and_preprocess_from_path_label, num_parallel_calls = AUTOTUNE)
    
    if training:
        dataset = prepare_for_training(dataset, cache=cache, shuffle_buffer_size=n_samples, augment=augment)
    else: 
        dataset = prepare_for_testing(dataset)

    return dataset, steps_per_epoch, labels

# ------------- DECODE PREDICTIONS -------------
def decode_batch(out):
    result = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        result.append(outstr)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--path", type=str, required=False) # path to test data
    args = parser.parse_args()
    
    # ------------- TRAIN -------------
    if args.train:
        train_ds, num_steps_train, _ = build_dataset('train', cache=True, augment=True)
        test_ds, num_steps_val, _ = build_dataset('test', training=False)
        model = mb.build_model(input_size=INPUT_SIZE, d_model=vocab_size+1, learning_rate=0.001)
        callbacks = mb.get_callbacks('weights_1.hdf5', 'val_loss', 1)
        batch_stats_callback = mb.CollectBatchStats()
        start_time = datetime.datetime.now()

        h = model.fit(train_ds,
                    steps_per_epoch = num_steps_train,
                    epochs=100,
                    validation_data = test_ds,
                    validation_steps = num_steps_val,
                    callbacks=callbacks)

        total_time = datetime.datetime.now() - start_time

        loss = h.history['loss']
        val_loss = h.history['val_loss']
        min_val_loss = min(val_loss)
        min_val_loss_i = val_loss.index(min_val_loss)

        time_epoch = (total_time / len(loss))

        t_corpus = "\n".join([
            "Batch:                   {}\n".format(BATCH_SIZE),
            "Time per epoch:          {}".format(time_epoch),
            "Total epochs:            {}".format(len(loss)),
            "Best epoch               {}\n".format(min_val_loss_i + 1),
            "Training loss:           {}".format(loss[min_val_loss_i]),
            "Validation loss:         {}".format(min_val_loss),
        ])

        with open(os.path.join("train.txt"), "w") as f:
            f.write(t_corpus)
            print(t_corpus)

    # ------------- TRAINING PROCESS -------------
    elif args.test:
        checkpoint = './weights_1.hdf5'
        assert os.path.isfile(checkpoint) and os.path.exists(args.path)
        type_ = pathlib.Path(args.path).name
        
        ds, num_steps, labels = build_dataset(type_, training=False)
        model = mb.build_model(input_size=INPUT_SIZE, d_model=vocab_size+1)
        # model.load_weights(checkpoint)
        model.summary()

        start_time = datetime.datetime.now()

        predictions = model.predict(ds, steps=num_steps)

        # CTC DECODE
        ctc_decode = True
        if ctc_decode:
            predicts, probabilities = [], []
            x_test = np.array(predictions)
            x_test_len = [MAX_LABEL_LENGTH for _ in range(len(x_test))]

            decode, log = K.ctc_decode(x_test,
                                    x_test_len,
                                    greedy=False,
                                    beam_width=10,
                                    top_paths=1)

            probabilities = [np.exp(x) for x in log]
            predicts = [[[int(p) for p in x if p != -1] for x in y] for y in decode]
            predicts = np.swapaxes(predicts, 0, 1)
            predicts = [labels_to_text(label[0]) for label in predicts]
        else:
            predicts = decode_batch(predictions)

        total_time = datetime.datetime.now() - start_time
        print(predicts)
        print(labels)
        
        # ------------- EVALUATE -------------
        prediction_file = os.path.join('.', 'predictions_{}.txt'.format(type_))

        with open(prediction_file, "w") as f:
            for pd, gt in zip(predicts, labels):
                f.write("Y {}\nP {}\n".format(gt, pd))

        evaluate = evaluation.ocr_metrics(predicts=predicts,
                                          ground_truth=labels,
                                          norm_accentuation=False,
                                          norm_punctuation=False)

        e_corpus = "\n".join([
            "Total test images:    {}".format(len(labels)),
            "Total time:           {}".format(total_time),
            "Metrics:",
            "Character Error Rate: {}".format(evaluate[0]),
            "Word Error Rate:      {}".format(evaluate[1]),
            "Sequence Error Rate:  {}".format(evaluate[2]),
        ])

        with open("evaluate.txt", "w") as lg:
            lg.write(e_corpus)
            print(e_corpus)
