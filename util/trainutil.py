
import numpy as np
import os

from datetime import datetime as dt
from keras.wrappers.scikit_learn import KerasClassifier
from keras import optimizers

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, CSVLogger
from keras.models import Model
from keras.models import load_model

from keras.callbacks import LambdaCallback
from operator import itemgetter

def _epochOutput(epoch, logs):

    print("Finished epoch: " + str(epoch))
    print(logs)

    # if os.listdir(dirname)

def _delete_oldest_weightfile(dirname):
    weight_files = []
    for file in os.listdir(dirname):
        base, ext = os.path.splitext(file)
        if ext == 'hdf5':
            weight_files.append([file, os.path.getctime(file)])

    weight_files.sort(key=itemgetter(1), reverse=True)
    os.remove(weight_files[-1][0])

def _get_date_str():
    tdatetime = dt.now()
    tstr = tdatetime.strftime('%Y_%m%d_%H%M')
    return tstr

def _make_dir(dir_name):
    if not os.path.exists('dir_name'):
        os.makedirs(dir_name)

def train(model, train_gen, val_gen, steps_per_epoch=None, optimizer='adam', log_dir='./log', epochs=100, loss='binary_crossentropy', metrics=['accuracy']):
    if steps_per_epoch is None:
        steps_per_epoch = len(train_gen)

    sub_dir = _get_date_str()
    log_dir = log_dir + '/' + sub_dir
    # make log dir
    _make_dir(log_dir)
    # saved model path
    fpath = log_dir + '/weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
    # callback
    tb_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)
    cp_cb = ModelCheckpoint(filepath=log_dir+'/best_weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    batchLogCallback = LambdaCallback(on_epoch_end=_epochOutput)
    csv_logger = CSVLogger(log_dir + '/training.log')
    callbacks = [batchLogCallback, csv_logger, cp_cb]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print(model.summary())
    model.fit_generator(
        train_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=2,
        validation_data=val_gen,
        workers=8,
        callbacks=callbacks,
        )

    model.save(log_dir + '/' + str(epochs) + 'epochs_final_save')