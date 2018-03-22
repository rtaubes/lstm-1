"""
    LSTM model and settings.
    LSTM model is based on
    1) collect the last rows from an each batch in prediction.
    A prediction is an array with shape (batch_size, x_steps, neurons in the last layer).
    For this model the result is the same as a 'state.h' returned by dynamic_rnn.
    2) use a 'dense' layer for the collected values with output shape equal 'y_features'
    3) the result of (2) is an array with shape (batch_size, y_features).  It can be used
    directly to calculate 'loss' and as a result.
"""

import os
import pathlib
import shutil
import json
import tensorflow as tf
import numpy as np

from utils import (frozen_cls, TimeRec)


class Settings:
    """ Model parameters.  """
    __slots__ = (
    "state_fold", "data_file", "model_name", "neurons", "loss",
    "x_steps", "x_features", "y_features", "learning_rate",
    "train_sz", "validate_sz", "batch_sz", "epochs"
    )
    slots = __slots__

    def __init__(self, *, model_name, state_fold, data_file, neurons, loss, x_steps,
                 x_features, y_features, learning_rate, train_sz, validate_sz, batch_sz, epochs):
        self.state_fold = state_fold
        self.data_file = data_file
        self.model_name = model_name
        self.neurons = neurons
        self.loss = loss
        self.x_steps = x_steps
        self.x_features = x_features
        self.y_features = y_features
        self.learning_rate = learning_rate
        self.train_sz = train_sz
        self.validate_sz = validate_sz
        self.batch_sz = batch_sz
        self.epochs = epochs

    @staticmethod
    def from_json(fname='settings.json'):
        """ Read settings from a json file """
        with open(fname, 'r') as jfile:
            file_sets = json.load(jfile)
            fkeys = set(file_sets.keys())
            miss_keys = set(Settings.slots) - fkeys
            if miss_keys:
                raise AttributeError("Undefined attributes in the json file: {}".format(miss_keys))
            extra_keys = fkeys - set(Settings.slots)
            if extra_keys:
                raise AttributeError("Unused attributes in the json file: {}".format(extra_keys))
            print(file_sets)
            obj = Settings(**file_sets)
            return obj

    def to_json(self, fname='settings.json'):
        file_sets = {}
        for key in self.__slots__:
            file_sets[key] = self.__getattribute__(key)
        with open(fname, 'w') as jfile:
            json.dump(file_sets, jfile)


class Model:
    """ LSTM model """
    MAKE_NEW = 0
    FROM_LATEST = 1
    FROM_CUSTOM = 2
    VERSION = 1.1

    def __init__(self, settings, data_src):
        self._sets = settings
        self._data_src = data_src
        # Tensorflow
        self._init_vars = None
        self._x = None
        self._y = None
        self._keep_prob = None
        self._calc_y = None
        self._calc_loss = None
        self._training_op = None
        self._global_step = None
        # Internals
        self._state = Model.MAKE_NEW
        self._restore_path = None
        self._ckpt_fold = os.path.join(self._sets.state_fold, self._sets.model_name, "ckpt")
        self._log_tb_fold = os.path.join(self._sets.state_fold, self._sets.model_name, "tb_logs")
        self._store_file = None

    def __del__(self):
        print("Model.delete()")

    def _create_folders(self):
        """ Create folders structure """
        path1 = pathlib.Path(self._ckpt_fold)
        path1.mkdir(parents=True, exist_ok=True)
        path2 = pathlib.Path(self._log_tb_fold)
        path2.mkdir(parents=True, exist_ok=True)

    def create(self):
        shutil.rmtree(self._ckpt_fold, ignore_errors=True)
        shutil.rmtree(self._log_tb_fold, ignore_errors=True)
        self._create_folders()
        self._state = Model.MAKE_NEW

    def from_latest_point(self):
        self._state = Model.FROM_LATEST

    def from_custom_point(self, path):
        self._state = Model.FROM_CUSTOM
        self._restore_path = path

    def _create_structure(self, graph):
        with graph.as_default():
            with tf.name_scope('inputs'):
                self._keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')
                self._x = tf.placeholder(tf.float32,
                                         [None, self._sets.x_steps, self._sets.x_features],
                                         name='x')
                self._y = tf.placeholder(tf.float32,
                                         [None, self._sets.y_features],name='y')

            self._global_step = tf.Variable(0, trainable=False, name='global_step')
            cells = [tf.contrib.rnn.LSTMCell(num_units=neurons, activation=tf.nn.relu,
                                             state_is_tuple=True) for neurons in self._sets.neurons]
            cells_drop = [
                tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self._keep_prob) for cell in
                cells
            ]
            ml_cell = tf.contrib.rnn.MultiRNNCell(cells_drop)
            self._lstm_out, self._states = tf.nn.dynamic_rnn(ml_cell, self._x, dtype=tf.float32)

            # self._out1 = tf.transpose(self._lstm_out, [1, 0, 2], name='out1')
            # self._out2 = tf.gather(self._out1, self._out1.get_shape()[0] - 1, name='out2')
            #
            # self.weight = tf.Variable(tf.truncated_normal([self._sets.neurons[-1], self._sets.y_features]), name='weight')
            # self.bias = tf.Variable(tf.constant(0.1, shape=[self._sets.y_features]), name='bias')
            # self._calc_y = tf.matmul(self._out2, self.weight) + self.bias

            self._calc_y = tf.layers.dense(self._states[-1].h, self._sets.y_features,
                                           name='y_out')

            self._calc_loss = tf.reduce_mean(tf.square(self._calc_y - self._y))
            optimizer = tf.train.AdamOptimizer(learning_rate=self._sets.learning_rate)
            self._training_op = optimizer.minimize(self._calc_loss)
            self._init_vars = tf.global_variables_initializer()
        self._store_file = None

    def train(self):
        """ Train. Graph should """
        graph = tf.Graph()
        mse_lst = []
        time_rec = TimeRec()
        with graph.as_default():
            self._create_structure(graph)
            saver = tf.train.Saver()
            with tf.Session(graph=graph,
                            config=tf.ConfigProto(allow_soft_placement=True,
                                                  log_device_placement=True)) as sess:
                if self._state == Model.MAKE_NEW:
                    sess.run(self._init_vars)
                else:
                    self._restore_env(sess=sess, saver=saver)
                ckpt_file = os.path.join(self._ckpt_fold, "checkpoint")
                epoch_offset = self._global_step.eval(sess)
                print("epoch_offset:", epoch_offset)
                # train
                for epoch in range(self._sets.epochs):
                    mse_coll = []
                    self._data_src.reset()
                    for y_idx, x_batches, y_batches in self._data_src:
                        lstm_out, calc_y = sess.run(
                            [self._lstm_out, self._calc_y],
                            feed_dict={
                                self._x: x_batches, self._y: y_batches,
                                self._keep_prob: 1 - self._sets.loss})
                        # print("lstm_out.shape:", lstm_out.shape, ", calc_y.shape:", calc_y.shape)
                        mse_, _ = sess.run(
                            [self._calc_loss, self._training_op],
                            feed_dict={
                                self._x: x_batches, self._y: y_batches,
                                self._keep_prob: 1 - self._sets.loss})
                        mse_coll.append(mse_)
                    mse = np.mean(mse_coll)
                    prefix = ''
                    if epoch % 10 == 0:
                        self._global_step.load(epoch_offset + epoch, sess)
                        prefix = saver.save(sess, ckpt_file, global_step=epoch_offset + epoch)
                        prefix = ", saved to '{}'".format(prefix)
                    print("{:3d}/{}: {:.1f} sec:  MSE: {:.6f}{}".format(
                        epoch + 1, self._sets.epochs, time_rec.step(), mse, prefix))
                    mse_lst.append(mse)
                self._global_step.load(epoch_offset + self._sets.epochs, sess)
                # always save results of the last epoch
                self._store_file = saver.save(sess, ckpt_file,
                                              global_step=epoch_offset + self._sets.epochs - 1)
                print("model saved to:", self._store_file)
                print("total time: {:.3f}".format(time_rec.total()))
        self._state = Model.FROM_LATEST
        return mse_lst

    def _find_latest_ckpt_path(self):
        ckpt = tf.train.get_checkpoint_state(self._ckpt_fold)
        if ckpt and ckpt.model_checkpoint_path:
            return ckpt.model_checkpoint_path
        else:
            raise RuntimeError("No checkpoints found into the path {}".format(self._ckpt_fold))

    def evaluate(self):
        """ Evalute the model on the train set. The model must be trained using train(),
            because evaluation uses the same generator as the last train() procedure.
            Moreover, graph should be produced by train too.
        """
        predicts = []
        actual = []
        last_idx = -1
        graph = tf.Graph()
        time_rec = TimeRec()
        with graph.as_default():
            self._create_structure(graph)
            saver = tf.train.Saver()
            with tf.Session(graph=graph) as sess:
                self._restore_env(sess=sess, saver=saver)
                self._data_src.reset()
                train_batch_size = self._data_src.batch_size
                self._data_src.batch_size = 1
                bnum = 0
                for idx, x_batches, y_batches in self._data_src:
                    # TODO: x_batches has the first size==1. Remove the next line?
                    xtest = x_batches[0].reshape(1, self._sets.x_steps, 1)
                    y_calc = sess.run(self._calc_y,
                                      feed_dict={self._x: xtest, self._keep_prob: 1.0})
                    # print("p:", y_calc.shape, y_batches.shape)
                    # print("2:", x_batches)
                    # print("3:", y_batches)
                    # print("4:", idx, self._data_src.data_size)

                    pred_elem = y_calc[-1, 0]
                    gen_elem = y_batches[-1, 0]
                    predicts.append(pred_elem)
                    actual.append(gen_elem)
                    last_idx = idx
                    if bnum % 2000 == 0:
                        print("step:", bnum)
                    bnum += 1
                print("test steps: {}, total time: {:.3f}".format(bnum, time_rec.total()))
                self._data_src.batch_size = train_batch_size
                actual = actual[:-1]
                predicts = predicts[1:]
        return last_idx, np.array(actual), np.array(predicts)

    def predict(self):
        """ Make prediction """
        graph = tf.Graph()
        with graph.as_default():
            self._create_structure(graph)
            saver = tf.train.Saver()
            with tf.Session(graph=graph) as sess:
                self._restore_env(sess=sess, saver=saver)
                x_batches = self._data_src.get_last_x_batches(batch_size=1)
                y_calc = sess.run(self._calc_y,
                                  feed_dict={self._x: x_batches, self._keep_prob: 1.0})
                pred_elem = y_calc[0, -1, 0]
        return pred_elem

    @property
    def data_source(self):
        return self._data_src

    def _restore_env(self, sess=None, saver=None):
        """" Restore session and graph """
        if self._state == Model.FROM_LATEST:
            if self._store_file:
                path = self._store_file  # train() returns the latest path
            else:
                path = self._find_latest_ckpt_path()
            saver.restore(sess, path)
            print("model restored from", path)
        elif self._state == Model.FROM_CUSTOM:
            path = self._restore_path
            saver = tf.train.Saver()
            saver.restore(sess, path)
            print("model restored from", path)
        else:
            raise ValueError("Could not evaluate undefined model")

    @staticmethod
    def graph_content(graph):
        """ Return collections of the current graph for debugging purpose """
        res = {}
        if graph is None:
            return {}
        for name in graph.get_all_collection_keys():
            coll = graph.get_collection_ref(name)
            res[name] = coll
        return res
