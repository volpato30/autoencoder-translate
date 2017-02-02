from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pickle
with open('test_set.pkl', 'rb') as f:
    test_set = pickle.load(f)


import math
import os
import random
import sys
import time
import logging

import numpy as np
from six.moves import xrange    # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.rnn.translate import data_utils
import seq2seq_model
import scipy.io as sio


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                                                    "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                                                    "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                                                        "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("fr_vocab_size", 40000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                                                        "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                                                        "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                                                        "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                                                        "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                                                        "Train using fp16 instead of fp32.")




FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


def create_model(session, forward_only):
    """Create translation model and initialize or load parameters in session."""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = seq2seq_model.Seq2SeqModel(
            FLAGS.en_vocab_size,
            FLAGS.fr_vocab_size,
            _buckets,
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.max_gradient_norm,
            FLAGS.batch_size,
            FLAGS.learning_rate,
            FLAGS.learning_rate_decay_factor,
            forward_only=forward_only,
            dtype=dtype)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


# FLAGS.data_dir = "/home/rui/Data/Translation/training-giga-fren"
# FLAGS.train_dir = "./en_model_weights"
sess = tf.InteractiveSession()

model = create_model(sess, True)
feature = model.encode_testset(sess, test_set, language_id=FLAGS.language_id)

if FLAGS.language_id == 0:
    file_name = 'encode_en.mat'
else:
    file_name = 'encode_fr.mat'

sio.savemat(file_name, {'feature': feature})
