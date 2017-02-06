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
from translate_en2en import create_model
import scipy.io as sio

tf.app.flags.DEFINE_integer("language_id", 0, "0 for english, 1 for french")


FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]




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
