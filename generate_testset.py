from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from tensorflow.models.rnn.translate import seq2seq_model

import pickle

tf.app.flags.DEFINE_integer("en_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("fr_vocab_size", 40000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/home/rui/Data/Translation/training-giga-fren", "Data directory")
tf.app.flags.DEFINE_integer('test_size', 1000, "the size of test set")


FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


def main(_):
    t_size = 0
    test_set = [[] for _ in _buckets]
    en_train, fr_train, en_dev, fr_dev, _, _ = data_utils.prepare_wmt_data(
            FLAGS.data_dir, FLAGS.en_vocab_size, FLAGS.fr_vocab_size)
    with tf.gfile.GFile(en_train, mode='r') as en_file:
        with tf.gfile.GFile(fr_train, mode='r') as fr_file:
            en_sentence, fr_sentence = en_file.readline(), fr_file.readline()
            counter = 0
            while counter < FLAGS.test_size:
                counter += 1
                en_sentence_ids = [int(x) for x in en_sentence.split()]
                fr_sentence_ids = [int(x) for x in fr_sentence.split()]
                for bucket_id, (size, _) in enumerate(_buckets):
                    if len(en_sentence_ids) < size and len(fr_sentence_ids) < size:
                        test_set[bucket_id].append([en_sentence_ids, fr_sentence_ids])
                        t_size += 1
                        break
                en_sentence, fr_sentence = en_file.readline(), fr_file.readline()
    print("successful generate test set, now saving")
    with open('test_set.pkl', 'wb') as f:
        pickle.dump(test_set, f)
    print('All set, saved {} sentence pairs'.format(t_size))

if __name__ == "__main__":
  tf.app.run()
