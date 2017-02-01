import tensorflow as tf
with tf.variable_scope('ss') as scope:
    a = tf.get_variable('a', [0])
    b = tf.get_variable('b', [1])
    c = tf.add(a,b)
print c.name
