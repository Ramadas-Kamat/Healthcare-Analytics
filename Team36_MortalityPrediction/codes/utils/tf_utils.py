import tensorflow as tf


def top_k_gpu(x, k):
    if k > 10:
        return tf.nn.top_k(x, k)
    values = []
    indices = []
    depth = tf.shape(x)[-1]
    for i in range(k):
        values.append(tf.reduce_max(x, -1))
        argmax = tf.argmax(x, -1)
        indices.append(argmax)
        if i + 1 < k:
            x += tf.one_hot(argmax, depth, float('-inf'))
    return tf.stack(values, axis=-1), tf.to_int32(tf.stack(indices, axis=-1))
