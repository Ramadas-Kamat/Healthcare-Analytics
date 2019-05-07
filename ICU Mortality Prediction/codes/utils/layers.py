import tensorflow as tf
from utils.tf_utils import top_k_gpu


def maxout_layer(inputs, output_size=200, pool_size=16):
    pool = tf.layers.dense(inputs, output_size * pool_size)
    pool = tf.reshape(pool, (-1, tf.shape(inputs)[1], output_size, pool_size))
    output = tf.reduce_max(pool, -1)
    return output


def sparsify(matrix, k=2):
    _, top_indices = top_k_gpu(matrix, k)
    output = tf.one_hot(top_indices, tf.shape(matrix)[-1], 1.0, 0.0)
    output = tf.multiply(tf.reduce_sum(output, axis=-2), matrix)
    output = tf.nn.softmax(
        output + -1e30 * tf.cast(tf.equal(output, 0), tf.float32))
    return output


def sparse_mixture_of_experts_layer(inputs, output_size=200, num_experts=16):
    non_noise = tf.layers.dense(
        inputs, output_size * num_experts, use_bias=False, name='gating_non_noise')
    noise = tf.layers.dense(inputs, output_size * num_experts,
                            activation=tf.nn.softplus, use_bias=False, name='gating_noise')
    h = non_noise + \
        tf.random_normal((tf.shape(inputs)[0], tf.shape(
            inputs)[1], output_size * num_experts)) * noise
    h = tf.reshape(h, (-1, tf.shape(inputs)[1], output_size, num_experts))
    g = sparsify(h)
    e = tf.layers.dense(inputs, output_size * num_experts,
                        activation=tf.tanh, name='experts')
    e = tf.reshape(e, (-1, tf.shape(inputs)[1], output_size, num_experts))
    output = tf.reduce_sum(tf.multiply(g, e), -1)
    return output


def feature_embedding_layer(age, ethnicity, gender, language, martial_status, religion, embedding_size, drop_rate):
    emb1 = tf.layers.dense(age, units=embedding_size, activation=tf.nn.relu)
    emb1 = tf.layers.dropout(emb1, rate=drop_rate)
    emb2 = tf.layers.dense(ethnicity, units=embedding_size, activation=tf.nn.relu)
    emb2 = tf.layers.dropout(emb2, rate=drop_rate)
    emb3 = tf.layers.dense(gender, units=embedding_size, activation=tf.nn.relu)
    emb3 = tf.layers.dropout(emb3, rate=drop_rate)
    emb4 = tf.layers.dense(language, units=embedding_size, activation=tf.nn.relu)
    emb4 = tf.layers.dropout(emb4, rate=drop_rate)
    emb5 = tf.layers.dense(martial_status, units=embedding_size, activation=tf.nn.relu)
    emb5 = tf.layers.dropout(emb5, rate=drop_rate)
    emb6 = tf.layers.dense(religion, units=embedding_size, activation=tf.nn.relu)
    emb6 = tf.layers.dropout(emb6, rate=drop_rate)
    return tf.concat([emb1, emb2, emb3, emb4, emb5, emb6], axis=1)
