import tensorflow as tf
from utils.layers import maxout_layer, sparse_mixture_of_experts_layer, feature_embedding_layer


def highway_maxout(input):
    expanded = tf.expand_dims(input, 1)
    layer_1 = sparse_mixture_of_experts_layer(expanded)
    layer_2 = maxout_layer(layer_1)
    output = maxout_layer(tf.concat([layer_1, layer_2], axis=2), 2)
    return tf.squeeze(output)


def embedding_plus_highway_maxout(age, ethnicity, gender, language, martial_status, religion, embedding_size,
                                  drop_rate):
    input = feature_embedding_layer(age, ethnicity, gender, language, martial_status, religion, embedding_size,
                                    drop_rate)
    expanded = tf.expand_dims(input, 1)
    layer_1 = sparse_mixture_of_experts_layer(expanded)
    layer_2 = maxout_layer(layer_1)
    output = maxout_layer(tf.concat([layer_1, layer_2], axis=2), 2)
    return tf.squeeze(output)


def embedding_plus_mlp(age, ethnicity, gender, language, martial_status, religion, embedding_size, num_units,
                       drop_rate):
    input = feature_embedding_layer(age, ethnicity, gender, language, martial_status, religion, embedding_size,
                                    drop_rate)
    return multi_layer_perceptron(input, num_units, drop_rate)


def multi_layer_perceptron(input, num_units, drop_rate):
    layer1 = tf.layers.dense(input, units=num_units, activation=tf.nn.relu)
    layer1 = tf.layers.dropout(layer1, rate=drop_rate)
    layer2 = tf.layers.dense(layer1, units=num_units / 2, activation=tf.nn.relu)
    layer2 = tf.layers.dropout(layer2, rate=drop_rate)
    layer3 = tf.layers.dense(layer2, units=num_units / 4, activation=tf.nn.relu)
    layer3 = tf.layers.dropout(layer3, rate=drop_rate)
    output = tf.layers.dense(layer3, units=2)
    return output
