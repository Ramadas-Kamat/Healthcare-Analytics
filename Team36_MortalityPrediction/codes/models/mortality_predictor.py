import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from utils.data_processing import get_batch
from models.encoders import highway_maxout, embedding_plus_highway_maxout, multi_layer_perceptron, embedding_plus_mlp


class MortalityPredictor():
    def __init__(self, config):

        self.age = tf.placeholder(tf.float32, (None, 1), 'age')
        self.ethnicity = tf.placeholder(tf.float32, (None, 4), 'ethnicity')
        self.gender = tf.placeholder(tf.float32, (None, 2), 'gender')
        self.language = tf.placeholder(tf.float32, (None, 102), 'language')
        self.marital_status = tf.placeholder(tf.float32, (None, 6), 'marital_status')
        self.religion = tf.placeholder(tf.float32, (None, 6), 'religion')

        combined = tf.concat([self.age, self.ethnicity, self.gender, self.language, self.marital_status, self.religion],
                             axis=1)

        self.labels = tf.placeholder(tf.int32, None, 'inputs')
        self.drop_rate = tf.placeholder(tf.float32, [], 'dropout_rate')

        if config['classifier_type'] == 'highway_maxout':
            self.logits = highway_maxout(combined)

        elif config['classifier_type'] == 'embedding_plus_highway_maxout':
            self.logits = embedding_plus_highway_maxout(self.age, self.ethnicity, self.gender, self.language,
                                                        self.marital_status, self.religion, config['embedding_dim'],
                                                        self.drop_rate)

        elif config['classifier_type'] == 'multilayer_perceptron':
            self.logits = multi_layer_perceptron(combined, config['hidden_size'], self.drop_rate)

        elif config['classifier_type'] == 'embedding_plus_multilayer_perceptron':
            self.logits = embedding_plus_mlp(self.age, self.ethnicity, self.gender, self.language,
                                             self.marital_status, self.religion, config['embedding_dim'],
                                             config['hidden_size'], self.drop_rate)

        else:
            raise ValueError('Invalid Classifier Type')

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=config['learning_rate']).minimize(self.loss)

    def train(self, sess, train_data, dev_data, config):
        saver = tf.train.Saver()
        best_dev_loss = float('inf')
        best_dev_auc = 0.0
        sess.run(tf.global_variables_initializer())
        tf.summary.scalar('loss', self.loss)
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(config['log_dir'] + config['classifier_type'], sess.graph)
        model_directory = config['save_dir'] + config['classifier_type'] + '/'
        output_csv = open(config['log_dir'] + config['classifier_type'] + '.csv', 'w', encoding='utf8')
        for iteration_no in range(config['num_iterations']):
            features, labels = get_batch(list(train_data[0]), list(train_data[1]), iteration_no, config['batch_size'],
                                         True)
            feed_dict = self.create_feed_dict(features, labels, config['drop_rate'])
            loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
            if iteration_no % config['checkpoint'] == 0:
                dev_features, dev_labels = get_batch(list(dev_data[0]), list(dev_data[1]), 0, len(dev_data[0]), False)
                feed_dict = self.create_feed_dict(dev_features, dev_labels, 1.0)
                dev_logits, dev_loss, summary_val = sess.run([self.logits, self.loss, summary], feed_dict=feed_dict)
                dev_auc = roc_auc_score(dev_labels, dev_logits[:, 1])
                output_csv.write(str(dev_loss) + ',' + str(dev_auc) + '\n')
                summary_writer.add_summary(summary_val, global_step=iteration_no)
                print(dev_loss)
                if dev_auc > best_dev_auc:
                    best_dev_auc = dev_auc
                    saver.save(sess, model_directory)
                    print('Saved best model')
        output_csv.close()

    def test(self, sess, test_data, config):
        saver = tf.train.Saver()
        saver.restore(sess, config['save_dir'] + config['classifier_type'] + '/')
        print('model restored')
        test_features, test_labels = get_batch(list(test_data[0]), list(test_data[1]), 0, len(test_data[0]), False)
        feed_dict = self.create_feed_dict(test_features, test_labels, 1.0)
        test_logits = sess.run([self.logits], feed_dict=feed_dict)[0]
        test_predictions = np.argmax(test_logits, axis=1)
        accuracy = accuracy_score(test_labels, test_predictions)
        auc = roc_auc_score(test_labels, test_logits[:, 1])
        print('Test accuracy is ' + str(accuracy))
        print('Test AUC is ' + str(auc))

    def create_feed_dict(self, features, labels, drop_rate):
        feed_dict = {
            self.age: features['age'],
            self.ethnicity: features['ethnicity'],
            self.gender: features['gender'],
            self.language: features['language'],
            self.marital_status: features['marital_status'],
            self.religion: features['religion'],
            self.labels: labels,
            self.drop_rate: drop_rate
        }
        return feed_dict
