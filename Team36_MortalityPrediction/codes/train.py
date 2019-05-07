import yaml
import tensorflow as tf
from models.mortality_predictor import MortalityPredictor
from utils.data_processing import clean_and_split


classifier = 'mixture_of_experts'

if __name__ == '__main__':
    config = yaml.safe_load(open('config.yml', 'r'))
    train_data, dev_data, test_data = clean_and_split(config)
    sess = tf.Session()
    model = MortalityPredictor(config)
    model.train(sess, train_data, dev_data, config)
    model.test(sess, test_data, config)
