import pandas as pd
import numpy as np
import random


def get_batch(dataset, labels, iteration_no, batch_size, wraparound):
    start = (iteration_no * batch_size) % (len(dataset))
    end = start + batch_size
    if end < len(dataset):
        batch = dataset[start:end]
        outputs = labels[start:end]
    else:
        batch = dataset[start: len(dataset)]
        outputs = labels[start: len(dataset)]
        if wraparound is True:
            batch.extend(dataset[0:(end - len(dataset))])
            outputs.extend(labels[0:(end - len(dataset))])
    batch = np.array(batch)
    features = {
        'age': batch[:, 0:1],
        'ethnicity': batch[:, 1:5],
        'gender': batch[:, 5:7],
        'language': batch[:, 7:109],
        'marital_status': batch[:, 109:115],
        'religion': batch[:, 115:121]
    }
    return features, outputs


def clean_and_split(config):
    demographics = pd.read_csv(config['preprocessed_input_file'])
    demographics.head()

    athero_pre = demographics[demographics['OLD_FLAG'] == 0]
    athero_pos = athero_pre[athero_pre['ATHERO_DIAGNOSIS_FLAG'] == 1]
    athero_neg = athero_pre[athero_pre['ATHERO_DIAGNOSIS_FLAG'] == 0]

    # Clean data sets
    del athero_neg['CAUSE']
    del athero_pos['CAUSE']

    del athero_neg['ATHERO_DIAGNOSIS_FLAG']
    del athero_pos['ATHERO_DIAGNOSIS_FLAG']

    del athero_neg['OLD_FLAG']
    del athero_pos['OLD_FLAG']

    del athero_neg['OUTSIDE_DEATH_FLAG']
    del athero_pos['OUTSIDE_DEATH_FLAG']

    del athero_neg['SUBJECT_ID']
    del athero_pos['SUBJECT_ID']

    del athero_neg['DOB']
    del athero_pos['DOB']

    del athero_neg['DOD']
    del athero_pos['DOD']

    del athero_pos['DOA']
    del athero_neg['DOA']

    del athero_neg['HEART_ATTACK_FLAG']
    del athero_pos['HEART_ATTACK_FLAG']

    del athero_pos['Unnamed: 0']
    athero_pos.head()

    athero_death = pd.Series(athero_pos['DEATH_FLAG'])

    del athero_pos['HEART_DEATH_FLAG']
    del athero_pos['DEATH_FLAG']

    athero_pos = pd.get_dummies(athero_pos,
                                columns=['GENDER', 'ETHNICITY', 'MARITAL_STATUS', 'LANGUAGE', 'RELIGION', 'INSURANCE',
                                         'ADMISSION_LOCATION'])
    combined = list(zip(athero_pos.values, athero_death.values))
    random.shuffle(combined)

    features, labels = zip(*combined)
    train_data = (features[:-2000], labels[:-2000])
    dev_data = (
        features[(len(features) - 2000):(len(features) - 1000)], labels[(len(features) - 2000):(len(features) - 1000)])
    test_data = (features[(len(features) - 1000):], labels[(len(features) - 1000):])

    return train_data, dev_data, test_data
