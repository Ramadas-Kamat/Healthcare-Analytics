import tensorflow as tf
import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
import numpy

demographics = pd.read_csv('data/Demographics.csv')
demographics.head()

# First, define atherosclerosis diagnoses from non-atherosclerosis diagnoses
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

athero_pos['DOA']
del athero_pos['DOA']
del athero_neg['DOA']

athero_neg['HEART_ATTACK_FLAG']
del athero_neg['HEART_ATTACK_FLAG']
del athero_pos['HEART_ATTACK_FLAG']

del athero_pos['Unnamed: 0']
athero_pos.head()


# Create Outcome data sets
athero_heartdeath = pd.Series(athero_pos['HEART_DEATH_FLAG'])
athero_death = pd.Series(athero_pos['DEATH_FLAG'])


del athero_pos['HEART_DEATH_FLAG']
del athero_pos['DEATH_FLAG']

# Get dummies
athero_pos = pd.get_dummies(athero_pos, columns=['GENDER', 'ETHNICITY', 'MARITAL_STATUS', 'LANGUAGE',
                                                 'RELIGION', 'INSURANCE', 'ADMISSION_LOCATION'])

# Check outcome numbers
print(athero_heartdeath.value_counts())
print(athero_death.value_counts())


X_train, X_test, y_train, y_test = train_test_split(athero_pos, athero_death, test_size=0.20, random_state=42)

# fix random seed for reproducibility
numpy.random.seed(0)

# Create model
model = Sequential()
model.add(Dense(80, input_dim=121 , activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# checkpoint
from keras.callbacks import ModelCheckpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X_train, y_train, batch_size=20, nb_epoch=150, verbose=1, callbacks=callbacks_list, validation_data=(X_test, y_test), shuffle=True)


model.load_weights("weights.best.hdf5")

# estimate accuracy on test data set using loaded weights
scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



