#!/usr/bin/env python
# -- special statements for sbatch --
#SBATCH -p timed-gpu
#SBATCH --gres gpu:1
# -- end special statements for sbatch --
"""
2024-01-16
"v03": First generic run for GPU nodes


Complete Python script to do the ML pipeline of Sherlock 19F17C dataset
using neural networks:

* 512 + 512 hidden layers
* all dense

Use all remaining features.

Goal of this script:

* find the right way to control resource usage of TensorFlow according to what we expect
"""

import sys
import os

## HACK
# Force OMP_NUM_THREADS=1
os.environ['OMP_NUM_THREADS'] = "1"
## END HACK

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import sklearn
from sklearn import preprocessing
import tensorflow as tf
import tensorflow.keras as keras

from analysis_sherlock_ML import *

# Import KERAS objects
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

# CUSTOMIZATIONS
numpy.set_printoptions(linewidth=1000)

print("OMP_NUM_THREADS = ", os.environ.get('OMP_NUM_THREADS', None))

## Tensorflow would have been initialized by this time::
print("\nPhysical devices:")
print(tf.config.list_physical_devices())

print("\nLogical devices:")
print(tf.config.list_logical_devices())

print("------------")

df = pd.read_csv("sherlock_apps_yhe_test.csv")
summarize_dataset(df)
df2 = preprocess_sherlock_19F17C(df)

print()
Rec = step0_label_features(df2)

#df_corr1_cols_to_drop = ['UidRxPackets', 'UidTxPackets', 'lru', 'rss', 'utime', 'vsize']
#print("Drop the following columns:")
#print(list(df_corr1_cols_to_drop))

#tmp = step_drop_columns(Rec, df_corr1_cols_to_drop)
tmp = step_onehot_encoding(Rec)
tmp = step_feature_scaling(Rec)
print("After scaling:")
print(Rec.df_features.head(10))

print()
tmp = step_train_test_split(Rec, test_size=0.2, random_state=34)

print()

# Neural network part is here

Rec.train_L_onehot = pd.get_dummies(Rec.train_labels)
Rec.test_L_onehot = pd.get_dummies(Rec.test_labels)

def NN_Model(hidden_neurons,learning_rate):
    """Definition of deep learning model with one dense hidden layer"""
    model = Sequential([
        Dense(hidden_neurons, activation='relu',input_shape=(19,),kernel_initializer='random_normal'),
        Dense(18, activation='softmax')
    ])
    adam=tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model


def NN_Model_2H(hidden_neurons,learning_rate):
    """Definition of deep learning model with one dense hidden layer.
    The `hidden_neurons` must be a list of two ints (layer widths)."""
    model = Sequential([
        Dense(hidden_neurons[0], activation='relu',input_shape=(19,),kernel_initializer='random_normal'),
        Dense(hidden_neurons[1], activation='relu'),
        Dense(18, activation='softmax')
    ])
    adam=tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model


print("Training neural network model now...")
model_1 = NN_Model_2H([512,512],0.0003)
model_1.fit(Rec.train_features,
            Rec.train_L_onehot,
            epochs=10, batch_size=32,
            validation_data=(Rec.test_features, Rec.test_L_onehot),
            verbose=2)

