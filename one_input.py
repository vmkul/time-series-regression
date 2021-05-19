import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import models, layers

raw_data = pd.read_csv('daily-website-visitors.csv')
data = np.array(raw_data)

inputs = []

for row in data:
  date = row[3].split('/')
  month = tf.one_hot(int(date[0]), 12)
  day = tf.one_hot(int(date[1]), 31)
  day_of_week = tf.one_hot(int(row[2]), 7)
  inputs.append(np.concatenate([month, day, day_of_week]))

inputs = np.array(inputs)

labels = np.array([int(x.replace(',', '')) for x in data[:, 4]])

train_data = inputs[:1000]
train_labels = np.array(labels[:1000])

val_data = inputs[1000:1782]
val_labels = np.array(labels[1000:1782])

test_data = inputs[1782:]
test_labels = np.array(labels[1782:])

model = tf.keras.Sequential([
#    layers.Dense(units=8, activation='relu'),
#    layers.Dense(units=4, activation='relu'),
    layers.Dense(units=128, activation='relu'),
    layers.Dense(units=1)
])

model.compile(
    optimizer='rmsprop',
    loss='mae')

history = model.fit(
    train_data, train_labels,
    validation_data=(val_data, val_labels),
    verbose=0,
    epochs=300)


def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  #plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)
  plt.show()

plot_loss(history)

test_predictions = model.predict(test_data)

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = np.array([0, 8000])
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
_ = plt.plot(lims, lims + 800)
_ = plt.plot(lims, lims - 800)
plt.show()
  