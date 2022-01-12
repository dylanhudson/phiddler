#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 20:57:54 2019
Updated Nov 2021

"""

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.callbacks import ModelCheckpoint
import os


os.environ['KMP_DUPLICATE_LIB_OK']='True'

with open("rawnotesDmaj-new.txt") as corpus_file:
    corpus = corpus_file.read()

print("loaded a corpus of {0} chars.".format(len(corpus)))

chars = sorted(list(set(corpus)))
num_chars = len(chars)
encoding = {c: i for i, c in enumerate(chars)}
decoding = {i: c for i, c in enumerate(chars)}


bar_length = 75
skip = 1
x_data = []
y_data = []
for i in range (0, len(corpus) - bar_length, skip):
    bar = corpus[i:i + bar_length]
    next_char = corpus[i + bar_length]
    x_data.append([encoding[char] for char in bar])
    y_data.append(encoding[next_char])

num_bars = len(x_data)


print("vectorizing...")
X = np.zeros((num_bars, bar_length, num_chars), dtype=np.bool)
y = np.zeros((num_bars, num_chars), dtype=np.bool)

for i, bar in enumerate(x_data):
    for n, encoded_char in enumerate(bar):
        X[i, n, encoded_char] = 1
    y[i, y_data[i]] = 1

print("y Dimension: {0} # Sentences {1} Characters {2}".format(y.shape, num_bars, len(chars)))
print("X Dimension: {0} # Sentences {1}".format(X.shape, bar_length))

model = Sequential()
model.add(LSTM(256, input_shape=(bar_length, num_chars)))
model.add(Dense(num_chars))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

#saving the model to a file
arch = model.to_json()
with open('modelDmaj-new.json', 'w') as model_file:
    model_file.write(arch)


#setting up checkpoints to be called at the end of each epoch
file_path = "weightsDmaj-new-{epoch:02d}-{loss:.3f}.hdf5"
dump_weights = ModelCheckpoint(file_path, monitor="loss", verbose=1, save_best_only=True, mode="min")
callbacks = [dump_weights]

#train
model.fit(X, y, epochs=10, batch_size=128, callbacks=callbacks)
