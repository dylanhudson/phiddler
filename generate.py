#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 19:43:28 2019

"""

import numpy as np
from keras.models import model_from_json
from random import randint


with open("rawnotesDmaj-new.txt") as corpus_file:
    corpus = corpus_file.read()


chars = sorted(list(set(corpus)))
num_chars = len(chars)
encoding = {c: i for i, c in enumerate(chars)}
decoding = {i: c for i, c in enumerate(chars)}
print("Corpus contains {0} unique chars.".format(num_chars))

bar_length = 75
corpus_length = len(corpus)

with open("modelDmaj-new.json") as model_file:
    architecture = model_file.read()

model = model_from_json(architecture)
model.load_weights("weightsDmaj-new-10-0.929.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='adam')

seed = randint(0, corpus_length - bar_length)
seed_bar = corpus[seed:seed + bar_length]
X = np.zeros((1, bar_length, num_chars), dtype=np.bool)
for i, c in enumerate(seed_bar):
    X[0, i, encoding[c]] = 1

composed_tune = ""
for i in range(400):
    prediction = np.argmax(model.predict(X, verbose=0))

    composed_tune += decoding[prediction]

    activations = np.zeros((1, 1, num_chars), dtype=np.bool)
    activations[0, 0, prediction] = 1
    X = np.concatenate((X[:, 1:, :], activations), axis=1)


print(composed_tune)
