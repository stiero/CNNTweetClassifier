#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
import os

import keras.backend as K
import multiprocessing
import tensorflow as tf

from gensim.models.word2vec import Word2Vec

from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam

from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer

import numpy as np

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

#Configs

np.random.seed(1000)

use_gpu = True

config = tf.ConfigProto(intra_op_parallelism_threads=multiprocessing.cpu_count(), 
                        inter_op_parallelism_threads=multiprocessing.cpu_count(), 
                        allow_soft_placement=True, 
                        device_count = {'CPU' : 1, 
                                        'GPU' : 1 if use_gpu else 0})

#Start session
session = tf.Session(config=config)
K.set_session(session)

dataset_dir = '/home/ht/Projects/W2V/Sentiment Analysis Dataset.csv'
model_dir = '/home/ht/Projects/W2V/model/'
corpus = []
labels = []

#Parse tweets
with open(dataset_dir, 'r') as df:
    for i, line in tqdm(enumerate(df)):
        #Skipping header
        if i == 0:
            continue

        #Split line to components
        comps = line.strip().split(',')
        
        #Store sentiment scores
        labels.append(int(comps[1].strip()))
        
        # Tweet
        tweet = comps[3].strip()
        if tweet.startswith('"'):
            tweet = tweet[1:]
        if tweet.endswith('"'):
            tweet = tweet[::-1]
        
        corpus.append(tweet.strip().lower())
        
print('Corpus size: {}'.format(len(corpus)))

#Tokenise and stem
tokeniser = RegexpTokenizer('[a-zA-Z0-9@]+')
stemmer = LancasterStemmer()

ts_corpus = []

for i, tweet in tqdm(enumerate(corpus[0:10000])):
    tokens = [stemmer.stem(t) for t in tokeniser.tokenize(tweet) if not t.startswith('@')]
    ts_corpus.append(tokens)
    
    
#Generate word embeddings
vector_size = 512
window_size = 5

word2vec = Word2Vec(sentences=ts_corpus,
                    size=vector_size, 
                    window=window_size, 
                    negative=20,
                    iter=50,
                    seed=1000,
                    workers=multiprocessing.cpu_count())

word2vec.save("/home/ht/Projects/W2V/word2vec/word2vec_vs512_ws5_n20")

word2vec = Word2Vec.load("word2vec_vs512_ws5_n20")

#Copy word vectors and delete Word2Vec model and original corpus to save memory
X_vecs = word2vec.wv
del word2vec
del corpus

# Train subset size (0 < size < len(tokenized_corpus))
#train_size = 1500000
train_size = 9500

# Test subset size (0 < size < len(tokenized_corpus) - train_size)
test_size = 500
#test_size = 10000

# Compute average and max tweet length
tot_length = 0.0
max_length = 0

for tweet in tqdm(ts_corpus):
    if len(tweet) > max_length:
        max_length = len(tweet)
    tot_length += float(len(tweet))
    
avg_length = float(tot_length/len(ts_corpus))
    
print("Average tweet length", avg_length)
print("Max tweet length", max_length)

#Setting max length for input tweets
max_tweet_length = 19

# Create train and test sets
# Generate random indexes
indexes = set(np.random.choice(len(ts_corpus), train_size + test_size, replace=False))

X_train = np.zeros((train_size, max_tweet_length, vector_size), dtype=K.floatx())
Y_train = np.zeros((train_size, 2), dtype=np.int32)
X_test = np.zeros((test_size, max_tweet_length, vector_size), dtype=K.floatx())
Y_test = np.zeros((test_size, 2), dtype=np.int32)

for i, index in tqdm(enumerate(indexes)):
    for t, token in enumerate(ts_corpus[index]):
        if t >= max_tweet_length:
            break
        
        if token not in X_vecs:
            continue
    
        if i < train_size:
            X_train[i, t, :] = X_vecs[token]
        else:
            X_test[i - train_size, t, :] = X_vecs[token]
            
    if i < train_size:
        Y_train[i, :] = [1.0, 0.0] if labels[index] == 0 else [0.0, 1.0]
    else:
        Y_test[i - train_size, :] = [1.0, 0.0] if labels[index] == 0 else [0.0, 1.0]
        
    
#Defining Keras convolutional neural network
batch_size = 100
nb_epochs = 100

model = Sequential()

model.add(Conv1D(100, kernel_size=3, activation='elu', padding='same', input_shape=(max_tweet_length, vector_size)))
model.add(Conv1D(100, kernel_size=3, activation='elu', padding='same'))
model.add(Conv1D(100, kernel_size=3, activation='elu', padding='same'))
model.add(Conv1D(100, kernel_size=3, activation='elu', padding='same'))
model.add(Dropout(0.25))

model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
model.add(Dropout(0.25))

model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
model.add(Dropout(0.25))

model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
model.add(Dropout(0.25))


model.add(Flatten())

model.add(Dense(256, activation='tanh'))
model.add(Dense(256, activation='tanh'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

#Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])

vizmodel = TensorBoard(log_dir='./Graph', histogram_freq=0.005,
                       write_graph=True, write_images=True)

#Fit the model
model.fit(X_train, Y_train,
          batch_size=batch_size,
          shuffle=True,
          epochs=nb_epochs,
          validation_data=(X_test, Y_test),
callbacks=[EarlyStopping(min_delta=0.00025, patience=2), vizmodel])


os.chdir(model_dir)
model.save('Keras_Bon.h5')
os.chdir(dataset_dir)

#score = model.evaluate(X_test, Y_test, batch_size=128, verbose=1)

vals = [str(datetime.now()), "S, 4xC(32), Dr(0.25), 4xC(32, Dr(0.25), F, 2xDe(256), Dr(0.5), De(2)",
         vector_size, window_size, max_tweet_length]

names = ["Time", "Layer scheme", "vector_size", "window_size", "max_tweet_length"]

score = dict(zip(names, vals))

score["loss and accuracy"] = model.evaluate(X_test, Y_test, batch_size=128, verbose=2)


import json

with open("keras_output.json", "a") as outfile:
    json.dump(score, outfile)
    outfile.write('\n')

#results = []

#for line in open("keras_output.json", "r"):
#    results.append(json.loads(line))



