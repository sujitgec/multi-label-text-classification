#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:46:45 2019

@author: suk
"""

import tensorflow as tf
import numpy as np
import pandas as pd
 
import os

# Prepare Dataset
TRAIN_DATA = "/home/suk/Desktop/mls/news_classification/text_classification_using_gans_hans/data/jigsaw-toxic-comment-classification-challenge/train.csv"
GLOVE_EMBEDDING = "/home/suk/Desktop/mls/news_classification/glove_6B/glove.6B.100d.txt"
 
train = pd.read_csv(TRAIN_DATA)
 
train["comment_text"].fillna("fillna")
 
x_train = train["comment_text"].str.lower()
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
 
max_words = 100000
max_len = 150
 
embed_size = 100
 
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words, lower=True)
 
tokenizer.fit_on_texts(x_train)
 
x_train = tokenizer.texts_to_sequences(x_train)
 
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)

# Use pre-train embeddings
embeddings_index = {}
 
with open(GLOVE_EMBEDDING, encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        embed = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embed
 
word_index = tokenizer.word_index
 
num_words = min(max_words, len(word_index) + 1)
 
embedding_matrix = np.zeros((num_words, embed_size), dtype='float32')
 
for word, i in word_index.items():
 
    if i >= max_words:
        continue
 
    embedding_vector = embeddings_index.get(word)
 
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
input = tf.keras.layers.Input(shape=(max_len,))
 
x = tf.keras.layers.Embedding(max_words, embed_size, weights=[embedding_matrix], trainable=False)(input)

# Bidirectional Layer
x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
 
x = tf.keras.layers.Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x)
 
avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
 
x = tf.keras.layers.concatenate([avg_pool, max_pool])
 
preds = tf.keras.layers.Dense(6, activation="sigmoid")(x)
 
model = tf.keras.Model(input, preds)
 
model.summary()

from keras import backend as K

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-3), metrics=['accuracy',f1_m, precision_m, recall_m])

# Train Model
batch_size = 128
 
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
 
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
 
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    cp_callback
]
 
model.fit(x_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=1, callbacks=callbacks, verbose=1)

# Predictions
latest = tf.train.latest_checkpoint(checkpoint_dir)
 
model.load_weights(latest)
 
predictions = model.predict(np.expand_dims(x_train[15], 0))
 
print(tokenizer.sequences_to_texts([x_train[15]]))
print(y_train[15])
print(predictions)
