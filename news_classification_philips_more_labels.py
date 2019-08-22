#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:46:45 2019

@author: suk
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from keras.backend import clear_session
from imblearn.keras import BalancedBatchGenerator
from imblearn.over_sampling import SMOTE
import timeit
from sklearn.model_selection import train_test_split
clear_session()

import os

# Prepare Dataset
DATA = "/home/suk/Desktop/mls/news_classification/text_classification_using_gans_hans/data/philips_categories_cleaned.xlsx"
DATA_1 = "/home/suk/Desktop/mls/news_classification/text_classification_using_gans_hans/data/sim_cos_df_full_20190725.csv"
GLOVE_EMBEDDING = "/home/suk/Desktop/mls/news_classification/glove_6B/glove.6B.100d.txt"
 
#train = pd.read_excel(TRAIN_DATA)
data = pd.read_csv(DATA_1, encoding = "ISO-8859-1")
data.info()
#train = train.astype(str)
data.head()
'''
tr = pd.DataFrame(np.array([[0.001, 2, 3], [4, 0.01, 6], [7, 1, 0]]), columns=['a', 'b', 'c'])
tr = tr.astype(str)
tr.info()
tr = tr.replace(to_replace=r'^-?0(\.\d+(e\d+)?)?$', value='1', regex=True)
#tr.loc[tr.a == 0, tr.a] = 1
print(tr) 
tr.iloc[:,1:] = tr.iloc[:,2:].apply(lambda x: 0 if x == 0 else 1)
'''
#train.iloc[1:,2:] = train.iloc[1:,2:].apply(lambda x: 1 if x > 0 else 0)
data['female'] = data['female'].apply(lambda x: 1 if x > 0 else 0)
data['male'] = data['male'].apply(lambda x: 1 if x > 0 else 0)
data['shavers.1'] = data['shavers.1'].apply(lambda x: 1 if x > 0 else 0)
data['shavers'] = data['shavers'].apply(lambda x: 1 if x > 0 else 0)
data['oneblade.1'] = data['oneblade.1'].apply(lambda x: 1 if x > 0 else 0)
data['oneblade'] = data['oneblade'].apply(lambda x: 1 if x > 0 else 0)
data['multigroom'] = data['multigroom'].apply(lambda x: 1 if x > 0 else 0)
data['hair_clippers'] = data['hair_clippers'].apply(lambda x: 1 if x > 0 else 0)
data['detail_trimmer'] = data['detail_trimmer'].apply(lambda x: 1 if x > 0 else 0)
data['body_grooming'] = data['body_grooming'].apply(lambda x: 1 if x > 0 else 0)
data['beard_trimmer'] = data['beard_trimmer'].apply(lambda x: 1 if x > 0 else 0)
data['grooming'] = data['grooming'].apply(lambda x: 1 if x > 0 else 0)
data['male_grooming'] = data['male_grooming'].apply(lambda x: 1 if x > 0 else 0)
data['cleansing'] = data['cleansing'].apply(lambda x: 1 if x > 0 else 0)
data['anti_aging'] = data['anti_aging'].apply(lambda x: 1 if x > 0 else 0)
data['skincare'] = data['skincare'].apply(lambda x: 1 if x > 0 else 0)
data['lady_trimmer'] = data['lady_trimmer'].apply(lambda x: 1 if x > 0 else 0)
data['lady_shaver'] = data['lady_shaver'].apply(lambda x: 1 if x > 0 else 0)
data['ipl_depilation'] = data['ipl_depilation'].apply(lambda x: 1 if x > 0 else 0)
data['epilators'] = data['epilators'].apply(lambda x: 1 if x > 0 else 0)
data['hair_removal'] = data['hair_removal'].apply(lambda x: 1 if x > 0 else 0)
data['straightener'] = data['straightener'].apply(lambda x: 1 if x > 0 else 0)
data['multi_stylers'] = data['multi_stylers'].apply(lambda x: 1 if x > 0 else 0)
data['straightening_brush'] = data['straightening_brush'].apply(lambda x: 1 if x > 0 else 0)
data['dryer'] = data['dryer'].apply(lambda x: 1 if x > 0 else 0)
data['brush'] = data['brush'].apply(lambda x: 1 if x > 0 else 0)
data['air_styler'] = data['air_styler'].apply(lambda x: 1 if x > 0 else 0)
data['auto_curler'] = data['auto_curler'].apply(lambda x: 1 if x > 0 else 0)
data['haircare'] = data['haircare'].apply(lambda x: 1 if x > 0 else 0)
data['beauty'] = data['beauty'].apply(lambda x: 1 if x > 0 else 0)

data.head()
data["input"].fillna("NA")
 
x_data = data["input"].str.lower()
#y_train = train[["haircare", "hair_removal", "skincare", "grooming", "oneblade", "shavers"]].values
y_data = data[["beauty","haircare","auto_curler","air_styler","brush","dryer","straightening_brush",
                 "multi_stylers","straightener","hair_removal","epilators","ipl_depilation","lady_shaver",
                 "lady_trimmer","skincare","anti_aging","cleansing","male_grooming","grooming","beard_trimmer",
                 "body_grooming","detail_trimmer","hair_clippers","multigroom","oneblade","oneblade.1","shavers",
                 "shavers.1","male","female"]].values

max_words = 35000
max_len = 150
 
embed_size = 100
 
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words, lower=True)
 
tokenizer.fit_on_texts(x_data)
 
x_data = tokenizer.texts_to_sequences(x_data)
 
x_data = tf.keras.preprocessing.sequence.pad_sequences(x_data, maxlen=max_len)
tf.keras.backend.clear_session()

# Save keras tokenizer
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# Split train-test
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

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
 
preds = tf.keras.layers.Dense(30, activation="sigmoid")(x)
 
model = tf.keras.Model(input, preds)

global graph
graph = tf.get_default_graph()
 
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
 
checkpoint_path = "/home/suk/Desktop/mls/news_classification/text_classification_using_gans_hans/checkpoint/training_6/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
 
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
 
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    cp_callback
]

# Handling class imbalance
"""
haircare - 155 (4.77)
hair_removal - 102 (7.25)
skincare - 739 (1)
grooming - 178 (4.15)
oneblade - 125 (5.91)
shavers - 9 (82.11)

class_weight = {0: 4.77,
                1: 7.25,
                2: 1.,
                3: 4.15,
                4: 5.91,
                5: 82.11}
training_generator = BalancedBatchGenerator(x_train, y_train,
                                            batch_size=1000,
                                            random_state=42)
model.fit_generator(generator=training_generator, epochs=5, verbose=1)

x_train.shape, y_train.shape

smote = SMOTE('minority')
x_sm, y_sm = smote.fit_sample(x_train, y_train)

from skmultilearn.problem_transformation import LabelPowerset
from imblearn.over_sampling import RandomOverSampler

# Import a dataset with X and multi-label y

lp = LabelPowerset()
ros = RandomOverSampler(random_state=42)

# Applies the above stated multi-label (ML) to multi-class (MC) transformation.
yt = lp.transform(y)

X_resampled, y_resampled = ros.fit_sample(X, yt)

# Inverts the ML-MC transformation to recreate the ML set
y_resampled = lp.inverse_transform(y_resampled)
"""


#model.fit(x_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=100, class_weight=class_weight, callbacks=callbacks, verbose=1)
model.fit(x_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=100, callbacks=callbacks, verbose=1)

scores = model.evaluate(x_test, y_test, verbose=0) 
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
print("%s: %.2f%%" % (model.metrics_names[3], scores[3]*100))
print("%s: %.2f%%" % (model.metrics_names[4], scores[4]*100))

# Save weights in HDF5 file
model.save("news_classification.h5")
model.save("saved_model.h5")
print("Saved model to disk")

# serialize model to JSON
model_json = model.to_json()
with open("model_config.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_weights.h5")
print("Saved model to disk")

# Predictions
latest = tf.train.latest_checkpoint(checkpoint_dir)
 
model.load_weights(latest)

import time
start = time.time()

with graph.as_default():
    predictions = model.predict(np.expand_dims(x_test[92], 0))
    print(tokenizer.sequences_to_texts([x_test[92]]))
    print(y_train[92])
    print(predictions)

end = time.time()
print('Time taken to predict:', end - start)

model.predict(np.expand_dims(x_train[100], 0))
preds = model.predict(x_test)
preds[preds>=0.5] = 1
preds[preds<0.5] = 0
print(x_test.iloc[100,1])

#print('Time taken for prediction:', predictions.timeit())
