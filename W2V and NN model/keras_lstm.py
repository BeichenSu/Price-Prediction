# Load packages
import gensim
import pickle
import keras
import os
import math
import pydot
import graphviz
import random
import pandas as pd
import tensorflow as tf
import numpy as np
from numpy import array
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, Embedding, LSTM, Dense,\
    BatchNormalization, Dropout
from keras.models import Model, load_model
from keras.utils import plot_model
from keras import losses
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import plot

# GPU configuration, allow GPU memory
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

# Set working directory
os.chdir("C:\\Users\Lala No.5\Desktop\Final_Thesis")
# Add path to Graphviz package
os.environ["PATH"] += os.pathsep + \
                      'C:/Program Files (x86)/Graphviz2.38/bin/'
# Read data
print("Loading data...")
df = pd.read_csv("train_clean.csv")
# Number of observations
n = df.shape[0]
print("Number of total observations:", n)

# Numeric column and reshape
condition = np.array(df.item_condition_id).reshape(n, 1)
shipping = np.array(df.shipping).reshape(n, 1)
price = np.array(df.price).reshape(n, 1)

## One-Hot encoding for categorical column
category_name = np.array(df.category_name)
brand_name = np.array(df.brand_name)

# One-hot brand name
print("One hot encoding for brand and category name...")
values_brand = array(brand_name)
label_encoder_brand = LabelEncoder()
integer_encoded_brand = label_encoder_brand.\
    fit_transform(values_brand)
onehot_encoder_brand = OneHotEncoder(sparse=False)
integer_encoded_brand = integer_encoded_brand.\
    reshape(len(integer_encoded_brand),1)
onehot_encoded_brand = onehot_encoder_brand.\
    fit_transform(integer_encoded_brand)

# One-hot category name
values_cate = array(category_name)
label_encoder_cate = LabelEncoder()
integer_encoded_cate = label_encoder_cate.\
    fit_transform(values_cate)
onehot_encoder_cate = OneHotEncoder(sparse=False)
integer_encoded_cate = integer_encoded_cate.\
    reshape(len(integer_encoded_cate),1)
onehot_encoded_cate = onehot_encoder_cate.\
    fit_transform(integer_encoded_cate)

# Conmbine all those column except for description as the second input
print("Combining the second level inputs...")
train_x_2 = np.hstack([condition, shipping, onehot_encoded_brand,
                       onehot_encoded_cate])

# Release some memory
print("Releasing memory...")
del df, condition, brand_name, category_name, values_brand, \
    label_encoder_brand, integer_encoded_brand\
    , onehot_encoder_brand, values_cate, label_encoder_cate,\
    integer_encoded_cate, onehot_encoder_cate

## Deal with description:
# Read the sentence token, this is a list of list of token
print("Loading description tokens...")
with open("decription_token.txt", "rb") as fp:  # Unpickling
    description_token = pickle.load(fp)

# Load the pretrained w2v model with skip gram
model_sg = gensim.models.Word2Vec.load('model_sg_2')
print("Loading W2V model(Skip gram)...")


# Restore tokens of description to sentence
description_sentence = [' '.join(sentence) for
                        sentence in description_token]
del description_token
# Fit keras tokenizer and obtain sequence for token of sentence
word_num = len(model_sg.wv.vocab)
tokenizer = Tokenizer(word_num)
tokenizer.fit_on_texts(description_sentence)
description_token_sequence = tokenizer.\
    texts_to_sequences(description_sentence)
# The tokenizer.word_index is a dictionary
# which maps word into index in sequence above

# Create a embedding matrix to map the index in
# description_token_sequence to the word vector

print("Creating embedding matrix...")
vector_size = 300
embedding_matrix = np.zeros((word_num, vector_size))
for word, i in tokenizer.word_index.items():
    if word in list(model_sg.wv.vocab):
        vector = model_sg.wv[word]
        embedding_matrix[i] = vector
    if i > word_num:
        break

# Pad the sentences to a fixed length
print("Padding sentence to same length")
max_length = 60
description_token_sequence = pad_sequences(
    description_token_sequence, maxlen=max_length)
del model_sg
del tokenizer


# Model building part
print("Start building model...")
# First level nput layer for sentence sequence
main_input = Input(shape=(60,), dtype='int32', name='First_Input')

# This embedding layer will encode the input sequence
# into a sequence of dense 300-dimensional vectors.
# Using pretrained word2vec skip gram model
x = Embedding(output_dim=vector_size, input_dim=word_num,
              input_length=60, weights=[embedding_matrix],
              trainable=False, name="Embedding")(main_input)

# A LSTM will transform the vector sequence into a single vector,
# containing information about the entire sequence
lstm_out = LSTM(200, recurrent_dropout=0.1,
                dropout=0.1, name="LSTM")(x)

# Second level input layer for other predictors
auxiliary_input = Input(shape=(455,), name="Second_Input")

# Merge layer to concatenate LSTM output,
# ie, sentence vector with second layer input
x = keras.layers.concatenate([lstm_out, auxiliary_input])

# Fully-connected layer and batch normalization layers
x = Dense(256, activation='relu', name="FC_1")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', name="FC_2")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', name="FC_3")(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu', name="FC_4")(x)
x = Dropout(0.5)(x)
# Output layer for price
main_output = Dense(1, activation='relu', name='Output')(x)
# Define model and compile
model = Model(inputs=[main_input, auxiliary_input],
              outputs=[main_output])
model.compile(loss="mean_squared_logarithmic_error",
              optimizer='Adagrad', metrics=["mae"])

# Prepare input data
train_x_1 = description_token_sequence
train_x_2 = train_x_2
X = np.hstack([train_x_1,train_x_2])
y = price
# Cut the data into train, validation and test set
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.1, random_state=42)
del train_x_1, train_x_2, X
train_x_1 = X_train[:,0:60]
train_x_2 = X_train[:,60:515]
test_x_1 = X_test[:,0:60]
test_x_2 = X_test[:,60:515]


# Collect the training result for tensorboard visualization
tensorboard = keras.callbacks.TensorBoard(
    log_dir='C:\\Users\Lala No.5\Desktop\Final_Thesis\graph',
    histogram_freq=0,
    write_graph=True, write_images=True)
csv_logger = keras.callbacks.CSVLogger('log.csv',
                                       append=True, separator=',')

# Train the model
model.fit([train_x_1, train_x_2], y_train,
          batch_size=5000, epochs=5, validation_split=0.3,
          callbacks=[tensorboard, csv_logger])
# Save the model
model.save('model_deep.h5')
# Plot and save the model structure
plot_model(model, to_file='model1.png')

# model_adam = load_model('model_1_Adam.h5')
# model_adagrad = load_model("model_Adagrad.h5")
# model_sgd = load_model("model_SGD.h5")
# pred_adam = model_adam.predict([test_x_1,test_x_2])
# pred_sgd = model_sgd.predict([test_x_1,test_x_2])
# pred_adagrad = model_adagrad.predict([test_x_1,test_x_2])
#
#
# def RMSLE(y, y_pred):
# 	terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1))
#  ** 2.0 for i,pred in enumerate(y_pred)]
# 	return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5
#
# def MAE(y,y_pred):
#     return np.mean(np.abs(y-y_pred))
#
#
# model_2_64 = load_model("model_2_64.h5")
# model_2_128 = load_model("model_2_128.h5")
# pred_2_64 = model_2_64.predict([test_x_1,test_x_2])
# pred_2_128 = model_2_128.predict([test_x_1,test_x_2])
#
# RMSLE(y_test,pred_2_64)
# RMSLE(y_test,pred_2_128)
# MAE(y_test,pred_2_64)
# MAE(y_test,pred_2_128)



### Test set evaluation

# Read data
print("Loading data...")
df = pd.read_csv("test_clean.csv")
# Number of observations
n = df.shape[0]
print("Number of total observations:", n)

# Numeric column and reshape
condition = np.array(df.item_condition_id).reshape(n, 1)
shipping = np.array(df.shipping).reshape(n, 1)

## One-Hot encoding for categorical column
category_name = np.array(df.category_name)
brand_name = np.array(df.brand_name)

# One-hot brand name
print("One hot encoding for brand and category name...")
values_brand = array(brand_name)
label_encoder_brand = LabelEncoder()
integer_encoded_brand = label_encoder_brand.\
    fit_transform(values_brand)
onehot_encoder_brand = OneHotEncoder(sparse=False)
integer_encoded_brand = integer_encoded_brand.\
    reshape(len(integer_encoded_brand),1)
onehot_encoded_brand = onehot_encoder_brand.\
    fit_transform(integer_encoded_brand)

# One-hot category name
values_cate = array(category_name)
label_encoder_cate = LabelEncoder()
integer_encoded_cate = label_encoder_cate.\
    fit_transform(values_cate)
onehot_encoder_cate = OneHotEncoder(sparse=False)
integer_encoded_cate = integer_encoded_cate.\
    reshape(len(integer_encoded_cate),1)
onehot_encoded_cate = onehot_encoder_cate.\
    fit_transform(integer_encoded_cate)

# Conmbine all those column except for description as the second input
print("Combining the second level inputs...")
test_x_2 = np.hstack([condition, shipping, onehot_encoded_brand,
                       onehot_encoded_cate])

# Deal with sentences of descriptions: pad to a sentence of length 60
model_sg = gensim.models.Word2Vec.load('model_sg_2')
description_sentence = df.item_description
word_num = len(model_sg.wv.vocab)
tokenizer = Tokenizer(word_num)
tokenizer.fit_on_texts(description_sentence)
description_token_sequence = tokenizer.\
    texts_to_sequences(description_sentence)

print("Padding sentence to same length")
max_length = 60
description_token_sequence = pad_sequences(
    description_token_sequence, maxlen=max_length)

my_model = load_model('model_Adagrad.h5')
out = my_model.predict([description_token_sequence, test_x_2], batch_size= 5000)

submission = df[["test_id"]]
submission = np.hstack([submission, out])

import csv
with open("submission.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in submission:
            writer.writerow(line)