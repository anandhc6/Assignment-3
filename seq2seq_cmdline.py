#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 17:09:37 2022

@author: stephen & Anandh
"""
#python seq2seq_cmdline.py 'LSTM' 3 3 256 512 0.3 30

# Required packages
import sys
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend
from random import randrange 
from tensorflow import keras
from google.colab import files
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import EarlyStopping

# parameters

cell_type = sys.argv[1]
n_encoder_layers = sys.argv[2]
n_decoder_layers = sys.argv[3]
embedding_size =sys.argv[4]
latent_dimension = sys.argv[5]
dropout = sys.argv[6]
epochs = sys.argv[7]


print('Passed arguments are:')
print('cell_type:', cell_type)
print('num_encoder_layers:',n_encoder_layers)
print('num_decoder_layers:',n_decoder_layers)
print('embedding_size:', embedding_size)
print('latent_dimension:',latent_dimension)
print('dropout:',dropout)
print('epochs:',epochs)


 # Paths of train and valid datasets.
train_data_path = "dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.train.tsv"
val_data_path = "dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.dev.tsv"

# Saving the files in list
with open(train_data_path, "r", encoding="utf-8") as file:
    train_data_lines = file.read().split("\n")

with open(val_data_path, "r", encoding="utf-8") as file:
    val_data_lines = file.read().split("\n")

# popping the empty character of the lists 
val_data_lines.pop()
train_data_lines.pop()

# Fixed parameter
batch_size = 64 

# embedding train data

def embed_train_data(train_data_lines):

    lenk = len(train_data_lines) - 1
    train_input_data = []
    train_target_data = []
    input_data_characters = set()
    target_data_characters = set()
    
    for line in train_data_lines[: lenk]:
        target_data, input_data, _ = line.split("\t")

        # We are using "tab" as the "start sequence" and "\n" as "end sequence".
        target_data = "\t" + target_data + "\n"
        train_input_data.append(input_data)
        train_target_data.append(target_data)

        # Finding unique characters.
        for ch in input_data:
            if ch not in input_data_characters:
                input_data_characters.add(ch)
        for ch in target_data:
            if ch not in target_data_characters:
                target_data_characters.add(ch)

    print("Number of samples:", len(train_input_data))
    # adding space 
    input_data_characters.add(" ")
    target_data_characters.add(" ")

    # sorting
    input_data_characters = sorted(list(input_data_characters))
    target_data_characters = sorted(list(target_data_characters))

    # maximum length of the words
    encoder_max_length = max([len(txt) for txt in train_input_data])
    decoder_max_length = max([len(txt) for txt in train_target_data])

    print("Max sequence length for inputs:", encoder_max_length)
    print("Max sequence length for outputs:", decoder_max_length)

    # number of input and target characters
    num_encoder_tokens = len(input_data_characters)
    num_decoder_tokens = len(target_data_characters)  
    
    print("Number of unique input tokens:", num_encoder_tokens)
    print("Number of unique output tokens:", num_decoder_tokens)

    # create an index
    input_token_idx = dict([(char, i) for i, char in enumerate(input_data_characters)])
    target_token_idx = dict([(char, i) for i, char in enumerate(target_data_characters)])
   
    # creating 0 array for encoder,decoder 
    encoder_input_data = np.zeros((len(train_input_data), encoder_max_length), dtype="float32")

    decoder_input_data = np.zeros((len(train_input_data), decoder_max_length), dtype="float32")

    decoder_target_data = np.zeros((len(train_input_data), decoder_max_length, num_decoder_tokens), dtype="float32")

    # index of the character is encoded for all the sample whereas target data is one hot encoded.
    for i, (input_data, target_data) in enumerate(zip(train_input_data, train_target_data)):
        for t, char in enumerate(input_data):
            encoder_input_data[i, t] = input_token_idx[char]
        
        encoder_input_data[i, t + 1:] = input_token_idx[" "]
        
        # decoder data
        for t, char in enumerate(target_data):
            # decoder_target_data is one timestep ahead of decoder_input_data
            decoder_input_data[i, t] = target_token_idx[char]

            if t > 0:
                # excluding the start character since decoder target data is one timestep ahead.
                decoder_target_data[i, t - 1, target_token_idx[char]] = 1.0
        # append the remaining positions with empty space
       
        decoder_input_data[i, t + 1:] = target_token_idx[" "]
        decoder_target_data[i, t:, target_token_idx[" "]] = 1.0

    return encoder_input_data,decoder_input_data,decoder_target_data,num_encoder_tokens,num_decoder_tokens,input_token_idx,target_token_idx,encoder_max_length,decoder_max_length

# embedding validation data

def embed_val_data(val_data_lines,num_decoder_tokens,input_token_idx,target_token_idx):
    val_input_data = []
    val_target_data = []
    lenk = len(val_data_lines) - 1

    for line in val_data_lines[: lenk]:
        target_data, input_data, _ = line.split("\t")
        
        # We use "tab" as the "start sequence" character and "\n" as "end sequence" character.
        target_data = "\t" + target_data + "\n"
        val_input_data.append(input_data)
        val_target_data.append(target_data)

    val_encoder_max_length = max([len(txt) for txt in val_input_data])
    val_decoder_max_length = max([len(txt) for txt in val_target_data])

    val_encoder_input_data = np.zeros((len(val_input_data), val_encoder_max_length), dtype="float32")
    val_decoder_input_data = np.zeros((len(val_input_data), val_decoder_max_length), dtype="float32")
    val_decoder_target_data = np.zeros((len(val_input_data), val_decoder_max_length, num_decoder_tokens), dtype="float32")

    for i, (input_data, target_data) in enumerate(zip(val_input_data, val_target_data)):
        for t, ch in enumerate(input_data):
            val_encoder_input_data[i, t] = input_token_idx[ch]
        val_encoder_input_data[i, t + 1:] = input_token_idx[" "]
        
        for t, ch in enumerate(target_data):
            # decoder_target_data is one timestep ahead of decoder_input_data
            val_decoder_input_data[i, t] = target_token_idx[ch]
            if t > 0:
                # excluding the start character since decoder target data is one timestep ahead.
                val_decoder_target_data[i, t - 1, target_token_idx[ch]] = 1.0
       
        val_decoder_input_data[i, t + 1:] = target_token_idx[" "]
        val_decoder_target_data[i, t:, target_token_idx[" "]] = 1.0

    return val_encoder_input_data,val_decoder_input_data,val_decoder_target_data,target_token_idx,val_target_data

# Embedding data
encoder_input_data,decoder_input_data,decoder_target_data,num_encoder_tokens,num_decoder_tokens,input_token_idx,target_token_idx,encoder_max_length,decoder_max_length = embed_train_data(train_data_lines)

val_encoder_input_data,val_decoder_input_data,val_decoder_target_data,target_token_idx,val_target_data = embed_val_data(val_data_lines,num_decoder_tokens,input_token_idx,target_token_idx)

reverse_input_char_index = dict((i, char) for char, i in input_token_idx.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_idx.items())

# Build RNN model

def seq2seq(embedding_size, n_encoder_tokens, n_decoder_tokens, n_encoder_layers,n_decoder_layers, latent_dimension, cell_type,
            target_token_idx, decoder_max_length, reverse_target_char_index,dropout,encoder_input_data, decoder_input_data,
            decoder_target_data,batch_size,epochs):
  
  encoder_inputs = keras.Input(shape=(None,), name='encoder_input')

  encoder = None
  encoder_outputs = None
  state_h = None
  state_c = None
  e_layer= n_encoder_layers
  
  # RNN

  if cell_type=="RNN":
    embed = tf.keras.layers.Embedding(input_dim=n_encoder_tokens, output_dim=embedding_size,name='encoder_embedding')(encoder_inputs)
    encoder = keras.layers.SimpleRNN(latent_dimension, return_state=True, return_sequences=True,name='encoder_hidden_1', dropout=dropout)
    print("Embed done")
    encoder_outputs, state_h = encoder(embed)
    
    for i in range(2,e_layer+1):
      layer_name = ('encoder_hidden_%d') % i
      print("Starting 2nd")
      encoder = keras.layers.SimpleRNN(latent_dimension, return_state=True, return_sequences=True,name=layer_name, dropout=dropout)
      print("Ending 2nd")

      encoder_outputs, state_h = encoder(encoder_outputs, initial_state=[state_h])

    encoder_states = None
    encoder_states = [state_h]
    decoder_inputs = keras.Input(shape=(None,), name='decoder_input')
    embed_dec = tf.keras.layers.Embedding(n_decoder_tokens, embedding_size, name='decoder_embedding')(decoder_inputs)
    
    # number of decoder layers
    d_layer = n_decoder_layers
    decoder = None
    decoder = keras.layers.SimpleRNN(latent_dimension, return_sequences=True, return_state=True,name='decoder_hidden_1', dropout=dropout)
    
    # initial state of decoder is encoder's last state of last layer
    decoder_outputs, _ = decoder(embed_dec, initial_state=encoder_states)
    for i in range(2,d_layer+1):
      layer_name = 'decoder_hidden_%d' % i
      decoder = keras.layers.SimpleRNN(latent_dimension, return_sequences=True, return_state=True,name=layer_name, dropout=dropout)
      decoder_outputs, _ = decoder(decoder_outputs, initial_state=encoder_states)
    
    decoder_dense = keras.layers.Dense(n_decoder_tokens, activation="softmax", name='decoder_output')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy",metrics=['accuracy'])                 

    model.fit(
          [encoder_input_data, decoder_input_data],
          decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
      )
    
    # Inference model
    encoder_inputs = model.input[0]
    encoder_outputs, state_h_enc = model.get_layer('encoder_hidden_' + str(n_encoder_layers)).output
    encoder_states = [state_h_enc]
    encoder_model = keras.Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1] 
    decoder_outputs = model.get_layer('decoder_embedding')(decoder_inputs)
    decoder_states_inputs = []
    decoder_states = []

    for j in range(1, n_decoder_layers + 1):
        decoder_state_input_h = keras.Input(shape=(latent_dimension,))
        current_states_inputs = [decoder_state_input_h]
        decoder = model.get_layer('decoder_hidden_' + str(j))
        decoder_outputs, state_h_dec = decoder(decoder_outputs, initial_state=current_states_inputs)
        decoder_states += [state_h_dec]
        decoder_states_inputs += current_states_inputs

    decoder_dense = model.get_layer('decoder_output')
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    return encoder_model, decoder_model

  # GRU

  elif cell_type=="GRU":
    embed = tf.keras.layers.Embedding(input_dim=n_encoder_tokens, output_dim=embedding_size,name='encoder_embedding')(encoder_inputs)
    encoder = keras.layers.GRU(latent_dimension, return_state=True, return_sequences=True,name='encoder_hidden_1', dropout=dropout)
    encoder_outputs, state_h = encoder(embed)
    
    for i in range(2,e_layer+1):
      layer_name = ('encoder_hidden_%d') % i
      encoder = keras.layers.GRU(latent_dimension, return_state=True, return_sequences=True,name=layer_name, dropout=dropout)
      encoder_outputs, state_h = encoder(encoder_outputs, initial_state=[state_h])
    
    encoder_states = None
    encoder_states = [state_h]
    decoder_inputs = keras.Input(shape=(None,), name='decoder_input')
    embed_dec = tf.keras.layers.Embedding(n_decoder_tokens, embedding_size, name='decoder_embedding')(decoder_inputs)
    
    # number of decoder layers
    d_layer = n_decoder_layers
    decoder = None
    decoder = keras.layers.GRU(latent_dimension, return_sequences=True, return_state=True,name='decoder_hidden_1', dropout=dropout)
    
    # initial state of decoder is encoder's last state of last layer
    decoder_outputs, _ = decoder(embed_dec, initial_state=encoder_states)
    for i in range(2,d_layer+1):
      layer_name = 'decoder_hidden_%d' % i
      decoder = keras.layers.GRU(latent_dimension, return_sequences=True, return_state=True,name=layer_name, dropout=dropout)
      decoder_outputs, _ = decoder(decoder_outputs, initial_state=encoder_states)
    
    decoder_dense = keras.layers.Dense(n_decoder_tokens, activation="softmax", name='decoder_output')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy",metrics=['accuracy'])#, metrics=[my_metric]                 

    model.fit(
          [encoder_input_data, decoder_input_data],
          decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
      )
    
    # Inference Model
    encoder_inputs = model.input[0]
    encoder_outputs, state_h_enc = model.get_layer('encoder_hidden_' + str(n_encoder_layers)).output
    encoder_states = [state_h_enc]
    encoder_model = keras.Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]
    decoder_outputs = model.get_layer('decoder_embedding')(decoder_inputs)
    decoder_states_inputs = []
    decoder_states = []

    for j in range(1, n_decoder_layers + 1):
        decoder_state_input_h = keras.Input(shape=(latent_dimension,))
        current_states_inputs = [decoder_state_input_h]
        decoder = model.get_layer('decoder_hidden_' + str(j))
        decoder_outputs, state_h_dec = decoder(decoder_outputs, initial_state=current_states_inputs)
        decoder_states += [state_h_dec]
        decoder_states_inputs += current_states_inputs
    
    decoder_dense = model.get_layer('decoder_output')
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    return encoder_model, decoder_model

  # LSTM

  elif cell_type=="LSTM":
    embed = tf.keras.layers.Embedding(input_dim=n_encoder_tokens, output_dim=embedding_size,name='encoder_embedding')(encoder_inputs)
    encoder = keras.layers.LSTM(latent_dimension, return_state=True, return_sequences=True,name='encoder_hidden_1', dropout=dropout)
    encoder_outputs, state_h, state_c = encoder(embed)
    
    for i in range(2,e_layer+1):
      layer_name = ('encoder_hidden_%d') % i
      encoder = keras.layers.LSTM(latent_dimension, return_state=True, return_sequences=True,name=layer_name, dropout=dropout)
      encoder_outputs, state_h, state_c = encoder(encoder_outputs, initial_state=[state_h,state_c])
    
    encoder_states = None
    encoder_states = [state_h, state_c]

    decoder_inputs = keras.Input(shape=(None,), name='decoder_input')
    embed_dec = tf.keras.layers.Embedding(n_decoder_tokens, embedding_size, name='decoder_embedding')(decoder_inputs)
    
    # number of decoder layers
    d_layer = n_decoder_layers
    decoder = None
    decoder = keras.layers.LSTM(latent_dimension, return_sequences=True, return_state=True,name='decoder_hidden_1', dropout=dropout)
    
    # initial state of decoder is encoder's last state of last layer
    decoder_outputs, _,_ = decoder(embed_dec, initial_state=encoder_states)
    for i in range(2,d_layer+1):
      layer_name = 'decoder_hidden_%d' % i
      decoder = keras.layers.LSTM(latent_dimension, return_sequences=True, return_state=True,name=layer_name, dropout=dropout)
      decoder_outputs, _,_ = decoder(decoder_outputs, initial_state=encoder_states)
    
    decoder_dense = keras.layers.Dense(n_decoder_tokens, activation="softmax", name='decoder_output')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy",metrics=['accuracy'])#, metrics=[my_metric]                 
    
    model.fit(
          [encoder_input_data, decoder_input_data],
          decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
      )
    
    encoder_inputs = model.input[0]
    encoder_outputs, state_h_enc, state_c_enc = model.get_layer('encoder_hidden_' + str(n_encoder_layers)).output
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = keras.Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]
    decoder_outputs = model.get_layer('decoder_embedding')(decoder_inputs)
    decoder_states_inputs = []
    decoder_states = []

    for j in range(1,n_decoder_layers + 1):
        decoder_state_input_h = keras.Input(shape=(latent_dimension,))
        decoder_state_input_c = keras.Input(shape=(latent_dimension,))
        current_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder = model.get_layer('decoder_hidden_' + str(j))
        decoder_outputs, state_h_dec, state_c_dec = decoder(decoder_outputs, initial_state=current_states_inputs)
        decoder_states += [state_h_dec, state_c_dec]
        decoder_states_inputs += current_states_inputs
    
    decoder_dense = model.get_layer('decoder_output')
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    return encoder_model, decoder_model
      

#Accuracy  
def accuracy(val_encoder_input_data, val_target_data,n_decoder_layers,encoder_model,decoder_model, verbose=False):
    n_correct = 0
    n_total = 0
    for seq_index in range(len(val_encoder_input_data)):
        # Taking one sequence 
        input_seq = val_encoder_input_data[seq_index: seq_index + 1]

        # decoded_sentence = decode_textsequence(input_seq,n_decoder_layers,'LSTM',encoder_model,decoder_model)
        
        states_val = [encoder_model.predict(input_seq)]*n_decoder_layers

        empty_sequence = np.zeros((1, 1))
        # adding first character of target sequence with the start character.
        empty_sequence[0, 0] = target_token_idx["\t"]
        target_sequence = empty_sequence

 
        stop_cond = False
        decoded_sentence = ""
        while not stop_cond:
            if cell_type is not None and (cell_type.lower() == 'rnn' or cell_type.lower() == 'gru'):
                temp = decoder_model.predict([target_sequence] + [states_val])
                output_tokens, states_val = temp[0], temp[1:]
            else:
                temp = decoder_model.predict([target_sequence] + states_val )
                output_tokens, states_val = temp[0], temp[1:]

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            if sampled_char == "\n" or len(decoded_sentence) > decoder_max_length:
                stop_cond = True

            # Updating the target sequence.
            target_sequence = np.zeros((1, 1))
            target_sequence[0, 0] = sampled_token_index
        
        
        
        
        if decoded_sentence.strip() == val_target_data[seq_index].strip():
            n_correct += 1

        n_total += 1

        if verbose:
            print('Prediction ', decoded_sentence.strip(), ',Ground Truth ', val_target_data[seq_index].strip())
    
    accuracy =n_correct * 100.0 / n_total
    return accuracy
  


#Best model
encoder_model, decoder_model=seq2seq(embedding_size,num_encoder_tokens,num_decoder_tokens,n_encoder_layers, n_decoder_layers,latent_dimension,
                cell_type, target_token_index, max_decoder_seq_length,reverse_target_char_index, dropout ,encoder_input_data, decoder_input_data,
                decoder_target_data,batch_size,epochs)

# encoder_model.save('encoder_model.h5')
# decoder_model.save('decoder_model.h5')

val_accuracy= accuracy(val_encoder_input_data, val_target_texts,n_decoder_layers,encoder_model,decoder_model)
print('Validation accuracy: ', val_accuracy)


#Test Accuracy

# compute test accuracy

print('Reading test data')
test_data_path = "dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.test.tsv"

with open(test_data_path, "r", encoding="utf-8") as f:
    test_lines = f.read().split("\n")
# popping the last element since it is empty character
test_lines.pop()

# embedding test
test_input_data = []
test_target_data = []
for line in test_lines[: (len(test_lines) - 1)]:
    target_text, input_text, _ = line.split("\t")
    # We use "tab" as the "start sequence" character and "\n" as "end sequence" character.
    target_text = "\t" + target_text + "\n"
    test_input_data.append(input_text)
    test_target_data.append(target_text)

test_max_encoder_seq_length = max([len(txt) for txt in test_input_data])
test_max_decoder_seq_length = max([len(txt) for txt in test_target_data])
test_encoder_input_data = np.zeros((len(test_input_data), test_max_encoder_seq_length), dtype="float32")
test_decoder_input_data = np.zeros((len(test_input_data), test_max_decoder_seq_length), dtype="float32")
test_decoder_target_data = np.zeros((len(test_input_data), test_max_decoder_seq_length, num_decoder_tokens), dtype="float32")
for i, (input_text, target_text) in enumerate(zip(test_input_data, test_target_data)):
    for t, char in enumerate(input_text):
        test_encoder_input_data[i, t] = input_token_idx[char]
    test_encoder_input_data[i, t + 1:] = input_token_idx[" "]
    for t, char in enumerate(target_text):
        # decoder_target_data is one timestep ahead of decoder_input_data
        test_decoder_input_data[i, t] = target_token_idx[char]
        if t > 0:
            # excluding the start character since decoder target data is one timestep ahead.
            test_decoder_target_data[i, t - 1, target_token_idx[char]] = 1.0
    test_decoder_input_data[i, t + 1:] = target_token_idx[" "]
    test_decoder_target_data[i, t:, target_token_idx[" "]] = 1.0


test_accuracy= accuracy(test_encoder_input_data, test_target_texts,n_decoder_layers,encoder_model,decoder_model)
print('Test accuracy: ', test_accuracy)


