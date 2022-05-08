#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 10:46:48 2022

@author: tenet
"""
#python Attention_cmdline.py 10 128 256 'LSTM' 0.2

# Required packages 

import os
import sys
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from random import randrange 
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


epochs = sys.argv[1]
embedding_size =sys.argv[2]
latent_dimension = sys.argv[3]
cell_type = sys.argv[4]
dropout = sys.argv[5]

# parameters
n_encoder_layers=1
n_decoder_layers=1


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

#Attention Layer

class AttentionLayer(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
    
    Credits to Tensorflow.org and https://github.com/thushv89/attention_keras/blob/master/src/layers/attention.py
     """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.        

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(AttentionLayer, self).build(input_shape) 

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print('encoder_out_seq>', encoder_out_seq.shape)
            print('decoder_out_seq>', decoder_out_seq.shape)

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state
            inputs: (batchsize * 1 * de_in_dim)
            states: (batchsize * 1 * de_latent_dim)
            """

            assert_msg = "States must be an iterable. Got {} of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            
            W_a_dot_s = K.dot(encoder_out_seq, self.W_a)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
            if verbose:
                print('Ua.h>', U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            
            Ws_plus_Uh = K.tanh(W_a_dot_s + U_a_dot_h)
            if verbose:
                print('Ws+Uh>', Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
           
            e_i = K.squeeze(K.dot(Ws_plus_Uh, self.V_a), axis=-1)
           
            e_i = K.softmax(e_i)

            if verbose:
                print('ei>', e_i.shape)
            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """
            assert_msg = "States must be an iterable. Got {} of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print('ci>', c_i.shape)
            return c_i, [c_i]
        fake_state_c = K.sum(encoder_out_seq, axis=1)
        fake_state_e = K.sum(encoder_out_seq, axis=2)  
        """ Computing energy outputs """
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )
        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c],
        )
        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]

# RNN model 

def seq2seq(embedding_size, n_encoder_tokens, n_decoder_tokens, n_encoder_layers,n_decoder_layers, latent_dimension, cell_type,
            target_token_idx, decoder_max_length, reverse_target_char_index,dropout,encoder_input_data, decoder_input_data,
            decoder_target_data,batch_size,epochs
):
  encoder_inputs = keras.Input(shape=(None,), name='encoder_input')

  encoder = None
  encoder_outputs = None
  state_h = None
  state_c = None
  e_layer=n_encoder_layers

  # RNN 

  if cell_type=="RNN":
    embed = tf.keras.layers.Embedding(input_dim=n_encoder_tokens, output_dim=embedding_size,name='encoder_embedding')(encoder_inputs)

    encoder = keras.layers.SimpleRNN(latent_dimension, return_state=True, return_sequences=True,name='encoder_hidden_1', dropout=dropout)
    encoder_outputs, state_h = encoder(embed)

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
    
    attn_out, attn_states = AttentionLayer(name='attention_layer')([encoder_outputs, decoder_outputs])
    decoder_concat_input = tf.keras.layers.Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

    decoder_dense = keras.layers.Dense(n_decoder_tokens, activation="softmax",name="dense_1")
    decoder_outputs = decoder_dense(decoder_concat_input)
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy",metrics=['accuracy'])                

    model.fit(
          [encoder_input_data, decoder_input_data],
          decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=WandbCallback()
      )
    model.summary()

    # Inference model
    encoder_inputs = model.input[0]
    encoder_outputs, state_h_enc = model.get_layer('encoder_hidden_' + str(n_encoder_layers)).output
    encoder_states = [encoder_outputs, state_h_enc]
    encoder_model = keras.Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]  
    decoder_outputs = model.get_layer('decoder_embedding')(decoder_inputs)
    decoder_states_inputs = []
    decoder_states = []
    decoder_hidden_state = keras.Input(shape=(None,latent_dimension), name = "input_4")

    for j in range(1, n_decoder_layers + 1):
        decoder_state_input_h = keras.Input(shape=(latent_dimension,))
        current_states_inputs = [decoder_state_input_h]
        decoder = model.get_layer('decoder_hidden_' + str(j))
        decoder_outputs, state_h_dec = decoder(decoder_outputs, initial_state=current_states_inputs)
        decoder_states += [state_h_dec]
        decoder_states_inputs += current_states_inputs


    attention_layer = model.get_layer('attention_layer')
    attention_out, attention_states = attention_layer([decoder_hidden_state, decoder_outputs])

    decoder_concat_inf = keras.layers.Concatenate(axis=-1, name='concat_layer_inf')([decoder_outputs, attention_out])
    decoder_dense = model.get_layer("dense_1")
    decoder_dense_outputs = decoder_dense(decoder_concat_inf)
    decoder_model=keras.Model([decoder_inputs]+[decoder_hidden_state,decoder_state_input_h],[decoder_dense_outputs]+decoder_states+[attention_states])
    return encoder_model, decoder_model

  # GRU

  elif cell_type=="GRU":
    embed = tf.keras.layers.Embedding(input_dim=n_encoder_tokens, output_dim=embedding_size,name='encoder_embedding')(encoder_inputs)
    encoder = keras.layers.GRU(latent_dimension, return_state=True, return_sequences=True,name='encoder_hidden_1', dropout=dropout)
    encoder_outputs, state_h = encoder(embed)
   
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
    attn_out, attn_states = AttentionLayer(name='attention_layer')([encoder_outputs, decoder_outputs])
    decoder_concat_input = tf.keras.layers.Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])
    decoder_dense = keras.layers.Dense(n_decoder_tokens, activation="softmax", name='dense_1')
    decoder_outputs = decoder_dense(decoder_concat_input)
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy",metrics=['accuracy'])#, metrics=[my_metric]                 

    model.fit(
          [encoder_input_data, decoder_input_data],
          decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=WandbCallback()
      )
    model.summary()

    # Inference model
    encoder_inputs = model.input[0]
    encoder_outputs, state_h_enc = model.get_layer('encoder_hidden_' + str(n_encoder_layers)).output
    encoder_states = [encoder_outputs,state_h_enc]
    encoder_model = keras.Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]  
    decoder_outputs = model.get_layer('decoder_embedding')(decoder_inputs)
    decoder_states_inputs = []
    decoder_states = []
    decoder_hidden_state = keras.Input(shape=(None,latent_dimension), name = "input_4")


    for j in range(1, n_decoder_layers + 1):
        decoder_state_input_h = keras.Input(shape=(latent_dimension,))
        current_states_inputs = [decoder_state_input_h]
        decoder = model.get_layer('decoder_hidden_' + str(j))
        decoder_outputs, state_h_dec = decoder(decoder_outputs, initial_state=current_states_inputs)
        decoder_states += [state_h_dec]
        decoder_states_inputs += current_states_inputs

    attention_layer = model.get_layer('attention_layer')
    attention_out, attention_states = attention_layer([decoder_hidden_state, decoder_outputs])
    decoder_concat_inf = keras.layers.Concatenate(axis=-1, name='concat_layer_inf')([decoder_outputs, attention_out])

    decoder_dense = model.get_layer('dense_1')
    decoder_dense_outputs = decoder_dense(decoder_concat_inf)
    decoder_model=keras.Model([decoder_inputs]+[decoder_hidden_state,decoder_state_input_h],[decoder_dense_outputs]+decoder_states+[attention_states])

    return encoder_model, decoder_model

  # LSTM

  elif cell_type=="LSTM":
    embed = tf.keras.layers.Embedding(input_dim=n_encoder_tokens, output_dim=embedding_size,name='encoder_embedding')(encoder_inputs)
    encoder = keras.layers.LSTM(latent_dimension, return_state=True, return_sequences=True,name='encoder_hidden_1', dropout=dropout)
    encoder_outputs, state_h, state_c = encoder(embed)
  
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
    attn_out, attn_states = AttentionLayer(name='attention_layer')([encoder_outputs, decoder_outputs])
    decoder_concat_input = tf.keras.layers.Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])
    decoder_dense = keras.layers.Dense(n_decoder_tokens, activation="softmax", name='dense_1')
    decoder_outputs = decoder_dense(decoder_concat_input)
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy",metrics=['accuracy'])#, metrics=[my_metric]                 
    
    model.fit(
          [encoder_input_data, decoder_input_data],
          decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=WandbCallback()
      )
    model.summary()

    # Inference model
    encoder_inputs = model.input[0]
    encoder_outputs, state_h_enc, state_c_enc = model.get_layer('encoder_hidden_' + str(n_encoder_layers)).output
    encoder_states = [encoder_outputs,state_h_enc, state_c_enc]
    encoder_model = keras.Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]  
    decoder_outputs = model.get_layer('decoder_embedding')(decoder_inputs)
    decoder_states_inputs = []
    decoder_states = []
    decoder_hidden_state = keras.Input(shape=(None,latent_dimension), name = "input_4")

    for j in range(1,n_decoder_layers + 1):
        decoder_state_input_h = keras.Input(shape=(latent_dimension,))
        decoder_state_input_c = keras.Input(shape=(latent_dimension,))
        current_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder = model.get_layer('decoder_hidden_' + str(j))
        decoder_outputs_inf, state_h_dec, state_c_dec = decoder(decoder_outputs, initial_state=current_states_inputs)
        decoder_states += [state_h_dec, state_c_dec]
        decoder_states_inputs += current_states_inputs

    attention_layer = model.get_layer('attention_layer')
    attention_out, attention_states = attention_layer([decoder_hidden_state, decoder_outputs_inf])
    decoder_concat_inf = keras.layers.Concatenate(axis=-1, name='concat_layer_inf')([decoder_outputs_inf, attention_out])

    decoder_dense = model.get_layer('dense_1')
    decoder_dense_outputs = decoder_dense(decoder_concat_inf)
    decoder_model=keras.Model([decoder_inputs]+[decoder_hidden_state,decoder_state_input_h,decoder_state_input_c],[decoder_dense_outputs]+decoder_states+[attention_states])

    return encoder_model, decoder_model
      

#Accuracy 

def accuracy(val_encoder_input_data, val_target_texts,n_decoder_layers,encoder_model,decoder_model,cell_type,latent_dimension, verbose=False):
        correct_count = 0
        total_count = 0
        n_val_data=len(val_encoder_input_data)
        for seq_idx in range(n_val_data):
            #Taking one sequence
            input_charseq = val_encoder_input_data[seq_idx: seq_idx + 1]

            # Encode the input as state vectors.
            if cell_type=='LSTM':
              encoder_first_outputs,state_h,state_c= encoder_model.predict(input_charseq)
              states_value=[state_h,state_c]
            else:
              encoder_first_outputs,states_value= encoder_model.predict(input_charseq)

            empty_charseq = np.zeros((1, 1))
            empty_charseq[0, 0] = target_token_idx["\t"]
            target_charseq = empty_charseq

            # Loop for batch
            stop_condition = False
            decoded_sentence = ""
            attention_weights = []
            while not stop_condition:
                if cell_type == "LSTM":
                    output_tokens, h, c,attn = decoder_model.predict([target_charseq, encoder_first_outputs] + states_value)
                elif cell_type == "RNN" or cell_type == "GRU":
                    states_value = states_value[0].reshape((1, latent_dimension))
                    output_tokens, h ,attn= decoder_model.predict([target_charseq] + [encoder_first_outputs] + [states_value])

                sampled_token_idx = np.argmax(output_tokens[0, -1, :])
                sampled_character = reverse_target_char_index[sampled_token_idx]
                decoded_sentence += sampled_character

                if sampled_character == "\n" or len(decoded_sentence) > decoder_max_length:
                    stop_condition = True

                # Updating the target sequence
                target_charseq = np.zeros((1, 1))
                target_charseq[0, 0]= sampled_token_idx

                # Update states
                if cell_type == "LSTM":
                    states_value = [h, c]
                elif cell_type == "RNN" or cell_type == "GRU":
                    states_value = [h]


            if decoded_sentence.strip() == val_target_texts[seq_idx].strip():
                correct_count += 1

            total_count += 1
            print(correct_count)

            if verbose:
                print('Prediction ', decoded_sentence.strip(), ',Ground Truth ', val_target_texts[seq_idx].strip())
        accuracy =correct_count * 100.0 / total_count
        return accuracy
    

# calling model build
encoder_model, decoder_model=seq2seq(embedding_size, n_encoder_tokens,n_decoder_tokens,n_encoder_layers, n_decoder_layers,latent_dimension,
                cell_type, target_token_index, max_decoder_seq_length,reverse_target_char_index, dropout ,encoder_input_data, decoder_input_data,
                decoder_target_data,batch_size,1)


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
print('Test accuracy: ', test_accuracy)