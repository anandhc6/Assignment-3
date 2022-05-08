# Overview of Assignment-3

The purpose of this assignment is to,

  1. Build and train a RNN based seq2seq model from scratch.
  2. Implement attention based model.

Link for wandb report: https://wandb.ai/anandh/CS6910_Assignment3_Attention/reports/Assignment-3--VmlldzoxOTYyNzgx

To train a Seq2Seq model for Dakshina Dataset transliteration from English to Tamil, use 
**seq2seq.ipynb**

To train a Seq2Seq model with Attention for Dakshina Dataset transliteration from English to Tamil, use 
**Attention.ipynb**

# seq2seq.ipynb

This notebook trains the model and computes the validation accuracy.

**encoder_input_data,decoder_input_data,decoder_target_data,num_encoder_tokens,num_decoder_tokens,input_token_idx,target_token_idx,encoder_max_length,decoder_max_length = embed_train_data(train_data_lines)**

This function is used for data embedding of train dataset.
The same kind of function is used forve validation data.

**encoder_model, decoder_model=seq2seq(embedding_size, num_encoder_tokens,num_decoder_tokens,n_encoder_layers, n_decoder_layers,latent_dimension,
                cell_type, target_token_index, max_decoder_seq_length,reverse_target_char_index, dropout ,encoder_input_data, decoder_input_data,
                decoder_target_data,batch_size,epochs)**
                
This function is used to build and train a seq2seq model.
And it also gives us the inference encoder model and decdder model.

**decode_textsequence(input_seq,n_decoder_layers,'LSTM',encoder_model,decoder_model)**

The above function gets an input sequence and then it decodes them into decoded sequences.

**val_accuracy= accuracy(val_encoder_input_data, val_target_data,n_decoder_layers,encoder_model,decoder_model)**

This function gives the accuracy of validation dataset.

**Sweep**

The sweep configuration allows us to run several number of experiments with the hyperparameters.
The sweep calls the fit function which builds and trains the model. The best model and its hyperparameters could be obtained from the sweep.

**Passing hyperparameters as command line arguments in seq2seq model**

You can pass the following command with the 'dakshina_dataset_v1.0' data folder in the present working directory.

python <filename> <cell_type> <n_encoder_layers> <n_decoder_layers> <embedding_size> <latent_dimension> <dropout> <epochs>
 
* ```cell_type```,     this argument requires the cell to be used by the model. 
* ```n_encoder_layers```, this argument requires the number of layers to be used by the encoder.
* ```n_decoder_layers```, this argument requires the number of layers to be used by the decoder.
* ```embedding_size```, this argument requires the embedding size to train the model according to it..
* ```latent_dimension```, this argument requires the number of filters to be used by the model.
* ```dropout```,  this argument requires dropout to be passed.
* ```epochs```,this argument requires number of epochs to train the model.
  
  Example:```python seq2seq_cmdline.py 'LSTM' 3 3 256 512 0.3 30```

# Best_model_seq2seq.ipynb

The best model is built and trained using best hyperparameters and the predictions are made on test data and then test accuracy is computed. 

**Best model performance**

Test accuracy: 49.16%

The best model obtained was:

![model](https://user-images.githubusercontent.com/99970529/167281833-f72205e8-a4fc-4bcf-a3e4-c4fe09d8b673.png)

 
# Attention.ipynb

This file contains code to configure and run wandb sweeps for attention models.
This uses all the functions used in seq2seq, adding to it we have a class namely, AttentionLayer 
  
 **AttentionLayer(name='attention_layer')([encoder_outputs, decoder_outputs])** 
  
  This class uses Bahdanau attention and implements it.

# Best_attention_model.ipynb

The best model is built and trained using best hyperparameters and the predictions are made on test data and then test accuracy is computed. 

**Best model performance**

Test accuracy: **53.16%**
  
The best model obtained was:  
  
![model (1)](https://user-images.githubusercontent.com/99970529/167287182-7d3be43d-c3bc-4f38-aab4-4af552bbccd2.png)

# Passing hyperparameters as command line arguments in seq2seq with Attention.

You can pass the following command with the 'dakshina_dataset_v1.0' data folder in the present working directory.

python <filename> <epochs> <embedding_size> <latent_dimension> <cell_type> <dropout> 
  
* ```epochs```,this argument requires number of epochs to train the model.
* ```embedding_size```, this argument requires the embedding size to train the model according to it..
* ```latent_dimension```, this argument requires the number of filters to be used by the model.
* ```cell_type```,     this argument requires the cell to be used by the model. 
* ```dropout```,  this argument requires dropout to be passed.

  
  Example:```python Attention_cmdline.py 10 128 256 'LSTM' 0.2```
  
  
# Code for Visualisation is in **Best_attention_model.ipynb**


 
