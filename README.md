# Overview of Assignment-3

The purpose of this assignment is to,

  1. Build and train a RNN based seq2seq model from scratch.
  2. Implement attention based model.

Link for wandb report:

To train a Seq2Seq model for Dakshina Dataset transliteration from English to Tamil, use 
**seq2seq.ipynb**

To train a Seq2Seq model with Attention for Dakshina Dataset transliteration from English to Tamil, use 
**Attention.ipynb**

# seq2seq.ipynb

Trains the model and computes the validation accuracy.

**embedding_train(train_lines)**

The data embedding of train dataset is done here,

**encoder_model, decoder_model=seq2seq(embedding_size, num_encoder_tokens,num_decoder_tokens,n_encoder_layers, n_decoder_layers,latent_dimension,
                cell_type, target_token_index, max_decoder_seq_length,reverse_target_char_index, dropout ,encoder_input_data, decoder_input_data,
                decoder_target_data,batch_size,epochs)**
                
This function is used to build and train a seq2seq model.

# Best_model_seq2seq.ipynb

The best model is built and trained using best hyperparameters and the predictions are made on test data and then test accuracy is computed. 

# Attention.ipynb

This file contains code to configure and run wandb sweeps for attention models.

# Best_model_attention.ipynb

The best model is built and trained using best hyperparameters and the predictions are made on test data and then test accuracy is computed. 



 
