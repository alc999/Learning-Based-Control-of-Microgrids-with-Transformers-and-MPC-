
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:32:10 2023

@author: bobvanderwoude
"""


from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import time 
# from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import gurobipy as gp
from gurobipy import GRB
import torch.onnx
import onnxruntime as ort




## Use Apple GPU for trainig.
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print ("MPS available, current device MPS")
else:
    print ("MPS device not found.")
    



class SourceFeatureEmbedder(nn.Module):
    def __init__(self, feature_dim, d_model):
        super(SourceFeatureEmbedder, self).__init__()
        self.linear = nn.Linear(feature_dim, d_model)
    
    def forward(self, x):
        # Directly apply the linear transformation.
        return self.linear(x)

## Multi-Head Attention
#The Multi-Head Attention mechanism computes the attention between each pair of positions in a sequence. 
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize parameters 
        self.d_model = d_model                          # Dimensionality of the input/output
        self.num_heads = num_heads                      # Number of parallel attention heads
        self.d_k = d_model // num_heads                 # Dimensionality of keys and queries per head
        
        # Linear transformations for queires, keys, values and output 
        self.W_q = nn.Linear(d_model, d_model)          
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-1e9'))
        attn_probs = torch.softmax(attn_scores, dim=-1)  # Apply softmax to get probabilities
        output = torch.matmul(attn_probs, V)  # Weighted sum of values based on attention probs
        return output

    
    # Split heads for parallel computation
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()        
        assert d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        x = x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        return x
        
    def combine_heads(self, x):
        batch_size, num_heads, seq_length, d_k = x.size()
        
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)  # -1 implies num_heads * d_k
        return x
        
  
       
    
    # Forward pass of the multi-head attention layer
    def forward(self, Q, K, V, mask=None):
        # Linearly transform inputs for queries, keys, and values
        Q = self.split_heads(self.W_q(Q))                       
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Apply scaled dot-product attention mechanism
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        # print("Attn Output shape:", attn_output.shape)
        # Combine heads and perform final linear transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output
 
    
## Position-wise Feed-Forward Networks
# The PositionWiseFeedForward class extends PyTorch’s nn.Module and implements a position-wise feed-forward network.
# This process enables the model to consider the position of input elements while making predictions.

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)                     # First fully connected layer
        self.fc2 = nn.Linear(d_ff, d_model)                     # Second fully connected layer
        self.relu = nn.ReLU()                                   # ReLU activation function

    def forward(self, x):
        # Define the forward pass of the position-wise feed-forward network
        return self.fc2(self.relu(self.fc1(x)))                 # Output of the feed-forward network


## Positional Encoding
#Positional Encoding is used to inject the position information of each token in the input sequence. 
#The PositionalEncoding class initializes with input parameters d_model and max_seq_length, creating a tensor to store positional encoding values. 
  
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length,dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout) 
        pe = torch.zeros(max_seq_length, d_model)
        # position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        position = torch.arange(0, max_seq_length, dtype=torch.float).view(-1,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # print(f"Before positional encoding - shape of x: {x.shape}")
        x = x + self.pe[:, :x.size(1)]
        # print(f"After positional encoding - shape of x: {x.shape}")
        return self.dropout(x)

## Encoder Layer
# The Encoder layer consists of a Multi-Head Attention layer, a Position-wise Feed-Forward layer, and two Layer Normalization layers.

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        # Components of the encoder layer
        self.self_attn = MultiHeadAttention(d_model, num_heads)         # Multi-Head Self-Attention mechanism
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)      # Position-wise Feed-Forward network
        self.norm1 = nn.LayerNorm(d_model)                              # Layer Normalization for self-attention output
        self.norm2 = nn.LayerNorm(d_model)                              # Layer Normalization for feed-forward output
        self.dropout = nn.Dropout(dropout)                              # Dropout for regularization
        
    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask) if mask is not None else self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        # Position-wise Feed-Forward network followed by normalization and dropout
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

## Decoder Layer
## The Decoder layer consists of two Multi-Head Attention layers, a Position-wise Feed-Forward layer, and three Layer Normalization layers.

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        # Components of the decoder layer
        self.self_attn = MultiHeadAttention(d_model, num_heads)         # Multi-Head Self-Attention mechanism
        self.cross_attn = MultiHeadAttention(d_model, num_heads)        # Multi-Head Cross-Attention mechanism
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)      # Position-wise Feed-Forward network
        self.norm1 = nn.LayerNorm(d_model)                              # Layer Normalization for self-attention output
        self.norm2 = nn.LayerNorm(d_model)                              # Layer Normalization for cross-attention output
        self.norm3 = nn.LayerNorm(d_model)                              # Layer Normalization for feed-forward output
        self.dropout = nn.Dropout(dropout)                              # Dropout for regularization
        
    def forward(self, x, enc_output, tgt_mask, src_mask = None):
        # print("Decoder Layer - Input x shape:", x.shape)
        # Self-Attention mechanism followed by normalization and dropout
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-Attention mechanism followed by normalization and dropout
        if src_mask is not None: 
            attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        else: 
            attn_output = self.cross_attn(x, enc_output, enc_output)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Position-wise Feed-Forward network followed by normalization and dropout
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, feature_dim, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.d_model = d_model  # Define d_model as an instance variable FOR FASTER INFERENCE FUNCTION

        
        # Initialize the source feature embedder here
        self.source_feature_embedder = SourceFeatureEmbedder(feature_dim, d_model)
        
        ##NEW 
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length,dropout)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    # Generate masks for target sequences

    def generate_mask(self, tgt):
          # tgt_seq_len = tgt.size(1)
          tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
          seq_length = tgt.size(1)
          nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
          tgt_mask = nopeak_mask
    
          return tgt_mask
    
    #Forward pass of the Transformer model
    def forward(self, src, tgt):
        # Embed the source features first
        src = self.source_feature_embedder(src)
        tgt_mask = self.generate_mask(tgt)
        
        # print("Before positional encoding - shape of src:", src.shape)
        src_embedded = self.dropout(self.positional_encoding(src))
       
        tgt_embedded = self.tgt_embedding(tgt)
        tgt_embedded = self.dropout(self.positional_encoding(tgt_embedded))
        # print("After embedding and positional encoding - shape of tgt:", tgt_embedded.shape)  # Confirm shape

        # Encoder process
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output)
        
        # Decoder process
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, tgt_mask)
        # Final linear layer for output prediction
        output = self.fc(dec_output)
        return output
    



## DATA FOR TRANSFORMER
## Loading numpy data and changing to tensor format
def load_data_npz(file_path):
    # Load the .npz file
    data = np.load(file_path, allow_pickle=True)
    
    # Extract arrays using the exact keys used when saving
    cbuy_tensor = torch.tensor(np.vstack(data['cbuy_all']).astype(np.float32))[:,:-1]
    csell_tensor = torch.tensor(np.vstack(data['csell_all']).astype(np.float32))[:,:-1]
    cprod_tensor = torch.tensor(np.vstack(data['cprod_all']).astype(np.float32))[:,:-1]
    power_res_tensor = torch.tensor(np.vstack(data['power_res_all']).astype(np.float32))[:,:-1]
    power_load_tensor = torch.tensor(np.vstack(data['power_load_all']).astype(np.float32))[:,:-1]
    x0_tensor = torch.tensor(np.vstack(data['x0_all']).astype(np.float32))
    delta_transformed_tensor = torch.tensor(np.vstack(data['delta_transformed_all']).astype(np.int32))
    
    
    return cbuy_tensor, csell_tensor, cprod_tensor, power_res_tensor, power_load_tensor, x0_tensor, delta_transformed_tensor


file_path = "path_to_microgrid_data.npz"

cbuy_tensor, csell_tensor, cprod_tensor, power_res_tensor, power_load_tensor, x0_tensor, delta_transformed_tensor = load_data_npz(file_path)




net_power_load = power_load_tensor - power_res_tensor


src_data = torch.cat([
    cbuy_tensor.unsqueeze(-1),  # Adding an extra dimension for feature alignment
    csell_tensor.unsqueeze(-1),
    cprod_tensor.unsqueeze(-1),
    net_power_load.unsqueeze(-1),
    x0_tensor.unsqueeze(1).repeat(1, cbuy_tensor.size(1), 1)  # Repeating x0 across the sequence length
], dim=-1)  # Concatenate along the last dimension to combine features


#Create target data 
tgt_data = delta_transformed_tensor.long()


def normalize_batch(batch):

    mean = batch.mean(dim=(0, 1), keepdim=True)
    std = batch.std(dim=(0, 1), keepdim=True)
    normalized_batch = (batch - mean) / (std + 1e-5)  # Normalize per feature across batch and seq_length
    return normalized_batch



mini_batch_size = 32 
dataset_size = src_data.size(0)
val_size = int(dataset_size * 0.2) 
train_size = dataset_size - val_size

full_dataset = TensorDataset(src_data, tgt_data)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=mini_batch_size, shuffle=False)



# # ## 
config = {
    'src_vocab_size': None,  # Assume set elsewhere
    'tgt_vocab_size': 32,
    'd_model': 128,
    'num_heads': 4,
    'num_layers': 1,
    'd_ff': 2048,
    'feature_dim': 5,
    'max_seq_length': src_data.size(1),  # Assume set elsewhere
    'dropout': 0.1,
    'nr_epochs': 20,
    'lr': 0.0001  # Learning rate
}



def save_model_and_config(model, save_path, config):
    # Combine the model's state dictionary and its configuration in a single dictionary
    model_info = {
        'model_state_dict': model.state_dict(),
        'config': config
    }
    torch.save(model_info, save_path)
    print(f"Model and configuration saved to {save_path}")
    
    
def load_model_and_config(load_path):
    model_info = torch.load(load_path)
    model_state_dict = model_info['model_state_dict']
    config = model_info['config']
    
    # Recreate the model instance using the saved configuration
    loaded_model = Transformer(
            src_vocab_size=config['src_vocab_size'], 
            tgt_vocab_size=config['tgt_vocab_size'], 
            d_model=config['d_model'], 
            num_heads=config['num_heads'], 
            num_layers=config['num_layers'], 
            d_ff=config['d_ff'], 
            feature_dim=config['feature_dim'], 
            max_seq_length=config['max_seq_length'], 
            dropout=config['dropout'])
    loaded_model.load_state_dict(model_state_dict)
    # print(f"Model loaded from {load_path} with configuration: {config}")
    return loaded_model, config


    
    
def train_transformer(transformer, train_dataloader, val_dataloader, config, save_path='shallow_transformer_testd_model.pth'):
    criterion = nn.CrossEntropyLoss()
            
    optimizer = optim.Adam(transformer.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-9)  #earlier we used lr = 0.0001
    transformer.train()
    
    # Lists to store average losses per epoch
    avg_training_losses = []
    avg_validation_losses = []
    
    
    
    start_time = time.time()  
        

    for epoch in range(config['nr_epochs']):
        epoch_loss = 0
        for src_batch, tgt_batch in train_dataloader:
            src_batch_normalized = normalize_batch(src_batch)
            optimizer.zero_grad()
                        
            output = transformer(src_batch_normalized, tgt_batch[:,:-1])
            loss = criterion(output.contiguous().view(-1, config['tgt_vocab_size']),tgt_batch[:,1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        avg_training_losses.append(avg_epoch_loss)
        
        val_loss = validate_transformer(transformer, val_dataloader, criterion)
        avg_validation_losses.append(val_loss) 
        print(f"Epoch: {epoch+1}, Training Loss: {avg_epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total training time: {training_time} seconds")
    
    # Save the model's state dict
    save_model_and_config(transformer, save_path, config)
    print(f"Model saved to {save_path}")
    
    return avg_training_losses, avg_validation_losses
    

def validate_transformer(model, val_dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src_batch, tgt_batch in val_dataloader:
            src_batch_normalized = normalize_batch(src_batch)
            output = model(src_batch_normalized, tgt_batch[:,:-1])
            loss = criterion(output.contiguous().view(-1, config['tgt_vocab_size']), tgt_batch[:,1:].contiguous().view(-1))
            total_loss += loss.item()
    avg_val_loss = total_loss / len(val_dataloader)
    # print(f"Average Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss
        

# ##Inference on transformer 


    
def perform_inference(model, src_tensor, max_length=config['max_seq_length']):
    model.eval()  # Set the model to evaluation mode.

    timings = {}  # Dictionary to store timing for each section

    with torch.no_grad():
        # Initialize a sequence with the start token.

        # Timing source embedding
        start = time.time()
        src_tensor = model.source_feature_embedder(src_tensor)
        timings['source_embedding'] = time.time() - start

        # Timing positional encoding on source
        start = time.time()
        src_tensor = model.positional_encoding(src_tensor)
        timings['source_positional_encoding'] = time.time() - start

        # Timing encoder
        start = time.time()
        enc_output = src_tensor
        for layer in model.encoder_layers:
            enc_output = layer(enc_output)
        timings['encoder'] = time.time() - start

        generated_sequence = torch.zeros((src_tensor.size(0), max_length), dtype=torch.long).to(src_tensor.device)

        timing_details = {'tgt_embedding': 0, 'positional_encoding': 0, 'mask_generation': 0, 'decoder_pass': 0, 'logit_computation': 0}

        # Generate sequence one token at a time.

        for t in range(max_length):
            
            tgt_subsequence = generated_sequence[:, :t+1]
            start_embedding = time.time()
            tgt_subsequence = model.tgt_embedding(tgt_subsequence)
            timing_details['tgt_embedding'] += time.time() - start_embedding

            start_positional = time.time()
            tgt_subsequence = model.positional_encoding(tgt_subsequence)
            timing_details['positional_encoding'] += time.time() - start_positional

            start_mask = time.time()
            tgt_mask = model.generate_mask(tgt_subsequence)
            timing_details['mask_generation'] += time.time() - start_mask

            start_decoder = time.time()
            dec_output = tgt_subsequence
            for layer in model.decoder_layers:
                dec_output = layer(dec_output, enc_output, tgt_mask)
            timing_details['decoder_pass'] += time.time() - start_decoder

            start_logits = time.time()
            logits = model.fc(dec_output[:, -1, :])  # Only consider the last token.
            next_token = logits.argmax(-1)
            generated_sequence[:, t] = next_token
            timing_details['logit_computation'] += time.time() - start_logits

        # Print timing details
        for step, duration in timing_details.items():
            print(f"{step}: {duration:.4f}s")

        return generated_sequence

 




if __name__ == "__main__":
    #initialize the transformer model
    transformer = Transformer(
        src_vocab_size=config['src_vocab_size'], 
        tgt_vocab_size=config['tgt_vocab_size'], 
        d_model=config['d_model'], 
        num_heads=config['num_heads'], 
        num_layers=config['num_layers'], 
        d_ff=config['d_ff'], 
        feature_dim=config['feature_dim'], 
        max_seq_length=config['max_seq_length'], 
        dropout=config['dropout'])
    
    total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    # print(f"Total number of trainable parameters in the Transformer model: {total_params}")
    # # for name, param in transformer.named_parameters():
    # #     if param.requires_grad:
    # #         print(f"Layer: {name} | Size: {param.numel()}")
            
    perform_training = False      
    
    if perform_training:
        # train_transformer(transformer, train_dataloader, val_dataloader, nr_epochs, 'trained_transformer.pth')
        avg_training_losses, avg_validation_losses = train_transformer(transformer, train_dataloader, val_dataloader, config)

        
    else:
        
        model_state_path = 'transformer_name.pth'
        

        transformer, config = load_model_and_config(model_state_path)
        
 
        val_iter = iter(val_dataloader)
        src_batch, tgt_batch = next(val_iter)
            
        random_idx = torch.randint(0, src_batch.size(0), (1,)).item()
        src_tensor = src_batch[random_idx:random_idx+1]
        src_tensor_normalized = normalize_batch(src_tensor)
        predicted_sequence = perform_inference(transformer, src_tensor_normalized)

        # print("Predicted Sequence:", predicted_sequence)
        
        actual_sequence = tgt_batch[random_idx]
        # print("Actual Sequence:", actual_sequence)






def transform_predicted_to_binary_matrix(predicted_sequence):

    """
    Transforms a predicted sequence tensor of shape (1, N) into an (N x 5) binary matrix.
    
    :param predicted_sequence: A tensor of shape (1, N) with integers representing decimal values.
    :return: An (N x 5) matrix of binary values (as a list of lists).
    """
    # Ensure the input is a tensor, flatten it to 1D, and convert to a list of integers
    if torch.is_tensor(predicted_sequence):
        predicted_sequence = predicted_sequence.view(-1).tolist()  # Flatten and convert to list
    
    # Convert each integer to its 5-bit binary representation
    binary_matrix = np.array([list(map(int, '{:05b}'.format(num))) for num in predicted_sequence])

    
    return binary_matrix





#MLD equations
Ts = 1/2 # Ts = 15m
nd = 0.99
nc = 0.9 

Amld = np.array([
    [1]
])
B1 = np.array([
    [Ts/nd,0,0,0,0]
])
B2 = np.zeros((1,5))
B3 = np.array([
    [Ts*(nc-1/nd),0]
])
B5 = np.zeros((1,1))

#parameters
H = Ts*(nc - 1/nd)
F = Ts/nd

#ESS(battery)
Mb = 100
mb = -100

#grid
Mg = 1000
mg = -1000

#dispatchable generators
Md = 150
md = 6
mdg = 6

eps = 1e-6

# state constraint

E2_sc = np.zeros((2,5))
E3_sc = np.zeros((2,2))
E1_sc = np.zeros((2,5))
E4_sc = np.array([
    [-1],
    [1]
])
E5_sc = np.array([
    [250],
    [-25]
])

# input constraints

E2_ic = np.zeros((10,5))
E3_ic = np.zeros((10,2))
E1_ic = np.array([
    [-1,0,0,0,0],
    [1,0,0,0,0],
    [0,-1,0,0,0],
    [0,1,0,0,0],
    [0,0,-1,0,0],
    [0,0,1,0,0],
    [0,0,0,-1,0],
    [0,0,0,1,0],
    [0,0,0,0,-1],
    [0,0,0,0,1]
])
E4_ic = np.zeros((10,1))
E5_ic = np.array([
    [100],
    [100],
    [1000],
    [1000],
    [150],
    [0],
    [150],
    [0],
    [150],
    [0],
])

# continuous auxiliary variables

#z_b

E2_zb = np.array([
    [-Mb,0,0,0,0],
    [mb,0,0,0,0],
    [-mb,0,0,0,0],
    [Mb,0,0,0,0]
])
E3_zb = np.array([
    [1, 0],
    [-1, 0],
    [1, 0],
    [-1, 0]
])
E1_zb = np.array([
    [0,0,0,0,0],
    [0,0,0,0,0],
    [1,0,0,0,0],
    [-1,0,0,0,0]
])
E4_zb = np.zeros((4,1))
E5_zb = np.array([
    [0],
    [0],
    [-mb],
    [Mb]
])

#z_grid

E2_zg = np.array([
    [0,-Mg,0,0,0],
    [0,mg,0,0,0],
    [0,-mg,0,0,0],
    [0,Mg,0,0,0]
])
E3_zg = np.array([
    [0,1],
    [0,-1],
    [0,1],
    [0,-1]
])
E1_zg = np.array([
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,1,0,0,0],
    [0,-1,0,0,0]
])
E4_zg = np.zeros((4,1))
E5_zg = np.array([
    [0],
    [0],
    [-mg],
    [Mg]
])

# discrete variales

#E2,E3,E1,E4,E5

#battery (ESS)
E2_db = np.array([
    [-mb,0,0,0,0],
    [-(Mb+eps),0,0,0,0]
])
E3_db = np.zeros((2,2))
E1_db = np.array([
    [1,0,0,0,0],
    [-1,0,0,0,0]
])
E4_db = np.zeros((2,1))
E5_db = np.array([
    [-mb],
    [-eps]
])

#grid
E2_dg = np.array([
    [0,-mg,0,0,0],
    [0,-(Mg+eps),0,0,0]
])
E3_dg = np.zeros((2,2))
E1_dg = np.array([
    [0,1,0,0,0],
    [0,-1,0,0,0]
])
E4_dg = np.zeros((2,1))
E5_dg = np.array([
    [-mg],
    [-eps]
])

#gen 1
E2_d1 = np.array([
    [0,0,-md,0,0],
    [0,0,-(Md+eps),0,0]
])
E3_d1 = np.zeros((2,2))
E1_d1 = np.array([
    [0,0,1,0,0],
    [0,0,-1,0,0]
])
E4_d1 = np.zeros((2,1))
E5_d1 = np.array([
    [-md],
    [-eps]
])

#gen 2
E2_d2 = np.array([
    [0,0,0,-md,0],
    [0,0,0,-(Md+eps),0]
])
E3_d2 = np.zeros((2,2))
E1_d2 = np.array([
    [0,0,0,1,0],
    [0,0,0,-1,0]
])
E4_d2 = np.zeros((2,1))
E5_d2 = np.array([
    [-md],
    [-eps]
])

#gen 3
E2_d3 = np.array([
    [0,0,0,0,-md],
    [0,0,0,0,-(Md+eps)]
])
E3_d3 = np.zeros((2,2))
E1_d3 = np.array([
    [0,0,0,0,1],
    [0,0,0,0,-1]
])
E4_d3 = np.zeros((2,1))
E5_d3 = np.array([
    [-md],
    [-eps]
])

# generator constraint

# E2 delta , E3 zed, E1 u, E4 x, E5

E2_gc = np.array([
    [0,0,6,0,0],
    [0,0,0,6,0],
    [0,0,0,0,6],
    [0,0,-150,0,0],
    [0,0,0,-150,0],
    [0,0,0,0,-150]
])
E3_gc = np.zeros((6,2))
E1_gc = np.array([
    [0,0,1,0,0],
    [0,0,0,1,0],
    [0,0,0,0,1],
    [0,0,-1,0,0],
    [0,0,0,-1,0],
    [0,0,0,0,-1]
])
E4_gc = np.zeros((6,1))
E5_gc = np.zeros((6,1))

E1 = np.block([
    [E1_sc],
     [E1_ic],
    [E1_zb],
    [E1_zg],
    [E1_db],
    [E1_dg],
    [E1_d1],
    [E1_d2],
    [E1_d3],
    [E1_gc]
])

E2 = np.block([
    [E2_sc],
     [E2_ic],
    [E2_zb],
    [E2_zg],
    [E2_db],
    [E2_dg],
    [E2_d1],
    [E2_d2],
    [E2_d3],
    [E2_gc]
])

E3 = np.block([
    [E3_sc],
     [E3_ic],
    [E3_zb],
    [E3_zg],
    [E3_db],
    [E3_dg],
    [E3_d1],
    [E3_d2],
    [E3_d3],
    [E3_gc]
])

E4 = np.block([
    [E4_sc],
     [E4_ic],
    [E4_zb],
    [E4_zg],
    [E4_db],
    [E4_dg],
    [E4_d1],
    [E4_d2],
    [E4_d3],
    [E4_gc]
])

E5 = np.block([
    [E5_sc],
     [E5_ic],
    [E5_zb],
    [E5_zg],
    [E5_db],
    [E5_dg],
    [E5_d1],
    [E5_d2],
    [E5_d3],
    [E5_gc]
])
    

n=Amld.shape[1]; m=B1.shape[1]; N_bin = B2.shape[1]; N_z = B3.shape[1]


xmax = 250; xmin = 25
umax = [100,1000,150,150,150]
umin = [-100,-1000,0,0,0]
zmax = [100,1000]
zmin = [-100,-1000]




def gurobi_qp(x0, N, net_power, cbuy, csell, cprod, delta):
    mdl = gp.Model("hybridMPC")
    mdl.Params.LogToConsole = 0
    
    xmin_tile = np.tile(xmin, (N+1,1))
    xmax_tile = np.tile(xmax, (N+1,1))
    zmin_tile = np.tile(zmin, (N,1))
    zmax_tile = np.tile(zmax, (N,1))
    umin_tile = np.tile(umin, (N,1))
    umax_tile = np.tile(umax, (N,1))

    x = mdl.addMVar(shape=(N+1, n), lb=xmin_tile, ub=xmax_tile, name='x') #1*5= 5
    z = mdl.addMVar(shape=(N, N_z), lb=zmin_tile, ub=zmax_tile, name='z') #2*4= 8
    u = mdl.addMVar(shape=(N, m), lb=umin_tile, ub=umax_tile, name='u') # 5*4 = 20



    # 1 + 1*4 + 30*4 + 1*4= 129 (number of constraints)
    mdl.addConstr(x[0, :] == x0.reshape(Amld.shape[0],))
    for k in range(N):
        mdl.addConstr(x[k+1, :] == Amld @ x[k, :] + B1 @ u[k, :] + B2 @ delta[k, :] + B3 @ z[k,:] + B5.reshape(B1.shape[0],)) # dynamics
        mdl.addConstr(E2 @ delta[k, :] + E3 @ z[k, :] <= E1 @ u[k,:] + E4 @ x[k,:] + E5.reshape(E1.shape[0],)) # mld constraints
        mdl.addConstr(u[k,0]-u[k,1]-u[k,2]-u[k,3]-u[k,4] + net_power[k] == 0) # power balance

    obj1 = sum(cbuy[k]*z[k,1] - csell[k]*z[k,1] + csell[k]*u[k,1]  for k in range(N)) # cost for power exchanged with grid
    obj2 = sum(cprod[k]*u[k,2:].sum() for k in range(N)) # cost for energy production by dispatchable generators
    mdl.setObjective(obj1 + obj2, GRB.MINIMIZE)

    mdl.optimize()
    
    return mdl

    # return mdl


    
    
    
    
def plot_training_validation_loss(training_loss, validation_loss, title='Training and Validation Loss', xlabel='Epoch', ylabel='Loss'):
    """
    Plot the training and validation loss over epochs.

    Parameters:
    - training_loss: List of training loss values.
    - validation_loss: List of validation loss values.
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    """
    epochs = list(range(1, len(training_loss) + 1))
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, training_loss, label='Training Loss', marker='o', linestyle='-')
    plt.plot(epochs, validation_loss, label='Validation Loss', marker='x', linestyle='--')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(epochs, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# plot_training_validation_loss(training_loss=avg_training_losses, validation_loss=avg_validation_losses, title='Training and Validation Loss', xlabel='Epoch', ylabel='Loss')

def plot_predicted_vs_actual(predicted_sequence, actual_sequence, title='Predicted vs Actual Sequence', xlabel='Sequence Position', ylabel='Value'):
    """
    Plot the predicted sequence against the actual sequence for comparison and calculate the accuracy of the prediction.

    Parameters:
    - predicted_sequence: A tensor or list of predicted values.
    - actual_sequence: A tensor or list of actual values.
    - title: Title of the plot.
    - xlabel: Label for the x-axis.p
    - ylabel: Label for the y-axis.
    """
    
    # Convert tensors to lists if they are not already, assuming predicted_sequence might be a tensor with extra dimension
    if not isinstance(predicted_sequence, list):
        predicted_sequence = predicted_sequence.squeeze().tolist()  # Remove extra dimensions and convert to list
    if not isinstance(actual_sequence, list):
        actual_sequence = actual_sequence.tolist()  # Convert to list
    
    # Calculate accuracy
    correct_predictions = sum(p == a for p, a in zip(predicted_sequence, actual_sequence))
    accuracy = correct_predictions / len(actual_sequence)
    
    positions = list(range(1, len(actual_sequence) + 1))
    
    plt.figure(figsize=(14, 7))
    plt.plot(positions, predicted_sequence, label='Predicted Sequence', marker='o', linestyle='-', color='blue')
    plt.plot(positions, actual_sequence, label='Actual Sequence', marker='x', linestyle='--', color='red')
    plt.title(f"{title} - Accuracy: {accuracy*100:.2f}%")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(positions)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()





def try_model(model, src_tensor_normalized):
    # Start timing inference
    start_time_inference = time.time()
    predicted_sequence = perform_inference(model, src_tensor_normalized)
    elapsed_time_inference = time.time() - start_time_inference
    print(f"Inference Time: {elapsed_time_inference:.4f} seconds")

    # Start timing transformation to binary matrix
    start_time_transformation = time.time()
    delta_predicted = transform_predicted_to_binary_matrix(predicted_sequence)
    elapsed_time_transformation = time.time() - start_time_transformation
    print(f"Transformation Time: {elapsed_time_transformation:.4f} seconds")

    return delta_predicted






def run_optimization(delta, src_tensor, model_type):
    try:
        # Ensure input to gurobi_qp is correctly formatted
        x0 = np.array(src_tensor[0,1,4])
        net_power_load = src_tensor[:,:,3].numpy().squeeze()
        cbuy = src_tensor[:,:,0].numpy().squeeze()
        csell = src_tensor[:,:,1].numpy().squeeze()
        cprod = src_tensor[:,:,2].numpy().squeeze()
        
        # Call gurobi_qp with validated inputs
        mdl = gurobi_qp(x0, config['max_seq_length'], net_power_load, cbuy, csell, cprod, delta)

        # Check and log the optimization status
        if mdl.status == gp.GRB.OPTIMAL:
            print(f"{model_type} model optimization succeeded with objective value: {mdl.ObjVal}")
        else:
            print(f"{model_type} model optimization failed with status: {mdl.status}")
        return mdl
    except Exception as e:
        print(f"Exception during {model_type} model optimization: {e}")
        return None


def cascaded_transformer(models, val_dataloader, num_instances):
    it = iter(val_dataloader)  # Create an iterator over the validation dataloader
    stage_successes = [0] * len(models)
    cost_actual_mem = []
    cost_pred_mem = []
    successful_optimizations = 0
    detailed_timings = {'data_loading': [], 'normalization': [], 'prediction': [], 'optimization1': [], 'optimization2': []}
    instances_processed = 0
    unique_data_samples = set()

    while instances_processed < num_instances:
        try:
            src_batch, tgt_batch = next(it)  # Fetch the next batch
        except StopIteration:
            print("Ran out of validation data.")
            break  # Exit the loop if there are no more batches

        for idx in range(src_batch.size(0)):
            if instances_processed >= num_instances:
                break  # Stop processing if we've reached the desired number of instances

            # Optional: Check for uniqueness of data samples
            data_sample_id = src_batch[idx].numpy().tobytes()  # Convert tensor to bytes for hashability
            if data_sample_id in unique_data_samples:
                print(f"Duplicate data sample detected at instance {instances_processed}")
            unique_data_samples.add(data_sample_id)
            
            instances_processed += 1
            # Data loading
            start_time_data_loading = time.time()
            src_tensor = src_batch[idx:idx + 1]
            detailed_timings['data_loading'].append(time.time() - start_time_data_loading)

            # Normalization
            start_time_normalization = time.time()
            src_tensor_normalized = normalize_batch(src_tensor)
            detailed_timings['normalization'].append(time.time() - start_time_normalization)

            # Actual model optimization
            delta_actual = transform_predicted_to_binary_matrix(tgt_batch[idx])
            start_time_opt_actual = time.time()
            mdl_act = run_optimization(delta_actual, src_tensor, 'actual')
            detailed_timings['optimization1'].append(time.time() - start_time_opt_actual)
            if mdl_act and mdl_act.status == gp.GRB.OPTIMAL:
                cost_actual_mem.append(mdl_act.ObjVal)

            # Model predictions and subsequent optimizations
            optimization_found = False
            for i, model in enumerate(models):
                model.eval()
                start_time_prediction = time.time()
                delta_predicted = try_model(model, src_tensor_normalized)
                detailed_timings['prediction'].append(time.time() - start_time_prediction)

                start_time_opt_pred = time.time()
                mdl_pred = run_optimization(delta_predicted, src_tensor, model.__class__.__name__)
                detailed_timings['optimization2'].append(time.time() - start_time_opt_pred)

                if mdl_pred and mdl_pred.status == gp.GRB.OPTIMAL:
                    if not optimization_found:
                        optimization_found = True
                        cost_pred_mem.append(mdl_pred.ObjVal)
                        successful_optimizations += 1
                        for j in range(i, len(models)):
                            stage_successes[j] += 1
                        break

    # Calculate average costs and the optimality gap
    avg_cost_actual = np.mean(cost_actual_mem) if cost_actual_mem else 0
    avg_cost_pred = np.mean(cost_pred_mem) if cost_pred_mem else 0
    optimality_gap = abs((avg_cost_pred - avg_cost_actual)) / avg_cost_actual if avg_cost_actual != 0 else 0

    # Output results
    for i in range(len(models)):
        success_rate = (stage_successes[i] / instances_processed) * 100
        print(f"Success rate with up to {i+1} model(s): {success_rate:.2f}%")
    print(f"Optimality gap: {optimality_gap:.2%}")
    print(f"Time for lp without prediction: {sum(detailed_timings['optimization2']) / len(detailed_timings['optimization2']):.2f}s")
    print(f"Total unique samples encountered: {len(unique_data_samples)}")



def hybrid_fhocp_milp(x0, N, net_power, cbuy, csell, cprod):
    mdl = gp.Model("hybridMPC")
    mdl.Params.LogToConsole = 0
    
    xmin_tile = np.tile(xmin, (N+1,1))
    xmax_tile = np.tile(xmax, (N+1,1))
    zmin_tile = np.tile(zmin, (N,1))
    zmax_tile = np.tile(zmax, (N,1))
    umin_tile = np.tile(umin, (N,1))
    umax_tile = np.tile(umax, (N,1))

    x = mdl.addMVar(shape=(N+1, n), lb=xmin_tile, ub=xmax_tile, name='x') #1*5= 5
    z = mdl.addMVar(shape=(N, N_z), lb=zmin_tile, ub=zmax_tile, name='z') #2*4= 8
    u = mdl.addMVar(shape=(N, m), lb=umin_tile, ub=umax_tile, name='u') # 5*4 = 20
    delta = mdl.addMVar(shape=(N, N_bin), vtype=gp.GRB.BINARY, name='delta') # 5*4=20, total = 53

    # 1 + 1*4 + 30*4 + 1*4= 129 (number of constraints)
    mdl.addConstr(x[0, :] == x0.reshape(Amld.shape[0],))
    for k in range(N):
        mdl.addConstr(x[k+1, :] == Amld @ x[k, :] + B1 @ u[k, :] + B2 @ delta[k, :] + B3 @ z[k,:] + B5.reshape(B1.shape[0],)) # dynamics
        mdl.addConstr(E2 @ delta[k, :] + E3 @ z[k, :] <= E1 @ u[k,:] + E4 @ x[k,:] + E5.reshape(E1.shape[0],)) # mld constraints
        mdl.addConstr(u[k,0]-u[k,1]-u[k,2]-u[k,3]-u[k,4]+ net_power[k] == 0) # power balance

    obj1 = sum(cbuy[k]*z[k,1] - csell[k]*z[k,1] + csell[k]*u[k,1]  for k in range(N)) # cost for power exchanged with grid
    obj2 = sum(cprod[k]*u[k,2:].sum() for k in range(N)) # cost for energy production by dispatchable generators
    mdl.setObjective(obj1 + obj2, GRB.MINIMIZE)

    mdl.optimize()
    
    return mdl


def timed_hybrid_fhocp_milp(x0, N, net_power, cbuy, csell, cprod):
    start_time = time.time()  # Start timing
    result = hybrid_fhocp_milp(np.array(src_tensor[0,1,4]), config['max_seq_length'], src_tensor[:,:,3].numpy().squeeze(), src_tensor[:,:,0].numpy().squeeze(), src_tensor[:,:,1].numpy().squeeze(), src_tensor[:,:,2].numpy().squeeze())
    objective_value_milp = result.ObjVal
    end_time = time.time()  # End timing
    time_milp = end_time - start_time 

    return result, objective_value_milp, time_milp  

result, objective_value_milp, time_milp = timed_hybrid_fhocp_milp(np.array(src_tensor[0,1,4]), config['max_seq_length'], src_tensor[:,:,3].numpy().squeeze(), src_tensor[:,:,0].numpy().squeeze(), src_tensor[:,:,1].numpy().squeeze(), src_tensor[:,:,2].numpy().squeeze())

print("time for milp", result.Runtime)




def export_model_to_onnx(model, src_tensor, tgt_tensor, model_name):
    model.eval()
    onnx_file_path = f"{model_name}.onnx"

    torch.onnx.export(
        model,
        (src_tensor, tgt_tensor),
        onnx_file_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_src', 'input_tgt'],
        output_names=['output'],
        dynamic_axes={
            'input_src': {0: 'batch_size'},  # Adjust the dynamic axes as necessary
            'input_tgt': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {onnx_file_path}")

    
# Assuming src_batch and tgt_batch are loaded from your DataLoader
src_batch, tgt_batch = next(iter(val_dataloader))  # Get a batch of data
src_tensor_normalized = normalize_batch(src_batch[0:1])  # Normalize and use the first item as an example

# Use a realistic target tensor from your data
real_tgt_tensor = tgt_batch[0:1]  # Use the first target sequence from the batch

# Export your model using actual batch data
export_model_to_onnx(transformer, src_tensor_normalized, real_tgt_tensor, 'transformer_backup.onnx')
onnx_file_path="shallow_model.onnx"

def run_onnx_inference(onnx_path, src_input, tgt_input):
    session = ort.InferenceSession(onnx_path)

    # Convert PyTorch tensors to numpy arrays with appropriate data types
    src_input = src_input.detach().cpu().numpy().astype(np.float32)
    tgt_input = tgt_input.detach().cpu().numpy().astype(np.int64)

    # Start timing
    start_time = time.time()

    # Execute the model
    outputs = session.run(None, {"input_src": src_input, "input_tgt": tgt_input})

    # Stop timing
    elapsed_time = time.time() - start_time
    print(f"ONNX Runtime Inference Time: {elapsed_time:.4f} seconds")

    return outputs[0]

# Ensure src_tensor_normalized and real_tgt_tensor are prepared correctly
output = run_onnx_inference(onnx_file_path, src_tensor_normalized, real_tgt_tensor)
# print("ONNX Runtime Output:", output)

def convert_logits_to_tokens(logits):
    # Assuming logits shape is (batch_size, sequence_length, num_classes)
    # and you want the class (token id) with the highest score per position
    start_time = time.time()
    predicted_tokens = np.argmax(logits, axis=-1)
    elapsed_time = time.time() - start_time
    print(f"ONNX conversion logits Time: {elapsed_time:.4f} seconds")
    return predicted_tokens

# Run your ONNX inference to get logits
logits = run_onnx_inference(onnx_file_path, src_tensor_normalized, real_tgt_tensor)
predicted_sequence_onnx = convert_logits_to_tokens(logits)

predicted_sequence_inference = perform_inference(transformer, src_tensor_normalized)



def cascaded_transformer_onnx(models, val_dataloader, num_instances):
    it = iter(val_dataloader)
    stage_successes = [0] * len(models)
    cost_actual_mem = []
    cost_pred_mem = []
    successful_optimizations = 0
    instances_processed = 0

    while instances_processed < num_instances:
        try:
            src_batch, tgt_batch = next(it)
        except StopIteration:
            print("Ran out of validation data.")
            break

        for idx in range(src_batch.size(0)):
            if instances_processed >= num_instances:
                break
            
            instances_processed += 1
            src_tensor = src_batch[idx:idx + 1]

            src_tensor_normalized = normalize_batch(src_tensor)

            delta_actual = transform_predicted_to_binary_matrix(tgt_batch[idx])
            mdl_act = run_optimization(delta_actual, src_tensor, 'actual')
            if mdl_act and mdl_act.status == gp.GRB.OPTIMAL:
                cost_actual_mem.append(mdl_act.ObjVal)

            optimization_found = False
            for i, model_path in enumerate(models):
                logits = run_onnx_inference(model_path, src_tensor_normalized, tgt_batch[idx:idx+1])
                # detailed_timings['prediction'].append(time.time() - start_time_prediction)
                predicted_sequence = convert_logits_to_tokens(logits).flatten()
                delta_predicted = transform_predicted_to_binary_matrix(predicted_sequence)
 
                mdl_pred = run_optimization(delta_predicted, src_tensor, 'prediction')

                if mdl_pred and mdl_pred.status == gp.GRB.OPTIMAL:
                    if not optimization_found:
                        optimization_found = True
                        cost_pred_mem.append(mdl_pred.ObjVal)
                        successful_optimizations += 1
                        for j in range(i, len(models)):
                            stage_successes[j] += 1
                        break

    avg_cost_actual = np.mean(cost_actual_mem) if cost_actual_mem else 0
    avg_cost_pred = np.mean(cost_pred_mem) if cost_pred_mem else 0
    optimality_gap = abs((avg_cost_pred - avg_cost_actual)) / avg_cost_actual if avg_cost_actual != 0 else 0

    for i in range(len(models)):
        success_rate = (stage_successes[i] / instances_processed) * 100
        print(f"Success rate with up to {i+1} model(s): {success_rate:.2f}%")
    print(f"Optimality gap: {optimality_gap:.2%}")

print("Predicted Tokens from ONNX:", predicted_sequence_onnx)
print("Predicted Tokens from regular inference:", predicted_sequence_inference)





