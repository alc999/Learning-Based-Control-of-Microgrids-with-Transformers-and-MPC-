#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:34:21 2024

@author: bobvanderwoude
"""

import numpy as np 
import torch 
from microgrid_fun import hybrid_fhocp, gurobi_qp, qp_feasible, count_parameters, state_norm, build_delta, build_stacked_input
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
import time  



if torch.backends.mps.is_available():
    device = torch.device("mps")
    print ("MPS available, current device MPS")
else:
    print ("MPS device not found.")

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


file_path_data = "file_path_microgrid_data.npz"

cbuy_tensor, csell_tensor, cprod_tensor, power_res_tensor, power_load_tensor, x0_tensor, delta_transformed_tensor = load_data_npz(file_path_data)


def build_delta_vector(predicted_sequence):

    """
    Transforms a predicted sequence tensor of shape (1, N) into an (N x 5) binary matrix.
    
    :param predicted_sequence: A tensor of shape (1, N) with integers representing decimal values.
    :return: An (N x 5) matrix of binary values (as a list of lists).
    """
    # Ensure the input is a tensor, flatten it to 1D, and convert to a list of integers
    if torch.is_tensor(predicted_sequence):
        predicted_sequence = predicted_sequence.view(-1).tolist()  # Flatten and convert to list
    
    # Convert each integer to its 5-bit binary representation
    delta = np.array([list(map(int, '{:05b}'.format(num))) for num in predicted_sequence])

    
    return delta



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
    # This function assumes 'batch' is a tensor where the last dimension
    # corresponds to different features, e.g., [batch_size, seq_length, num_features]
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
n_actions = 32


def compute_global_stats(dataset):
    # Concatenate all tensors in the dataset along the batch dimension
    all_data = torch.cat([data[0].unsqueeze(0) for data in dataset], dim=0)
    global_mean = all_data.mean(dim=(0, 1), keepdim=True)
    global_std = all_data.std(dim=(0, 1), keepdim=True)
    return global_mean, global_std

val_iter = iter(val_dataloader)
src_batch, tgt_batch = next(val_iter)

random_idx = torch.randint(0, src_batch.size(0), (1,)).item()
src_tensor = src_batch[random_idx:random_idx+1]

global_mean, global_std = compute_global_stats(val_dataset)

# Normalize a single batch using pre-computed global stats
def normalize_tensor(batch, mean, std):
    normalized_batch = (batch - mean) / (std + 1e-5)
    return normalized_batch

src_tensor_normalized = normalize_tensor(src_tensor, global_mean, global_std)


class Network(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, lr, n_actions, batch_first=True):
        super(Network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers  # Ensure this is correctly set as an instance variable
        self.lr = lr
        self.n_actions = n_actions
        self.batch_first = batch_first
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.dense = nn.Linear(hidden_size, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification problems

    def forward(self, input, h0, c0):
        output, (hn, cn) = self.lstm(input, (h0, c0))
        output = output.contiguous().view(-1, output.shape[2])  # Flatten the output for the dense layer
        output = self.dense(output)
        output = output.view(input.shape[0], input.shape[1], -1)  # Reshape back to (batch_size, sequence_length, n_actions)
        return output

def train_network(network, train_loader, val_loader, num_epochs, device):
    network.to(device)
    best_val_loss = float('inf')
    
    start_time = time.time()  # Start timing

    for epoch in range(num_epochs):
        network.train()
        total_train_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).long()
            h0 = torch.zeros(network.num_layers, inputs.size(0), network.hidden_size).to(device)
            c0 = torch.zeros(network.num_layers, inputs.size(0), network.hidden_size).to(device)

            # Reset the gradients to zero
            network.optimizer.zero_grad()

            outputs = network(inputs, h0, c0)
            outputs = outputs.view(-1, network.n_actions)  # Reshape outputs to (batch_size*sequence_length, n_actions)
            targets = targets.view(-1)  # Flatten targets to match output shape

            loss = network.loss(outputs, targets)
            loss.backward()
            network.optimizer.step()
            
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validate the model
        network.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).long()
                h0 = torch.zeros(network.num_layers, inputs.size(0), network.hidden_size).to(device)
                c0 = torch.zeros(network.num_layers, inputs.size(0), network.hidden_size).to(device)

                outputs = network(inputs, h0, c0)
                outputs = outputs.view(-1, network.n_actions)
                targets = targets.view(-1)

                loss = network.loss(outputs, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        # Save the model if the validation loss is the best we've seen so far.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(network.state_dict(), 'LSTM2_network_model.pth')
            print("Saved Best Model")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")


# Initialize network with appropriate parameters
network = Network(input_size=5, hidden_size=128, num_layers=1, lr=1e-4, n_actions=32)

# Define the number of epochs
num_epochs = 500


def predict_sequences(network, dataloader, device):
    network.eval()  # Set the network to evaluation mode
    predictions = []
    targets_list = []

    with torch.no_grad():  # No need to track gradients during inference
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            h0 = torch.zeros(network.num_layers, inputs.size(0), network.hidden_size).to(device)
            c0 = torch.zeros(network.num_layers, inputs.size(0), network.hidden_size).to(device)

            outputs = network(inputs, h0, c0)
            _, predicted = torch.max(outputs, 2)  # Get the class with the highest probability for each timestep
            predictions.append(predicted.cpu().numpy())
            targets_list.append(targets.numpy())

    return predictions, targets_list




def evaluate_predictions(predictions, targets_list):
    # Flatten the lists
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets_list)

    # Compute accuracy
    accuracy = (predictions == targets).mean()
    return accuracy



def try_model(model, src_tensor_normalized):
    """
    Evaluates an LSTM model on the provided input tensor and returns the predicted sequence.

    :param model: The trained LSTM model.
    :param src_tensor_normalized: A tensor containing normalized input data.
    :return: The predicted output (e.g., a tensor of predicted sequences or classes).
    """
    # Assuming the LSTM model expects an initial hidden and cell state,
    # we initialize them to zeros. 
    h0 = torch.zeros(model.num_layers, src_tensor_normalized.size(0), model.hidden_size).to(model.device)
    c0 = torch.zeros(model.num_layers, src_tensor_normalized.size(0), model.hidden_size).to(model.device)
    
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients
        outputs = model(src_tensor_normalized, h0, c0)
  
        predicted = outputs.argmax(dim=2)  # Get the class index with the highest probability for each time step
        return predicted
    

