import argparse
import os
import random

import torch
import numpy as np

# Set random seed
os.environ['PYTHONHASHSEED'] = str(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Get data
data = utils.get_anscombe_dataset()

# Set hyperparameters
learning_rate = 1e-3
epochs = 10
batch_size = 2

num_layers = 2
max_seq_len = 11
input_dim = 2
hidden_dim = 10

generator = Generator()
discriminator = Discriminator()

optim_G = torch.optim.Adam(model.parameters(), lr=learning_rate)
optim_D = torch.optim.SGD(model.parameters(), lr=learning_rate)

# train model
for epoch in range(epochs):
    iterations = int(len(data)/batch_size)
    for i in trange(iterations):
        model.zero_grad()
        # TODO: Get mini batch
        
        
        X_hat = generator(batch_Z) 
        pred = discriminator(X_hat)

        loss.backward()
        opt.step()

    print(f"\nEpoch: {epoch}\tLoss: {loss}")

if __name__ == "__main__":
    
    # Inputs for the main function
    parser = argparse.ArgumentParser()

    # Global Arguments
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cuda',
        type=str)
    parser.add_argument(
        '--exp',
        default='test',
        type=str)