## IMPORTS
import torch
import torch.nn as nn

## FILE DESCRIPTION
'''
This Python script defines the ESMDecoderMLP, a multi-layer perceptron model that processes input sequences to generate scalar outputs.

The ESMDecoderMLP takes pre-embedded input sequences of fixed dimensions and applies a stack of linear layers with non-linear activations 
to refine the sequence data into scalar outputs.

Usage:
    decoder = ESMDecoderMLP()
    x = torch.randn((2, 30, 1280))  # Example input tensor (batch_size, seq_len, embed_size) with random floating-point embeddings
    output = decoder(x)
    print(output)  # Outputs a single scalar value per batch

Requirements:
    - PyTorch library must be installed and properly configured.
'''

## MULTI-LAYER PERCEPTRON MODEL
class ESMDecoderMLP(nn.Module):
    """
    ESMDecoderMLP: A multi-layer perceptron model that processes input sequences to generate scalar outputs.

    This model takes input sequences with pre-embedded tokens and applies a series of linear layers 
    with non-linear activations and dropout to refine the embeddings into a single scalar value per batch.
    """
    def __init__(self, embed_size=1280, num_layers=4, hidden_size=256, dropout=0.4):
        super(ESMDecoderMLP, self).__init__()

        # Constructing the multi-layer perceptron (MLP)
        layers = [nn.Linear(embed_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(1, num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_size, 1))

        # Creating a sequential stack of the layers
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the ESMDecoderMLP model.

        Arguments:
        - x: input tensor of shape (batch_size, seq_len, embed_size)

        Returns:
        - A tensor of shape (batch_size, 1) representing the scalar outputs per batch.
        """

        # Perform mean pooling over the sequence length dimension to obtain a single embedding per batch
        x = x.mean(dim=1)

        # Apply the MLP to generate the final scalar outputs
        return self.mlp(x)
