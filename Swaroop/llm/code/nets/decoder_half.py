## IMPORTS
import torch
import torch.nn as nn

## FILE DESCRIPTION
'''
This Python script defines the ESMDecoderHalf, which uses self-attention layers to process input sequences and generate scalar outputs.

The ESMDecoderHalf integrates multiple self-attention layers to refine sequence data into scalar outputs. It's designed for
processing sequences that are already embedded, bypassing the need for an initial embedding layer.

Usage:
    decoder = ESMDecoderHalf(embed_size=1280, num_heads=64)
    x = torch.randn((2, 30, 1280))  # Example input tensor (batch_size, seq_len, embed_size) with pre-embedded data
    output = decoder(x)
    print(output)  # Outputs a single scalar value per batch

Requirements:
    - PyTorch library must be installed and properly configured.
'''

## SINGLE LAYER OF COMPUTATION
class ESMDecoderHalfLayer(nn.Module):
    """
    ESMDecoderHalfLayer: Incorporates self-attention into a single computation layer.
    
    Processes input through self-attention, integrating normalization and dropout after each step.
    The input to this layer must be transposed to shape (seq_len, batch_size, embed_size) to match the
    requirements of nn.MultiheadAttention.
    """
    def __init__(self, embed_size, num_heads, dropout=0.4):
        super(ESMDecoderHalfLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads)
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Transpose x to (seq_len, batch_size, embed_size) for attention
        x = x.transpose(0, 1)
        attn_output, _ = self.attention(x, x, x)

        # Apply residual connection, layer normalization and dropout
        x = x + attn_output
        x = self.norm(x)
        x = self.dropout(x)
        # Transpose x back to (batch_size, seq_len, embed_size)
        return x.transpose(0, 1)

## ESM DECODER HALF
class ESMDecoderHalf(nn.Module):
    """
    ESMDecoderHalf: Aggregates multiple ESMDecoderHalfLayers to process input sequences, producing a scalar output.
    
    Each layer applies self-attention to the input sequences, refining the data through multiple steps of attention
    and nonlinear transformations. The model outputs a single scalar value per batch after a final linear transformation.
    """
    def __init__(self, embed_size, num_layers=8, num_heads=64, dropout=0.4):
        super(ESMDecoderHalf, self).__init__()
        self.layers = nn.ModuleList([ESMDecoderHalfLayer(embed_size, num_heads, dropout) for _ in range(num_layers)])
        self.output_layer = nn.Linear(embed_size, 1)

    def forward(self, x):
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x)

        # Mean pooling across the sequence length and final linear transformation
        x = self.output_layer(x.mean(dim=1))
        return x
