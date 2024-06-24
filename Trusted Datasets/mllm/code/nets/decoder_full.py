## IMPORTS
import torch
import torch.nn as nn
import math

from code.nets.pos_enc import PositionalEncoding

## FILE DESCRIPTION
'''
This Python script defines the ESMDecoderFull, which integrates self-attention and cross-attention mechanisms along with an embedding layer and positional encoding to process input sequences. It's designed to generate scalar outputs from the combined sequence and encoder data.

The ESMDecoderFull efficiently manages sequence processing by leveraging self-attention and cross-attention, followed by linear transformations, with support for activation sharding and 8-bit quantization.

Usage:
    encoder_output = torch.rand(10, 20, 480)  # Example encoder output tensor (batch_size, seq_length, embed_size)
    decoder = ESMDecoderFull(vocab_size=10000)
    x = torch.randint(0, 10000, (10, 20))  # Example input tensor (batch_size, seq_length) with integer token IDs
    output = decoder(x, encoder_output)
    print(output)  # Outputs a single number per batch

Requirements:
    - PyTorch library must be installed and properly configured.
'''

## SINGLE LAYER OF COMPUTATION
class ESMDecoderFullLayer(nn.Module):
    """
    ESMDecoderFullLayer integrates both self-attention and cross-attention mechanisms into a single decoder layer.
    
    Processes input through self-attention followed by cross-attention using the encoder output, with linear transformations and normalization steps integrated to enhance the output's stability and quality.
    """
    def __init__(self, embed_size, num_heads, dropout=0.4):
        super(ESMDecoderFullLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads)
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads)
        self.linear = nn.Linear(embed_size, embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output):
        x = x.transpose(0, 1)
        encoder_output = encoder_output.transpose(0, 1)

        # Self-attention
        self_attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self_attn_output)
        x = self.dropout(x)

        # Cross-attention
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output)
        x = self.norm2(x + cross_attn_output)
        x = self.dropout(x)

        # Linear transformation
        linear_output = self.linear(x)
        x = self.norm3(x + linear_output)
        x = self.dropout(x)

        return x.transpose(0, 1)

## ESM DECODER FULL
class ESMDecoderFull(nn.Module):
    """
    ESMDecoderFull: Aggregates multiple ESMDecoderFullLayers to process sequence data in conjunction with encoder outputs,
    producing a scalar output. Incorporates input embeddings and positional encodings.
    
    Integrates multiple decoding layers, each with self-attention and cross-attention capabilities, to refine sequence embeddings
    into a single output value per batch through a final linear transformation.
    """
    def __init__(self, vocab_size, num_layers=8, embed_size=1280, num_heads=64, dropout=0.4):
        super(ESMDecoderFull, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size)
        self.layers = nn.ModuleList([ESMDecoderFullLayer(embed_size, num_heads, dropout) for _ in range(num_layers)])
        self.output_layer = nn.Linear(embed_size, 1)

    def forward(self, x, encoder_output):
        # shape of x (inputs): (batch_size, seq_len) -> (2, ..)
        # shape of encoder_output: (batch_size, seq_len, embed_size) -> (2, .., 1280)

        # Embedding and positional encoding
        x = self.embedding(x)
        x = self.positional_encoding(x)

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output)

        # Final linear transformation to output
        x = self.output_layer(x.mean(dim=1))
        return x
