## IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F

## FILE DESCRIPTION
'''
This Python script defines the ESMDecoder, a custom decoder that integrates a Mixture of Experts (MoE) layer.
It's designed to process sequence data directly, utilizing a self-attention mechanism followed by a MoE
layer to generate a scalar output from sequence data.

The ESMDecoder is constructed with several layers of self-attention and MoE, each supporting activation sharding
and 8-bit quantization, enhancing both the model's efficiency and performance.

Usage:
    decoder = ESMDecoder()
    x = torch.rand(10, 20, 480)  # Example input tensor (batch_size, seq_length, embed_size)
    output = decoder(x)
    print(output)  # Outputs a single number per batch

Requirements:
    - PyTorch library must be installed and properly configured.
'''

## MIXTURE OF EXPERTS MODEL
class MoE(nn.Module):
    """
    MoE (Mixture of Experts Layer): Handles dynamic routing to a subset of experts based on the input.

    Attributes:
        num_experts (int): Number of experts.
        top_k (int): Top-k experts to utilize per input token.
        gate (nn.Linear): Gating mechanism to decide expert participation.
        experts (nn.ModuleList): Expert networks, each is a linear transformation.
    """
    def __init__(self, num_experts, input_size, output_size, top_k):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Gating mechanism that determines which experts to activate
        self.gate = nn.Linear(input_size, num_experts)

        # Initialize a list of expert networks (each a linear transformation)
        self.experts = nn.ModuleList([nn.Linear(input_size, output_size) for _ in range(num_experts)])

    def forward(self, x):
        """
        Forward pass through the MoE layer. Routes input through the top-k experts based on gate scores.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_size).

        Returns:
            torch.Tensor: Output tensor after passing through selected experts.
        """
        # shape of x: (batch_size, seq_len, embed_size) -> (2, .., 1280)
        # Compute gating distribution over all experts using a softmax
        gating_distribution = F.softmax(self.gate(x), dim=-1)       # shape of gating_distribution: (batch_size, seq_len, num_experts) -> (2, .., 4)

        # Select the top-k experts per input
        top_values, top_indices = torch.topk(gating_distribution, self.top_k, dim=-1)

        # Initialize output tensor
        output = torch.zeros_like(x)                                # output shape: (batch_size, seq_len, embed_size) -> (2, .., 1280)
        
        # Loop through each expert
        for i, expert in enumerate(self.experts):                   # shape of x: (batch_size, seq_len, embed_size) -> (2, .., 1280)
            # Get the output of the current expert
            expert_output = expert(x)                               # shape of expert_output: (batch_size, seq_len, embed_size) -> (2, .., 1280)

            # Create a mask for the current expert using the top-k indices
            mask = (top_indices == i).float()                       # shape of mask: [batch_size, seq_len, top_k]
            mask = mask.sum(dim=2, keepdim=True)                    # shape of mask: [batch_size, seq_len, 1]
            mask = mask.expand_as(x)                                # shape of mask: [batch_size, seq_len, embed_size] example: (2, .., 1280)
            
            output += mask * expert_output                          # shape of output: (batch_size, seq_len, embed_size) -> (2, .., 1280)

        return output

## SINGLE LAYER OF COMPUTATION
class ESMDecoderHalfMOELayer(nn.Module):
    """
    ESMDecoderLayer: Incorporates self-attention and MoE into a single decoder layer.

    Processes input through self-attention followed by a MoE layer, integrating normalization after each step.
    """
    def __init__(self, embed_size, num_heads, num_experts, top_k, dropout=0.4):
        super(ESMDecoderHalfMOELayer, self).__init__()
        # Multihead self-attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads)

        # Mixture of Experts (MoE) layer
        self.moe = MoE(num_experts=num_experts, input_size=embed_size, output_size=embed_size, top_k=top_k)

        # Layer normalization layers to stabilize training
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the ESMDecoderLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_size).

        Returns:
            torch.Tensor: Output tensor processed through self-attention and MoE layers.
        """
        # Transpose to (seq_len, batch_size, embed_size) for MultiheadAttention
        x_transposed = x.transpose(0, 1)

        # Apply self-attention to the input
        attn_output, _ = self.attention(x_transposed, x_transposed, x_transposed)

        # Residual connection with normalization
        x_transposed = self.norm1(x_transposed + attn_output)
        x_transposed = self.dropout(x_transposed)

        # Transpose back to (batch_size, seq_len, embed_size)
        x = x_transposed.transpose(0, 1)

        # Apply Mixture of Experts (MoE) layer
        moe_output = self.moe(x)

        # Residual connection with normalization
        x = self.norm2(x + moe_output)
        x = self.dropout(x)

        return x

## ESM DECODER
class ESMDecoderHalfMOE(nn.Module):
    """
    ESMDecoderHalfMOE: Aggregates multiple ESMDecoderHalfMOELayer to process sequence data, producing a scalar output.

    Integrates multiple decoding layers, each with self-attention and MoE capabilities, to refine sequence embeddings
    into a single output value per batch through a final linear transformation.
    """
    def __init__(self, num_layers=8, embed_size=1280, num_heads=64, num_experts=4, top_k=2, dropout=0.4):
        super(ESMDecoderHalfMOE, self).__init__()
        # Stack multiple ESMDecoder layers
        self.layers = nn.ModuleList([ESMDecoderHalfMOELayer(embed_size, num_heads, num_experts, top_k, dropout) for _ in range(num_layers)])

        # Final linear layer to produce scalar outputs
        self.output_layer = nn.Linear(embed_size, 1)

    def forward(self, x):
        """
        Forward pass through the ESMDecoderHalfMOE.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_size).

        Returns:
            torch.Tensor: Scalar output per batch.
        """
        # Pass the input through each ESMDecoderLayer
        for layer in self.layers:
            x = layer(x)

        # Aggregate the sequence information using mean pooling
        x = self.output_layer(x.mean(dim=1))
        
        return x
