## IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F
from code.nets.pos_enc import PositionalEncoding

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
class ESMDecoderFullMOELayer(nn.Module):
    """
    ESMDecoderFullMOELayer: Incorporates self-attention, cross-attention, and MoE into a single decoder layer.
    """
    def __init__(self, embed_size, num_heads, num_experts, top_k, dropout=0.4):
        super(ESMDecoderFullMOELayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads)
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads)
        self.moe = MoE(num_experts=num_experts, input_size=embed_size, output_size=embed_size, top_k=top_k)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output):
        x = x.transpose(0, 1)
        encoder_output = encoder_output.transpose(0, 1)

        self_attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self_attn_output)
        x = self.dropout(x)

        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output)
        x = self.norm2(x + cross_attn_output)
        x = self.dropout(x)

        x = x.transpose(0, 1)
        moe_output = self.moe(x)
        x = self.norm3(x + moe_output)
        x = self.dropout(x)

        return x

## ESM DECODER FULL MOE
class ESMDecoderFullMOE(nn.Module):
    """
    ESMDecoderFullMOE: Aggregates multiple ESMDecoderFullMOELayers to process input sequences and encoder outputs, producing a scalar output.
    
    Integrates multiple decoding layers, each with self-attention, cross-attention, and MoE capabilities, to refine sequence embeddings
    into a single output value per batch through a final linear transformation.
    """
    def __init__(self, vocab_size, num_layers=8, embed_size=1280, num_heads=64, num_experts=4, top_k=2, dropout=0.4):
        super(ESMDecoderFullMOE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size)
        self.layers = nn.ModuleList([ESMDecoderFullMOELayer(embed_size, num_heads, num_experts, top_k, dropout) for _ in range(num_layers)])
        self.output_layer = nn.Linear(embed_size, 1)

    def forward(self, x, encoder_output):
        # Embedding and positional encoding
        x = self.embedding(x)
        x = self.positional_encoding(x)

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output)

        # Final linear transformation to output
        x = self.output_layer(x.mean(dim=1))
        return x
