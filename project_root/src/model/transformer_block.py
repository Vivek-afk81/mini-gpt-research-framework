import torch
import torch.nn as nn

from src.model.multihead_attention import MultiHeadCausalAttention

class TransformerBlock(nn.Module):

    def __init__(self,d_model,num_heads,context_length,dropout=0.1):
        super().__init__()

        self.attn=MultiHeadCausalAttention(
            d_model,num_heads,context_length,dropout
        )

        self.ln=nn.LayerNorm(d_model)
    
    def forward(self,x):
        #x (B,T,d_model)

        attn_out=self.attn(x)

        #residual connection

        x=x+attn_out

        #layerNorm
        x=self.ln(x)

        return x