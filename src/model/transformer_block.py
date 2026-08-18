import torch
import torch.nn as nn

from src.model.multihead_attention import MultiHeadCausalAttention

class TransformerBlock(nn.Module):

    def __init__(self,d_model,num_heads,context_length,dropout=0.1):
        super().__init__()

        self.attn=MultiHeadCausalAttention(
            d_model,num_heads,context_length,dropout
        )

        self.ln1=nn.LayerNorm(d_model)

        # feed forward network
        self.ffn=nn.sequential(
            nn.Linear(d_model,4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model,d_model),
            nn.Dropout(dropout)
        )

        self.ln2=nn.LayerNorm(d_model)
    
    def forward(self,x):
        #x (B,T,d_model)

        attn_out=self.attn(x)

        #residual connection

        x=x+attn_out

        #layerNorm
        x=self.ln1(x)

        #feed forward
        ffn_out=self.ffn(x)

        #residual connection
        x=x+ffn_out

        #layerNorm
        x=self.ln2(x)

        return x