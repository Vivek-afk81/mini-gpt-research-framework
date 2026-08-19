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
        self.ffn=nn.Sequential(
            nn.Linear(d_model,4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model,d_model),
            nn.Dropout(dropout)
        )

        self.ln2=nn.LayerNorm(d_model)
    
    def forward(self,x,return_weights=False):
        #x (B,T,d_model)

        if return_weights:
            attn_out,weights=self.attn(x,return_weights=True)
        else:
            attn_out=self.attn(x)
            weights=None

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

        if return_weights:
            return x,weights
        return x