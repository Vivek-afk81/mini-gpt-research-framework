import torch
import torch.nn as nn



class SelfAttentionV1(nn.Module):
    def __init__(self,d_in,d_out):
        super().__init__()

        self.W_q=nn.Linear(d_in,d_out)
        self.W_k=nn.Linear(d_in,d_out)
        self.W_v=nn.Linear(d_in,d_out)
    
    def forward(self,x):
        #x shape --> (batch,T(sequence length),d_in)

        Q=self.W_q(x)
        K=self.W_k(x)
        V=self.W_v(x)


        #computing attention scores
        attn_scores=Q @ K.transpose(-2,-1)   ## (B, T, T)
        d_k=K.shape[-1]
        attn_scores=attn_scores/(d_k**0.5)


        #softmax to get attnn weights
        attn_weights=torch.softmax(attn_scores,dim=-1)

        context=attn_weights @ V  ## (B, T, d_out)

        return context
