import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):

    def __init__(self,d_in,d_out,context_length,dropout=0.1):
        super().__init__()

        self.W_q =nn.Linear(d_in,d_out,bias=False)
        self.W_k =nn.Linear(d_in,d_out,bias=False)
        self.W_v =nn.Linear(d_in,d_out,bias=False)

        self.dropout=nn.Dropout(dropout)

        mask=torch.tril(torch.ones(context_length,context_length))
        self.register_buffer("mask",mask)

    def forward(self,x):

        B,T,C=x.shape
        Q=self.W_q(x)
        K=self.W_k(x)
        V=self.W_v(x)

        scores=Q @ K.transpose(-2,-1)
        scores=scores/(K.shape[-1]**0.5)

        scores=scores.masked_fill(self.mask[:T,:T]==0,float("-inf"))
        weights=torch.softmax(scores,dim=-1)
        weights=self.dropout(weights)

        context=weights @ V

        return context
