import torch
import torch.nn as nn

class MultiHeadCausalAttention(nn.Module):
    def __init__(self,d_model,num_heads,context_length,dropout=0.1):
        super().__init__()

        assert d_model%num_heads==0

        self.num_heads=num_heads
        self.head_dim=d_model//num_heads

        self.W_q=nn.Linear(d_model,d_model,bias=False)
        self.W_k=nn.Linear(d_model,d_model,bias=False)
        self.W_v=nn.Linear(d_model,d_model,bias=False)

        self.out_proj = nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)

        mask=torch.tril(torch.ones(context_length,context_length))
        self.register_buffer("mask",mask)

    def forward(self,x,return_weights=False):

        B,T,C=x.shape

        Q=self.W_q(x)
        K=self.W_k(x)
        V=self.W_v(x)

        Q=Q.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        K=K.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        V=V.view(B,T,self.num_heads,self.head_dim).transpose(1,2)


        scores=Q @K.transpose(-2,-1)
        scores=scores/(self.head_dim**0.5)


        scores=scores.masked_fill(self.mask[:T,:T]==0,float("-inf"))

        weights=torch.softmax(scores,dim=-1)
        weights=self.dropout(weights)


        context=weights@V

        context=context.transpose(1,2).contiguous().view(B,T,C)


        output=self.out_proj(context)


        #inspecting the weights

        if return_weights:
            return output,weights

        return output