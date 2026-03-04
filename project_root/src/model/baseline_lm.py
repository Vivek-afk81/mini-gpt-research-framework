import torch
import torch.nn as nn

class BaselineLM(nn.Module):
    def __init__(self,vocab_size,d_model=128,context_length=64):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size,d_model)
        self.pos_emb=nn.Embedding(context_length,d_model)

        self.lm_head=nn.Linear(d_model,vocab_size,bias=False)

        #weight tying
        self.lm_head_weight=self.token_emb.weight

    def forward(self, input_ids):
        B, T = input_ids.shape

        positions = torch.arange(T, device=input_ids.device)
        pos_embeddings = self.pos_emb(positions)

        token_embeddings = self.token_emb(input_ids)
        x = token_embeddings + pos_embeddings
        logits = self.lm_head(x)

        return logits