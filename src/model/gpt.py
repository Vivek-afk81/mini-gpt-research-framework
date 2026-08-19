import torch
import torch.nn as nn

from src.model.transformer_block import TransformerBlock


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, context_length=128, num_heads=4, num_layers=4):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_length, d_model)

        #stacking multiple transformer blocks
        self.blocks=nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                context_length=context_length,
            )
            for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids, return_weights=False):

        B, T = input_ids.shape

        positions = torch.arange(T, device=input_ids.device)
        pos_embeddings = self.pos_emb(positions)

        token_embeddings = self.token_emb(input_ids)

        x = token_embeddings + pos_embeddings

        # Pass through all transformer blocks
        all_weights = []
        for block in self.blocks:
            if return_weights:
                x, weights = block(x, return_weights=True)
                all_weights.append(weights)  # (B, num_heads, T, T)
            else:
                x = block(x)

        logits = self.lm_head(x)

        if return_weights:
            return logits, all_weights  # all_weights: list of (B, num_heads, T, T)
        return logits