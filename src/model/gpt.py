import torch
import torch.nn as nn

from src.model.transformer_block import TransformerBlock


class GPT(nn.Module):
    """
    GPT-style decoder-only language model.

    Components:
        - Token + positional embeddings
        - N × Pre-Norm Transformer blocks
        - Final LayerNorm
        - Language model head (weight-tied with token embeddings)
    """

    def __init__(self, vocab_size, d_model=128, context_length=128,
                 num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()

        self.context_length = context_length

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_length, d_model)
        self.drop = nn.Dropout(dropout)

        # Stacking transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, context_length, dropout)
            for _ in range(num_layers)
        ])

        # Final LayerNorm (essential for Pre-Norm architecture)
        self.ln_f = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: LM head shares weights with token embeddings
        self.lm_head.weight = self.token_emb.weight

        # GPT-2 style weight initialization
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2*num_layers) for stable training
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0,
                            std=0.02 / (2 * num_layers) ** 0.5)
            nn.init.normal_(block.ffn[2].weight, mean=0.0,
                            std=0.02 / (2 * num_layers) ** 0.5)

    def _init_weights(self, module):
        """GPT-2 weight initialization: N(0, 0.02) for linear/embedding, zeros for biases."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, return_weights=False):
        B, T = input_ids.shape

        positions = torch.arange(T, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.drop(x)

        # Pass through all transformer blocks
        all_weights = []
        for block in self.blocks:
            if return_weights:
                x, weights = block(x, return_weights=True)
                all_weights.append(weights)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if return_weights:
            return logits, all_weights
        return logits