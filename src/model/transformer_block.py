import torch
import torch.nn as nn

from src.model.multihead_attention import MultiHeadCausalAttention


class TransformerBlock(nn.Module):
    """
    Pre-Norm Transformer block (GPT-2 style).

    Architecture:
        x' = x + Attention(LayerNorm(x))
        y  = x' + FFN(LayerNorm(x'))
    """

    def __init__(self, d_model, num_heads, context_length, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalAttention(
            d_model, num_heads, context_length, dropout
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_weights=False):
        # Pre-Norm: LayerNorm → Attention → Residual
        if return_weights:
            attn_out, weights = self.attn(self.ln1(x), return_weights=True)
        else:
            attn_out = self.attn(self.ln1(x))
            weights = None

        x = x + attn_out

        # Pre-Norm: LayerNorm → FFN → Residual
        x = x + self.ffn(self.ln2(x))

        if return_weights:
            return x, weights
        return x