"""
Tests for LexiFlow model architecture and components.
"""

import unittest
import torch

from src.model.gpt import GPT
from src.model.transformer_block import TransformerBlock
from src.model.multihead_attention import MultiHeadCausalAttention


class TestGPTModel(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 1000
        self.d_model = 64
        self.context_length = 32
        self.num_heads = 4
        self.num_layers = 2

        self.model = GPT(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            context_length=self.context_length,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=0.1,
        )

    def test_forward_pass_shape(self):
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))

        logits = self.model(input_ids)
        self.assertEqual(logits.shape, (batch_size, seq_len, self.vocab_size))

    def test_weight_tying(self):
        # Weight tying: lm_head and token_emb must share identical tensor storage
        self.assertIs(self.model.lm_head.weight, self.model.token_emb.weight)

    def test_pre_norm_structure(self):
        # Each block should have 2 LayerNorms (one before MHA, one before FFN)
        for block in self.model.blocks:
            self.assertTrue(hasattr(block, "ln1"))
            self.assertTrue(hasattr(block, "ln2"))

        # Final LayerNorm before LM head
        self.assertTrue(hasattr(self.model, "ln_f"))

    def test_causal_attention_masking(self):
        block = TransformerBlock(self.d_model, self.num_heads, self.context_length)
        x = torch.randn(1, 10, self.d_model)

        out, weights = block(x, return_weights=True)
        self.assertEqual(out.shape, x.shape)
        # Check causal mask: weights for future positions must be 0
        upper_tri = torch.triu(weights[0, 0], diagonal=1)
        self.assertTrue(torch.all(upper_tri == 0))


if __name__ == "__main__":
    unittest.main()
