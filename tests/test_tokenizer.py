"""
Tests for LexiFlow tokenizer wrapper.
"""

import unittest
from src.tokenizer import GPT2Tokenizer


class TestGPT2Tokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = GPT2Tokenizer()

    def test_vocab_size(self):
        self.assertEqual(self.tokenizer.vocab_size, 50257)

    def test_eos_token_id(self):
        self.assertEqual(self.tokenizer.eos_token_id, 50256)

    def test_encode_decode_roundtrip(self):
        text = "Sir, do we need to submit the assignment tomorrow?"
        ids = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(ids)
        self.assertEqual(decoded, text)

    def test_add_eos_flag(self):
        text = "Hello world"
        ids_without_eos = self.tokenizer.encode(text, add_eos=False)
        ids_with_eos = self.tokenizer.encode(text, add_eos=True)

        self.assertEqual(len(ids_with_eos), len(ids_without_eos) + 1)
        self.assertEqual(ids_with_eos[-1], self.tokenizer.eos_token_id)

    def test_special_tokens_in_text(self):
        # GPT-2 tokenizer splits custom prompt tags into subwords without error
        text = "[TASK] assignment_query [TONE] formal [INPUT] sir [OUTPUT]"
        ids = self.tokenizer.encode(text)
        self.assertTrue(len(ids) > 0)
        self.assertEqual(self.tokenizer.decode(ids), text)


if __name__ == "__main__":
    unittest.main()
