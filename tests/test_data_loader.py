"""
Tests for LexiFlow data loading, prompt formatting, and label masking.
"""

import json
import tempfile
import unittest
import torch

from src.config import Config
from src.tokenizer import GPT2Tokenizer
from src.data_loader import (
    format_prompt,
    format_prompt_prefix,
    LexiFlowDataset,
)


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.tokenizer = GPT2Tokenizer()

        self.sample_example = {
            "id": 1,
            "category": "assignment_query",
            "input": "sir kal assignment submit karna hai kya",
            "output": "Sir, do we need to submit the assignment tomorrow?",
            "tone": "formal",
        }

    def test_format_prompt(self):
        prompt = format_prompt(self.sample_example, self.config)
        self.assertIn("[TASK] assignment_query", prompt)
        self.assertIn("[TONE] formal", prompt)
        self.assertIn("[INPUT] sir kal assignment submit karna hai kya", prompt)
        self.assertIn("[OUTPUT] Sir, do we need to submit the assignment tomorrow?", prompt)

    def test_format_prompt_prefix(self):
        prefix = format_prompt_prefix(self.sample_example, self.config)
        self.assertTrue(prefix.endswith("[OUTPUT]"))
        self.assertNotIn(self.sample_example["output"], prefix)

    def test_dataset_label_masking(self):
        # Create a temp jsonl file
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            f.write(json.dumps(self.sample_example) + "\n")
            temp_path = f.name

        dataset = LexiFlowDataset(temp_path, self.tokenizer, self.config)
        self.assertEqual(len(dataset), 1)

        input_ids, labels = dataset[0]

        # Verify shapes match context_length
        self.assertEqual(input_ids.shape, (self.config.model.context_length,))
        self.assertEqual(labels.shape, (self.config.model.context_length,))

        # Verify label masking: early tokens must be -100
        prefix_len = len(self.tokenizer.encode(format_prompt_prefix(self.sample_example, self.config)))
        for i in range(prefix_len - 1):
            self.assertEqual(labels[i].item(), -100, f"Token at {i} should be masked with -100")

        # Verify that output tokens are NOT masked with -100
        unmasked = [val.item() for val in labels if val.item() != -100]
        self.assertTrue(len(unmasked) > 0, "Target tokens should not be masked")

        # Verify padding at the end is masked with -100
        self.assertEqual(labels[-1].item(), -100)


if __name__ == "__main__":
    unittest.main()
