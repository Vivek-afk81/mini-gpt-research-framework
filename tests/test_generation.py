"""
Tests for LexiFlow generation utilities, input validation, and sampling functions.
"""

import unittest
import torch

from src.generation import (
    detect_category,
    validate_input,
    apply_repetition_penalty,
    top_k_top_p_filtering,
)


class TestGeneration(unittest.TestCase):
    def test_detect_category(self):
        # Category detection tests across all categories
        self.assertEqual(detect_category("sir assignment kab dena hai"), "assignment_query")
        self.assertEqual(detect_category("attendance short ho gayi 75% nahi hai"), "attendance_request")
        self.assertEqual(detect_category("sir mujhe do din ki leave chahiye fever hai"), "leave_request")
        self.assertEqual(detect_category("mid sem exam kab start hoga"), "exam_query")
        self.assertEqual(detect_category("sir deadline extend kar do please"), "deadline_extension")
        self.assertEqual(detect_category("lab me ac kaam nahi kar raha broken fan"), "complaint")

    def test_validate_input_valid(self):
        inp = "   sir kal class hai kya?   "
        cleaned = validate_input(inp)
        self.assertEqual(cleaned, "sir kal class hai kya?")

    def test_validate_input_empty(self):
        with self.assertRaises(ValueError):
            validate_input("")

        with self.assertRaises(ValueError):
            validate_input("     ")

        with self.assertRaises(ValueError):
            validate_input(None)

    def test_validate_input_truncation(self):
        long_input = "a" * 600
        cleaned = validate_input(long_input, max_chars=100)
        self.assertEqual(len(cleaned), 100)

    def test_repetition_penalty(self):
        logits = torch.tensor([2.0, 4.0, -2.0, 1.0])
        # Token 1 has positive logit 4.0, penalized by 2.0 -> 2.0
        # Token 2 has negative logit -2.0, penalized by 2.0 -> -4.0
        penalized = apply_repetition_penalty(logits.clone(), generated_tokens=[1, 2], penalty=2.0)
        self.assertAlmostEqual(penalized[1].item(), 2.0)
        self.assertAlmostEqual(penalized[2].item(), -4.0)
        self.assertEqual(penalized[0].item(), 2.0)  # Unpenalized

    def test_top_k_filtering(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        filtered = top_k_top_p_filtering(logits, top_k=2)
        # Only top 2 tokens (indices 3, 4) should remain finite
        self.assertTrue(torch.isneginf(filtered[0, 0]))
        self.assertTrue(torch.isneginf(filtered[0, 1]))
        self.assertTrue(torch.isneginf(filtered[0, 2]))
        self.assertFalse(torch.isneginf(filtered[0, 3]))
        self.assertFalse(torch.isneginf(filtered[0, 4]))


if __name__ == "__main__":
    unittest.main()
