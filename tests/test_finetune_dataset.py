"""
Tests for QLoRA dataset preparation formatting/validation.
"""

import tempfile
import unittest
from pathlib import Path

from src.finetune.dataset import (
    format_for_instruction_tuning,
    normalize_category,
    prepare_finetuning_data,
    validate_chat_example,
)


class TestFinetuneDataset(unittest.TestCase):
    def test_normalize_category_aliases(self):
        self.assertEqual(normalize_category("examination_query"), "exam_query")
        self.assertEqual(normalize_category("complaint_request"), "complaint")
        self.assertEqual(normalize_category("assignment_query"), "assignment_query")

    def test_format_for_instruction_tuning_roles(self):
        rows = [{
            "category": "examination_query",
            "input": "sir exam kab hai",
            "output": "Sir, when is the exam scheduled?",
        }]
        formatted = format_for_instruction_tuning(rows)
        self.assertEqual(len(formatted), 1)
        msgs = formatted[0]["messages"]
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant"])
        self.assertIn("exam query", msgs[0]["content"].lower())

    def test_validate_chat_example_rejects_missing_assistant(self):
        with self.assertRaises(ValueError):
            validate_chat_example(
                {"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]},
                index=0,
            )

    def test_prepare_finetuning_data_writes_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_rows = prepare_finetuning_data(
                raw_path="data/raw/campus_messages.jsonl",
                output_dir=tmp_dir,
                include_qa=False,
            )
            self.assertGreater(len(output_rows), 0)
            self.assertTrue((Path(tmp_dir) / "train_chat.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
