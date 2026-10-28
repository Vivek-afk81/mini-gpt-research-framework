"""
Tests for finetuned pipeline configuration helpers (no model download).
"""

import unittest
from unittest.mock import patch

try:
    import torch
    from src.pipeline import LexiFlowPipeline
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


class _DummyModelWithMap:
    hf_device_map = {"model.embed_tokens": 0}


class _DummyModelNoMap:
    hf_device_map = None

    def __init__(self):
        self._param = torch.nn.Parameter(torch.zeros(1)) if TORCH_AVAILABLE else None

    def parameters(self):
        return iter([self._param] if self._param is not None else [])


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for pipeline configuration tests")
class TestPipelineFinetunedConfig(unittest.TestCase):
    @patch("src.pipeline.torch.cuda.is_available", return_value=False)
    def test_adapter_path_and_cpu_load_kwargs(self, _):
        pipe = LexiFlowPipeline(
            backend="finetuned",
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            adapter_path="checkpoints/custom_adapter",
            device="auto",
        )
        self.assertEqual(pipe._adapter_path, "checkpoints/custom_adapter")
        kwargs = pipe._get_model_load_kwargs()
        self.assertEqual(kwargs["torch_dtype"], torch.float32)
        self.assertNotIn("device_map", kwargs)

    @patch("src.pipeline.torch.cuda.is_available", return_value=True)
    def test_cuda_load_kwargs(self, _):
        pipe = LexiFlowPipeline(device="auto")
        kwargs = pipe._get_model_load_kwargs()
        self.assertEqual(kwargs["torch_dtype"], torch.float16)
        self.assertEqual(kwargs["device_map"], "auto")

    @patch("src.pipeline.torch.cuda.is_available", return_value=True)
    def test_input_device_from_hf_device_map(self, _):
        pipe = LexiFlowPipeline(device="cuda")
        pipe._model = _DummyModelWithMap()
        self.assertEqual(pipe._get_input_device(), "cuda:0")

    @patch("src.pipeline.torch.cuda.is_available", return_value=False)
    def test_input_device_from_model_parameter(self, _):
        pipe = LexiFlowPipeline(device="cpu")
        pipe._model = _DummyModelNoMap()
        self.assertEqual(str(pipe._get_input_device()), "cpu")


if __name__ == "__main__":
    unittest.main()
