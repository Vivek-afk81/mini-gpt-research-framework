"""
LexiFlow generation module.

Handles:
- Keyword-based category auto-detection
- Input validation and length truncation
- Autoregressive generation with temperature, top-k, top-p, and repetition penalty
- Repetition suppression and empty-output prevention
- Safe prompt formatting and decoding
"""

import os
import re
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn.functional as F

from src.config import Config
from src.tokenizer import GPT2Tokenizer
from src.model.gpt import GPT


# Keyword mappings for category auto-detection
CATEGORY_KEYWORDS = {
    "assignment_query": [
        "assignment", "assign", "homework", "hw", "submit", "submission",
        "portal", "topic", "project", "presentation", "report", "handwritten", "typed"
    ],
    "attendance_request": [
        "attendance", "short", "shortage", "percent", "percentage",
        "75%", "seventy five", "present", "absent", "medical certificate", "short ho gayi"
    ],
    "leave_request": [
        "leave", "chhutti", "chutti", "fever", "sick", "ill", "hospital",
        "urgent work", "family function", "marriage", "home", "out of station"
    ],
    "exam_query": [
        "exam", "paper", "test", "mid sem", "midsem", "end sem", "endsem",
        "room", "date sheet", "schedule", "timing", "hall ticket", "admit card"
    ],
    "deadline_extension": [
        "extension", "extend", "extra time", "delay", "deadline", "late",
        "time chahiye", "due date", "unable to finish", "emergency"
    ],
    "complaint": [
        "complaint", "ac", "fan", "lab", "water", "wifi", "internet",
        "mess", "food", "broken", "light", "not working", "kaam nahi", "kharab"
    ],
}

DEFAULT_FALLBACKS = {
    "assignment_query": "Sir, could you please clarify the submission details for the assignment?",
    "attendance_request": "Sir, I would like to request clarification regarding my attendance record.",
    "leave_request": "Sir, I kindly request leave of absence due to unforeseen personal circumstances.",
    "exam_query": "Sir, could you please provide information regarding the examination schedule?",
    "deadline_extension": "Sir, I respectfully request an extension on the assignment deadline.",
    "complaint": "Sir, I would like to bring an issue to your attention for resolution.",
}


def detect_category(text: str) -> str:
    """
    Auto-detect the message category based on Hinglish/English keywords.
    Defaults to 'assignment_query' if no specific category match is found.
    """
    text_lower = text.lower()
    scores: Dict[str, int] = {cat: 0 for cat in CATEGORY_KEYWORDS}

    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower) or kw in text_lower:
                scores[cat] += 1

    best_cat = max(scores, key=lambda k: scores[k])
    if scores[best_cat] > 0:
        return best_cat
    return "assignment_query"


def validate_input(text: Optional[str], max_chars: int = 400) -> str:
    """
    Validate and clean input text.

    Raises:
        ValueError: If input is empty, whitespace only, or invalid.
    """
    if text is None:
        raise ValueError("Input text cannot be None.")

    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Input text is empty. Please enter a valid message.")

    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].strip()

    return cleaned


def apply_repetition_penalty(logits: torch.Tensor, generated_tokens: list, penalty: float = 1.2):
    """
    Penalize previously generated tokens to prevent repetition loops.
    """
    if penalty == 1.0 or not generated_tokens:
        return logits

    unique_tokens = set(generated_tokens)
    for token_id in unique_tokens:
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits


def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    """
    if top_k > 0:
        top_k = min(max(top_k, 1), logits.size(-1))
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")

    return logits


class TextGenerator:
    """
    Inference generator for the custom GPT model.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_model.pt",
        config: Optional[Config] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or Config()
        self.tokenizer = GPT2Tokenizer()
        self.checkpoint_path = checkpoint_path
        self.model = None

        self._load_model()

    def _load_model(self):
        """Load model weights from checkpoint."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found at '{self.checkpoint_path}'. "
                f"Please run training first (e.g., 'python scripts/train.py')."
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        cfg = checkpoint.get("config", self.config)
        self.config = cfg

        self.model = GPT(
            vocab_size=self.tokenizer.vocab_size,
            d_model=cfg.model.d_model,
            context_length=cfg.model.context_length,
            num_heads=cfg.model.num_heads,
            num_layers=cfg.model.num_layers,
            dropout=0.0,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def generate(
        self,
        input_text: str,
        category: Optional[str] = None,
        tone: str = "formal",
        max_new_tokens: int = 48,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.25,
        min_tokens: int = 3,
    ) -> Dict[str, Any]:
        """
        Generate formal English output given a Hinglish input string.

        Returns:
            dict containing:
                - 'input': cleaned input text
                - 'category': detected or specified category
                - 'tone': tone used
                - 'prompt': full formatted prompt
                - 'output': generated formal response
                - 'num_tokens': token count of output
        """
        cleaned_input = validate_input(input_text)
        cat = category if category else detect_category(cleaned_input)

        # Build prompt: [TASK] <category> [TONE] <tone> [INPUT] <text> [OUTPUT]
        prompt = (
            f"{self.config.task_tag} {cat} "
            f"{self.config.tone_tag} {tone} "
            f"{self.config.input_tag} {cleaned_input} "
            f"{self.config.output_tag}"
        )

        prompt_ids = self.tokenizer.encode(prompt)
        max_ctx = self.config.model.context_length

        # If prompt is too long for context window, truncate input part
        if len(prompt_ids) >= max_ctx - max_new_tokens:
            allowed_prompt_len = max_ctx - max_new_tokens - 5
            prompt_ids = prompt_ids[-allowed_prompt_len:]

        generated_ids = []
        stop_tags = [
            self.config.task_tag,
            self.config.input_tag,
            self.config.tone_tag,
            self.config.output_tag,
        ]

        with torch.no_grad():
            for step in range(max_new_tokens):
                current_seq = prompt_ids + generated_ids
                input_tensor = torch.tensor(
                    [current_seq[-max_ctx:]], dtype=torch.long, device=self.device
                )

                logits = self.model(input_tensor)
                next_token_logits = logits[0, -1, :].clone()

                # Mask out EOS before min_tokens to prevent empty outputs
                if step < min_tokens:
                    next_token_logits[self.tokenizer.eos_token_id] = -float("Inf")

                # Apply repetition penalty
                next_token_logits = apply_repetition_penalty(
                    next_token_logits, generated_ids, penalty=repetition_penalty
                )

                if temperature <= 1e-4:
                    next_token = torch.argmax(next_token_logits).item()
                else:
                    scaled_logits = next_token_logits / temperature
                    filtered_logits = top_k_top_p_filtering(
                        scaled_logits, top_k=top_k, top_p=top_p
                    )
                    probs = F.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()

                if next_token == self.tokenizer.eos_token_id:
                    break

                generated_ids.append(next_token)

                # Check if generated text contains any stop tag or delimiter
                partial_text = self.tokenizer.decode(generated_ids)
                if any(tag in partial_text for tag in stop_tags):
                    for tag in stop_tags:
                        if tag in partial_text:
                            partial_text = partial_text.split(tag)[0]
                    # Update generated_ids based on truncated text
                    generated_ids = self.tokenizer.encode(partial_text)
                    break

        output_text = self.tokenizer.decode(generated_ids).strip()

        # Clean punctuation and tags
        for tag in stop_tags:
            output_text = output_text.replace(tag, "").strip()

        # If model generated empty output despite guards, use safe category fallback
        if not output_text:
            output_text = DEFAULT_FALLBACKS.get(
                cat, "Sir, could you please provide guidance on this matter?"
            )

        return {
            "input": cleaned_input,
            "category": cat,
            "tone": tone,
            "prompt": prompt,
            "output": output_text,
            "num_tokens": len(generated_ids),
        }
