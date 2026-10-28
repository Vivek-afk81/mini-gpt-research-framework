"""LexiFlow configuration — single source of truth for all hyperparameters."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Transformer model hyperparameters."""
    vocab_size: int = 50257       # GPT-2 BPE vocabulary size
    d_model: int = 128            # Embedding / hidden dimension
    num_heads: int = 4            # Attention heads (head_dim = d_model / num_heads = 32)
    num_layers: int = 4           # Number of Transformer blocks
    context_length: int = 128     # Maximum sequence length
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training loop hyperparameters."""
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 20
    gradient_clip: float = 1.0
    warmup_steps: int = 100
    eval_interval: int = 1        # Evaluate every N epochs


@dataclass
class DataConfig:
    """Dataset paths and split ratios."""
    raw_path: str = "data/raw/campus_messages.jsonl"
    train_path: str = "data/processed/train.jsonl"
    val_path: str = "data/processed/validation.jsonl"
    test_path: str = "data/processed/test.jsonl"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class GenerationConfig:
    """Text generation hyperparameters."""
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_k: int = 0                # 0 = disabled


@dataclass
class Config:
    """Root configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    # Prompt tags (used as plain text — GPT-2 tokenizer splits them into subwords,
    # but the model learns to recognise the patterns from training data)
    task_tag: str = "[TASK]"
    tone_tag: str = "[TONE]"
    input_tag: str = "[INPUT]"
    output_tag: str = "[OUTPUT]"
