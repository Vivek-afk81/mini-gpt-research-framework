# ProjectX

A GPT-style language model built from scratch in PyTorch — implementing core transformer components from first principles, including multi-head causal self-attention, residual connections, and weight-tied decoding.

## Overview

This project is a **from-scratch implementation** of a decoder-only transformer language model, following the architecture described in [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) and [GPT-1 (Radford et al., 2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf).

Every component — tokenization pipeline, dataset construction, attention mechanism, transformer block, and training loop — is implemented manually to provide full visibility into how autoregressive language models work under the hood.

### Key Implementation Details

- **Causal self-attention** with explicit lower-triangular masking (`torch.tril`)
- **Multi-head attention** with head splitting, parallel attention computation, and output projection
- **Transformer block** with residual connections and LayerNorm
- **Weight tying** between the token embedding and the language model head (as in GPT-2)
- **GPT-2 BPE tokenizer** via `tiktoken` for subword tokenization
- **Sliding-window dataset** with configurable sequence length and stride
- **End-to-end training loop** with cross-entropy loss and AdamW optimizer

## Project Structure

```
ProjectX/
├── main.py                         # Training entry point
├── data/
│   └── The Call of the Wild.txt    # Training corpus (Jack London, public domain)
├── src/
│   ├── tokenizer.py                # GPT-2 BPE tokenizer wrapper
│   ├── dataset.py                  # Sliding-window dataset and text loading
│   └── model/
│       ├── attention.py            # Single-head causal self-attention
│       ├── multihead_attention.py  # Multi-head causal self-attention
│       ├── transformer_block.py    # Attention + residual + LayerNorm
│       └── gpt.py                  # Full language model (embeddings → transformer → LM head)
└── notebooks/
    └── data_preprocessing_pipeline_llm.ipynb
```

## Requirements

- Python 3.8+
- PyTorch
- tiktoken

## Usage

```bash
python main.py
```

This tokenizes the training corpus, constructs next-token-prediction sequences, and trains the model for 3 epochs, printing the average cross-entropy loss per epoch.

## References

- Vaswani, A., et al. (2017). *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Radford, A., et al. (2018). *Improving Language Understanding by Generative Pre-Training.* OpenAI.
