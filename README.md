# LexiFlow

> A Transformer-Based Assistant for Formalizing Code-Mixed Student Communication

LexiFlow converts informal Romanized Hinglish messages into clear, polite, formal English — purpose-built for student-to-professor and student-to-administration communication.

```
Input:  sir kal assignment submit karna hai kya
Output: Sir, do we need to submit the assignment tomorrow?
```

---

## Problem

Many Indian college students communicate using code-mixed language — a blend of Romanized Hindi and English:

```
maam mujhe topic samajh nahi aaya
sir ghar me emergency ki wajah se assignment complete nahi hua
```

This is natural in informal settings, but not always appropriate for emails to professors, formal requests, leave applications, or administrative complaints. Existing translation tools fail to preserve student intent, politeness, or campus-specific context.

LexiFlow addresses this by generating formal English versions that preserve the original meaning.

## Architecture

LexiFlow uses a **decoder-only causal Transformer** (GPT-style), trained from scratch on a domain-specific Hinglish → Formal English dataset.

```
Tokenizer:        GPT-2 BPE (tiktoken)
Embedding dim:    128
Transformer blocks: 4
Attention heads:  4
Head dimension:   32
FFN dimension:    512
Context length:   128
Norm style:       Pre-Norm (GPT-2)
Weight tying:     Token embedding ↔ LM head
```

### Model Pipeline

```
Input text
    ↓
[TASK] + [TONE] + [INPUT] tagging
    ↓
GPT-2 BPE tokenization
    ↓
Token embeddings + Positional embeddings
    ↓
N × Transformer blocks (Pre-Norm + MHA + FFN + Residual)
    ↓
Final LayerNorm → LM Head
    ↓
Autoregressive generation (greedy / temperature sampling)
    ↓
Formal English output
```

## Supported Message Categories

| # | Category | Example Input |
|---|---|---|
| 1 | Assignment query | `sir assignment kab submit karna hai` |
| 2 | Attendance request | `sir meri attendance short ho gayi hai` |
| 3 | Leave / absence request | `maam mujhe do din ki leave chahiye` |
| 4 | Examination query | `sir kal exam kis room me hoga` |
| 5 | Deadline extension | `sir emergency ki wajah se assignment complete nahi hua` |
| 6 | Complaint / admin request | `sir lab me AC kaam nahi kar raha` |

## Project Structure

```
LexiFlow/
├── README.md
├── requirements.txt
├── train.py                        # Training entry point
├── generate.py                     # Text generation CLI
├── evaluate.py                     # Evaluation pipeline
├── app.py                          # Streamlit demo interface
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Centralized hyperparameters
│   ├── tokenizer.py                # GPT-2 BPE wrapper + special tokens
│   ├── data_loader.py              # JSONL dataset with label masking
│   ├── training.py                 # Training loop utilities
│   ├── generation.py               # Autoregressive decoding
│   ├── evaluation.py               # BLEU, perplexity, intent metrics
│   └── model/
│       ├── __init__.py
│       ├── attention.py            # Single-head causal self-attention
│       ├── multihead_attention.py  # Multi-head causal self-attention
│       ├── transformer_block.py    # Pre-Norm block (MHA + FFN + residual)
│       └── gpt.py                  # Full language model
│
├── data/
│   ├── raw/
│   │   └── campus_messages.jsonl   # Source dataset
│   └── processed/
│       ├── train.jsonl
│       ├── validation.jsonl
│       └── test.jsonl
│
├── checkpoints/
│   ├── best_model.pt
│   └── latest_model.pt
│
├── results/
│   ├── training_loss.csv
│   ├── loss_plot.png
│   └── evaluation_results.json
│
├── notebooks/
│   └── training_colab.ipynb        # Google Colab training notebook
│
└── scripts/
    ├── train.py                    # Configurable training script
    └── visualize_attention.py      # Attention heatmap visualization
```

## Dataset

The dataset uses JSONL format with tagged examples:

```json
{
  "id": 1,
  "category": "assignment_query",
  "input": "sir kal assignment submit karna hai kya",
  "output": "Sir, do we need to submit the assignment tomorrow?",
  "tone": "formal"
}
```

Each example is converted into a training prompt:

```
[TASK] assignment_query [TONE] formal [INPUT] sir kal assignment submit karna hai kya [OUTPUT] Sir, do we need to submit the assignment tomorrow?
```

Loss is computed only on the `[OUTPUT]` portion (prompt tokens are masked with `-100`).

**Split:** 70% train / 15% validation / 15% test

## Training

Training uses next-token prediction with cross-entropy loss.

```
Optimizer:        AdamW
Learning rate:    3e-4
Weight decay:     0.01
Gradient clipping: 1.0
Batch size:       16
Epochs:           10–30
```

### Local (CPU)

```bash
python train.py --epochs 5 --batch-size 8
```

### Google Colab (GPU)

```python
!git clone <REPO_URL>
%cd LexiFlow
!pip install -r requirements.txt
!python train.py --epochs 20 --batch-size 16
```

## Generation

Three decoding modes are supported:

| Mode | Temperature | Use Case |
|---|---|---|
| Greedy | — | Most deterministic output |
| Conservative | 0.7 | Reliable, slightly varied |
| Creative | 1.0 | More variation, less precision |

```bash
python generate.py --input "sir kal assignment submit karna hai kya" --temperature 0.7
```

## Evaluation

### Automatic Metrics

- Validation loss and perplexity
- BLEU / ROUGE scores
- Intent preservation rate
- Empty / invalid output rate

### Human Evaluation (1–5 scale)

| Criterion | Question |
|---|---|
| Meaning preservation | Does the output preserve the original message? |
| Formality | Is the message suitable for a professor? |
| Grammar | Is the English grammatically correct? |
| Relevance | Does it answer the intended task? |
| Usefulness | Would a student actually use this? |

## Demo

```bash
pip install streamlit
streamlit run app.py
```

The Streamlit interface provides:

- Message type selector (6 categories)
- Tone selector
- Temperature and length controls
- One-click copy and regenerate

## Requirements

- Python 3.8+
- PyTorch
- tiktoken
- streamlit
- matplotlib, seaborn

```bash
pip install -r requirements.txt
```

## Limitations

- GPT-2 BPE tokenizer is not optimized for Romanized Hinglish subwords
- Small dataset size limits generalization to unseen phrasings
- No Devanagari Hindi support (Romanized only)
- Domain-specific — not a general-purpose chatbot or translator
- May alter student intent in edge cases

## Future Work

- Larger and crowd-sourced Hinglish dataset
- Fine-tuned multilingual tokenizer
- Multiple tone outputs (friendly, very formal, WhatsApp-short)
- Email subject line generation
- Hindi Devanagari input support
- Intent classification module

## References

- Vaswani, A., et al. (2017). *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Radford, A., et al. (2018). *Improving Language Understanding by Generative Pre-Training.* OpenAI.
- Radford, A., et al. (2019). *Language Models are Unsupervised Multitask Learners.* OpenAI.
