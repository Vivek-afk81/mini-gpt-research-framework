# LexiFlow

> A Transformer-Based Assistant for Formalizing Code-Mixed Student Communication & Campus Inquiry

LexiFlow converts informal Romanized Hinglish messages into clear, polite, formal English — purpose-built for student-to-professor and student-to-administration communication, backed by a local RAG retrieval system for campus policies.

```text
Input:  sir kal assignment submit karna hai kya
Output: Sir, do we need to submit the assignment tomorrow?
```

---

## 1. Problem Statement

Many Indian college students communicate using code-mixed language — a blend of Romanized Hindi and English:

```text
maam mujhe topic samajh nahi aaya
sir ghar me emergency ki wajah se assignment complete nahi hua
sir attendance 75% se kam hai to exam de sakte hain kya
```

While natural in informal peer conversations, this phrasing is not suitable for official emails to faculty, absence letters, or administrative queries. LexiFlow addresses this by:
1. Converting informal Hinglish into formal, polite English with intent preservation.
2. Grounding student queries against official campus policies (attendance, leave, examinations, assignment guidelines, hostel rules) using local semantic search (RAG).

---

## 2. Architecture & Design

### Custom Decoder-Only Transformer (Local CPU Baseline)
A compact 4-layer, 128-dimensional causal Transformer designed specifically for low-latency CPU inference on standard laptops:

```text
Tokenizer:         GPT-2 BPE (tiktoken) — 50,257 vocab
Embedding dim:     128
Transformer blocks: 4
Attention heads:   4 (head_dim = 32)
FFN hidden dim:    512
Context length:    128
Architecture:      Pre-Norm (GPT-2 style LayerNorm placement)
Weight Tying:      Token embedding ↔ Language model head
Total Parameters:  ~7.24 Million
```

### Local RAG Grounding (FAISS + Sentence Transformers)
- **Vector Index**: FAISS Inner-Product cosine similarity (`IndexFlatIP`).
- **Embedding Model**: `all-MiniLM-L6-v2` via `sentence-transformers`.
- **Knowledge Base**: Curated campus policy documents under `data/knowledge/` (Attendance, Leave, Exams, Assignments, Hostel, Calendar).
- **Index Persistence**: Pre-computed and cached in `data/index/faiss.index` to prevent rebuilding on every application start.

---

## 3. Project Structure

```text
LexiFlow/
├── README.md                           # Project documentation and guide
├── requirements.txt                    # Python dependencies
├── generate.py                         # Root shortcut for generation CLI
├── train.py                            # Root shortcut for training script
├── evaluate.py                         # Root shortcut for evaluation pipeline
│
├── src/
│   ├── __init__.py
│   ├── config.py                       # Centralized type-safe dataclass configs
│   ├── tokenizer.py                    # GPT-2 BPE wrapper with EOS handling
│   ├── data_loader.py                  # JSONL dataset with prompt label masking (-100)
│   ├── generation.py                   # Category auto-detection, repetition penalty, sampling
│   ├── pipeline.py                     # Orchestrator (Memory + RAG + LLM backend)
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── attention.py                # Single-head causal attention
│   │   ├── multihead_attention.py      # Multi-head causal self-attention
│   │   ├── transformer_block.py        # Pre-Norm block (LN → MHA → Res → LN → FFN → Res)
│   │   └── gpt.py                      # Full GPT model with GPT-2 initialization
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── document_loader.py          # Markdown document loader and chunker
│   │   └── retriever.py                # FAISS vector store and embedding retrieval
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   └── chat_memory.py              # Conversation turn buffer
│   │
│   └── finetune/
│       ├── __init__.py
│       ├── dataset.py                  # Instruction formatting for open-source LLMs
│       └── train_qlora.py              # Colab QLoRA fine-tuning script
│
├── data/
│   ├── raw/
│   │   └── campus_messages.jsonl       # 120 curated Hinglish → English examples
│   ├── processed/
│   │   ├── train.jsonl                 # Training split (70%)
│   │   ├── validation.jsonl            # Validation split (15%)
│   │   └── test.jsonl                  # Held-out test split (15%)
│   ├── knowledge/                      # Campus documents for RAG
│   │   ├── attendance_policy.md
│   │   ├── leave_policy.md
│   │   ├── examination_rules.md
│   │   ├── assignment_guidelines.md
│   │   ├── hostel_rules.md
│   │   └── academic_calendar.md
│   └── index/                          # Pre-built FAISS index
│       ├── faiss.index
│       └── chunks.pkl
│
├── checkpoints/
│   ├── best_model.pt                   # Model weights with lowest validation loss
│   └── latest_model.pt                 # Most recent checkpoint
│
├── results/
│   ├── training_loss.csv               # Epoch-by-epoch loss and learning rate history
│   ├── evaluation_results.json         # Automated evaluation report
│   └── manual_evaluation_ratings.md    # 20 rated responses with rubric
│
├── tests/
│   ├── __init__.py
│   ├── test_tokenizer.py               # Tokenizer encode/decode & vocab tests
│   ├── test_data_loader.py             # Prompt formatting & label masking tests
│   ├── test_model.py                   # Forward pass, weight-tying & causal mask tests
│   └── test_generation.py              # Category detection, penalty & validation tests
│
└── scripts/
    ├── train.py                        # Configurable training script
    ├── generate.py                     # Command-line inference and interactive shell
    ├── evaluate.py                     # Loss, perplexity, and quality evaluation
    ├── build_index.py                  # RAG document indexing utility
    └── visualize_attention.py          # Multi-head attention map visualizer
```

---

## 4. Supported Categories

LexiFlow automatically recognizes student intents across 6 college communication categories:

| # | Category | Sample Hinglish Input |
|---|---|---|
| 1 | `assignment_query` | `sir kal assignment submit karna hai kya` |
| 2 | `attendance_request` | `sir attendance 75% se kam hai to exam de sakte hain kya` |
| 3 | `leave_request` | `sir fever hai isliye do din college nahi aa paunga` |
| 4 | `exam_query` | `sir mid sem exam kis date se start ho rahe hain` |
| 5 | `deadline_extension` | `sir tabiyat kharab thi assignment ke liye thoda extra time de dijiye` |
| 6 | `complaint` | `sir computer lab me ac aur internet kaam nahi kar raha` |

---

## 5. Getting Started & Local Usage

### Installation

```bash
git clone <REPO_URL>
cd LexiFlow
python -m venv venv
venv\Scripts\activate       # On Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

### Running Tests

Run the complete unit test suite (18 tests verifying tokenizer, masking, model forward pass, and generation):

```bash
python -m unittest discover -s tests -v
```

### Text Generation CLI

**Single Message:**
```bash
python scripts/generate.py --input "sir kal assignment submit karna hai kya"
```

**Batch Test across All 6 Categories (18 sample messages):**
```bash
python scripts/generate.py --batch-test
```

**Interactive Mode:**
```bash
python scripts/generate.py --interactive
```

### Training the Custom Model (Local CPU)

```bash
python scripts/train.py --epochs 25 --batch-size 8 --lr 4e-4
```

### Running Evaluation

Computes validation loss, test loss, perplexity, empty output rate, repetition rate, and formality metrics:

```bash
python scripts/evaluate.py
```

### Testing Local RAG Retrieval

Builds or verifies the FAISS index and retrieves grounded campus documents:

```bash
python scripts/build_index.py
```

---

## 6. Local Evaluation Summary

From automated evaluation on held-out test and validation sets (`checkpoints/best_model.pt`):

| Metric | Result | Notes |
|---|---|---|
| **Validation Loss** | `4.3559` | Cross-entropy loss on output tokens only |
| **Validation Perplexity** | `77.94` | Down from ~48,000 initial untrained |
| **Test Loss** | `4.3936` | Consistent generalization on unseen data |
| **Empty Output Rate** | `0.0%` | Prevented via `min_tokens` and safe category fallbacks |
| **Repetition Rate** | `0.0%` | Suppressed via `repetition_penalty=1.25` |
| **Average Formality** | `0.64 / 1.0` | Consistent usage of polite honorifics and modal verbs |
| **Average CPU Inference** | `~300ms` | Fast, lightweight execution on consumer laptops |

---

## 7. Next Step: Google Colab for QLoRA Training

To respect local laptop resources, fine-tuning large language models (such as `Qwen/Qwen2.5-1.5B-Instruct`) should be performed in Google Colab's free T4 GPU environment:

1. Open `notebooks/` in Google Colab with GPU runtime enabled.
2. Run `src/finetune/train_qlora.py` with 4-bit quantization (QLoRA).
3. Download only the small LoRA adapter weights (`checkpoints/qlora/`, ~30–50 MB).
4. Place the adapter into `checkpoints/qlora/` locally.
5. Set `backend="finetuned"` in `src/pipeline.py` to seamlessly switch from the custom baseline to the fine-tuned assistant with source citations and conversation memory.
