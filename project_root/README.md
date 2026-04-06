# ProjectX

A simple PyTorch-based language model prototype built around a small GPT-style training pipeline.

## Overview

This project demonstrates basic tokenizer, dataset, and model components for autoregressive text modeling.

The current implementation:
- loads raw text data from `data/`
- tokenizes text using `src/tokenizer.py`
- creates token sequences using `src/data_loader.py`
- builds a baseline language modeling network in `src/model/baseline_lm.py`
- exercises attention with `src/model/attention.py`

## Structure

- `main.py` — entry point for running a quick data/model sanity check
- `data/` — raw text and supporting dataset files
- `src/data_loader.py` — dataset creation and sequence batching
- `src/tokenizer.py` — tokenizer wrapper and encoding logic
- `src/model/` — model components, including attention and transformer blocks

## Requirements

- Python 3.8+
- PyTorch

## Run

From the `project_root` folder:

```bash
python main.py
```

## Notes

This is a work in progress, so the project is currently intended for experimentation and prototyping rather than production use.
