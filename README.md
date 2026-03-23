<div align="center">

# 🏷️ NER System

**Named Entity Recognition** with BIO tagging, CRF, BiLSTM-CRF, and entity-level evaluation

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-passing-success?style=flat-square)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-009688?style=flat-square)](https://fastapi.tiangolo.com)

</div>

## Overview

A **Named Entity Recognition (NER)** system implementing the BIO tagging scheme with multiple model architectures (CRF, BiLSTM-CRF, DistilBERT). Features synthetic data generation, hand-crafted and learned feature extraction, entity-level evaluation with exact and partial matching, and a REST API with batch support.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Input Text                        │
│  "John Smith works at Google in New York"            │
└─────────────┬───────────────────────────────────────┘
              ▼
┌─────────────────────────────────────────────────────┐
│              Tokenization & Features                 │
│  ┌──────────┬──────────┬──────────┬──────────┐      │
│  │ "John"   │ "Smith"  │ "works"  │ ...      │      │
│  │ title=T  │ title=T  │ title=F  │          │      │
│  │ len=4    │ len=5    │ len=5    │          │      │
│  │ shape=Xx │ shape=Xx │ shape=xx │          │      │
│  └──────────┴──────────┴──────────┴──────────┘      │
└─────────────┬───────────────────────────────────────┘
              ▼
┌─────────────────────────────────────────────────────┐
│              Model (CRF / BiLSTM / BERT)             │
│                                                      │
│  CRF:     Feature templates → tag transitions        │
│  BiLSTM:  Embeddings → BiLSTM → Linear → tags       │
│  BERT:    Subwords → Transformer → token logits      │
└─────────────┬───────────────────────────────────────┘
              ▼
┌─────────────────────────────────────────────────────┐
│              BIO Tag Sequence                        │
│  B-PER  I-PER  O  O  B-ORG  O  B-LOC  I-LOC        │
└─────────────┬───────────────────────────────────────┘
              ▼
┌─────────────────────────────────────────────────────┐
│              Entity Extraction                       │
│  [PER: "John Smith"] [ORG: "Google"] [LOC: "New York"]│
└─────────────────────────────────────────────────────┘
```

## The BIO Tagging Scheme

NER is formulated as a **token classification** problem using BIO (Beginning-Inside-Outside) tags:

| Tag | Meaning | Example |
|-----|---------|---------|
| `B-PER` | **Beginning** of a person entity | "**John** Smith" → `B-PER` |
| `I-PER` | **Inside** (continuation) of a person entity | "John **Smith**" → `I-PER` |
| `O` | **Outside** any entity | "**works** at" → `O` |

**Why B- and I- prefixes?** Without the B- prefix, we couldn't distinguish adjacent entities of the same type. In "John Smith Sarah Lee", we need `B-PER I-PER B-PER I-PER` to know there are two separate people, not one four-word name.

### Supported Entity Types

| Type | Description | Examples |
|------|-------------|----------|
| 👤 PER | Person names | John Smith, Dr. Maria Garcia |
| 🏢 ORG | Organizations | Google, United Nations, Stanford University |
| 📍 LOC | Locations | New York, Silicon Valley, Tokyo |
| 📅 DATE | Temporal expressions | January 2024, last week, Monday |
| 📎 MISC | Miscellaneous | Nationalities, events, works of art |

## Features

- 🗂️ **BIO Tagging** — Standard BIO scheme for entity boundaries
- 🏗️ **Multiple Models** — CRF, BiLSTM-CRF (PyTorch), DistilBERT fine-tuning
- 🔤 **Feature Engineering** — 10+ hand-crafted features per token (word shape, prefix/suffix, capitalization)
- 📊 **Synthetic Data** — Configurable template-based NER dataset generation
- 📏 **Rich Evaluation** — Exact match + partial match metrics, confusion matrix, per-type breakdown
- 🌐 **REST API** — Single and batch NER endpoints with FastAPI
- 🖥️ **Streamlit Dashboard** — Interactive entity extraction and metrics visualization
- 💾 **Model Persistence** — Save/load trained CRF models
- 🧪 **Comprehensive Tests** — Full pipeline coverage with edge cases

## Quick Start

```bash
git clone https://github.com/mohamed-elkholy95/ner-system.git
cd ner-system
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Launch the dashboard
streamlit run streamlit_app/app.py

# Start the API server
python -m src.api.main
```

### API Usage

```bash
# Single text
curl -X POST http://localhost:8009/ner \
  -H "Content-Type: application/json" \
  -d '{"text": "John Smith works at Google in New York"}'

# Batch processing
curl -X POST http://localhost:8009/ner/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["John works at Google", "Visit Paris in spring"]}'

# List supported entity types
curl http://localhost:8009/ner/tags
```

## Project Structure

```
09-ner-system/
├── src/
│   ├── config.py           # Tags, colors, paths, entity descriptions
│   ├── tokenizer.py        # Vocabulary, tag encoder, word shape features
│   ├── ner_model.py        # CRF and BiLSTM-CRF implementations
│   ├── evaluation.py       # Exact/partial metrics, confusion matrix
│   ├── data_generator.py   # Synthetic NER data generation
│   └── api/main.py         # FastAPI endpoints (single + batch)
├── train_ner_conll.py      # DistilBERT fine-tuning on Few-NERD
├── streamlit_app/          # Interactive dashboard
├── tests/                  # Comprehensive test suite
├── models/                 # Saved model checkpoints
├── data/                   # Training data
└── docs/                   # Additional documentation
```

## Evaluation Metrics

The system provides three levels of evaluation:

1. **Exact Match** — Entity must match type AND exact span boundaries
2. **Partial Match** — Proportional credit for overlapping spans (useful during development)
3. **Confusion Matrix** — Token-level type confusion analysis (e.g., ORG↔LOC errors)

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95) · melkholy@techmatrix.com
