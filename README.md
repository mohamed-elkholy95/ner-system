<div align="center">

# 🏷️ NER System

**Named Entity Recognition** with BIO tagging, BiLSTM-CRF, and entity-level evaluation

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-39%20passed-success?style=flat-square)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-009688?style=flat-square)](https://fastapi.tiangolo.com)

</div>

## Overview

A **Named Entity Recognition (NER)** system implementing the BIO tagging scheme with a BiLSTM-CRF model. Features synthetic data generation, feature extraction (morphological, positional, capitalization), entity-level evaluation (precision, recall, F1), and a REST API.

## Features

- 🗂️ **BIO Tagging** — Standard BIO scheme for entity boundaries (PER, ORG, LOC, DATE, MISC)
- 🏗️ **BiLSTM-CRF Model** — Bidirectional LSTM with conditional random fields
- 🔤 **Feature Extraction** — 6 hand-crafted features per token (prefix, suffix, capitalization, position, etc.)
- 📊 **Synthetic Data** — Configurable template-based NER dataset generation
- 📏 **Entity-Level Metrics** — Exact match precision, recall, F1 per entity type
- 🧪 **39 Tests** — Full pipeline coverage

## Quick Start

```bash
git clone https://github.com/mohamed-elkholy95/ner-system.git
cd ner-system
pip install -r requirements.txt
python -m pytest tests/ -v
streamlit run streamlit_app/app.py
```

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95) · melkholy@techmatrix.com
