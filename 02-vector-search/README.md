# LLM Zoomcamp 2026 - Module 2 Homework Solution

This project contains my solution for the DataTalksClub LLM Zoomcamp 2026
Module 2 homework:
[Vector Search homework](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/cohorts/2026/02-vector-search/homework.md).

The homework focuses on search without the RAG generation step. It uses course
lesson pages as the knowledge base, turns text into embeddings with a lightweight
ONNX model, and compares vector search, keyword search, and hybrid search with
Reciprocal Rank Fusion (RRF).

## What It Does

- Downloads and uses the `Xenova/all-MiniLM-L6-v2` ONNX embedding model.
- Loads the LLM Zoomcamp lesson markdown files from the pinned course commit
  `8c1834d`.
- Embeds a query and lesson chunks.
- Computes cosine similarity manually with NumPy.
- Runs vector search with `minsearch.VectorSearch`.
- Runs keyword search with `minsearch.Index`.
- Combines vector and keyword results with RRF.

## Files

- `main.py` - runs the homework solution for Q1-Q6.
- `embedder.py` - ONNX-based embedding helper.
- `download.py` - downloads the tokenizer and ONNX model from Hugging Face.
- `pyproject.toml` - project dependencies.

## Setup

This project uses Python 3.13 and `uv`.

```bash
uv sync
```

Download the embedding model:

```bash
uv run python download.py
```

## Run

```bash
uv run python main.py
```

Or, if the local virtual environment is already activated:

```bash
python main.py
```

The script prints the outputs for each homework question, including embedding
values, similarity scores, top filenames from vector/text search, and the final
RRF result.

## Notes

The data reader fetches lesson files from GitHub, so running `main.py` requires
an internet connection unless the data has already been cached by the underlying
library.
