# LLM Zoomcamp 2026 - Module 4 Homework Solution

This project contains my solution for the DataTalksClub LLM Zoomcamp 2026
Module 4 homework:
[Evaluation homework](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/cohorts/2026/04-evaluation/homework.md).

The homework evaluates retrieval quality for keyword search, vector search, and
hybrid search over the LLM Zoomcamp lesson pages. It uses generated ground truth
questions, then measures retrieval with Hit Rate and Mean Reciprocal Rank (MRR).

## What It Does

- Loads 72 lesson markdown pages from the pinned course commit `8c1834d`.
- Splits the pages into chunks with `chunk_documents(size=2000, step=1000)`.
- Generates sample ground-truth questions for the first 3 pages with Gemini.
- Loads the full `ground-truth.csv` dataset with 360 questions.
- Builds a keyword index with `minsearch.Index`.
- Builds a vector index with `minsearch.VectorSearch`.
- Evaluates text, vector, and hybrid search with Hit Rate and MRR.
- Tests RRF `k` values: `1`, `50`, `100`, and `200`.

## Files

- `main.py` - runs the homework solution for Q1-Q6.
- `evaluate.py` - contains Hit Rate, MRR, evaluation, and RRF helpers.
- `embedder.py` - ONNX-based embedding helper.
- `download.py` - downloads the tokenizer and ONNX model.
- `ground-truth.csv` - full homework ground truth dataset.
- `evaluation_utils.py` and `rag_helper.py` - LLM helper utilities from the course.
- `pyproject.toml` - project dependencies.

## Setup

This project uses Python 3.13 and `uv`.

```bash
cd datatalks-llm-zoomcamp-2026/04-evaluation
uv sync
```

Download the embedding model:

```bash
uv run python download.py
```

Create a `.env` file with a Gemini API key:

```bash
GEMINI_API_KEY=your_key_here
```

The script uses `gemini-3.1-flash-lite` for Q1 question generation. The rest of
the homework uses the provided `ground-truth.csv`.

## Run

```bash
uv run python main.py
```

Or, if the virtual environment is already activated:

```bash
python main.py
```

## Answers

| Question | Result | Selected answer |
| --- | ---: | --- |
| Q1. Average input tokens across the first 3 calls | `1449.67` | `1400` |
| Q2. First result with text search | `01-agentic-rag/lessons/03-rag.md` | `01-agentic-rag/lessons/03-rag.md` |
| Q3. First result with vector search | `01-agentic-rag/lessons/01-intro.md` | `01-agentic-rag/lessons/01-intro.md` |
| Q4. Text search Hit Rate | `0.7583` | `0.76` |
| Q5. Vector search MRR | `0.5486` | `0.55` |
| Q6. Best RRF `k` by MRR | `50`, `100`, and `200` tied at `0.6468`; choose the smallest | `50` |

## Full Metrics

```text
Q1: Input tokens used: 1449.6666666666667

Q2 first text-search result:
01-agentic-rag/lessons/03-rag.md

Q3 first vector-search result:
01-agentic-rag/lessons/01-intro.md

Q4 text search:
Hit Rate: 0.7583333333333333
MRR: 0.5942592592592594

Q5 vector search:
Hit Rate: 0.725
MRR: 0.5486111111111112

Q6 hybrid search:
k=1, MRR=0.6458333333333337
k=50, MRR=0.6467592592592594
k=100, MRR=0.6467592592592594
k=200, MRR=0.6467592592592594
```

## Notes

The exact Q1 token count may vary slightly by model/provider response metadata,
but it should stay closest to `1400`. For Q6, the homework says to pick the
smallest `k` when there is a tie, so the selected answer is `50`.
