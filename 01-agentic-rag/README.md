# LLM Zoomcamp 2026 Code

This project is a small retrieval-augmented generation (RAG) and tool-calling
agent demo for the DataTalksClub LLM Zoomcamp materials.

The script loads lesson markdown files from the
[`DataTalksClub/llm-zoomcamp`](https://github.com/DataTalksClub/llm-zoomcamp)
GitHub repository, indexes them with `minsearch`, and uses Gemini to answer
questions from the retrieved context.

## What It Does

- Loads LLM Zoomcamp lesson documents from GitHub.
- Builds a searchable text index over the lesson content.
- Runs keyword search over the indexed documents.
- Runs a basic RAG pipeline using retrieved context.
- Chunks longer documents and repeats RAG over the chunk index.
- Demonstrates an agentic loop where Gemini can call a local `search` tool
  multiple times before returning an answer.

## Project Structure

```text
.
├── main.py          # End-to-end demo: loading, indexing, RAG, chunking, agent loop
├── rag_helper.py    # Reusable RAGBase class for search, prompt building, and LLM calls
├── pyproject.toml   # Project metadata and dependencies
├── uv.lock          # Locked dependency versions
└── README.md
```

## Requirements

- Python 3.13 or newer
- `uv`
- A Gemini API key

## Setup

Install dependencies:

```bash
uv sync
```

Create a `.env` file in this directory:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

## Run

From the project directory:

```bash
uv run python main.py
```

The script will:

1. Load lesson documents from the LLM Zoomcamp GitHub repository.
2. Print the number of loaded documents.
3. Show search results for a sample query.
4. Run RAG on the full documents.
5. Chunk the documents and run RAG again.
6. Run an agent that uses the search tool until it has enough context to answer.
   Each tool call is printed, followed by the total number of search calls.

## Main Components

### `main.py`

Contains the full demo workflow.

Important functions:

- `load_data()` loads markdown lesson files from GitHub.
- `build_index(documents)` creates a `minsearch.Index`.
- `search_documents(index, query)` searches the index.
- `print_search_results(results)` prints matching documents.
- `run_rag(index, query)` runs a RAG call through `RAGBase`.
- `AgentSearchTool` exposes the search index as a Gemini tool and counts calls.
- `run_agent(index, instructions, question)` runs the agentic tool-calling loop.

### `rag_helper.py`

Defines `RAGBase`, a small helper class that:

- Searches the index.
- Builds a context block from search results.
- Formats the final prompt.
- Sends the prompt to Gemini.
- Returns the model response.

## Configuration

The Gemini model is currently set to:

```python
gemini-3.1-flash-lite
```

You can change the model in:

- `rag_helper.py`, via the `RAGBase` default `model` argument.
- `main.py`, inside `run_agent()`.

## Notes

- The document loader is pinned to commit `8c1834d` of the LLM Zoomcamp repo, so
  results should be stable as the upstream repository changes.
- Only markdown files under `/lessons/` are loaded.
- The app expects `GEMINI_API_KEY` to be present in the environment or `.env`
  file before running.
