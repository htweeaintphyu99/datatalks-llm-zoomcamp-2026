# LLM Zoomcamp 2026 - Module 5 Homework (Monitoring)

This repository contains my solution for the **Module 5: Monitoring** homework from the **DataTalksClub LLM Zoomcamp 2026**. The homework focuses on instrumenting a Retrieval-Augmented Generation (RAG) application with **OpenTelemetry**, exporting traces to **SQLite**, and analyzing the collected trace data.

## Overview

The implementation demonstrates how to:

- Instrument a RAG pipeline using OpenTelemetry
- Record spans for different stages of the pipeline
- Export traces to both:
  - Console
  - SQLite database
- Analyze trace data with SQL and pandas
- Answer all homework questions programmatically

## Project Structure

```
.
├── main.py                 # Runs the homework and prints all answers
├── span_exporter.py        # Custom SQLite span exporter
├── starter.py              # RAG implementation
├── traces.db               # SQLite database storing traces
├── pyproject.toml
├── uv.lock
└── README.md
```

## Setup

This project uses **uv** for dependency management.

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Install dependencies
```bash
uv sync
```

This creates a virtual environment and installs all required packages.


### 3. Configure environment variables

If the project requires an API key (for example, Google Gemini), create a `.env` file:

```text
GOOGLE_API_KEY=your_api_key_here
```

### 4. Run the homework

```bash
uv run python main.py
```

## How It Works

1. Configure OpenTelemetry tracing.
2. Clear previous trace records.
3. Execute the RAG query.
4. Export spans to SQLite.
5. Load spans into pandas.
6. Compute the required statistics.
7. Print answers for each homework question.

## Homework Questions

The script automatically answers:

- **Q1** – Number of spans in the first trace
- **Q2** – Input token count
- **Q3** – Search and LLM execution time
- **Q4** – Span names stored in SQLite
- **Q5** – Total duration by span type
- **Q6** – Input token stability across multiple runs


## References

- LLM Zoomcamp: https://github.com/DataTalksClub/llm-zoomcamp
- Module 5 Homework: https://github.com/DataTalksClub/llm-zoomcamp/blob/main/cohorts/2026/05-monitoring/homework.md

## Notes

This repository contains my personal implementation of the homework. Running `uv run python main.py` reproduces all trace collection, analysis, and homework answers automatically.