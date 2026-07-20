import sqlite3
from pathlib import Path

import pandas as pd
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from span_exporter import SQLiteSpanExporter


QUERY = "How does the agentic loop keep calling the model until it stops?"
DB_PATH = Path("traces.db")
Q2_OPTIONS = [700, 7_000, 70_000, 700_000]


def setup_tracing(db_path=DB_PATH):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    provider.add_span_processor(SimpleSpanProcessor(SQLiteSpanExporter(db_path)))
    trace.set_tracer_provider(provider)

    return provider


def reset_spans_table(db_path=DB_PATH):
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM spans")
        conn.commit()


def load_spans(db_path=DB_PATH):
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql("SELECT * FROM spans", conn)


def add_duration_seconds(df):
    df = df.copy()
    df["duration_seconds"] = (df["end_time"] - df["start_time"]) / 1_000_000_000
    return df


def print_q1_to_q3(spans):
    first_trace = spans.tail(3)
    llm_span = first_trace[first_trace["name"] == "llm"].iloc[0]
    timed_spans = add_duration_seconds(first_trace)
    llm_duration = timed_spans[timed_spans["name"] == "llm"]["duration_seconds"].iloc[0]
    input_tokens = int(llm_span["input_tokens"])
    closest_token_option = min(Q2_OPTIONS, key=lambda option: abs(option - input_tokens))

    if llm_duration < 0.1:
        duration_answer = "Under 100ms"
    elif llm_duration < 0.5:
        duration_answer = "100-500ms"
    elif llm_duration < 2.0:
        duration_answer = "500-2000ms"
    else:
        duration_answer = "Over 2000ms"

    print("\nQ1. First trace")
    print(f"Span count: {len(first_trace)}")

    print("\nQ2. Capturing metrics as span attributes")
    print(f"Input tokens: {input_tokens}")
    print(f"Closest answer: {closest_token_option}")

    print("\nQ3. Span timing")
    print(f"Search duration: {timed_spans[timed_spans['name'] == 'search']['duration_seconds'].iloc[0]:.3f} seconds")
    print(f"LLM duration: {llm_duration:.3f} seconds")
    print(f"Answer for this run: {duration_answer}")


def print_q4(spans):
    span_names = sorted(spans["name"].unique())

    print("\nQ4. Saving traces to SQLite")
    print(f"Span names: {span_names}")


def print_q5(spans):
    child_spans = add_duration_seconds(spans[spans["name"] != "rag"])
    total_duration = (
        child_spans.groupby("name")["duration_seconds"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nQ5. Querying trace data")
    print(total_duration)
    print(f"Answer: {total_duration.index[0]}")


def print_q6(spans):
    llm_tokens = spans[spans["name"] == "llm"]["input_tokens"].astype(int)
    min_tokens = llm_tokens.min()
    max_tokens = llm_tokens.max()
    variation = (max_tokens - min_tokens) / min_tokens if min_tokens else 0

    if variation == 0:
        answer = "They're identical"
    elif variation <= 0.10:
        answer = "Within 10% of each other"
    elif variation <= 0.50:
        answer = "Within 50% of each other"
    else:
        answer = "They vary more than 50%"

    print("\nQ6. Token stability across runs")
    print(llm_tokens.to_list())
    print(f"Variation: {variation:.2%}")
    print(f"Answer: {answer}")


def main():
    print("Hello from 05-monitoring-hw!")
    setup_tracing()
    reset_spans_table()

    # Import after tracing setup so starter/rag_helper use the configured provider.
    from starter import rag

    print(f"\nRunning homework query:\n{QUERY}")

    print("\nRunning Q1-Q4 trace...")
    answer = rag.rag(QUERY)
    print(f"\nRAG answer:\n{answer}")

    spans = load_spans()
    print(spans)
    print_q1_to_q3(spans)
    print_q4(spans)

    print("\nRunning one more query for Q5...")
    rag.rag(QUERY)
    spans = load_spans()
    print_q5(spans)

    print("\nRunning two more queries for Q6, for 4 RAG calls total...")
    for _ in range(2):
        rag.rag(QUERY)

    spans = load_spans()
    print_q6(spans)


if __name__ == "__main__":
    main()
