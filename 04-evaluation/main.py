import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from dotenv import load_dotenv
from google import genai
from minsearch import Index, VectorSearch
from pydantic import BaseModel

from embedder import Embedder
from evaluate import evaluate, rrf
from evaluation_utils import llm_structured_gemini
from gitsource import GithubRepositoryDataReader, chunk_documents


BASE_DIR = Path(__file__).resolve().parent
MODEL = "gemini-3.1-flash-lite"
RRF_K_VALUES = [1, 50, 100, 200]
TARGET_PAGES = [
    "01-agentic-rag/lessons/01-intro.md",
    "01-agentic-rag/lessons/02-environment.md",
    "01-agentic-rag/lessons/03-rag.md",
]

DATA_GEN_INSTRUCTIONS = """
You emulate a student who is taking our LLM course.
You are given one lesson page from the course.
Formulate 5 questions this student might ask that are answered by this page.

Rules:
- The page should contain the answer to each question.
- Make the questions complete and not too short.
- Use as few words as possible from the page; don't copy its phrasing.
- The questions should resemble how people actually ask things online:
  not too formal, not too short, not too long.
- Ask about the content of the lesson, not about its formatting or filename.
""".strip()


class Questions(BaseModel):
    questions: list[str]


class SearchEngine:
    def __init__(self, chunks, embeddings, embedder):
        self.embedder = embedder
        self.text_index = build_text_index(chunks)
        self.vector_index = build_vector_index(embeddings, chunks)

    def text_search(self, query, num_results=5):
        return self.text_index.search(query, num_results=num_results)

    def vector_search(self, query, num_results=5):
        query_embedding = self.embedder.encode(query)
        return self.vector_index.search(query_embedding, num_results=num_results)

    def hybrid_search(self, query, num_results=5, rrf_k=60):
        text_results = self.text_search(query, num_results=num_results)
        vector_results = self.vector_search(query, num_results=num_results)
        return rrf([text_results, vector_results], k=rrf_k, num_results=num_results)


def get_documents():
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    return [file.parse() for file in reader.read()]


def embed_in_batches(texts, embedder, batch_size=50):
    embeddings = []

    for i in tqdm.tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        embeddings.extend(embedder.encode_batch(batch))

    return np.array(embeddings)


def build_text_index(chunks):
    index = Index(text_fields=["content"])
    index.fit(chunks)
    return index


def build_vector_index(embeddings, chunks):
    index = VectorSearch()
    index.fit(embeddings, chunks)
    return index


def build_search_engine(documents):
    chunks = chunk_documents(documents, size=2000, step=1000)
    texts = [chunk["content"] for chunk in chunks]

    embedder = Embedder()
    embeddings = embed_in_batches(texts, embedder)

    return SearchEngine(chunks, embeddings, embedder)


def generate_ground_truth(client, doc):
    user_prompt = json.dumps(doc)

    out, usage = llm_structured_gemini(
        client,
        MODEL,
        user_prompt,
        DATA_GEN_INSTRUCTIONS,
        Questions,
    )

    records = [
        {
            "question": question,
            "filename": doc["filename"],
        }
        for question in out.questions
    ]

    return records, usage


def generate_target_ground_truth(client, documents, target_pages=TARGET_PAGES):
    ground_truth = []
    prompt_tokens = 0

    for doc in documents:
        if doc["filename"] not in target_pages:
            continue

        records, usage = generate_ground_truth(client, doc)
        print(records)

        ground_truth.extend(records)
        prompt_tokens += usage.prompt_token_count

    avg_prompt_tokens = prompt_tokens / len(target_pages)
    print(f"Q1: Input tokens used: {avg_prompt_tokens}")

    return ground_truth


def load_ground_truth(path=BASE_DIR / "ground-truth.csv"):
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def print_search_results(title, question, results):
    print(f"\n{title}: {question}")

    for i, doc in enumerate(results, start=1):
        print(f"Result {i}: {doc['filename']}")


def print_metrics(title, metrics):
    print(f"\n{title}")
    print(f"Hit Rate: {metrics['hit_rate']}")
    print(f"MRR: {metrics['mrr']}")


def evaluate_rrf_k_values(ground_truth, search_engine, k_values=RRF_K_VALUES):
    print("\nQ6: Evaluating hybrid search with MRR for different k values")

    for k in k_values:
        search_function = lambda query, num_results=5, k=k: search_engine.hybrid_search(
            query,
            num_results=num_results,
            rrf_k=k,
        )
        metrics = evaluate(ground_truth, search_function)
        print(f"k={k}, MRR={metrics['mrr']}")


def main():
    load_dotenv(BASE_DIR / ".env")

    print("Hello from 04-evaluation!")

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    documents = get_documents()
    search_engine = build_search_engine(documents)

    generate_target_ground_truth(client, documents)

    ground_truth = load_ground_truth()
    question = ground_truth[0]["question"]

    text_results = search_engine.text_search(question, num_results=5)
    print_search_results("Q2: Text search results for question", question, text_results)

    vector_results = search_engine.vector_search(question, num_results=5)
    print_search_results("Q3: Vector search results for question", question, vector_results)

    text_search_metrics = evaluate(ground_truth, search_engine.text_search)
    print_metrics("Q4: Text search metrics", text_search_metrics)

    vector_search_metrics = evaluate(ground_truth, search_engine.vector_search)
    print_metrics("Q5: Vector search metrics", vector_search_metrics)

    evaluate_rrf_k_values(ground_truth, search_engine)


if __name__ == "__main__":
    main()
