import numpy as np

from tqdm.auto import tqdm
from embedder import Embedder
from minsearch import VectorSearch, Index
from gitsource import GithubRepositoryDataReader, chunk_documents

REPO_OWNER = "DataTalksClub"
REPO_NAME = "llm-zoomcamp"
COMMIT_ID = "8c1834d"
CHUNK_SIZE = 2000
CHUNK_STEP = 1000
BATCH_SIZE = 50
NUM_RESULTS = 5


def load_data():
    reader = GithubRepositoryDataReader(
        repo_owner=REPO_OWNER,
        repo_name=REPO_NAME,
        commit_id=COMMIT_ID,
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    documents = [file.parse() for file in reader.read()]
    return documents


def chunk_data(documents):
    return chunk_documents(documents, size=CHUNK_SIZE, step=CHUNK_STEP)


def embed_in_batches(texts, embedder, batch_size=BATCH_SIZE):
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        batch_vectors = embedder.encode_batch(batch)
        embeddings.extend(batch_vectors)

    return np.array(embeddings)


def build_vector_index(embeddings, chunks):
    index = VectorSearch()
    index.fit(embeddings, chunks)
    return index


def build_text_index(chunks):
    index = Index(text_fields=["content"])
    index.fit(chunks)
    return index


def find_most_relevant_chunk(query_embedding, embeddings, chunks):
    scores = embeddings.dot(query_embedding)
    return chunks[np.argmax(scores)]


def print_filenames(results):
    for doc in results:
        print(doc["filename"])


def run_text_and_vector_search(query, embedder, text_index, vector_index):
    query_embedding = embedder.encode(query)
    text_results = text_index.search(query, num_results=NUM_RESULTS)
    vector_results = vector_index.search(query_embedding, num_results=NUM_RESULTS)
    return text_results, vector_results


def rrf(result_lists, k=60, num_results=NUM_RESULTS):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


def main():
    print("Hello from 02-vector-search!")

    embedder = Embedder()
    documents = load_data()

    # Q1
    q1 = "How does approximate nearest neighbor search work?"
    q1_embedding = embedder.encode(q1)
    print("Q1 :")
    print("Embedding shape:", q1_embedding.shape)
    print("First embedding vector:", q1_embedding[0])

    # Q2
    target_filename = "02-vector-search/lessons/07-sqlitesearch-vector.md"
    filtered_doc = next(
        doc for doc in documents if target_filename in doc["filename"]
    )
    filtered_vector = embedder.encode(filtered_doc["content"])
    print("\nQ2 :")
    print(filtered_vector.dot(q1_embedding))

    # Q3
    chunks = chunk_data(documents)
    texts = [chunk["content"] for chunk in chunks]
    embeddings = embed_in_batches(texts, embedder)
    most_relevant_chunk = find_most_relevant_chunk(q1_embedding, embeddings, chunks)

    print("\nQ3 :")
    print("Most relevant chunk filename: ", most_relevant_chunk["filename"])

    # Q4
    vector_index = build_vector_index(embeddings, chunks)
    query = "What metric do we use to evaluate a search engine?"
    q_embed = embedder.encode(query)
    results = vector_index.search(q_embed, num_results=NUM_RESULTS)
    print("\nQ4 :")
    print_filenames(results)

    # Q5
    text_index = build_text_index(chunks)
    query = "How do I store vectors in PostgreSQL?"
    text_search_results, vector_search_results = run_text_and_vector_search(
        query, embedder, text_index, vector_index
    )

    print("\nQ5 :")
    print("Results from Text search:")
    print_filenames(text_search_results)

    print("\nResults from Vector search:")
    print_filenames(vector_search_results)

    # Q6
    query = "How do I give the model access to tools?"
    text_search_results, vector_search_results = run_text_and_vector_search(
        query, embedder, text_index, vector_index
    )

    print("\nQ6 :")
    print("Results from Text search:")
    print_filenames(text_search_results)

    print("\nResults from Vector search:")
    print_filenames(vector_search_results)

    results = rrf([vector_search_results, text_search_results])
    print("\nResults from RRF:")
    print_filenames(results)


if __name__ == "__main__":
    main()
