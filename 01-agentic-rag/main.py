import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from gitsource import GithubRepositoryDataReader, chunk_documents
from google import genai
from google.genai import types
from minsearch import Index

from rag_helper import RAGBase


MODEL = "gemini-3.1-flash-lite"
QUERY = "How does the agentic loop keep calling the model until it stops?"
AGENT_QUESTION = (
    "How does the agentic loop work, and how is it different from plain RAG?"
)
AGENT_INSTRUCTIONS = """
You're a course teaching assistant. Answer the student's question using the
search tool. Make multiple searches with different keywords before answering.
""".strip()


load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def print_section(title):
    print(f"\n{title}")


# ============================================================
# Data Loading
# ============================================================
def load_data():
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )

    files = reader.read()
    return [file.parse() for file in files]


def build_index(documents):
    index = Index(
        text_fields=["content"],
        keyword_fields=["filename"],
    )
    index.fit(documents)
    return index


# ============================================================
# Search
# ============================================================
def search_documents(index, query, num_results=1):
    return index.search(query, num_results=num_results)


def print_search_results(results):
    print("\nSearch results:")
    for doc in results:
        print(f"Filename: {doc['filename']}")
        print(f"Content: {doc['content'][:200]}...")


# ============================================================
# RAG
# ============================================================
def run_rag(index, query):
    assistant = RAGBase(
        index=index,
        llm_client=client,
    )
    return assistant.rag(query)


def print_token_usage(response):
    print(
        "Prompt tokens:",
        response.usage_metadata.prompt_token_count,
    )
    print(
        "Output tokens:",
        response.usage_metadata.candidates_token_count,
    )


def run_and_print_rag(index, query):
    response = run_rag(index, query)
    print_token_usage(response)
    return response


# ============================================================
# Agent
# ============================================================
class AgentSearchTool:
    def __init__(self, index):
        self.index = index
        self.calls = 0

    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the chunk index for documents relevant to a query.

        Args:
            query: User search query.
            num_results: Maximum number of results to return.

        Returns:
            A list of relevant document chunks.
        """
        self.calls += 1
        print(
            f"SEARCH CALLED #{self.calls}: "
            f"{query}"
        )
        return self.index.search(
            query=query,
            num_results=num_results,
        )


def execute_tool(function_call, search_tool):
    args = dict(function_call.args)

    if function_call.name == "search":
        result = search_tool.search(**args)
    else:
        raise ValueError(f"Unknown tool: {function_call.name}")

    return types.Part.from_function_response(
        name=function_call.name,
        response={"results": result},
    )


def get_function_calls(response):
    parts = response.candidates[0].content.parts
    return [
        part.function_call
        for part in parts
        if part.function_call
    ]


def run_agent(index, instructions, question, model=MODEL):
    search_tool = AgentSearchTool(index)
    contents = [
    types.Content(
        role="user",
        parts=[types.Part(text=question)]
    )
]

    while True:
        has_function_calls = False
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=instructions,
                tools=[search_tool.search],
            ),
        )

        contents.append(response.candidates[0].content)
        for part in response.candidates[0].content.parts:
            if part.function_call:
                tool_response = execute_tool(
                        part.function_call,
                        search_tool,
                    )   
                
                contents.append(
                types.Content(
                    role="tool",
                    parts=[tool_response],
                )
            )
                has_function_calls = True

            elif part.text:
                return part.text, search_tool.calls
            
        if not has_function_calls:
            break



# ============================================================
# Demo Steps
# ============================================================
def run_full_document_steps(documents, query):
    print(f"Q1: {len(documents)} documents")

    index = build_index(documents)

    print_section("Q2")
    results = search_documents(index, query)
    print_search_results(results)

    print_section("Q3 - RAG")
    run_and_print_rag(index, query)


def run_chunk_steps(documents, query):
    print_section("Q4 - Chunking")
    chunks = chunk_documents(
        documents,
        size=2000,
        step=1000,
    )
    print(f"Chunks created: {len(chunks)}")

    print_section("Q5 - RAG with chunks")
    chunk_index = build_index(chunks)
    run_and_print_rag(chunk_index, query)

    return chunk_index


def run_agent_step(index):
    print_section("Q6 - Agent")
    answer, search_calls = run_agent(
        index,
        AGENT_INSTRUCTIONS,
        AGENT_QUESTION,
    )

    # print(answer)
    print(f"Search calls: {search_calls}")


# ============================================================
# Main
# ============================================================
def main():
    print("Loading documents...")
    documents = load_data()

    run_full_document_steps(documents, QUERY)
    chunk_index = run_chunk_steps(documents, QUERY)
    run_agent_step(chunk_index)


if __name__ == "__main__":
    main()
