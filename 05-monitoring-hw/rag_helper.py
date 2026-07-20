from google.genai import types


INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model='gemini-3.1-flash-lite',
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        return self.index.search(query, num_results=num_results)

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc['filename'])
            lines.append(doc['content'])
            lines.append('')

        return '\n'.join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        response = self.llm_client.models.generate_content(
        model=self.model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=self.instructions,
            ),
        )
        return response

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)
        return response.text
    
from opentelemetry import trace
tracer = trace.get_tracer("llm-zoomcamp")

class RAGTraced(RAGBase):

    def rag(self, query):
        with tracer.start_as_current_span("rag") as span:
            # span.set_attribute("query", query)
            result = super().rag(query)
            return result

    def search(self, query, num_results=5):
        with tracer.start_as_current_span("search") as span:
            # span.set_attribute("query", query)
            return super().search(query, num_results=num_results)

    def llm(self, prompt):
        with tracer.start_as_current_span("llm") as span:
            # span.set_attribute("prompt_length", len(prompt))
            response = super().llm(prompt)

            usage = response.usage_metadata
            input_tokens = usage.prompt_token_count
            output_tokens = usage.candidates_token_count
            cost = calculate_cost(input_tokens, output_tokens)

            span.set_attribute("input_tokens", input_tokens)
            span.set_attribute("output_tokens", output_tokens)
            span.set_attribute("cost", cost)
            return response


def calculate_cost(input_tokens, output_tokens):
    input_price_per_million = 0.75
    output_price_per_million = 4.50

    input_cost = (input_tokens / 1_000_000) * input_price_per_million
    output_cost = (output_tokens / 1_000_000) * output_price_per_million

    return input_cost + output_cost
