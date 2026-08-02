"""
Answer Generation
-----------------
Stage 7 of the RAG pipeline: given a user question and the retrieved
context chunks, generate a grounded answer.

Supports several pluggable backends so you can use whatever you have
an API key for (or run fully local/free with BACKEND=local):

    BACKEND=anthropic   -> Claude models (needs ANTHROPIC_API_KEY)
    BACKEND=openai      -> GPT models   (needs OPENAI_API_KEY)
    BACKEND=cohere      -> Cohere models(needs COHERE_API_KEY)
    BACKEND=local       -> local Hugging Face flan-t5 model, no API key
"""

import os

SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context from the user's documents. If the answer is not "
    "contained in the context, say you don't know rather than guessing. "
    "Be concise and cite which part of the context you used when helpful."
)


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context_block = "\n\n---\n\n".join(context_chunks)
    return (
        f"Context from the document:\n\n{context_block}\n\n"
        f"---\n\nQuestion: {question}\n\nAnswer based only on the context above:"
    )


class Generator:
    def __init__(self, backend: str | None = None, model: str | None = None):
        self.backend = (backend or os.getenv("BACKEND", "local")).lower()
        self.model = model

    def generate(self, question: str, context_chunks: list[str]) -> str:
        prompt = build_prompt(question, context_chunks)

        if self.backend == "anthropic":
            return self._generate_anthropic(prompt)
        elif self.backend == "openai":
            return self._generate_openai(prompt)
        elif self.backend == "cohere":
            return self._generate_cohere(prompt)
        elif self.backend == "local":
            return self._generate_local(prompt)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ---- Backend implementations ------------------------------------

    def _generate_anthropic(self, prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        model = self.model or "claude-sonnet-4-6"
        response = client.messages.create(
            model=model,
            max_tokens=500,
            system=SYSTEM_INSTRUCTIONS,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _generate_openai(self, prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI()  # reads OPENAI_API_KEY from env
        model = self.model or "gpt-4o-mini"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content

    def _generate_cohere(self, prompt: str) -> str:
        import cohere
        client = cohere.ClientV2()  # reads COHERE_API_KEY from env
        model = self.model or "command-r"
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ],
        )
        return response.message.content[0].text

    _local_pipeline = None  # cached across calls

    def _generate_local(self, prompt: str) -> str:
        # Lazily load a small local seq2seq model so there's a fully free,
        # offline-after-download option that needs no API key at all.
        if Generator._local_pipeline is None:
            from transformers import pipeline
            model_name = self.model or "google/flan-t5-base"
            Generator._local_pipeline = pipeline(
                "text2text-generation", model=model_name
            )
        full_prompt = f"{SYSTEM_INSTRUCTIONS}\n\n{prompt}"
        result = Generator._local_pipeline(
            full_prompt, max_new_tokens=300, do_sample=False
        )
        return result[0]["generated_text"]
