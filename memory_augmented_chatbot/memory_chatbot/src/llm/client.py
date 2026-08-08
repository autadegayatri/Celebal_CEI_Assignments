"""
LLM client abstraction.

`LLM_PROVIDER=anthropic` calls the real Claude API (requires
ANTHROPIC_API_KEY in the environment).

`LLM_PROVIDER=mock` (the default) uses a deterministic, template-based
generator that composes an answer directly from the retrieved
RAG/KG/memory/tool context. This lets the entire orchestration graph run
end-to-end -- and be graded/tested -- without any API key or network
access, while still producing grounded, non-hallucinated answers.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from src import config


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class AnthropicLLMClient(LLMClient):
    def __init__(self, api_key: str = config.ANTHROPIC_API_KEY, model: str = config.ANTHROPIC_MODEL):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set. Set it in your environment or .env file.")
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class GroqLLMClient(LLMClient):
    """Simple Groq.ai API client. Requires `GROQ_API_KEY` and `GROQ_MODEL` in env/.env.

    This implementation uses the HTTP endpoint at
    `https://api.groq.ai/v1/models/{model}/generate` which accepts a JSON
    payload with `prompt` and returns a text response. Replace or extend
    this client if your Groq deployment uses a different API shape.
    """

    def __init__(self, api_key: str = config.GROQ_API_KEY, model: str = config.GROQ_MODEL):
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Set it in your environment or .env file.")
        try:
            import requests
        except Exception as exc:
            raise ImportError("Install the 'requests' package to use the Groq LLM client.") from exc
        self.requests = requests
        self.api_key = api_key
        self.model = model
        self.endpoint = f"https://api.groq.ai/v1/models/{self.model}/generate"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        payload = {"prompt": f"{system_prompt}\n\n{user_prompt}", "max_tokens": 1024}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        resp = self.requests.post(self.endpoint, json=payload, headers=headers, timeout=30)
        try:
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise RuntimeError(f"Groq API request failed: {exc} - status={getattr(resp, 'status_code', None)} body={getattr(resp, 'text', None)}")

        # Attempt to extract text from common response shapes.
        if isinstance(data, dict):
            if "text" in data:
                return data["text"]
            if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                first = data["choices"][0]
                if isinstance(first, dict) and "text" in first:
                    return first["text"]
        # Fallback to raw string
        return str(data)


class MockLLMClient(LLMClient):
    """
    Deterministic, extractive fallback. Does not call any external API.

    Strategy: pull the most query-relevant sentences out of the supplied
    context (RAG passages + KG facts + memory) and compose them into a
    short, direct answer. This keeps the system honest (no hallucination)
    and fully offline-runnable.
    """

    # Sections are listed most-specific-first: if a tool was called or a KG
    # lookup succeeded, that's almost always what the query is actually
    # asking for -- so it should win over generic RAG passages or stale
    # conversation history, rather than being diluted by word-overlap
    # scoring across the whole blended context.
    _SECTION_PRIORITY = ["TOOL RESULT", "KNOWLEDGE GRAPH FACTS", "KNOWLEDGE BASE", "USER MEMORY", "RECENT CONVERSATION"]
    _SECTION_HEADER_RE = re.compile(
        r"^(TOOL RESULT|KNOWLEDGE GRAPH FACTS|KNOWLEDGE BASE|USER MEMORY|RECENT CONVERSATION):\s*$",
        re.MULTILINE,
    )

    def _split_sections(self, context: str) -> dict[str, str]:
        sections: dict[str, str] = {}
        matches = list(self._SECTION_HEADER_RE.finditer(context))
        if not matches:
            return sections

        for i, m in enumerate(matches):
            name = m.group(1)
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(context)
            sections[name] = context[start:end].strip()
        return sections

    def _best_sentences(self, text: str, query_words: set[str], top_n: int = 3) -> list[str]:
        candidates = re.split(r"(?<=[.!?])\s+|\n", text)
        scored = []
        for sentence in candidates:
            sentence = sentence.strip(" -\n")
            if len(sentence) < 10:
                continue
            words = {w.lower() for w in re.findall(r"\w+", sentence)}
            overlap = len(words & query_words)
            scored.append((overlap, sentence))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [s for score, s in scored[:top_n] if score > 0]
        if not top:
            top = [s for _, s in scored[:1] if len(s) > 10]
        return top

    def _format_kg_facts(self, text: str) -> list[str]:
        formatted: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(r"(.+?)\s--\[(.+?)\]-->\s(.+)$", line)
            if match:
                subject, relation, obj = match.groups()
                formatted.append(f"{subject} {relation} {obj}.")
        return formatted

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        context_match = re.search(r"CONTEXT:\s*(.*?)\n\nUSER QUESTION:", user_prompt, re.DOTALL)
        question_match = re.search(r"USER QUESTION:\s*(.*)", user_prompt, re.DOTALL)

        context = context_match.group(1).strip() if context_match else ""
        question = question_match.group(1).strip() if question_match else user_prompt

        if not context:
            return (
                "I don't have enough information in my knowledge base, knowledge graph, "
                "or memory to answer that confidently yet. Could you rephrase, or is this "
                "something I should look up with a tool?"
            )

        query_words = {w.lower() for w in re.findall(r"\w+", question) if len(w) > 2}
        sections = self._split_sections(context)

        # Answer directly from the highest-priority non-empty, non-history
        # section available (tool output / KG facts / RAG passages).
        for section_name in self._SECTION_PRIORITY:
            section_text = sections.get(section_name, "")
            if not section_text:
                continue
            if section_name == "TOOL RESULT":
                return section_text  # tool output is already a direct answer
            if section_name == "RECENT CONVERSATION":
                continue  # never answer purely from stale history
            if section_name == "KNOWLEDGE GRAPH FACTS":
                formatted = self._format_kg_facts(section_text)
                if formatted:
                    return " ".join(formatted[:2])
            best = self._best_sentences(section_text, query_words)
            if best:
                return " ".join(best)

        return "I found some related context but couldn't extract a confident answer from it."


def get_llm_client(provider: str | None = None) -> LLMClient:
    provider = provider or config.LLM_PROVIDER
    if provider == "anthropic":
        return AnthropicLLMClient()
    if provider == "groq":
        return GroqLLMClient()
    return MockLLMClient()
