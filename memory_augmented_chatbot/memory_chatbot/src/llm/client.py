"""
LLM client: Groq API integration.

`LLM_PROVIDER=groq` (the default) calls the real Groq API via the official
`groq` Python SDK, so the chatbot actually *generates* answers -- it isn't
limited to copying/extracting sentences out of retrieved context. The
model is told to prioritize the supplied CONTEXT when one is provided
(RAG / Knowledge Graph / Tool results), but is free to draw on its own
knowledge to answer naturally, fill gaps, and handle general questions
that the knowledge base doesn't cover.

`LLM_PROVIDER=mock` is an offline, template-based fallback with no
external calls -- used only for tests / environments without a
GROQ_API_KEY, never as the default.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from memory_augmented_chatbot.memory_chatbot.src import config


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class GroqLLMClient(LLMClient):
    """Real Groq API client (OpenAI-compatible chat completions), via the
    official `groq` SDK. Requires GROQ_API_KEY."""

    def __init__(self, api_key: str = config.GROQ_API_KEY, model: str = config.GROQ_MODEL):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Get a key from https://console.groq.com/keys "
                "and set it in your environment or .env file."
            )
        try:
            from groq import Groq
        except ImportError as exc:
            raise ImportError("Run `pip install groq` to use the Groq LLM client.") from exc

        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""


class MockLLMClient(LLMClient):
    """
    Deterministic, extractive, offline fallback -- no external API calls.
    Used only when LLM_PROVIDER=mock (tests / no API key available). Pulls
    the most query-relevant sentences out of the supplied context so the
    rest of the pipeline (routing, retrieval, memory) can still be
    exercised end-to-end without network access.
    """

    _SECTION_PRIORITY = ["TOOL RESULT", "KNOWLEDGE GRAPH FACTS", "KNOWLEDGE BASE", "USER MEMORY", "RECENT CONVERSATION"]
    _SECTION_HEADER_RE = re.compile(
        r"^(TOOL RESULT|KNOWLEDGE GRAPH FACTS|KNOWLEDGE BASE|USER MEMORY|RECENT CONVERSATION):\s*$",
        re.MULTILINE,
    )

    def _split_sections(self, context: str) -> dict[str, str]:
        sections: dict[str, str] = {}
        matches = list(self._SECTION_HEADER_RE.finditer(context))
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

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        context_match = re.search(r"CONTEXT:\s*(.*?)\n\nUSER QUESTION:", user_prompt, re.DOTALL)
        question_match = re.search(r"USER QUESTION:\s*(.*)", user_prompt, re.DOTALL)
        context = context_match.group(1).strip() if context_match else ""
        question = question_match.group(1).strip() if question_match else user_prompt

        if not context:
            return "I don't have relevant context for that yet -- ask me something else, or try rephrasing."

        query_words = {w.lower() for w in re.findall(r"\w+", question) if len(w) > 2}
        sections = self._split_sections(context)

        for section_name in self._SECTION_PRIORITY:
            section_text = sections.get(section_name, "")
            if not section_text or section_name == "RECENT CONVERSATION":
                continue
            if section_name == "TOOL RESULT":
                return section_text
            best = self._best_sentences(section_text, query_words)
            if best:
                return " ".join(best)

        return "I found some related context but couldn't extract a confident answer from it."


def get_llm_client(provider: str | None = None) -> LLMClient:
    provider = provider or config.LLM_PROVIDER
    if provider == "mock":
        return MockLLMClient()
    return GroqLLMClient()
