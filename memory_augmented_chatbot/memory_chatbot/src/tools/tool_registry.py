"""
Tool registry: real-time / dynamic-intelligence tools the LangGraph "Tool"
node can invoke when a query needs live information the static RAG/KG
layers can't provide (current time, quick arithmetic, live web lookups).

Each tool is a plain Python callable registered under a name + a short
description (used by the router to decide whether a query needs it).
Add new tools by writing a function and calling `register`.
"""

from __future__ import annotations

import ast
import datetime
import operator
from dataclasses import dataclass
from typing import Callable

from memory_augmented_chatbot.memory_chatbot.src import config


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]


_REGISTRY: dict[str, Tool] = {}


def register(name: str, description: str):
    def decorator(func: Callable[[str], str]):
        _REGISTRY[name] = Tool(name=name, description=description, func=func)
        return func

    return decorator


def get_tool(name: str) -> Tool | None:
    return _REGISTRY.get(name)


def list_tools() -> list[Tool]:
    return list(_REGISTRY.values())


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------
@register("datetime", "Get the current date and time.")
def current_datetime(_query: str) -> str:
    now = datetime.datetime.now()
    return f"Current date and time: {now.strftime('%A, %d %B %Y, %H:%M:%S')}"


_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Unsupported constant")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


@register("calculator", "Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4)'.")
def calculator(query: str) -> str:
    import re

    match = re.search(r"[-0-9\.\s\+\-\*/\(\)%\^]{2,}", query)
    expr = match.group(0).strip() if match else query.strip()
    expr = expr.replace("^", "**")
    try:
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval(tree.body)
        return f"{expr.strip()} = {result}"
    except Exception:
        return "I couldn't parse that as a valid arithmetic expression."


@register(
    "web_search",
    "Search the live web for real-time / current-events information not "
    "covered by the static knowledge base.",
)
def web_search(query: str) -> str:
    if not config.WEB_SEARCH_ENABLED:
        return (
            "Live web search is disabled in this environment "
            "(set WEB_SEARCH_ENABLED=true and configure a search API key to enable it). "
            f"Query was: '{query}'"
        )
    # Placeholder for a real integration (SerpAPI, Bing, Tavily, etc.)
    # Kept intentionally pluggable: swap this body for a real HTTP call.
    return f"[web_search stub] No live results available for: '{query}'"
