"""
Thin, provider-agnostic LLM client.

Supports two zero-cost backends, chosen by whichever is configured:

  1. Groq free tier (default) — OpenAI-compatible chat completions API.
     Set GROQ_API_KEY. Model defaults to a fast free-tier model.
  2. Ollama (local) — for fully offline/zero-signup use.
     Set LLM_PROVIDER=ollama (and optionally OLLAMA_HOST, OLLAMA_MODEL).

Both are free/no-cost options explicitly allowed by the assessment brief.
Only one HTTP call shape is needed because both expose an
OpenAI-compatible /chat/completions-style endpoint.

This module deliberately knows nothing about tickets or business logic —
it only knows how to ask a model to return JSON and parse that JSON
robustly (models occasionally wrap JSON in prose or code fences).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()  # "groq" | "ollama"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")

REQUEST_TIMEOUT_S = 30


class LLMNotConfiguredError(RuntimeError):
    """Raised when no usable LLM backend is configured."""


class LLMResponseError(RuntimeError):
    """Raised when the LLM response can't be parsed as expected."""


def _extract_json(text: str) -> dict:
    """
    Models sometimes answer with ```json ... ``` fences or a short
    preamble sentence before the JSON object. Strip those defensively
    rather than trusting strict JSON-only compliance.
    """
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"Could not parse JSON from LLM output: {text[:500]!r}") from exc


def _call_groq(system_prompt: str, user_prompt: str) -> str:
    if not GROQ_API_KEY:
        raise LLMNotConfiguredError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
            "and set it as an environment variable, or set LLM_PROVIDER=ollama to use "
            "a local model instead."
        )
    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _call_ollama(system_prompt: str, user_prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


def complete_json(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    """Send a system+user prompt to the configured LLM and parse a JSON object back."""
    if LLM_PROVIDER == "ollama":
        raw = _call_ollama(system_prompt, user_prompt)
    elif LLM_PROVIDER == "groq":
        raw = _call_groq(system_prompt, user_prompt)
    else:
        raise LLMNotConfiguredError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r} (use 'groq' or 'ollama')")
    return _extract_json(raw)


def llm_status() -> dict:
    """Report which backend is configured, for the /health endpoint."""
    if LLM_PROVIDER == "groq":
        return {"provider": "groq", "model": GROQ_MODEL, "configured": bool(GROQ_API_KEY)}
    if LLM_PROVIDER == "ollama":
        return {"provider": "ollama", "model": OLLAMA_MODEL, "configured": True}
    return {"provider": LLM_PROVIDER, "configured": False}
