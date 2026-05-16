"""
LLM HTTP clients — keys come from environment variables only:

  OPENAI_API_KEY       — Chat Completions API
  ANTHROPIC_API_KEY    — Messages API
"""

from __future__ import annotations

import os


def call_anthropic(prompt: str, model: str = "claude-haiku-4-5-20251001") -> str:
    """Call Anthropic API. Requires: pip install anthropic"""
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        msg = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except ImportError:
        raise RuntimeError("pip install anthropic") from None
    except KeyError:
        raise RuntimeError("Set ANTHROPIC_API_KEY environment variable") from None


def call_openai(prompt: str, model: str = "gpt-4o") -> str:
    """Call OpenAI API. Requires: pip install openai"""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        )
        return resp.choices[0].message.content or ""
    except ImportError:
        raise RuntimeError("pip install openai") from None
    except KeyError:
        raise RuntimeError("Set OPENAI_API_KEY environment variable") from None


def call_llm(prompt: str, model: str = "gpt-4o", provider: str = "openai") -> str:
    if provider == "anthropic":
        return call_anthropic(prompt, model)
    return call_openai(prompt, model)
