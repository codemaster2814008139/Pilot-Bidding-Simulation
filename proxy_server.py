#!/usr/bin/env python3
"""
Local backend for pilot_bidding POC.

- Serves pilot_bidding_poc.html from this directory (same origin as fetch).
- Proxies POST /api/llm → OpenAI or Anthropic using keys from the machine only.

Secrets: `.env` beside this script (see `env.example`) or exported env vars:

  OPENAI_API_KEY / ANTHROPIC_API_KEY   — whichever provider you use
  LLM_PROVIDER=anthropic|openai         — optional; inferred from keys if omitted
  ANTHROPIC_MODEL=... OPENAI_MODEL=...  — model ids (defaults: Sonnet 4.5, gpt-4o)

The browser never chooses provider/model; `/api/llm` ignores client overrides.

Install: pip install openai anthropic python-dotenv
Run: python proxy_server.py

Optional: PILOT_BIDDING_PROXY_PORT=8765

Only binds to loopback — not reachable from other machines unless you tunnel.
"""

from __future__ import annotations

import json
import os
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv

    _HERE = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(_HERE, ".env"))
except ImportError:
    pass

from llm_api import call_llm

_PORT = int(os.environ.get("PILOT_BIDDING_PROXY_PORT", "8765"))
_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o").strip() or "gpt-4o"
_DEFAULT_ANTHROPIC_MODEL = (
    os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929").strip()
    or "claude-sonnet-4-5-20250929"
)


def resolved_llm() -> tuple[str, str]:
    """Provider and model selected only from environment."""
    explicit = os.environ.get("LLM_PROVIDER", "").strip().lower()
    has_o = bool(os.getenv("OPENAI_API_KEY"))
    has_a = bool(os.getenv("ANTHROPIC_API_KEY"))

    if explicit == "anthropic":
        provider = "anthropic"
    elif explicit == "openai":
        provider = "openai"
    elif has_a and not has_o:
        provider = "anthropic"
    elif has_o and not has_a:
        provider = "openai"
    else:
        # Both keys present (or neither): neutral default favors OpenAI ordering;
        # set LLM_PROVIDER when you use both APIs.
        provider = "openai"

    model = _DEFAULT_ANTHROPIC_MODEL if provider == "anthropic" else _DEFAULT_OPENAI_MODEL
    return provider, model


def _provider_ready(provider: str) -> bool:
    if provider == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    return bool(os.getenv("OPENAI_API_KEY"))


class POCRequestHandler(SimpleHTTPRequestHandler):
    """Static files under _ROOT + JSON API endpoints."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, directory=_ROOT, **kwargs)

    def log_message(self, fmt: str, *log_args: object) -> None:
        sys.stderr.write("%s - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % log_args))

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/api/llm", "/api/health"):
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            pv, mv = resolved_llm()
            self._serve_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "provider": pv,
                    "model": mv,
                    "provider_ready": _provider_ready(pv),
                    "openai_key_configured": bool(os.getenv("OPENAI_API_KEY")),
                    "anthropic_key_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
                },
            )
            return
        if parsed.path in ("/", ""):
            self.path = "/pilot_bidding_poc.html"
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/llm":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as e:
            self._serve_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"Invalid JSON: {e}"})
            return

        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            self._serve_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Missing non-empty \"prompt\""})
            return

        provider, model = resolved_llm()
        if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            self._serve_json(
                HTTPStatus.BAD_GATEWAY,
                {"ok": False, "error": 'Provider "anthropic" selected but ANTHROPIC_API_KEY is not set.'},
            )
            return
        if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            self._serve_json(
                HTTPStatus.BAD_GATEWAY,
                {"ok": False, "error": 'Provider "openai" selected but OPENAI_API_KEY is not set.'},
            )
            return

        try:
            text = call_llm(prompt, model=model, provider=provider)
        except RuntimeError as e:
            self._serve_json(HTTPStatus.BAD_GATEWAY, {"ok": False, "error": str(e)})
            return

        self._serve_json(HTTPStatus.OK, {"ok": True, "text": text})

    def _serve_json(self, status: HTTPStatus | int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", _PORT), POCRequestHandler)
    pv, mv = resolved_llm()
    print(f"Pilot bidding proxy + UI: http://127.0.0.1:{_PORT}/pilot_bidding_poc.html", file=sys.stderr)
    print(
        "LLM routing: provider=%s model=%s "
        "(set LLM_PROVIDER, ANTHROPIC_MODEL, OPENAI_MODEL; keys in .env)."
        % (pv, mv),
        file=sys.stderr,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.", file=sys.stderr)


if __name__ == "__main__":
    main()
