#!/usr/bin/env python3
"""
VS Code Bridge Test Script
==========================
Run this from the project root to verify the VS Code bridge is reachable
and can respond to a test prompt.

Usage:
    python test_vscode_bridge.py
    python test_vscode_bridge.py --model gpt-4o --prompt "Say hello in one sentence"
    python test_vscode_bridge.py --url http://127.0.0.1:8765/v1 --stream

Exit codes:
    0  bridge is online and responded successfully
    1  bridge offline or request failed
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

import httpx


BRIDGE_URL = "http://127.0.0.1:8765/v1"
API_KEY    = "vscode"
TIMEOUT    = 30.0


async def preflight(base_url: str, api_key: str) -> list[str]:
    """Hit /v1/models and return available model IDs."""
    url = f"{base_url}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=6.0) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return [m["id"] for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]


async def chat_once(base_url: str, api_key: str, model: str, prompt: str, system: str | None) -> str:
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {"model": model, "messages": messages, "stream": False}
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


async def chat_stream(base_url: str, api_key: str, model: str, prompt: str, system: str | None) -> None:
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {"model": model, "messages": messages, "stream": True}
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        evt = json.loads(data)
                        chunk = evt["choices"][0]["delta"].get("content", "")
                        if chunk:
                            print(chunk, end="", flush=True)
                    except Exception:
                        pass
    print()  # newline after stream


async def run_test(args: argparse.Namespace) -> int:
    base_url = args.url.rstrip("/")

    # ── 1. Preflight ──────────────────────────────────────────────────────────
    print(f"\n🔍  Checking bridge at {base_url}/models …")
    try:
        models = await preflight(base_url, args.api_key)
    except Exception as exc:
        print(f"\n✗  Bridge OFFLINE or unreachable.\n   {exc}")
        print("\n   Make sure the VS Code bridge extension is running,")
        print("   then retry.")
        return 1

    print(f"✓  Bridge online. {len(models)} model(s) available:")
    for m in models:
        print(f"     • {m}")

    # ── 2. Pick model ─────────────────────────────────────────────────────────
    model = args.model
    if not model:
        if models:
            model = models[0]
            print(f"\n   No --model specified. Using first available: {model}")
        else:
            print("\n✗  No models returned by bridge. Cannot send test prompt.")
            return 1

    if model not in models:
        print(f"\n⚠  Warning: '{model}' not listed by /models — sending anyway.")

    prompt = args.prompt or "In exactly one sentence, confirm you are reachable."

    # ── 3. Send request ───────────────────────────────────────────────────────
    print(f"\n📤  Sending test prompt to model '{model}' …")
    print(f"    Prompt: {prompt}")
    print()

    try:
        if args.stream:
            print("📥  Streaming response:")
            print("─" * 40)
            await chat_stream(base_url, args.api_key, model, prompt, args.system)
            print("─" * 40)
        else:
            reply = await chat_once(base_url, args.api_key, model, prompt, args.system)
            print("📥  Response:")
            print("─" * 40)
            print(reply)
            print("─" * 40)
    except Exception as exc:
        print(f"\n✗  Request failed: {exc}")
        return 1

    print("\n✓  VS Code bridge test PASSED.\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test the VS Code language-model bridge"
    )
    parser.add_argument("--url", default=BRIDGE_URL, help=f"Bridge base URL (default: {BRIDGE_URL})")
    parser.add_argument("--api-key", default=API_KEY, help="API key (default: 'vscode')")
    parser.add_argument("--model", default=None, help="Model ID to test (default: first from /models)")
    parser.add_argument("--prompt", default=None, help="Test prompt to send")
    parser.add_argument("--system", default=None, help="Optional system message")
    parser.add_argument("--stream", action="store_true", help="Test streaming mode")
    args = parser.parse_args()

    return asyncio.run(run_test(args))


if __name__ == "__main__":
    sys.exit(main())
