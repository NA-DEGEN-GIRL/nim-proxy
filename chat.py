#!/usr/bin/env python3
"""
Simple chat CLI for NIM proxy.

Usage:
    python chat.py

Connects to the running NIM proxy and provides a conversational interface.
"""

import os
import json

import httpx
from dotenv import load_dotenv

load_dotenv()

PROXY_URL = os.getenv("PROXY_URL", os.getenv("ANTHROPIC_BASE_URL", "http://localhost:8082"))
MODEL = os.getenv("CHAT_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "4096"))
SYSTEM = os.getenv("CHAT_SYSTEM", "You are a helpful assistant. Answer concisely.")

DIM = "\033[2m"
BOLD = "\033[1m"
RED = "\033[31m"
CYAN = "\033[36m"
RESET = "\033[0m"

history: list[dict] = []
show_thinking = True


def chat(user_input: str):
    history.append({"role": "user", "content": user_input})

    body = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM,
        "messages": history,
        "stream": True,
    }

    full_text = ""
    in_thinking = False

    with httpx.Client(timeout=180) as client:
        with client.stream(
            "POST",
            f"{PROXY_URL}/v1/messages",
            json=body,
            headers={"x-api-key": "dummy", "anthropic-version": "2023-06-01"},
        ) as resp:
            if resp.status_code != 200:
                resp.read()
                print(f"{RED}HTTP {resp.status_code}: {resp.text[:200]}{RESET}")
                history.pop()
                return

            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue

                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                etype = data.get("type", "")

                if etype == "content_block_start":
                    block = data.get("content_block", {})
                    if block.get("type") == "thinking":
                        in_thinking = True
                        if show_thinking:
                            print(f"{DIM}[thinking] ", end="", flush=True)

                elif etype == "content_block_delta":
                    delta = data.get("delta", {})
                    dt = delta.get("type", "")
                    if dt == "text_delta":
                        t = delta.get("text", "")
                        print(t, end="", flush=True)
                        full_text += t
                    elif dt == "thinking_delta" and show_thinking:
                        print(delta.get("thinking", ""), end="", flush=True)

                elif etype == "content_block_stop":
                    if in_thinking:
                        if show_thinking:
                            print(RESET)
                        in_thinking = False

                elif etype == "message_stop":
                    break

    print()
    history.append({"role": "assistant", "content": full_text})


def main():
    global show_thinking

    print(f"{CYAN}NIM Chat{RESET}  proxy={PROXY_URL}  model={MODEL}")
    print("─" * 50)

    while True:
        try:
            user_input = input(f"\n{BOLD}> {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input in ("/quit", "/exit", "/q"):
            break
        elif user_input == "/clear":
            history.clear()
            print("대화 초기화")
            continue
        elif user_input == "/think":
            show_thinking = not show_thinking
            print(f"thinking 표시: {'ON' if show_thinking else 'OFF'}")
            continue
        elif user_input == "/help":
            print("/clear   대화 초기화")
            print("/think   thinking 표시 토글")
            print("/quit    종료")
            continue

        try:
            chat(user_input)
        except httpx.ConnectError:
            print(f"{RED}프록시 연결 실패 - server.py 실행 확인{RESET}")
            if history and history[-1]["role"] == "user":
                history.pop()
        except Exception as e:
            print(f"{RED}Error: {e}{RESET}")
            if history and history[-1]["role"] == "user":
                history.pop()


if __name__ == "__main__":
    main()
