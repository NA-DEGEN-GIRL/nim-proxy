#!/usr/bin/env python3
"""
NIM Chat CLI - Gemini CLI style.

Usage:
    nim "질문"                     # one-shot
    echo "질문" | nim              # pipe
    nim -s "너는 번역가다" "번역해줘"  # system prompt override
    nim                            # interactive mode

Install as CLI:
    ln -s $(pwd)/chat.py ~/.local/bin/nim
"""

import argparse
import os
import sys
import json

import httpx
from dotenv import load_dotenv

load_dotenv()

PROXY_URL = os.getenv("PROXY_URL", os.getenv("ANTHROPIC_BASE_URL", "http://localhost:8082"))
MODEL = os.getenv("CHAT_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "4096"))
SYSTEM = os.getenv("CHAT_SYSTEM", "You are a helpful assistant. Answer concisely.")
TIMEOUT = int(os.getenv("CHAT_TIMEOUT", "180"))

# Colors (disabled when piping output)
_tty = sys.stdout.isatty()
DIM = "\033[2m" if _tty else ""
BOLD = "\033[1m" if _tty else ""
RED = "\033[31m" if _tty else ""
CYAN = "\033[36m" if _tty else ""
GREEN = "\033[32m" if _tty else ""
YELLOW = "\033[33m" if _tty else ""
RESET = "\033[0m" if _tty else ""

history: list[dict] = []


def stream_chat(user_input: str, *, system: str = SYSTEM, show_thinking: bool = True) -> str:
    """Send a message and stream the response. Returns full response text."""
    history.append({"role": "user", "content": user_input})

    body = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": history,
        "stream": True,
    }

    full_text = ""
    in_thinking = False

    with httpx.Client(timeout=TIMEOUT) as client:
        with client.stream(
            "POST",
            f"{PROXY_URL}/v1/messages",
            json=body,
            headers={"x-api-key": "dummy", "anthropic-version": "2023-06-01"},
        ) as resp:
            if resp.status_code != 200:
                resp.read()
                msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                history.pop()
                raise RuntimeError(msg)

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
    return full_text


def interactive(system: str):
    """Interactive chat mode."""
    show_thinking = True

    print(f"{CYAN}NIM Chat{RESET}  model={GREEN}{MODEL}{RESET}")
    print(f"{DIM}{PROXY_URL}{RESET}")
    print(f"{DIM}/help for commands{RESET}")
    print("─" * 50)

    while True:
        try:
            user_input = input(f"\n{BOLD}> {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}Bye!{RESET}")
            break

        if not user_input:
            continue
        if user_input in ("/quit", "/exit", "/q"):
            break
        elif user_input == "/clear":
            history.clear()
            print(f"{DIM}대화 초기화{RESET}")
            continue
        elif user_input == "/think":
            show_thinking = not show_thinking
            print(f"{DIM}thinking: {'ON' if show_thinking else 'OFF'}{RESET}")
            continue
        elif user_input == "/model":
            print(f"{DIM}{MODEL}{RESET}")
            continue
        elif user_input == "/help":
            print(f"{DIM}/clear   대화 초기화")
            print("/think   thinking 표시 토글")
            print("/model   현재 모델 확인")
            print(f"/quit    종료{RESET}")
            continue

        try:
            stream_chat(user_input, system=system, show_thinking=show_thinking)
        except httpx.ConnectError:
            print(f"{RED}프록시 연결 실패 - server.py 실행 확인{RESET}")
            if history and history[-1]["role"] == "user":
                history.pop()
        except Exception as e:
            print(f"{RED}{e}{RESET}")
            if history and history[-1]["role"] == "user":
                history.pop()


def main():
    parser = argparse.ArgumentParser(
        description="NIM Chat CLI",
        usage="nim [options] [prompt]",
    )
    parser.add_argument("prompt", nargs="*", help="prompt (없으면 interactive mode)")
    parser.add_argument("-s", "--system", default=None, help="system prompt override")
    parser.add_argument("-m", "--model", default=None, help="model override")
    parser.add_argument("-t", "--max-tokens", type=int, default=None, help="max tokens")
    parser.add_argument("-T", "--timeout", type=int, default=None, help="요청 타임아웃(초, 기본 180)")
    parser.add_argument("--no-think", action="store_true", help="thinking 숨김")
    args = parser.parse_args()

    global MODEL, MAX_TOKENS, TIMEOUT
    if args.model:
        MODEL = args.model
    if args.max_tokens:
        MAX_TOKENS = args.max_tokens
    if args.timeout:
        TIMEOUT = args.timeout

    system = args.system or SYSTEM

    # Determine prompt source: args > stdin pipe > interactive
    prompt = " ".join(args.prompt) if args.prompt else None

    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()

    if prompt:
        # One-shot mode
        try:
            stream_chat(prompt, system=system, show_thinking=not args.no_think)
        except httpx.ConnectError:
            print(f"{RED}프록시 연결 실패 - server.py 실행 확인{RESET}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"{RED}{e}{RESET}", file=sys.stderr)
            sys.exit(1)
    else:
        # Interactive mode
        interactive(system)


if __name__ == "__main__":
    main()
