"""
NIM Proxy - Anthropic <-> OpenAI format conversion and streaming.

Converts Claude Code's Anthropic API requests to OpenAI-compatible format
for NVIDIA NIM, and streams NIM responses back as Anthropic SSE events.
"""

import json
import uuid
from typing import AsyncGenerator

from openai import AsyncOpenAI


# ── Anthropic -> OpenAI Request Conversion ────────────────


def convert_request(body: dict, target_model: str) -> dict:
    """Convert Anthropic MessagesRequest to OpenAI ChatCompletion request."""
    messages = []

    # System prompt
    system = body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            parts = []
            for block in system:
                if isinstance(block, dict):
                    parts.append(block.get("text", ""))
                else:
                    parts.append(str(block))
            if parts:
                messages.append({"role": "system", "content": "\n\n".join(parts)})

    # Messages
    for msg in body.get("messages", []):
        role = msg["role"]
        content = msg.get("content", "")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            messages.append({"role": role, "content": str(content)})
            continue

        if role == "user":
            text_parts = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    text_parts.append(block["text"])
                elif btype == "tool_result":
                    if text_parts:
                        messages.append({"role": "user", "content": "\n".join(text_parts)})
                        text_parts = []
                    tc = block.get("content", "")
                    if isinstance(tc, list):
                        tc = "\n".join(
                            c["text"] if isinstance(c, dict) and "text" in c else str(c)
                            for c in tc
                        )
                    prefix = "[ERROR] " if block.get("is_error") else ""
                    messages.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": prefix + (str(tc) if tc else ""),
                    })
                elif btype == "image":
                    text_parts.append("[image content]")
            if text_parts:
                messages.append({"role": "user", "content": "\n".join(text_parts)})

        elif role == "assistant":
            text_parts = []
            thinking_parts = []
            tool_calls = []

            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    text_parts.append(block["text"])
                elif btype == "thinking":
                    thinking_parts.append(block.get("thinking", ""))
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })

            combined = ""
            if thinking_parts:
                combined += "<think>\n" + "\n".join(thinking_parts) + "\n</think>\n"
            combined += "\n".join(text_parts)

            assistant_msg = {"role": "assistant"}
            if combined:
                assistant_msg["content"] = combined
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
                if "content" not in assistant_msg:
                    assistant_msg["content"] = ""
            messages.append(assistant_msg)

    # Build request
    request = {
        "model": target_model,
        "messages": messages,
        "stream": True,
        "max_tokens": min(body.get("max_tokens", 4096), 81920),
    }

    if "temperature" in body:
        request["temperature"] = body["temperature"]
    if "top_p" in body:
        request["top_p"] = body["top_p"]
    if "stop_sequences" in body:
        request["stop"] = body["stop_sequences"]

    # Tools
    tools = body.get("tools")
    if tools:
        converted = []
        for tool in tools:
            ttype = tool.get("type", "")
            if ttype in ("computer_20250124", "bash_20250124", "text_editor_20250124"):
                continue
            converted.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })
        if converted:
            request["tools"] = converted

    # Tool choice
    tc = body.get("tool_choice")
    if tc and isinstance(tc, dict):
        t = tc.get("type", "auto")
        if t == "auto":
            request["tool_choice"] = "auto"
        elif t == "any":
            request["tool_choice"] = "required"
        elif t == "tool":
            request["tool_choice"] = {"type": "function", "function": {"name": tc["name"]}}

    return request


# ── Think Tag Parser ──────────────────────────────────────


class ThinkParser:
    """
    Streaming parser for <think>...</think> tags.
    Yields (type, text) tuples where type is 'thinking' or 'text'.
    Handles tags split across multiple chunks.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self):
        self.state = "init"  # init | thinking | text
        self.buf = ""

    def feed(self, chunk: str):
        self.buf += chunk
        yield from self._process()

    def flush(self):
        if self.buf:
            yield ("thinking" if self.state == "thinking" else "text", self.buf)
            self.buf = ""

    def _process(self):
        while self.buf:
            if self.state == "init":
                stripped = self.buf.lstrip()
                if not stripped:
                    return
                if stripped.startswith(self.OPEN):
                    idx = self.buf.index(self.OPEN)
                    self.buf = self.buf[idx + len(self.OPEN) :]
                    self.state = "thinking"
                    continue
                if len(stripped) < len(self.OPEN) and self.OPEN.startswith(stripped):
                    return  # partial match, wait
                self.state = "text"
                continue

            elif self.state == "thinking":
                idx = self.buf.find(self.CLOSE)
                if idx >= 0:
                    before = self.buf[:idx]
                    if before:
                        yield ("thinking", before)
                    self.buf = self.buf[idx + len(self.CLOSE) :]
                    self.state = "text"
                    continue
                # Check partial close tag at end
                for i in range(min(len(self.CLOSE) - 1, len(self.buf)), 0, -1):
                    if self.CLOSE[:i] == self.buf[-i:]:
                        safe = self.buf[:-i]
                        if safe:
                            yield ("thinking", safe)
                        self.buf = self.buf[-i:]
                        return
                yield ("thinking", self.buf)
                self.buf = ""
                return

            elif self.state == "text":
                yield ("text", self.buf)
                self.buf = ""
                return


# ── SSE Event Builder ─────────────────────────────────────


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Streaming Response ────────────────────────────────────


async def stream_response(
    client: AsyncOpenAI,
    openai_request: dict,
    original_model: str,
) -> AsyncGenerator[str, None]:
    """Call NIM API and yield Anthropic SSE events."""

    request_id = uuid.uuid4().hex[:24]

    # message_start
    yield sse("message_start", {
        "type": "message_start",
        "message": {
            "id": f"msg_{request_id}",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": original_model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        },
    })

    block_idx = 0
    state = "none"  # none | thinking | text | tool
    parser = ThinkParser()
    output_tokens = 0
    has_tools = False

    def _open_block(btype, **extra):
        nonlocal block_idx, state
        block = {"type": btype}
        if btype == "text":
            block["text"] = ""
        elif btype == "thinking":
            block["thinking"] = ""
        elif btype == "tool_use":
            block.update(extra)
            block["input"] = {}
        state = "tool" if btype == "tool_use" else btype
        return sse("content_block_start", {
            "type": "content_block_start",
            "index": block_idx,
            "content_block": block,
        })

    def _close_block():
        nonlocal block_idx, state
        ev = sse("content_block_stop", {
            "type": "content_block_stop",
            "index": block_idx,
        })
        block_idx += 1
        state = "none"
        return ev

    def _emit_parsed(ptype, ptext):
        nonlocal state
        events = ""
        if ptype == "thinking":
            if state == "text":
                events += _close_block()
            if state != "thinking":
                events += _open_block("thinking")
            events += sse("content_block_delta", {
                "type": "content_block_delta",
                "index": block_idx,
                "delta": {"type": "thinking_delta", "thinking": ptext},
            })
        elif ptype == "text":
            if state == "thinking":
                events += _close_block()
            if state != "text":
                events += _open_block("text")
            events += sse("content_block_delta", {
                "type": "content_block_delta",
                "index": block_idx,
                "delta": {"type": "text_delta", "text": ptext},
            })
        return events

    try:
        stream = await client.chat.completions.create(**openai_request)

        async for chunk in stream:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            # Text content
            if delta.content is not None and delta.content != "":
                for ptype, ptext in parser.feed(delta.content):
                    output_tokens += len(ptext) // 4
                    yield _emit_parsed(ptype, ptext)

            # Tool calls
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.id:
                        has_tools = True
                        if state != "none":
                            yield _close_block()
                        yield _open_block(
                            "tool_use",
                            id=tc.id,
                            name=tc.function.name if tc.function else "",
                        )
                    if tc.function and tc.function.arguments:
                        yield sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": tc.function.arguments,
                            },
                        })

            if choice.finish_reason:
                break

        # Flush remaining content
        for ptype, ptext in parser.flush():
            output_tokens += len(ptext) // 4
            yield _emit_parsed(ptype, ptext)

        # Close final block
        if state != "none":
            yield _close_block()

        # Empty response fallback
        if block_idx == 0:
            yield _open_block("text")
            yield _close_block()

        # Final events
        yield sse("message_delta", {
            "type": "message_delta",
            "delta": {
                "stop_reason": "tool_use" if has_tools else "end_turn",
                "stop_sequence": None,
            },
            "usage": {"output_tokens": max(output_tokens, 1)},
        })
        yield sse("message_stop", {"type": "message_stop"})

    except Exception as e:
        error_msg = f"NIM API Error: {type(e).__name__}: {e}"
        if state != "none":
            yield _close_block()
        yield _open_block("text")
        yield sse("content_block_delta", {
            "type": "content_block_delta",
            "index": block_idx,
            "delta": {"type": "text_delta", "text": error_msg},
        })
        yield _close_block()
        yield sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        })
        yield sse("message_stop", {"type": "message_stop"})
