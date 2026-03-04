"""
NIM Proxy - Anthropic <-> OpenAI format conversion and streaming.

Converts Claude Code's Anthropic API requests to OpenAI-compatible format
for NVIDIA NIM, and streams NIM responses back as Anthropic SSE events.
"""

import hashlib
import json
import logging
import uuid
from typing import AsyncGenerator

from openai import AsyncOpenAI


MAX_TOOL_NAME_LEN = 64

# 모델별 컨텍스트 한도 (tokens). 여기에 없으면 DEFAULT_CONTEXT_LIMIT 적용.
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "qwen/qwen3.5-397b-a17b": 202752,
    "deepseek-ai/deepseek-v3-1": 160000,
    "nvidia/glm-5-20b-chat": 205000,
    "z-ai/glm5": 205000,
}
DEFAULT_CONTEXT_LIMIT = 131072
# max_tokens(출력)을 뺀 뒤 추가로 빼는 안전 마진
CONTEXT_SAFETY_MARGIN = 4096


def _shorten_tool_name(name: str) -> str:
    """Shorten a tool name to fit within MAX_TOOL_NAME_LEN using a hash suffix."""
    if len(name) <= MAX_TOOL_NAME_LEN:
        return name
    # Keep as much of the name as possible, append a short hash for uniqueness
    h = hashlib.md5(name.encode()).hexdigest()[:8]
    # Reserve 1 char for separator + 8 for hash = 9
    prefix = name[:MAX_TOOL_NAME_LEN - 9]
    return f"{prefix}_{h}"


# ── Token Estimation & Auto-Truncation ────────────────────


def _estimate_tokens(messages: list[dict]) -> int:
    """메시지 리스트의 대략적인 토큰 수를 추정한다. (1 token ≈ 4 chars)"""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content) // 4
        elif isinstance(content, list):
            total += sum(len(json.dumps(b, ensure_ascii=False)) // 4 for b in content)
        # tool_calls
        if "tool_calls" in msg:
            total += len(json.dumps(msg["tool_calls"], ensure_ascii=False)) // 4
    return total


def _estimate_tools_tokens(tools: list[dict]) -> int:
    """도구 정의의 토큰 수를 추정한다."""
    if not tools:
        return 0
    return len(json.dumps(tools, ensure_ascii=False)) // 4


def truncate_messages(messages: list[dict], max_input_tokens: int, tools_tokens: int = 0) -> list[dict]:
    """메시지를 max_input_tokens 이하로 잘라낸다.

    전략:
    - system 메시지(첫 번째)는 항상 유지
    - 최근 메시지를 최대한 유지
    - 오래된 메시지부터 제거
    - tool 호출/결과 쌍은 함께 제거 (고아 방지)
    """
    if not messages:
        return messages

    available = max_input_tokens - tools_tokens
    current = _estimate_tokens(messages)
    if current <= available:
        return messages

    # system 메시지 분리
    system_msgs = []
    rest_msgs = []
    for msg in messages:
        if msg.get("role") == "system":
            system_msgs.append(msg)
        else:
            rest_msgs.append(msg)

    system_tokens = _estimate_tokens(system_msgs)
    budget = available - system_tokens

    if budget <= 0:
        # system 메시지조차 너무 길면, system을 잘라냄
        sys_text = system_msgs[0].get("content", "") if system_msgs else ""
        max_chars = max_input_tokens * 4 // 2  # system에 절반 할당
        system_msgs = [{"role": "system", "content": sys_text[:max_chars]}]
        system_tokens = _estimate_tokens(system_msgs)
        budget = available - system_tokens

    # 최근 메시지부터 역순으로 추가
    kept = []
    used = 0
    for msg in reversed(rest_msgs):
        msg_tokens = _estimate_tokens([msg])
        if used + msg_tokens <= budget:
            kept.append(msg)
            used += msg_tokens
        else:
            break

    kept.reverse()

    # tool_result가 있는데 대응하는 assistant tool_call이 없으면 제거
    kept = _fix_orphan_tool_messages(kept)

    # 잘린 메시지가 있으면 컨텍스트 압축 표시 삽입
    n_dropped = len(rest_msgs) - len(kept)
    if n_dropped > 0:
        notice = {
            "role": "user",
            "content": f"[시스템: 컨텍스트 한도 초과로 이전 대화 {n_dropped}개 메시지가 생략되었습니다. 최근 대화만 유지됩니다.]",
        }
        return system_msgs + [notice] + kept

    return system_msgs + kept


def _fix_orphan_tool_messages(messages: list[dict]) -> list[dict]:
    """고아 tool 메시지를 제거한다.

    - tool_result가 있는데 대응하는 tool_call이 없으면 제거
    - tool_call이 있는데 대응하는 tool_result가 없으면 제거
    """
    # tool_call ID 수집
    call_ids = set()
    result_ids = set()

    for msg in messages:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                call_ids.add(tc.get("id", ""))
        if msg.get("role") == "tool":
            result_ids.add(msg.get("tool_call_id", ""))

    # 대응되지 않는 것이 있으면 제거
    orphan_call_ids = call_ids - result_ids
    orphan_result_ids = result_ids - call_ids

    if not orphan_call_ids and not orphan_result_ids:
        return messages

    filtered = []
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("tool_call_id", "") in orphan_result_ids:
            continue
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            # 고아 tool_call만 있는 assistant 메시지에서 해당 call 제거
            remaining_calls = [tc for tc in msg["tool_calls"] if tc.get("id", "") not in orphan_call_ids]
            if not remaining_calls and not msg.get("content"):
                continue  # 메시지 전체가 고아 tool_call만이면 제거
            msg = dict(msg)
            if remaining_calls:
                msg["tool_calls"] = remaining_calls
            else:
                msg.pop("tool_calls", None)
        filtered.append(msg)

    return filtered


# ── Anthropic -> OpenAI Request Conversion ────────────────


def convert_request(body: dict, target_model: str) -> tuple[dict, dict]:
    """Convert Anthropic MessagesRequest to OpenAI ChatCompletion request.

    Returns:
        (openai_request, name_map) where name_map maps shortened names back to originals.
    """
    # Build tool name mapping (original -> short, short -> original)
    name_map: dict[str, str] = {}  # short_name -> original_name
    fwd_map: dict[str, str] = {}   # original_name -> short_name

    tools_raw = body.get("tools", [])
    for tool in tools_raw:
        orig = tool.get("name", "")
        short = _shorten_tool_name(orig)
        if short != orig:
            name_map[short] = orig
            fwd_map[orig] = short

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
                    tc_name = block["name"]
                    tc_name = fwd_map.get(tc_name, tc_name)
                    tool_calls.append({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": tc_name,
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

    # Tools 변환
    converted_tools = []
    if tools_raw:
        for tool in tools_raw:
            ttype = tool.get("type", "")
            if ttype in ("computer_20250124", "bash_20250124", "text_editor_20250124"):
                continue
            orig_name = tool["name"]
            short_name = fwd_map.get(orig_name, orig_name)
            converted_tools.append({
                "type": "function",
                "function": {
                    "name": short_name,
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })

    # Auto-truncation: 컨텍스트 한도 초과 시 오래된 메시지 제거
    max_tokens = min(body.get("max_tokens", 4096), 81920)
    context_limit = MODEL_CONTEXT_LIMITS.get(target_model, DEFAULT_CONTEXT_LIMIT)
    max_input_tokens = context_limit - max_tokens - CONTEXT_SAFETY_MARGIN
    tools_tokens = _estimate_tools_tokens(converted_tools)

    pre_count = _estimate_tokens(messages)
    messages = truncate_messages(messages, max_input_tokens, tools_tokens)
    post_count = _estimate_tokens(messages)
    if post_count < pre_count:
        logging.getLogger("nim-proxy").warning(
            f"Auto-truncated: {pre_count} -> {post_count} tokens "
            f"(limit {max_input_tokens}, dropped {pre_count - post_count})"
        )

    # Build request
    request = {
        "model": target_model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
    }

    if "temperature" in body:
        request["temperature"] = body["temperature"]
    if "top_p" in body:
        request["top_p"] = body["top_p"]
    if "stop_sequences" in body:
        request["stop"] = body["stop_sequences"]

    if converted_tools:
        request["tools"] = converted_tools

    # Tool choice
    tc = body.get("tool_choice")
    if tc and isinstance(tc, dict):
        t = tc.get("type", "auto")
        if t == "auto":
            request["tool_choice"] = "auto"
        elif t == "any":
            request["tool_choice"] = "required"
        elif t == "tool":
            tc_name = fwd_map.get(tc["name"], tc["name"])
            request["tool_choice"] = {"type": "function", "function": {"name": tc_name}}

    return request, name_map


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
    name_map: dict[str, str] | None = None,
) -> AsyncGenerator[str, None]:
    """Call NIM API and yield Anthropic SSE events."""
    name_map = name_map or {}

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
                        short_name = tc.function.name if tc.function else ""
                        restored_name = name_map.get(short_name, short_name)
                        yield _open_block(
                            "tool_use",
                            id=tc.id,
                            name=restored_name,
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
