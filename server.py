"""
NIM Proxy Server for Claude Code

Minimal proxy that translates between Anthropic API format and
OpenAI-compatible format for NVIDIA NIM.

Usage:
    python server.py

Then launch Claude Code with:
    ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_AUTH_TOKEN=dummy claude
"""

import os
import json
import logging

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from openai import AsyncOpenAI
from dotenv import load_dotenv
import httpx

from proxy import convert_request, stream_response, _estimate_text_tokens

load_dotenv()

# ── Config ────────────────────────────────────────────────

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen/qwen3.5-397b-a17b")
MODEL_MAP: dict = json.loads(os.getenv("MODEL_MAP", "{}"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8082"))
TIMEOUT = int(os.getenv("TIMEOUT", "600"))  # seconds, generous for slow models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("nim-proxy")

# ── OpenAI Client (for NIM) ──────────────────────────────

client = AsyncOpenAI(
    base_url=NIM_BASE_URL,
    api_key=NVIDIA_API_KEY,
    timeout=httpx.Timeout(TIMEOUT, connect=10),
)


# ── Model Resolution ─────────────────────────────────────


def resolve_model(model: str) -> str:
    """Map Claude model name to NIM model."""
    if model in MODEL_MAP:
        return MODEL_MAP[model]
    return DEFAULT_MODEL


# ── FastAPI App ───────────────────────────────────────────

app = FastAPI(title="NIM Proxy for Claude Code")


@app.post("/v1/messages")
async def create_message(request: Request):
    body = await request.json()
    original_model = body.get("model", "")
    target_model = resolve_model(original_model)

    # 디버그: 요청 메타 정보 로깅
    msgs = body.get("messages", [])
    tool_count = len(body.get("tools", []))
    log.info(f"{original_model} -> {target_model} | msgs={len(msgs)} tools={tool_count} max_tokens={body.get('max_tokens', '?')} system={'yes' if body.get('system') else 'no'}")
    for i, m in enumerate(msgs[:5]):
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, str):
            preview = content[:100]
        elif isinstance(content, list):
            preview = str([c.get("type", "?") for c in content[:3]])
        else:
            preview = str(content)[:100]
        log.info(f"  msg[{i}] {role}: {preview}")
    if len(msgs) > 5:
        log.info(f"  ... +{len(msgs)-5} more messages")

    openai_request, name_map = convert_request(body, target_model)

    # 변환 후 요청 크기 로깅
    oai_msgs = openai_request.get("messages", [])
    total_chars = sum(len(str(m.get("content", ""))) for m in oai_msgs)
    log.info(f"  -> converted: msgs={len(oai_msgs)} tools={len(openai_request.get('tools', []))} ~{total_chars} chars")

    if name_map:
        log.info(f"Shortened {len(name_map)} tool name(s): {list(name_map.values())}")

    return StreamingResponse(
        content=stream_response(client, openai_request, original_model, name_map),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    body = await request.json()
    text = json.dumps(body.get("messages", []), ensure_ascii=False)
    return JSONResponse({"input_tokens": _estimate_text_tokens(text)})


@app.get("/health")
async def health():
    return {"status": "ok", "model": DEFAULT_MODEL}


# ── Entry Point ───────────────────────────────────────────

if __name__ == "__main__":
    if not NVIDIA_API_KEY:
        log.error("NVIDIA_API_KEY is not set!")
        log.error("Get a free key at https://build.nvidia.com")
        exit(1)
    log.info(f"Starting NIM proxy on {HOST}:{PORT}")
    log.info(f"Default model: {DEFAULT_MODEL}")
    uvicorn.run(app, host=HOST, port=PORT, timeout_keep_alive=TIMEOUT)
