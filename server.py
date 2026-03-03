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

from proxy import convert_request, stream_response

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

    log.info(f"{original_model} -> {target_model}")

    openai_request = convert_request(body, target_model)

    return StreamingResponse(
        content=stream_response(client, openai_request, original_model),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    body = await request.json()
    text = json.dumps(body.get("messages", []))
    return JSONResponse({"input_tokens": len(text) // 4})


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
