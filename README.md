# NIM Proxy for Claude Code

NVIDIA NIM API를 Claude Code에서 사용할 수 있게 해주는 최소 프록시 서버.

Claude Code가 보내는 Anthropic API 형식의 요청을 OpenAI 호환 형식으로 변환하여 NIM에 전달하고,
NIM의 응답을 다시 Anthropic SSE 형식으로 변환하여 Claude Code에 반환합니다.

## 작동 방식

```
Claude Code ──POST /v1/messages (Anthropic 형식)──> NIM Proxy
NIM Proxy ──POST /v1/chat/completions (OpenAI 형식)──> NVIDIA NIM
NVIDIA NIM ──OpenAI 스트리밍──> NIM Proxy
NIM Proxy ──Anthropic SSE 스트리밍──> Claude Code
```

## 준비물

- Python 3.10+
- NVIDIA NIM API Key (무료: https://build.nvidia.com 에서 발급)

## 설치

```bash
git clone <repo-url>
cd nim-proxy

# uv 사용 (권장)
uv sync

# 또는 pip
pip install -e .
```

## 설정

```bash
cp .env.example .env
```

`.env` 파일을 열고 `NVIDIA_API_KEY`를 설정합니다:

```env
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxx
```

### 환경변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `NVIDIA_API_KEY` | NIM API 키 | (필수) |
| `NIM_BASE_URL` | NIM API URL | `https://integrate.api.nvidia.com/v1` |
| `DEFAULT_MODEL` | 기본 모델 | `meta/llama-3.3-70b-instruct` |
| `MODEL_MAP` | 모델 매핑 (JSON) | `{}` |
| `HOST` | 서버 호스트 | `0.0.0.0` |
| `PORT` | 서버 포트 | `8082` |

## 실행

```bash
# uv
uv run python server.py

# 또는 pip 환경
python server.py
```

## Claude Code 연결

```bash
# 환경변수 설정 후 Claude Code 실행
ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_AUTH_TOKEN=dummy claude
```

또는 셸 프로필에 추가:

```bash
# ~/.bashrc 또는 ~/.zshrc
export ANTHROPIC_BASE_URL=http://localhost:8082
export ANTHROPIC_AUTH_TOKEN=dummy
```

## 모델 매핑

Claude Code는 `claude-sonnet-4-6` 같은 모델 이름을 보냅니다.
기본적으로 모든 요청은 `DEFAULT_MODEL`로 라우팅됩니다.

특정 Claude 모델을 다른 NIM 모델로 매핑하려면:

```env
MODEL_MAP={"claude-opus-4-6":"meta/llama-3.1-405b-instruct","claude-sonnet-4-6":"meta/llama-3.3-70b-instruct","claude-haiku-4-5-20251001":"meta/llama-3.3-70b-instruct"}
```

## 추천 모델

| 모델 | 도구 호출 | 추론 | 설명 |
|------|----------|------|------|
| `meta/llama-3.3-70b-instruct` | O | X | 범용, 빠름 (기본값) |
| `meta/llama-3.1-405b-instruct` | O | X | 가장 큰 무료 모델 |
| `deepseek-ai/deepseek-r1` | X | O | 추론 모델 (`<think>` 태그 지원) |

> **중요**: Claude Code는 도구 호출(function calling)에 크게 의존합니다.
> 도구 호출을 지원하는 모델을 사용해야 정상 동작합니다.

## 지원 기능

- Anthropic 메시지 형식 <-> OpenAI 형식 변환
- 스트리밍 응답 (SSE)
- 도구 호출 (function calling) 변환
- `<think>...</think>` 태그 기반 추론 블록 파싱
- 시스템 프롬프트, 멀티턴 대화 지원

## 제한사항

- 이미지 입력 미지원 (대부분의 NIM 모델이 미지원)
- 도구 호출은 모델에 따라 다름 (위 추천 모델 참고)
- 토큰 카운팅은 추정치 반환
- Anthropic 전용 기능 (캐시, 배치 등) 미지원

## 라이선스

MIT
