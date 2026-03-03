# NIM Proxy for Claude Code

**Anthropic API 결제 없이, Claude Code를 무료로 사용하는 방법.**

이 프록시 서버를 사용하면 [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) (Anthropic의 AI 코딩 CLI 도구)를 **NVIDIA NIM의 무료 AI 모델**로 구동할 수 있습니다.

## 이게 뭔가요?

Claude Code는 원래 Anthropic의 유료 API 키가 있어야 동작합니다. 하지만 이 프록시를 중간에 끼우면:

1. Claude Code가 Anthropic API로 보내는 요청을 **가로채서**
2. NVIDIA NIM의 **무료 AI 모델**로 대신 전달하고
3. 응답을 Claude Code가 이해할 수 있는 형식으로 **변환해서** 돌려줍니다

결과적으로 Claude Code의 모든 기능(파일 읽기/쓰기, 코드 검색, 터미널 실행 등)을 **무료**로 사용할 수 있습니다.

```
Claude Code ──요청──> NIM Proxy (localhost) ──변환──> NVIDIA NIM (무료)
Claude Code <──응답── NIM Proxy (localhost) <──변환── NVIDIA NIM (무료)
```

## 준비물

1. **Python 3.10 이상** — 터미널에서 `python --version`으로 확인
2. **NVIDIA NIM API Key** — 무료로 발급 (아래 설명)
3. **Claude Code** — `npm install -g @anthropic-ai/claude-code`로 설치

---

## Step 1: NVIDIA NIM API Key 발급 (무료)

1. https://build.nvidia.com 접속
2. 회원가입 / 로그인
3. 아무 모델 페이지로 이동 (예: [Qwen 3.5](https://build.nvidia.com/qwen/qwen3-5-397b-a17b))
4. 우측 "Get API Key" 클릭
5. `nvapi-` 로 시작하는 키를 복사해 둡니다

> 신용카드 없이 무료 크레딧이 제공됩니다.

## Step 2: 프로젝트 다운로드 및 설치

```bash
git clone https://github.com/NA-DEGEN-GIRL/nim-proxy.git
cd nim-proxy
```

의존성 설치 (둘 중 하나 선택):

```bash
# 방법 1: uv 사용 (권장, 더 빠름)
uv sync

# 방법 2: pip 사용
pip install -e .
```

> `uv`가 없다면 `pip install uv`로 먼저 설치하거나, 그냥 pip을 사용하세요.

## Step 3: API Key 설정

```bash
cp .env.example .env
```

`.env` 파일을 편집기로 열고, Step 1에서 복사한 API 키를 붙여넣습니다:

```env
NVIDIA_API_KEY=nvapi-여기에-본인-키-붙여넣기
```

나머지 설정은 기본값으로 두면 됩니다.

## Step 4: 프록시 서버 실행

```bash
# uv 사용시
uv run python server.py

# pip 사용시
python server.py
```

아래와 같은 로그가 나오면 정상입니다:

```
INFO     Starting NIM proxy on 0.0.0.0:8082
INFO     Default model: qwen/qwen3.5-397b-a17b
```

> 이 터미널은 **켜 둔 상태로 유지**합니다. 프록시 서버가 계속 돌아야 합니다.

## Step 5: Claude Code 실행

**새 터미널**을 열고 다음 명령어로 Claude Code를 실행합니다:

```bash
ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_AUTH_TOKEN=dummy claude
```

이게 전부입니다! Claude Code가 정상적으로 실행되며, 모든 요청이 NVIDIA NIM을 통해 **무료로** 처리됩니다.

### 매번 입력하기 귀찮다면

셸 설정 파일에 추가해두면 `claude`만 입력해도 됩니다:

```bash
# ~/.bashrc 또는 ~/.zshrc 에 추가
export ANTHROPIC_BASE_URL=http://localhost:8082
export ANTHROPIC_AUTH_TOKEN=dummy
```

추가 후 `source ~/.bashrc` (또는 `source ~/.zshrc`) 실행하거나 터미널을 재시작하세요.

---

## 간단 채팅 CLI (선택)

Claude Code 없이도 프록시를 통해 AI와 대화할 수 있는 가벼운 CLI가 포함되어 있습니다:

```bash
python chat.py
```

`/help`로 명령어를 확인할 수 있습니다.

---

## 설정 가이드

### 환경변수 (.env)

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `NVIDIA_API_KEY` | NIM API 키 (필수) | - |
| `NIM_BASE_URL` | NIM API URL | `https://integrate.api.nvidia.com/v1` |
| `DEFAULT_MODEL` | 기본 AI 모델 | `qwen/qwen3.5-397b-a17b` |
| `MODEL_MAP` | 모델 매핑 (고급, 아래 참고) | `{}` |
| `HOST` | 서버 호스트 | `0.0.0.0` |
| `PORT` | 서버 포트 | `8082` |
| `TIMEOUT` | NIM 응답 대기 시간 (초) | `600` |

### 모델 변경

기본 모델을 바꾸려면 `.env` 파일에서 수정합니다:

```env
DEFAULT_MODEL=meta/llama-3.3-70b-instruct
```

### 모델 매핑 (고급)

Claude Code 내부에서 모델을 전환할 때 (`/model` 명령어), 각 Claude 모델을 다른 NIM 모델로 보내고 싶다면:

```env
MODEL_MAP={"claude-opus-4-6":"meta/llama-3.1-405b-instruct","claude-sonnet-4-6":"qwen/qwen3.5-397b-a17b","claude-haiku-4-5-20251001":"meta/llama-3.3-70b-instruct"}
```

설정하지 않으면 모든 요청이 `DEFAULT_MODEL`로 보내집니다.

---

## 추천 NIM 모델

| 모델 | 도구 호출 | 추론 | 특징 |
|------|----------|------|------|
| `qwen/qwen3.5-397b-a17b` | O | X | 고성능, 범용 (기본값) |
| `meta/llama-3.3-70b-instruct` | O | X | 빠른 응답 |
| `meta/llama-3.1-405b-instruct` | O | X | 가장 큰 무료 모델 |
| `deepseek-ai/deepseek-r1` | X | O | 추론 특화 (thinking 지원) |

> **참고**: Claude Code는 도구 호출(function calling)에 크게 의존합니다.
> 도구 호출을 지원하는 모델을 사용해야 파일 읽기/쓰기, 코드 검색 등이 정상 동작합니다.
> DeepSeek R1은 추론 능력은 뛰어나지만 도구 호출 미지원으로 Claude Code와의 호환성이 낮습니다.

---

## 속도에 대해

NVIDIA NIM은 **무료**이지만, 유료 API 대비 **응답 속도가 상당히 느릴 수 있습니다**:

- **무료 크레딧 기반**: 요청이 몰리면 대기열에 걸릴 수 있음
- **큰 모델**: 모델이 클수록 응답 생성이 느림
- **Reasoning 모델** (DeepSeek R1): 첫 응답까지 수십 초 이상 걸릴 수 있음

timeout은 기본 600초(10분)로 충분히 넉넉하게 설정되어 있어 중간에 끊기진 않습니다.
**무료인 대신 느린 것**이므로, 속도가 필요하다면 Anthropic API 직접 사용을 권장합니다.

---

## 제한사항

- Anthropic Claude가 아닌 **다른 AI 모델**이 응답하므로 품질 차이가 있을 수 있습니다
- 이미지 입력은 지원되지 않습니다
- 토큰 카운팅은 추정치입니다
- Anthropic 전용 기능 (prompt caching, batch 등)은 지원되지 않습니다

## 라이선스

MIT
