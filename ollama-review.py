#!/usr/bin/env python3
"""
Ollama Editorial Review CLI - 소설 에피소드를 로컬 Ollama 모델로 편집 리뷰한다.

Usage:
    python3 ollama-review.py --file chapter.md --novel no-title-011
    python3 ollama-review.py --file chapter.md --model gpt-oss:120b --novel no-title-011
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

NOVEL_ROOT = os.environ.get("NOVEL_ROOT", "/root/novel")
OLLAMA_PATH = os.environ.get("OLLAMA_PATH", "/usr/local/bin/ollama")
DEFAULT_MODEL = os.environ.get("OLLAMA_REVIEW_MODEL", "gpt-oss:120b")

REVIEW_PROMPT = """너는 소설 원고의 시니어 편집자다. 아래 원고를 리뷰하고 EDITOR_FEEDBACK 형식으로 피드백을 작성하라.

[리뷰 카테고리]
1. [Language/Prose] - 한국어 문장의 자연스러움, 감정 접속사 vs 논리 접속사, 동사의 무게감, 번역투, 반복 표현
2. [Continuity/Logic] - 연속성 오류, 논리 모순, 메타 참조 금지 위반 ("X화에서" 등)
3. [Character] - 캐릭터 일관성, 말투, 심리적 개연성, 행동과 설정의 부합
4. [Setting/Worldbuilding] - 세계관 설정 위반, 시대 착오, 고유명사 일관성

[출력 형식]

### [Language/Prose]
#### 1. {지적 제목}
- **위치**: "{해당 문장 원문 인용}"
- **문제**: {문제 설명}
- **제안**: {구체적 대안 1~2개}

### [Continuity/Logic]
(동일 형식)

### [Character]
(동일 형식)

[규칙]
- 해당 카테고리에 지적이 없으면 그 섹션을 생략하라.
- 구체적 문장을 인용하라.
- 대안을 반드시 제시하라.
- 캐릭터 대사의 의도적 비문, 사투리, 문체 선택은 지적하지 마라.
- 문법보다 예술성(어감, 리듬, 몰입도)에 집중하라."""


def read_file(path: str, max_lines: int = 0) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8")
        if max_lines > 0:
            return "\n".join(text.split("\n")[:max_lines])
        return text
    except (FileNotFoundError, PermissionError):
        return ""


def extract_body(text: str) -> str:
    marker = "### EPISODE_META"
    idx = text.find(marker)
    if idx != -1:
        body = text[:idx].rstrip()
        if body.endswith("---"):
            body = body[:-3].rstrip()
        return body
    return text


def load_context(novel_id: str) -> dict:
    novel_dir = os.path.join(NOVEL_ROOT, novel_id)
    return {
        "style": read_file(os.path.join(novel_dir, "settings/01-style-guide.md"), 50),
        "characters": read_file(os.path.join(novel_dir, "settings/03-characters.md"), 80),
        "world": read_file(os.path.join(novel_dir, "settings/04-worldbuilding.md"), 30),
    }


def run_ollama(prompt: str, model: str, timeout: int = 600) -> str:
    cmd = [OLLAMA_PATH, "run", model, "--hidethinking"]
    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "TERM": "dumb"},
        )
        output = result.stdout
        output = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', output)
        output = re.sub(r'[⠀-⣿]', '', output)
        output = re.sub(r'\n{3,}', '\n\n', output).strip()
        return output
    except subprocess.TimeoutExpired:
        return f"[오류] Ollama 응답 시간 초과 (제한: {timeout}초)"
    except FileNotFoundError:
        return f"[오류] Ollama를 찾을 수 없습니다: {OLLAMA_PATH}"
    except Exception as e:
        return f"[오류] {e}"


def main():
    parser = argparse.ArgumentParser(description="Ollama Editorial Review CLI")
    parser.add_argument("--file", required=True, help="에피소드 파일 절대 경로")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama 모델명 (기본: {DEFAULT_MODEL})")
    parser.add_argument("--novel", default="", help="소설 ID (예: no-title-011). 없으면 경로에서 자동 추출")
    parser.add_argument("--timeout", type=int, default=900, help="응답 대기 시간(초, 기본 900)")
    args = parser.parse_args()

    # 파일 읽기
    text = read_file(args.file)
    if not text:
        print(f"[오류] 파일을 읽을 수 없습니다: {args.file}", file=sys.stderr)
        sys.exit(1)

    body = extract_body(text)
    if len(body) < 50:
        print(f"[오류] 본문이 너무 짧습니다 ({len(body)}자)", file=sys.stderr)
        sys.exit(1)

    # 소설 ID 추출
    novel_id = args.novel
    if not novel_id:
        match = re.search(r"(no-title-\d+)", args.file)
        novel_id = match.group(1) if match else ""

    # 맥락 로드
    ctx = load_context(novel_id) if novel_id else {"style": "", "characters": "", "world": ""}

    # 프롬프트 구성
    prompt = REVIEW_PROMPT
    if ctx["style"]:
        prompt += f"\n\n[문체 가이드]\n{ctx['style']}"
    if ctx["characters"]:
        prompt += f"\n\n[캐릭터 설정]\n{ctx['characters']}"
    if ctx["world"]:
        prompt += f"\n\n[세계관]\n{ctx['world']}"
    prompt += f"\n\n[원고 시작]\n{body}\n[원고 끝]\n\n위 원고를 리뷰하여 피드백을 출력하라."

    # Ollama 실행
    print(f"모델: {args.model}", file=sys.stderr)
    print(f"대상: {args.file}", file=sys.stderr)
    raw = run_ollama(prompt, args.model, args.timeout)

    if raw.startswith("[오류]"):
        print(raw, file=sys.stderr)
        sys.exit(1)

    # 보고서 구성
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    basename = Path(args.file).stem
    header = (
        f"# Ollama 편집 리뷰\n\n"
        f"- **모델**: `{args.model}`\n"
        f"- **대상**: `{args.file}`\n"
        f"- **일시**: {now}\n\n---\n"
    )
    report = header + "\n" + raw

    # 저장
    if novel_id:
        novel_dir = os.path.join(NOVEL_ROOT, novel_id)
        out_path = os.path.join(novel_dir, "EDITOR_FEEDBACK_ollama.md")
        Path(out_path).write_text(report, encoding="utf-8")
        print(f"저장됨: {out_path}", file=sys.stderr)

    # stdout 출력
    print(report)


if __name__ == "__main__":
    main()
