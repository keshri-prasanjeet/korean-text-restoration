"""Chat prompt shared by training and evaluation."""

from __future__ import annotations

# "You are a Korean word-spacing corrector. Fix only the spacing of the input
#  sentence according to standard orthography. Do not change, add or remove any
#  character. Output only the corrected sentence."
SYSTEM_PROMPT = (
    "당신은 한국어 띄어쓰기 교정기입니다. 입력 문장의 띄어쓰기만 한글 맞춤법에 맞게 고치고, "
    "글자는 하나도 바꾸거나 더하거나 빼지 마세요. 교정된 문장만 출력하세요."
)


def build_messages(text: str, target: str | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


def clean_output(text: str) -> str:
    """Keep the first non-empty line; models sometimes add an explanation after it."""
    for line in text.strip().splitlines():
        line = line.strip().strip('"').strip()
        if line:
            return line
    return ""
