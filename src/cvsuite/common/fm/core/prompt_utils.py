from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

CLASS_TEMPLATE_TOKEN = "<class>"
DEFAULT_CLASS_TEMPLATE_PROMPT = f"a photo of a {CLASS_TEMPLATE_TOKEN}"


def parse_prompt_legacy(raw: str) -> Tuple[str | None, List[str]]:
    txt = (raw or "").strip()
    label: str | None = None
    prompt_body = txt
    if ":" in txt:
        label_part, rest = txt.split(":", 1)
        label = label_part.strip() or None
        prompt_body = rest
    parts: List[str] = []
    for chunk in re.split(r"[\n]", prompt_body):
        for piece in re.split(r"[.;,]", chunk):
            text = piece.strip()
            if text:
                parts.append(text)
    if not parts and label:
        parts.append(label)
    return (label.lower().strip() if label else None), [p.strip() for p in parts]


def render_class_template(label: str, template_prompt: str = DEFAULT_CLASS_TEMPLATE_PROMPT) -> str:
    clean_label = str(label).strip()
    if not clean_label:
        raise ValueError("A non-empty class label is required when rendering a template prompt.")
    template = str(template_prompt or "").strip()
    if not template:
        raise ValueError("Class template prompt must not be empty.")
    if CLASS_TEMPLATE_TOKEN not in template:
        raise ValueError(
            f"Class template prompt must contain {CLASS_TEMPLATE_TOKEN!r}. "
            f"Example: {DEFAULT_CLASS_TEMPLATE_PROMPT!r}"
        )
    return template.replace(CLASS_TEMPLATE_TOKEN, clean_label)


def _normalize_prompt_values(value: Any) -> List[str]:
    if isinstance(value, str):
        prompt = value.strip()
        return [prompt] if prompt else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def build_prompt_map(
    raw: str,
    *,
    template_prompt: str = DEFAULT_CLASS_TEMPLATE_PROMPT,
) -> Dict[str, List[str]]:
    txt = (raw or "").strip()
    if not txt:
        raise ValueError("A prompt is required.")
    parsed_json = False
    try:
        data = json.loads(txt)
        parsed_json = True
    except Exception:
        data = None
    if parsed_json and isinstance(data, dict):
        out: Dict[str, List[str]] = {}
        for raw_key, raw_value in data.items():
            key = str(raw_key).strip()
            if not key:
                continue
            prompts = _normalize_prompt_values(raw_value)
            if prompts:
                out[key] = prompts
        if out:
            return out
    if parsed_json and isinstance(data, list):
        labels = [str(item).strip() for item in data if str(item).strip()]
        if labels:
            return {label: [render_class_template(label, template_prompt)] for label in labels}

    label, prompt_parts = parse_prompt_legacy(txt)
    if not prompt_parts:
        raise ValueError("At least one prompt is required.")
    if label:
        return {label: prompt_parts}
    return {prompt: [render_class_template(prompt, template_prompt)] for prompt in prompt_parts}
