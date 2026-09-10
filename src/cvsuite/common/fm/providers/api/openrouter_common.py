from __future__ import annotations

import base64
import json
import mimetypes
import os
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from PIL import Image

from cvsuite.common.fm.core import utils

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_API_KEY_ENV_ALIASES = (
    OPENROUTER_API_KEY_ENV,
    "OPENROUTER_API_KEY_ENV",
)


def require_openrouter_api_key() -> str:
    for _key, api_key, _source in utils.iter_env_var_candidates(OPENROUTER_API_KEY_ENV_ALIASES):
        os.environ.setdefault(OPENROUTER_API_KEY_ENV, api_key)
        return api_key
    expected = ", ".join(f"`{key}`" for key in OPENROUTER_API_KEY_ENV_ALIASES)
    raise RuntimeError(
        f"OpenRouter API key not found. Set {expected} in the environment or in a nearby `.env` file."
    )


def image_path_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def decode_data_url_image(data_url: str) -> Image.Image:
    if "," not in data_url:
        raise ValueError("Expected data URL payload for generated image response.")
    _prefix, payload = data_url.split(",", 1)
    raw = base64.b64decode(payload)
    with Image.open(BytesIO(raw)) as image:
        return image.convert("RGB")


def post_openrouter_chat(*, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        OPENROUTER_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=120.0) as response:
        return json.loads(response.read().decode("utf-8"))


def assistant_text_from_response(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


def assistant_image_data_urls_from_response(payload: dict[str, Any]) -> list[str]:
    choices = payload.get("choices") or []
    if not choices:
        return []
    message = choices[0].get("message") or {}
    images = message.get("images") or []
    urls: list[str] = []
    for image in images:
        if not isinstance(image, dict):
            continue
        image_url = image.get("image_url") or image.get("imageUrl") or {}
        if isinstance(image_url, dict):
            url = str(image_url.get("url") or "").strip()
            if url:
                urls.append(url)
    return urls


def usage_from_response(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage")
    return dict(usage) if isinstance(usage, dict) else {}
