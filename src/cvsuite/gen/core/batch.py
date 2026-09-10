from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Sequence

import yaml

from cvsuite.common.core import ImageRecord, Record
from cvsuite.common.core import VisionDataset

_PROMPT_FILE_EXTS = {".yaml", ".yml", ".json"}
_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_PROMPT_FILE_TOP_LEVEL_ERROR = (
    "Gen prompt files must contain a top-level YAML/JSON list of strings, "
    "a mapping of id to prompt, or a list of single-entry id-to-prompt mappings."
)
_PROMPT_FILE_ITEM_ERROR = "Gen prompt list items must be strings or single-entry id-to-prompt mappings."
_PROMPT_FILE_ITEM_MAPPING_ERROR = "Gen prompt list mapping items must contain exactly one id-to-prompt entry."
_PROMPT_FILE_VALUE_ERROR = "Gen prompt files must contain string prompt values only."
_PROMPT_FILE_EMPTY_PROMPT_ERROR = "Gen prompt files must not contain empty prompt strings."
_PROMPT_FILE_EMPTY_ID_ERROR = "Gen prompt mapping ids must be non-empty."
GEN_PROMPT_ATTR = "gen_prompt"
GEN_PROMPT_INDEX_ATTR = "gen_prompt_index"
GEN_SAMPLE_INDEX_ATTR = "gen_sample_index"
GEN_OUTPUT_KEY_ATTR = "gen_output_key"
GEN_SOURCE_RECORD_IDX_ATTR = "gen_source_record_idx"
GEN_SOURCE_IMAGE_ATTR = "gen_source_image"
GEN_SOURCE_KEY_ATTR = "gen_source_key"


class PromptItem(str):
    prompt_id: str

    def __new__(cls, prompt: str, prompt_id: str | int):
        obj = str.__new__(cls, prompt)
        obj.prompt_id = str(prompt_id)
        return obj


def _resolve_prompt_file(raw: str, caller_cwd: Path) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.suffix.lower() not in _PROMPT_FILE_EXTS:
        return None
    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append((caller_cwd / candidate).resolve())
        cwd_candidate = (Path.cwd() / candidate).resolve()
        if cwd_candidate not in candidates:
            candidates.append(cwd_candidate)
    for path in candidates:
        if path.is_file():
            return path
        if path.exists():
            raise SystemExit(f"Prompt path is not a file: {path}")
    raise FileNotFoundError(f"Prompt file not found: {candidates[0] if candidates else candidate}")


def _load_prompt_file(path: Path) -> list[PromptItem | str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    prompts = _normalize_prompt_file_data(data)
    if not prompts:
        raise SystemExit("Gen prompt file did not contain any prompts.")
    return prompts


def parse_prompt_items(raw_items: Sequence[str] | None, *, caller_cwd: Path | None = None) -> list[PromptItem]:
    caller = caller_cwd or Path.cwd()
    prompts: list[PromptItem] = []
    for raw in list(raw_items or []):
        text = str(raw or "").strip()
        if not text:
            raise SystemExit("`--prompt` values must be non-empty.")
        prompt_file = _resolve_prompt_file(text, caller)
        if prompt_file is None:
            prompts.append(PromptItem(text, f"{len(prompts):04d}"))
            continue
        for prompt in _load_prompt_file(prompt_file):
            if isinstance(prompt, PromptItem):
                prompts.append(prompt)
            else:
                prompts.append(PromptItem(prompt, f"{len(prompts):04d}"))
    if not prompts:
        raise SystemExit("`--prompt` is required.")
    _ensure_unique_prompt_ids(prompts)
    return prompts


def prompt_payload(prompts: Sequence[str]) -> str:
    return json.dumps(list(prompts), ensure_ascii=False)


def _sanitize_token(text: str) -> str:
    cleaned = _SANITIZE_RE.sub("_", str(text or "").strip().lower())
    cleaned = cleaned.strip("._")
    return cleaned or "prompt"


def _sanitize_id(text: str) -> str:
    cleaned = _SANITIZE_RE.sub("_", str(text or "").strip())
    cleaned = cleaned.strip("._")
    return cleaned or "prompt"


def _normalize_prompt_text(value: object) -> str:
    if not isinstance(value, str):
        raise SystemExit(_PROMPT_FILE_VALUE_ERROR)
    prompt = value.strip()
    if not prompt:
        raise SystemExit(_PROMPT_FILE_EMPTY_PROMPT_ERROR)
    return prompt


def _normalize_prompt_id(value: object) -> str:
    prompt_id = "" if value is None else str(value).strip()
    if not prompt_id:
        raise SystemExit(_PROMPT_FILE_EMPTY_ID_ERROR)
    return prompt_id


def _normalize_prompt_mapping_entry(prompt_id: object, value: object) -> PromptItem:
    return PromptItem(_normalize_prompt_text(value), _normalize_prompt_id(prompt_id))


def _normalize_prompt_list_item(item: object) -> PromptItem | str:
    if isinstance(item, str):
        return _normalize_prompt_text(item)
    if not isinstance(item, dict):
        raise SystemExit(_PROMPT_FILE_ITEM_ERROR)
    if len(item) != 1:
        raise SystemExit(_PROMPT_FILE_ITEM_MAPPING_ERROR)
    prompt_id, value = next(iter(item.items()))
    return _normalize_prompt_mapping_entry(prompt_id, value)


def _normalize_prompt_file_data(data: object) -> list[PromptItem | str]:
    if isinstance(data, dict):
        return [_normalize_prompt_mapping_entry(prompt_id, value) for prompt_id, value in data.items()]
    if isinstance(data, list):
        return [_normalize_prompt_list_item(item) for item in data]
    raise SystemExit(_PROMPT_FILE_TOP_LEVEL_ERROR)


def _ensure_unique_prompt_ids(prompts: Sequence[PromptItem]) -> None:
    seen: dict[str, str] = {}
    for idx, prompt in enumerate(prompts):
        prompt_id = _prompt_id(prompt, idx)
        sanitized = _sanitize_id(prompt_id)
        previous = seen.get(sanitized)
        if previous is not None:
            raise SystemExit(
                "Gen prompt ids must be unique after filename sanitization. "
                f"{prompt_id!r} collides with {previous!r} as {sanitized!r}."
            )
        seen[sanitized] = prompt_id


def source_key(record: Record, idx: int) -> str:
    candidate = record.rel_image_path or record.image.path
    if isinstance(candidate, Path):
        base = candidate.stem or candidate.name
    else:
        base = str(candidate)
    return f"{idx:04d}_{_sanitize_token(base)}"


def output_key(
    *,
    prompt_id: str | int,
    prompt_index: int,
    source_key_value: str | None = None,
    sample_index: int | None = None,
) -> str:
    prompt_id_text = str(prompt_id if prompt_id is not None else f"{prompt_index:04d}").strip()
    parts = [_sanitize_id(prompt_id_text)]
    if source_key_value:
        parts.append(source_key_value)
    parts.append(f"{(sample_index or 0) + 1:04d}")
    return "_".join(parts)


def _prompt_id(prompt: str, fallback_index: int) -> str:
    return str(getattr(prompt, "prompt_id", fallback_index))


def build_create_dataset(prompts: Sequence[str], *, root: Path, num_images: int = 1) -> VisionDataset:
    records: list[Record] = []
    if num_images < 1:
        raise SystemExit("`--num-images` must be >= 1.")
    for prompt_index, prompt in enumerate(prompts):
        for sample_index in range(num_images):
            key = output_key(
                prompt_id=_prompt_id(prompt, prompt_index),
                prompt_index=prompt_index,
                sample_index=sample_index,
            )
            records.append(
                Record(
                    image=ImageRecord(path=Path("pending") / f"{key}.png", width=0, height=0),
                    split="train",
                    attributes={
                        GEN_PROMPT_ATTR: prompt,
                        GEN_PROMPT_INDEX_ATTR: prompt_index,
                        GEN_SAMPLE_INDEX_ATTR: sample_index,
                        GEN_OUTPUT_KEY_ATTR: key,
                    },
                )
            )
    return VisionDataset(
        records=records,
        root=root,
        meta={
            "gen_mode": "create",
            "gen_prompt_count": len(prompts),
            "gen_num_images_per_prompt": num_images,
            "gen_source_count": 0,
            "gen_fanout_mode": "prompt-list",
        },
    )


def expand_edit_dataset(dataset: VisionDataset, prompts: Sequence[str], num_images: int = 1) -> VisionDataset:
    source_records = list(dataset.records)
    expanded_records: list[Record] = []
    if num_images < 1:
        raise SystemExit("`--num-images` must be >= 1.")
    for source_idx, source_record in enumerate(source_records):
        key = source_key(source_record, source_idx)
        for prompt_index, prompt in enumerate(prompts):
            for sample_index in range(num_images):
                record = copy.deepcopy(source_record)
                record.attributes[GEN_PROMPT_ATTR] = prompt
                record.attributes[GEN_PROMPT_INDEX_ATTR] = prompt_index
                record.attributes[GEN_SAMPLE_INDEX_ATTR] = sample_index
                record.attributes[GEN_OUTPUT_KEY_ATTR] = output_key(
                    prompt_id=_prompt_id(prompt, prompt_index),
                    prompt_index=prompt_index,
                    source_key_value=key,
                    sample_index=sample_index,
                )
                record.attributes[GEN_SOURCE_RECORD_IDX_ATTR] = source_idx
                record.attributes[GEN_SOURCE_IMAGE_ATTR] = str(source_record.image.path)
                record.attributes[GEN_SOURCE_KEY_ATTR] = key
                expanded_records.append(record)
    meta = dict(dataset.meta)
    meta["gen_prompt_count"] = len(prompts)
    meta["gen_num_images_per_prompt"] = num_images
    meta["gen_source_count"] = len(source_records)
    meta["gen_fanout_mode"] = "cartesian"
    return VisionDataset(
        records=expanded_records,
        classes=list(dataset.classes),
        task=dataset.task,
        root=dataset.root,
        meta=meta,
        fm_request=dataset.fm_request,
    )


__all__ = [
    "build_create_dataset",
    "expand_edit_dataset",
    "output_key",
    "parse_prompt_items",
    "PromptItem",
    "prompt_payload",
    "source_key",
]
