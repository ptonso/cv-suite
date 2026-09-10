from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata
from typing import Iterable, Sequence

from cvsuite.common.core import Record, VQA
from cvsuite.common.core import VisionDataset

from ...core import io

FALLBACK_LABEL = "unclear"

_EDGE_PUNCTUATION = " \t\r\n\"'`“”‘’.,;:!?()[]{}<>"
_WS_RE = re.compile(r"\s+")
_TAIL_CONNECTOR_RE = re.compile(r"\b(?:because|since|but)\b", re.IGNORECASE)
_LEADIN_PATTERNS = (
    re.compile(r"\bfinal\s+answer\s*[:\-]\s*", re.IGNORECASE),
    re.compile(r"\banswer\s*[:\-]\s*", re.IGNORECASE),
    re.compile(r"\bthe\s+answer\s+is\s+", re.IGNORECASE),
    re.compile(r"\boption\s*[:\-]?\s*", re.IGNORECASE),
    re.compile(r"\bchoice\s*[:\-]?\s*", re.IGNORECASE),
    re.compile(r"\blabel\s*[:\-]\s*", re.IGNORECASE),
    re.compile(r"\bclass\s*[:\-]\s*", re.IGNORECASE),
    re.compile(r"\bcategory\s*[:\-]\s*", re.IGNORECASE),
)
_SYMBOLIC_PATTERNS = (
    re.compile(r"^\s*[\(\[]?\s*([A-Za-z0-9])\s*[\)\]]?\s*[\.\!\?]*\s*$", re.IGNORECASE),
    re.compile(
        r"\b(?:option|choice|answer|pick|picked|choose|chose|select|selected|label|class|category)\s*[:\-]?\s*[\(\[]?\s*([A-Za-z0-9])\s*[\)\]]?\b",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class AnswerMatch:
    label: str
    matched_alias: str | None
    rule: str


@dataclass(frozen=True)
class LabelRegistry:
    labels: tuple[str, ...]
    alias_to_label: dict[str, str]
    symbolic_alias_to_label: dict[str, str]


def _normalize_folder_label(raw: object) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    normalized = normalize_alias(value)
    if normalized == normalize_alias(FALLBACK_LABEL):
        return FALLBACK_LABEL
    return value


def _validate_folder_label(label: str) -> None:
    path = Path(label)
    if path.is_absolute():
        raise SystemExit(f"Mapped labels must be relative folder paths, got absolute path: {label!r}")
    for part in path.parts:
        if part in {"", ".", ".."}:
            raise SystemExit(f"Mapped labels must not contain empty, '.' , or '..' path segments: {label!r}")


def normalize_alias(raw: object) -> str:
    text = unicodedata.normalize("NFKC", str(raw or ""))
    text = text.casefold().replace("_", " ").replace("-", " ")
    text = _WS_RE.sub(" ", text)
    text = text.strip(_EDGE_PUNCTUATION)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def _normalize_sequence(values: Iterable[object]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = _normalize_folder_label(raw)
        if not value or value in seen:
            continue
        _validate_folder_label(value)
        seen.add(value)
        out.append(value)
    return out


def _parse_labels_file(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    data = io.read_yaml_or_json(path)
    if isinstance(data, list):
        labels = _normalize_sequence(data)
        return labels, {}
    if isinstance(data, dict):
        aliases: dict[str, list[str]] = {}
        ordered: list[str] = []
        for raw_label, raw_values in data.items():
            label = _normalize_folder_label(raw_label)
            if not label:
                continue
            _validate_folder_label(label)
            if not isinstance(raw_values, list):
                raise SystemExit(f"--labels-file object values must be lists of synonyms: {path}")
            values = [str(item) for item in raw_values if str(item or "").strip()]
            if label not in aliases:
                aliases[label] = []
                ordered.append(label)
            aliases[label].extend(values)
        return ordered, aliases
    raise SystemExit(f"--labels-file must be a list of labels or a mapping of label -> synonym list: {path}")


def load_label_registry(labels: Sequence[str], labels_file: Path | None = None) -> LabelRegistry:
    ordered_labels = _normalize_sequence(labels)
    file_labels: list[str] = []
    file_aliases: dict[str, list[str]] = {}
    if labels_file is not None:
        file_labels, file_aliases = _parse_labels_file(labels_file)

    label_order = _normalize_sequence([*ordered_labels, *file_labels])
    label_aliases: dict[str, list[str]] = {label: [] for label in label_order}
    for label, aliases in file_aliases.items():
        label_aliases.setdefault(label, [])
        for alias in aliases:
            text = str(alias).strip()
            if text:
                label_aliases[label].append(text)

    if FALLBACK_LABEL in label_order:
        label_order = [label for label in label_order if label != FALLBACK_LABEL]
    label_order.append(FALLBACK_LABEL)

    if not any(label != FALLBACK_LABEL for label in label_order):
        raise SystemExit("map-answers requires at least one canonical label besides 'unclear'.")

    alias_to_label: dict[str, str] = {}
    symbolic_alias_to_label: dict[str, str] = {}
    for label in label_order:
        aliases = [label, *label_aliases.get(label, [])]
        for raw_alias in aliases:
            normalized = normalize_alias(raw_alias)
            if not normalized:
                continue
            existing = alias_to_label.get(normalized)
            if existing is not None and existing != label:
                raise SystemExit(
                    f"Normalized synonym collision for {raw_alias!r}: maps to both {existing!r} and {label!r}."
                )
            alias_to_label[normalized] = label
            if re.fullmatch(r"[a-z0-9]", normalized):
                symbolic_alias_to_label[normalized] = label

    return LabelRegistry(
        labels=tuple(label_order),
        alias_to_label=alias_to_label,
        symbolic_alias_to_label=symbolic_alias_to_label,
    )


def _non_empty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _extract_after_leadins(text: str) -> list[str]:
    out: list[str] = []
    for pattern in _LEADIN_PATTERNS:
        for match in pattern.finditer(text):
            candidate = text[match.end() :].strip()
            if candidate:
                out.append(candidate)
    return out


def _trim_rationale_tail(text: str) -> str:
    candidate = str(text or "").strip()
    if not candidate:
        return ""
    if "\n" in candidate:
        candidate = candidate.split("\n", 1)[0].strip()
    match = _TAIL_CONNECTOR_RE.search(candidate)
    if match is not None:
        candidate = candidate[: match.start()].strip()
    return candidate.strip()


def _stage_exact_match(registry: LabelRegistry, *, candidates: Iterable[str], rule: str) -> AnswerMatch | None:
    matches: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        normalized = normalize_alias(candidate)
        if not normalized:
            continue
        label = registry.alias_to_label.get(normalized)
        if label is None:
            continue
        key = (label, normalized)
        if key in seen:
            continue
        seen.add(key)
        matches.append(key)
    if not matches:
        return None
    labels = {label for label, _alias in matches}
    if len(labels) > 1:
        return AnswerMatch(label=FALLBACK_LABEL, matched_alias=None, rule=f"{rule}:ambiguous")
    label, matched_alias = matches[0]
    return AnswerMatch(label=label, matched_alias=matched_alias, rule=rule)


def _symbolic_match(registry: LabelRegistry, raw_answer: str) -> AnswerMatch | None:
    if not registry.symbolic_alias_to_label:
        return None
    matches: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pattern in _SYMBOLIC_PATTERNS:
        for match in pattern.finditer(raw_answer):
            normalized = normalize_alias(match.group(1))
            label = registry.symbolic_alias_to_label.get(normalized)
            if label is None:
                continue
            key = (label, normalized)
            if key in seen:
                continue
            seen.add(key)
            matches.append(key)
    if not matches:
        return None
    labels = {label for label, _alias in matches}
    if len(labels) > 1:
        return AnswerMatch(label=FALLBACK_LABEL, matched_alias=None, rule="symbolic_token:ambiguous")
    label, matched_alias = matches[0]
    return AnswerMatch(label=label, matched_alias=matched_alias, rule="symbolic_token")


def match_answer_text(answer: str, registry: LabelRegistry) -> AnswerMatch:
    raw_answer = unicodedata.normalize("NFKC", str(answer or ""))
    if not normalize_alias(raw_answer):
        return AnswerMatch(label=FALLBACK_LABEL, matched_alias=None, rule="empty")

    lines = _non_empty_lines(raw_answer)
    first_last_lines = [lines[0]] if lines else []
    if len(lines) >= 2:
        first_last_lines.append(lines[-1])

    leadin_candidates = _extract_after_leadins(raw_answer)
    for line in lines:
        leadin_candidates.extend(_extract_after_leadins(line))

    trimmed_candidates: list[str] = []
    for candidate in [raw_answer, *first_last_lines, *leadin_candidates]:
        trimmed = _trim_rationale_tail(candidate)
        if trimmed:
            trimmed_candidates.append(trimmed)

    for rule, candidates in (
        ("full_answer", [raw_answer]),
        ("edge_lines", first_last_lines),
        ("leadin_span", leadin_candidates),
        ("trimmed_span", trimmed_candidates),
    ):
        match = _stage_exact_match(registry, candidates=candidates, rule=rule)
        if match is not None:
            return match

    symbolic = _symbolic_match(registry, raw_answer)
    if symbolic is not None:
        return symbolic

    return AnswerMatch(label=FALLBACK_LABEL, matched_alias=None, rule="no_match")


def _copy_qa(qa: VQA, *, mapped_label: str, matched_alias: str | None, match_rule: str) -> VQA:
    meta = dict(qa.meta or {})
    meta["mapped_label"] = mapped_label
    meta["match_rule"] = match_rule
    if matched_alias is not None:
        meta["matched_alias"] = matched_alias
    return VQA(
        question=qa.question,
        answer=qa.answer,
        score=qa.score,
        model=qa.model,
        meta=meta,
    )


def _qa_suffix(qa: VQA, index: int) -> str:
    qa_id = qa.meta.get("id") if isinstance(qa.meta, dict) else None
    if qa_id is not None:
        return f"qid{io.sanitize_sample_key(str(qa_id))}"
    return f"q{index:03d}"


def _explode_record(record: Record, *, registry: LabelRegistry) -> list[Record]:
    if not record.vqas:
        sample_key = io.record_sample_key(record)
        raise SystemExit(f"map-answers requires QA entries on every record. Missing QA list for sample {sample_key!r}.")

    out: list[Record] = []
    base_item_meta = io.record_item_meta(record)
    for index, qa in enumerate(record.vqas):
        match = match_answer_text(qa.answer, registry)
        copied_qa = _copy_qa(qa, mapped_label=match.label, matched_alias=match.matched_alias, match_rule=match.rule)
        attrs = dict(record.attributes)
        attrs[io.SAMPLE_KEY_ATTR] = f"{io.record_sample_key(record)}__{_qa_suffix(qa, index)}"
        attrs[io.ANSWER_BUCKET_ATTR] = match.label
        if base_item_meta:
            attrs[io.ITEM_META_ATTR] = dict(base_item_meta)

        qa_id = copied_qa.meta.get("id")
        if qa_id is not None:
            copied_qa.meta.setdefault("source_question_id", qa_id)
        copied_qa.meta.setdefault("source_question_index", index)

        out.append(
            Record(
                image=record.image,
                split=record.split,
                task=record.task,
                boxes=list(record.boxes),
                polys=list(record.polys),
                kpts=list(record.kpts),
                labelme_shapes=list(record.labelme_shapes),
                attributes=attrs,
                rel_image_path=record.rel_image_path,
                rel_label_path=record.rel_label_path,
                classification=record.classification,
                vqas=[copied_qa],
                embeddings=list(record.embeddings),
            )
        )
    return out


def apply_answer_mapping(dataset: VisionDataset, registry: LabelRegistry) -> VisionDataset:
    io.ensure_has_vqas(dataset)

    records: list[Record] = []
    for record in dataset.records:
        records.extend(_explode_record(record, registry=registry))

    meta = dict(dataset.meta)
    meta[io.ANSWER_BUCKETS_META] = list(registry.labels)
    return VisionDataset(
        records=records,
        classes=list(dataset.classes),
        task=dataset.task,
        root=dataset.root,
        meta=meta,
        fm_request=dataset.fm_request,
    )
