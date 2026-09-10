from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import imagehash
from tqdm import tqdm

from ..actions import Action
from ..config import FlattenMode, ArrangeConfig, NonImagePolicy
from ..fs import PREP_MANIFEST_NAME, PREP_NON_IMAGE_DIR, file_hash, file_size, is_image, iter_files, perceptual_hash
from ..logging import Logger


@dataclass
class FileEntry:
    src: Path
    rel_src: Path
    rel_dst: Path
    is_image: bool
    size: int
    hash: str | None
    keep: bool
    is_duplicate: bool
    phash: Optional[imagehash.ImageHash] = None


def build_existing_hashes(dst: Path, logger: Logger) -> set[tuple[int, str]]:
    if not dst.exists():
        return set()

    paths = sorted(path for path in iter_files(dst) if path.name != PREP_MANIFEST_NAME)
    hashes: set[tuple[int, str]] = set()
    for path in tqdm(paths, desc="Indexing dst", unit=" file"):
        hashes.add((file_size(path), file_hash(path)))
    logger.info(f"Indexed {len(hashes)} existing destination files.")
    return hashes


def build_existing_phashes(dst: Path, logger: Logger) -> list[imagehash.ImageHash]:
    if not dst.exists():
        return []

    paths = sorted(
        path for path in iter_files(dst) if path.name != PREP_MANIFEST_NAME and is_image(path)
    )
    phashes = [perceptual_hash(path) for path in tqdm(paths, desc="Indexing dst", unit=" file")]
    logger.info(f"Indexed {len(phashes)} existing destination images.")
    return phashes


def build_entries(config: ArrangeConfig, logger: Logger) -> list[FileEntry]:
    paths = sorted(iter_files(config.src))
    entries: list[FileEntry] = []
    for path in tqdm(paths, desc="Scanning src", unit=" file"):
        rel_src = path.relative_to(config.src)
        image_flag = is_image(path)
        size = file_size(path)
        entry = FileEntry(
            src=path,
            rel_src=rel_src,
            rel_dst=rel_src,
            is_image=image_flag,
            size=size,
            hash=None,
            keep=True,
            is_duplicate=False,
        )
        entries.append(entry)
    logger.info(f"Discovered {len(entries)} files.")
    return entries


def apply_flatten(entries: Sequence[FileEntry], mode: FlattenMode) -> None:
    if mode == "none":
        return
    for entry in entries:
        parts = list(entry.rel_src.parts)
        name = parts[-1]
        if mode == "plain":
            entry.rel_dst = Path(name)
        elif mode == "enc-prefix":
            encoded = "__".join(parts)
            entry.rel_dst = Path(encoded)
        elif mode == "enc-suffix":
            stem, dot, suffix = name.partition(".")
            prefix = "__".join(parts[:-1]) if len(parts) > 1 else ""
            encoded = f"{stem}__{prefix}" if prefix else stem
            full = encoded + (dot + suffix if dot else "")
            entry.rel_dst = Path(full)


def apply_exact_dedup(
    entries: Sequence[FileEntry],
    logger: Logger,
    existing_hashes: set[tuple[int, str]] | None = None,
) -> int:
    seen = set(existing_hashes or set())
    duplicates = 0
    for entry in tqdm(entries, desc="Deduplicating", unit=" file"):
        if not entry.keep:
            continue
        if entry.hash is None:
            entry.hash = file_hash(entry.src)
        key = (entry.size, entry.hash)
        if key in seen:
            entry.keep = False
            entry.is_duplicate = True
            duplicates += 1
        else:
            seen.add(key)
    if duplicates:
        logger.info(f"Marked {duplicates} duplicates.")
    return duplicates


def apply_perceptual_dedup(
    entries: Sequence[FileEntry],
    logger: Logger,
    threshold: int,
    existing_phashes: list[imagehash.ImageHash] | None = None,
) -> int:
    seen = list(existing_phashes or [])
    duplicates = 0
    for entry in tqdm(entries, desc="Deduplicating", unit=" file"):
        if not entry.keep or not entry.is_image:
            continue
        if entry.phash is None:
            entry.phash = perceptual_hash(entry.src)
        if any(entry.phash - other <= threshold for other in seen):
            entry.keep = False
            entry.is_duplicate = True
            duplicates += 1
        else:
            seen.append(entry.phash)
    if duplicates:
        logger.info(f"Marked {duplicates} duplicates.")
    return duplicates


def _sequence_value(path: Path, flatten_mode: FlattenMode) -> int | None:
    stem = path.stem
    if flatten_mode == "enc-prefix":
        token = stem.split("__")[-1]
    elif flatten_mode == "enc-suffix":
        token = stem.split("__")[0]
    else:
        token = stem
    return int(token) if token.isdigit() else None


def next_sequence_id(dst: Path, flatten_mode: FlattenMode) -> int:
    if not dst.exists():
        return 1

    paths = sorted(iter_files(dst))
    values = [
        value
        for path in tqdm(paths, desc="Scanning sequence", unit=" file")
        if is_image(path)
        for value in [_sequence_value(path, flatten_mode)]
        if value is not None
    ]
    return max(values, default=0) + 1


def apply_rename_seq(entries: Sequence[FileEntry], flatten_mode: FlattenMode, start: int = 1) -> None:
    counter = start
    for entry in tqdm(entries, desc="Renaming sequence", unit=" file"):
        if not entry.keep:
            continue
        if not entry.is_image:
            continue

        ext = entry.rel_dst.suffix.lower()
        stem = entry.rel_dst.stem
        new_token = f"{counter:06d}"

        if flatten_mode == "none":
            entry.rel_dst = entry.rel_dst.with_name(f"{new_token}{ext}")
        elif flatten_mode == "enc-prefix":
            parts = stem.split("__")
            prefix = "__".join(parts[:-1]) if len(parts) > 1 else ""
            if prefix:
                new_name = f"{prefix}__{new_token}{ext}"
            else:
                new_name = f"{new_token}{ext}"
        elif flatten_mode == "enc-suffix":
            parts = stem.split("__")
            suffix_part = "__".join(parts[1:]) if len(parts) > 1 else ""
            if suffix_part:
                new_name = f"{new_token}__{suffix_part}{ext}"
            else:
                new_name = f"{new_token}{ext}"
        elif flatten_mode == "plain":
            new_name = f"{new_token}{ext}"
        else:
            raise ValueError(f"Unknown flatten mode: {flatten_mode}")
        if flatten_mode != "none":
            entry.rel_dst = Path(new_name)
        counter += 1


def route_non_images(entries: Sequence[FileEntry], policy: NonImagePolicy) -> None:
    if policy == "keep":
        return
    if policy == "skip":
        for entry in entries:
            if not entry.is_image:
                entry.keep = False
        return
    if policy == "collect":
        for entry in entries:
            if not entry.keep:
                continue
            if not entry.is_image:
                entry.rel_dst = Path(PREP_NON_IMAGE_DIR) / entry.rel_dst


def build_actions(config: ArrangeConfig, entries: Iterable[FileEntry]) -> list[Action]:
    actions: list[Action] = []
    for entry in entries:
        if not entry.keep:
            if entry.is_duplicate:
                actions.append(
                    Action(
                        kind="drop",
                        src=entry.src,
                        dst=None,
                        reason="duplicate",
                    )
                )
            else:
                actions.append(
                    Action(
                        kind="drop",
                        src=entry.src,
                        dst=None,
                        reason="non-image-skip",
                    )
                )
            continue
        rel_dst = Path(config.dst_subdir) / entry.rel_dst if config.dst_subdir else entry.rel_dst
        dst = config.dst / rel_dst
        if config.link_mode == "copy":
            kind = "copy"
        elif config.link_mode == "move":
            kind = "move"
        elif config.link_mode == "hard":
            kind = "link"
        else:
            raise ValueError(f"Unknown link mode: {config.link_mode}")
        actions.append(
            Action(
                kind=kind,
                src=entry.src,
                dst=dst,
                reason=None,
            )
        )
    return actions
