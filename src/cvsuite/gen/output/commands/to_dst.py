"""Move the generated image artifact to a destination file or directory."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import os
import shutil

from cvsuite.common.stats import attach_with_stats_flag, build_gen_stats, emit_stats_yaml, make_export_context

GEN_OUTPUT_KEY_ATTR = "gen_output_key"
GEN_SOURCE_IMAGE_ATTR = "gen_source_image"
GEN_SOURCE_KEY_ATTR = "gen_source_key"


def attach(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dst", type=Path, help="Destination file path or directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing destination image files.")
    parser.add_argument(
        "--no-original",
        action="store_true",
        help="For edit outputs, do not export hardlinked copies of the original source images.",
    )
    attach_with_stats_flag(parser)


def _resolve_record_path(rec_path: Path, root: Path | None, caller_cwd: Path) -> Path:
    if rec_path.is_absolute():
        return rec_path

    candidates: list[Path] = []
    if root is not None:
        base_root = root if isinstance(root, Path) else Path(root)
        if not base_root.is_absolute():
            base_root = caller_cwd / base_root
        if rec_path.parts and rec_path.parts[0] == base_root.name:
            candidates.append(base_root.parent / rec_path)
        candidates.append(base_root / rec_path)

    candidates.append(caller_cwd / rec_path)
    candidates.append(rec_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _generated_record_path(dataset, record) -> Path:
    resolved = _resolve_record_path(record.image.path, dataset.root, Path.cwd())
    if not resolved.exists():
        raise SystemExit(f"Generated image not found: {resolved}")
    return resolved


def _directory_destination(dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return dst / f"{stamp}.png"


def _resolve_destination(dst: Path) -> Path:
    if dst.exists() and dst.is_dir():
        return _directory_destination(dst)
    if dst.suffix:
        dst.parent.mkdir(parents=True, exist_ok=True)
        return dst
    return _directory_destination(dst)


def _resolve_multi_output_dir(dst: Path) -> Path:
    if dst.suffix:
        dst = dst.with_suffix("")
    dst.mkdir(parents=True, exist_ok=True)
    return dst


def _record_output_name(record, idx: int) -> str:
    key = str(record.attributes.get(GEN_OUTPUT_KEY_ATTR) or "").strip()
    if not key:
        key = datetime.now().strftime(f"%Y%m%d_%H%M%S_{idx:04d}")
    return f"{key}.png"


def _move_generated(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(str(dst))
        if src.exists() and src.samefile(dst):
            return
        if dst.is_dir():
            raise IsADirectoryError(str(dst))
        dst.unlink()
    shutil.move(str(src), str(dst))


def _should_export_originals(dataset, args: argparse.Namespace) -> bool:
    if bool(getattr(args, "no_original", False)):
        return False
    return str(dataset.meta.get("gen_mode") or "") == "edit"


def _resolve_original_source_path(dataset, record) -> Path | None:
    source_value = str(record.attributes.get(GEN_SOURCE_IMAGE_ATTR) or "").strip()
    if not source_value:
        return None
    return _resolve_record_path(Path(source_value), dataset.meta.get("gen_source"), Path.cwd())


def _original_output_name(record, resolved_source: Path) -> str:
    source_key = str(record.attributes.get(GEN_SOURCE_KEY_ATTR) or "").strip()
    suffix = resolved_source.suffix or ".img"
    if source_key:
        return f"{source_key}{suffix}"
    return resolved_source.name


def _link_original(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(str(dst))
        if src.exists() and src.samefile(dst):
            return
        if dst.is_dir():
            raise IsADirectoryError(str(dst))
        dst.unlink()
    os.link(src, dst)


def _export_originals(dataset, dst_dir: Path, *, overwrite: bool) -> list[Path]:
    artifacts: list[Path] = []
    seen: set[str] = set()
    for record in dataset.records:
        source_path = _resolve_original_source_path(dataset, record)
        if source_path is None:
            continue
        if not source_path.exists():
            raise SystemExit(f"Original source image not found: {source_path}")
        source_key = str(record.attributes.get(GEN_SOURCE_KEY_ATTR) or source_path)
        if source_key in seen:
            continue
        seen.add(source_key)
        dst = dst_dir / _original_output_name(record, source_path)
        _link_original(source_path, dst, overwrite=overwrite)
        artifacts.append(dst)
    return artifacts


def run(dataset, args: argparse.Namespace):
    primary_artifacts: list[Path] = []
    overwrite = bool(getattr(args, "overwrite", False))
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    if len(dataset.records) == 1:
        src = _generated_record_path(dataset, dataset.records[0])
        dst = _resolve_destination(args.dst)
        _move_generated(src, dst, overwrite=overwrite)
        dataset.records[0].image.path = dst
        dataset.root = dst.parent
        primary_artifacts = [dst]
        if _should_export_originals(dataset, args):
            primary_artifacts.extend(_export_originals(dataset, dst.parent, overwrite=overwrite))
    else:
        dst_dir = _resolve_multi_output_dir(args.dst)
        for idx, record in enumerate(dataset.records):
            src = _generated_record_path(dataset, record)
            dst = dst_dir / _record_output_name(record, idx)
            _move_generated(src, dst, overwrite=overwrite)
            record.image.path = Path(dst.name)
            primary_artifacts.append(dst)
        dataset.root = dst_dir
        if _should_export_originals(dataset, args):
            primary_artifacts.extend(_export_originals(dataset, dst_dir, overwrite=overwrite))
    if bool(getattr(args, "with_stats", False)):
        ctx = make_export_context(
            branch="gen",
            command="to-dst",
            dst=args.dst,
            primary_artifacts=primary_artifacts,
        )
        emit_stats_yaml(build_gen_stats(dataset, ctx), ctx)
    return dataset
