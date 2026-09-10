from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

from ..config import ProcessConfig
from ..fs import PREP_NON_IMAGE_DIR, ensure_parent, is_image, iter_files, require_image_source
from ..logging import Logger
from ..manifest import emit_prep_manifest

from .ops import apply_process_ops, read_process_write_opencv, resolve_output_path
from .vsample import sample_video_frames


def handle_non_image(
    path: Path,
    rel: Path,
    config: ProcessConfig,
    logger: Logger,
) -> None:
    if config.non_image == "skip":
        if config.dry_run or config.verbose:
            logger.info(f"[SKIP NON-IMAGE] {path}")
        return
    if config.non_image == "collect":
        if rel.parts and rel.parts[0] == PREP_NON_IMAGE_DIR:
            rel_out = rel
        else:
            rel_out = Path(PREP_NON_IMAGE_DIR) / rel
    else:
        rel_out = rel
    dst = config.dst / rel_out
    if config.dry_run:
        logger.info(f"[NON-IMAGE {config.transfer.upper()}] {path} -> {dst}")
        return
    ensure_parent(dst)
    if config.transfer == "copy":
        dst.write_bytes(path.read_bytes())
    else:
        ensure_parent(dst)
        path.replace(dst)
    if config.verbose:
        logger.info(f"[NON-IMAGE {config.transfer.upper()}] {path} -> {dst}")


def run_process_core(config: ProcessConfig) -> None:
    require_image_source(config.src, allow_video=True)
    logger = Logger(verbose=config.verbose)
    frame_entries, temp_dirs = sample_video_frames(config, logger)

    def _process_image(path: Path, rel: Path) -> None:
        dst = resolve_output_path(rel, config.dst, config.to_format)
        ops = []
        if config.orient:
            ops.append("orient")
        if config.resize_mode != "none" and config.size is not None:
            ops.append(f"resize:{config.resize_mode}")
        if config.grayscale:
            ops.append("grayscale")
        if config.to_format is not None:
            ops.append(f"to_format:{config.to_format}")
        ops_str = ",".join(ops) if ops else "none"

        if config.dry_run:
            logger.info(f"[PROCESSED] {path} -> {dst} ({ops_str})")
            return
        try:
            ensure_parent(dst)
            if config.backend == "opencv":
                read_process_write_opencv(path, dst, config)
            else:
                with Image.open(path) as img:
                    img = apply_process_ops(img, config)
                    save_kwargs = {}
                    save_format = None
                    if config.to_format is not None:
                        ext = "." + config.to_format.lower().lstrip(".")
                        save_format = Image.registered_extensions().get(ext, config.to_format.upper())
                    output_format = save_format or img.format or Image.registered_extensions().get(dst.suffix.lower())
                    if output_format == "JPEG" or dst.suffix.lower() in (".jpg", ".jpeg"):
                        save_kwargs["quality"] = config.jpeg_quality
                    if config.to_format is not None:
                        img.save(dst, format=save_format, **save_kwargs)
                    else:
                        img.save(dst, **save_kwargs)
        except Exception as exc:
            logger.error(f"[ERROR] Failed to process {path}: {exc}")
            return
        if config.verbose:
            logger.info(f"[PROCESSED] {path} -> {dst} ({ops_str})")
        if config.transfer == "move":
            path.unlink()

    try:
        paths = sorted(iter_files(config.src))
        for path in tqdm(paths, desc="Processing files", unit=" file"):
            rel = path.relative_to(config.src)
            if not is_image(path):
                handle_non_image(path, rel, config, logger)
                continue
            _process_image(path, rel)

        if frame_entries:
            for frame_path, rel in tqdm(frame_entries, desc="Processing frames", unit=" frame"):
                _process_image(frame_path, rel)
    finally:
        for tmp in temp_dirs:
            shutil.rmtree(tmp, ignore_errors=True)

    if config.manifest and not config.dry_run:
        emit_prep_manifest(config, "process")
