from __future__ import annotations

import inspect
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageFile, ImageOps

from cvsuite.common.core import ImageRecord, Record
from cvsuite.common.core import VisionDataset

from ..config import OrientConfig
from ..fs import ensure_parent, is_image, iter_files
from ..logging import Logger
from .deep_orientation import OrientationPrediction, predict_dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

EXIF_ORIENTATION_TAG = 274


@dataclass(frozen=True)
class OrientationDecision:
    method: str
    exif_orientation: int | None
    ccw_degrees: int
    scores: tuple[float, ...] | None = None

    @property
    def unchanged(self) -> bool:
        if self.method == "exif":
            return self.exif_orientation == 1
        return self.ccw_degrees == 0

    def apply(self, image: Image.Image) -> Image.Image:
        if self.method == "exif":
            return ImageOps.exif_transpose(image)
        if self.ccw_degrees == 0:
            return image.copy()
        return image.rotate(self.ccw_degrees, expand=True)

    def describe(self) -> str:
        if self.method == "exif":
            return f"exif:{self.exif_orientation}"
        if self.scores is None:
            return f"{self.method}:{self.ccw_degrees}"
        score_text = ",".join(f"{score:.4f}" for score in self.scores)
        return f"{self.method}:{self.ccw_degrees} [{score_text}]"


@dataclass
class OrientStats:
    images: int = 0
    exif_resolved: int = 0
    model_resolved: int = 0
    rewritten: int = 0
    copied: int = 0
    hardlinked: int = 0
    skipped_non_images: int = 0


@dataclass(frozen=True)
class ImageEntry:
    path: Path
    rel: Path
    width: int
    height: int
    exif_orientation: int | None


def _is_subpath(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _read_exif_orientation(image: Image.Image) -> int | None:
    try:
        exif = image.getexif()
    except Exception:
        return None
    value = exif.get(EXIF_ORIENTATION_TAG)
    if isinstance(value, int) and 1 <= value <= 8:
        return value
    return None


def _build_records_dataset(entries: Sequence[ImageEntry], root: Path) -> VisionDataset:
    records = [
        Record(
            image=ImageRecord(path=entry.path, width=entry.width, height=entry.height),
            split="train",
        )
        for entry in entries
    ]
    return VisionDataset(records=records, root=root)


def _prediction_decisions(
    entries: Sequence[ImageEntry],
    predictions: Sequence[OrientationPrediction],
) -> list[tuple[ImageEntry, OrientationDecision]]:
    if len(entries) != len(predictions):
        raise RuntimeError(f"Expected {len(entries)} fallback-model predictions, received {len(predictions)}.")

    decisions: list[tuple[ImageEntry, OrientationDecision]] = []
    for entry, prediction in zip(entries, predictions):
        decisions.append(
            (
                entry,
                OrientationDecision(
                    method="deep_orientation",
                    exif_orientation=None,
                    ccw_degrees=prediction.ccw_degrees,
                    scores=prediction.scores,
                ),
            )
        )
    return decisions


def _normalized_exif_bytes(image: Image.Image) -> bytes | None:
    try:
        exif = image.getexif()
    except Exception:
        return None
    if not exif:
        return None
    exif[EXIF_ORIENTATION_TAG] = 1
    try:
        return exif.tobytes()
    except Exception:
        return None


def _build_save_kwargs(image: Image.Image) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    exif_bytes = _normalized_exif_bytes(image)
    if exif_bytes is not None:
        kwargs["exif"] = exif_bytes

    icc_profile = image.info.get("icc_profile")
    if icc_profile is not None:
        kwargs["icc_profile"] = icc_profile

    dpi = image.info.get("dpi")
    if dpi is not None:
        kwargs["dpi"] = dpi

    xmp = image.info.get("xmp")
    if xmp is not None:
        kwargs["xmp"] = xmp

    if image.format == "JPEG":
        kwargs["quality"] = "keep"
        kwargs["subsampling"] = "keep"

    return kwargs


def _copy_or_hardlink(src: Path, dst: Path, try_hardlink: bool) -> str:
    ensure_parent(dst)
    if try_hardlink:
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def _save_with_metadata(src_path: Path, source: Image.Image, oriented: Image.Image, dst: Path) -> None:
    ensure_parent(dst)
    save_kwargs = _build_save_kwargs(source)
    save_format = source.format
    try:
        if save_format is None:
            oriented.save(dst, **save_kwargs)
        else:
            oriented.save(dst, format=save_format, **save_kwargs)
    except Exception:
        fallback = {key: value for key, value in save_kwargs.items() if key in {"exif", "icc_profile", "dpi"}}
        if save_format is None:
            oriented.save(dst, **fallback)
        else:
            oriented.save(dst, format=save_format, **fallback)
    shutil.copystat(src_path, dst)


def _materialize_image(
    path: Path,
    rel: Path,
    decision: OrientationDecision,
    *,
    config: OrientConfig,
    logger: Logger,
    stats: OrientStats,
) -> None:
    dst = config.dst / rel
    action = "hardlink" if decision.unchanged and config.try_hardlink else "copy"
    if not decision.unchanged:
        action = "rewrite"

    if config.dry_run:
        logger.info(f"[ORIENT {action.upper()}] {path} -> {dst} ({decision.describe()})")
        if decision.unchanged:
            if config.try_hardlink:
                stats.hardlinked += 1
            else:
                stats.copied += 1
        else:
            stats.rewritten += 1
        return

    if decision.unchanged:
        actual_action = _copy_or_hardlink(path, dst, config.try_hardlink)
        if actual_action == "hardlink":
            stats.hardlinked += 1
        else:
            stats.copied += 1
        if config.verbose:
            logger.info(f"[ORIENT {actual_action.upper()}] {path} -> {dst} ({decision.describe()})")
        return

    try:
        with Image.open(path) as source:
            oriented = decision.apply(source)
            try:
                _save_with_metadata(path, source, oriented, dst)
            finally:
                oriented.close()
    except Exception as exc:
        logger.error(f"[ERROR] Failed to orient {path}: {exc}")
        return

    stats.rewritten += 1
    if config.verbose:
        logger.info(f"[ORIENT REWRITE] {path} -> {dst} ({decision.describe()})")


def run_orient_core(config: OrientConfig) -> None:
    if _is_subpath(config.dst, config.src):
        raise SystemExit("Destination must be outside the source tree.")

    logger = Logger(verbose=config.verbose)
    stats = OrientStats()
    image_entries: list[ImageEntry] = []

    for path in iter_files(config.src):
        rel = path.relative_to(config.src)
        if not is_image(path):
            stats.skipped_non_images += 1
            logger.debug(f"[SKIP NON-IMAGE] {path}")
            continue

        stats.images += 1
        try:
            with Image.open(path) as image:
                exif_orientation = _read_exif_orientation(image)
                width, height = image.size
        except Exception as exc:
            logger.error(f"[ERROR] Failed to inspect {path}: {exc}")
            continue

        image_entries.append(
            ImageEntry(
                path=path,
                rel=rel,
                width=width,
                height=height,
                exif_orientation=exif_orientation,
            )
        )

    predictions_by_path: dict[Path, OrientationDecision] = {}
    needs_model = any(entry.exif_orientation is None for entry in image_entries)

    if needs_model:
        try:
            predict_kwargs = {
                "device_hint": config.device,
                "precision": config.precision,
                "batch_size": config.batch,
                "weights_dir": config.weights,
            }
            if "no_resume" in inspect.signature(predict_dataset).parameters:
                predict_kwargs["no_resume"] = config.no_resume
            predictions = predict_dataset(
                _build_records_dataset(image_entries, config.src),
                **predict_kwargs,
            )
            for entry, decision in _prediction_decisions(image_entries, predictions):
                predictions_by_path[entry.path] = decision
        except Exception as exc:
            raise SystemExit(f"Orientation model fallback failed: {exc}") from exc

    for entry in image_entries:
        if entry.exif_orientation is not None:
            stats.exif_resolved += 1
            decision = OrientationDecision(
                method="exif",
                exif_orientation=entry.exif_orientation,
                ccw_degrees=0,
            )
        else:
            decision = predictions_by_path.get(entry.path)
            if decision is None:
                logger.error(f"[ERROR] Missing fallback-model prediction for {entry.path}")
                continue
            stats.model_resolved += 1

        _materialize_image(entry.path, entry.rel, decision, config=config, logger=logger, stats=stats)

    if config.verbose:
        logger.info(
            "[SUMMARY] "
            f"images={stats.images} exif={stats.exif_resolved} model={stats.model_resolved} "
            f"rewritten={stats.rewritten} copied={stats.copied} hardlinked={stats.hardlinked} "
            f"skipped_non_images={stats.skipped_non_images}"
        )
