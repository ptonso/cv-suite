from __future__ import annotations

import math
from pathlib import Path

from ..config import ProcessConfig
from ..fs import PREP_NON_IMAGE_DIR, PREP_VIDEO_FRAMES_DIR, is_image, iter_files
from ..logging import Logger

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency at runtime only when sampling videos.
    cv2 = None  # type: ignore[assignment]


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
}


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS and not is_image(path)


def _choose_stride(
    frame_count: int,
    fps: float,
    max_frames: int | None,
    min_gap_seconds: float | None,
) -> int:
    if min_gap_seconds is not None and fps > 0:
        return max(1, int(math.ceil(min_gap_seconds * fps)))
    if max_frames and max_frames > 0 and frame_count > 0:
        return max(1, frame_count // max_frames)
    return 1


def _frame_rel_path(rel_video: Path, base_name: str, mode: str) -> Path:
    if mode == "collect":
        rel_parent = rel_video.parent
        if rel_parent.parts and rel_parent.parts[0] == PREP_NON_IMAGE_DIR:
            rel_parent = Path(*rel_parent.parts[1:]) if len(rel_parent.parts) > 1 else Path()
        return Path(PREP_VIDEO_FRAMES_DIR) / rel_parent / base_name
    return rel_video.parent / f"{rel_video.stem}_frames" / base_name


def sample_video_frames(
    config: ProcessConfig,
    logger: Logger,
) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """Return (sampled_frame_path, rel_path) pairs and temp dirs to cleanup."""
    if config.video_sample_mode == "skip":
        return [], []
    if cv2 is None:
        logger.error("OpenCV is required for video sampling but is not installed.")
        return [], []

    base_ext = (config.to_format or "jpg").lower().lstrip(".") or "jpg"
    tmp_root = config.dst / f"{PREP_VIDEO_FRAMES_DIR}_tmp"
    temp_dirs: list[Path] = [] if config.dry_run else [tmp_root]
    if not config.dry_run:
        tmp_root.mkdir(parents=True, exist_ok=True)

    samples: list[tuple[Path, Path]] = []
    frame_seq = 1

    for video_path in iter_files(config.src):
        if not is_video(video_path):
            continue
        rel_video = video_path.relative_to(config.src)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video for sampling: {video_path}")
            continue

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stride = _choose_stride(frame_count, fps, config.video_sample_k, config.video_sample_min_gap)

        if config.dry_run:
            # Estimate count without decoding all frames.
            estimated = frame_count // stride + (1 if frame_count % stride else 0)
            if config.video_sample_k is not None:
                estimated = min(estimated, config.video_sample_k)
            for _ in range(estimated):
                fname = f"vid_{frame_seq:06d}.{base_ext}"
                rel_frame = _frame_rel_path(rel_video, fname, config.video_sample_mode)
                samples.append((config.src / rel_frame, rel_frame))
                frame_seq += 1
            logger.info(f"Planned {estimated} frames from {video_path.name} (stride={stride}, fps={fps:.2f}).")
            cap.release()
            continue

        out_dir = tmp_root / rel_video.parent / rel_video.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        saved_for_video = 0
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                if config.video_sample_k is None or saved_for_video < config.video_sample_k:
                    fname = f"vid_{frame_seq:06d}.{base_ext}"
                    frame_seq += 1
                    out_path = out_dir / fname
                    try:
                        ok = cv2.imwrite(str(out_path), frame)
                    except Exception as exc:  # pragma: no cover - best-effort logging
                        logger.error(f"Failed to write frame from {video_path}: {exc}")
                        break
                    if ok:
                        saved_for_video += 1
                        rel_frame = _frame_rel_path(rel_video, fname, config.video_sample_mode)
                        samples.append((out_path, rel_frame))
                if config.video_sample_k is not None and saved_for_video >= config.video_sample_k:
                    break
            frame_idx += 1

        cap.release()
        if saved_for_video:
            logger.info(f"Sampled {saved_for_video} frames from {video_path.name} (stride={stride}, fps={fps:.2f}).")
        else:
            logger.error(f"No frames sampled from video: {video_path}")

    return samples, temp_dirs
