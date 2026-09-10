from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageOps

from ..config import ProcessConfig, ResizeMode, SquareMode


def compute_resize_size(
    mode: ResizeMode,
    square_mode: Optional[SquareMode],
    size: Optional[int],
    width: int,
    height: int,
) -> Tuple[int, int]:
    if mode == "none" or size is None:
        return width, height
    if mode in ("long", "cap-long"):
        if width >= height:
            scale = size / float(width)
        else:
            scale = size / float(height)
        if mode == "cap-long" and max(width, height) <= size:
            return width, height
        return max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    if mode in ("short", "cap-short"):
        if width <= height:
            scale = size / float(width)
        else:
            scale = size / float(height)
        if mode == "cap-short" and min(width, height) <= size:
            return width, height
        return max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    if mode == "square":
        if square_mode == "distort":
            return size, size
        if width == height:
            return size, size
        if square_mode in ("pad", "crop"):
            if width >= height:
                scale = size / float(width)
            else:
                scale = size / float(height)
            new_w = max(1, int(round(width * scale)))
            new_h = max(1, int(round(height * scale)))
            return new_w, new_h
    return width, height


def apply_process_ops(image: Image.Image, config: ProcessConfig) -> Image.Image:
    img = image
    if config.orient:
        img = ImageOps.exif_transpose(img)  # type: ignore[name-defined]
    if config.resize_mode != "none" and config.size is not None:
        new_w, new_h = compute_resize_size(
            config.resize_mode,
            config.square_mode,
            config.size,
            img.width,
            img.height,
        )
        img = img.resize((new_w, new_h), Image.BILINEAR)
        if config.resize_mode == "square" and config.square_mode in ("pad", "crop") and config.size is not None:
            target = config.size
            if config.square_mode == "pad":
                padded = Image.new(img.mode, (target, target))
                x = (target - img.width) // 2
                y = (target - img.height) // 2
                padded.paste(img, (x, y))
                img = padded
            elif config.square_mode == "crop":
                left = max(0, (img.width - target) // 2)
                upper = max(0, (img.height - target) // 2)
                right = min(img.width, left + target)
                lower = min(img.height, upper + target)
                img = img.crop((left, upper, right, lower))
    if config.grayscale:
        img = img.convert("L")
    return img


def read_process_write_opencv(path: Path, dst: Path, config: ProcessConfig) -> None:
    """Apply process operations with OpenCV while handling EXIF explicitly."""
    import cv2
    import numpy as np

    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not decode {path}")

    if config.orient:
        try:
            with Image.open(path) as metadata_image:
                orientation = int(metadata_image.getexif().get(274, 1))
        except Exception:
            orientation = 1
        if orientation == 2:
            image = np.fliplr(image)
        elif orientation == 3:
            image = np.rot90(image, 2)
        elif orientation == 4:
            image = np.flipud(image)
        elif orientation == 5:
            image = np.swapaxes(image, 0, 1)
        elif orientation == 6:
            image = np.rot90(image, 3)
        elif orientation == 7:
            image = np.flip(np.swapaxes(image, 0, 1), axis=(0, 1))
        elif orientation == 8:
            image = np.rot90(image, 1)
        image = np.ascontiguousarray(image)

    height, width = image.shape[:2]
    if config.resize_mode != "none" and config.size is not None:
        new_w, new_h = compute_resize_size(
            config.resize_mode,
            config.square_mode,
            config.size,
            width,
            height,
        )
        if (new_w, new_h) != (width, height):
            shrinking = new_w < width or new_h < height
            interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        if config.resize_mode == "square" and config.square_mode in ("pad", "crop"):
            target = config.size
            if config.square_mode == "pad":
                top = (target - image.shape[0]) // 2
                bottom = target - image.shape[0] - top
                left = (target - image.shape[1]) // 2
                right = target - image.shape[1] - left
                image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            else:
                top = max(0, (image.shape[0] - target) // 2)
                left = max(0, (image.shape[1] - target) // 2)
                image = image[top:top + target, left:left + target]

    if config.grayscale and image.ndim == 3:
        conversion = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        image = cv2.cvtColor(image, conversion)

    suffix = dst.suffix.lower()
    params: list[int] = []
    if suffix in (".jpg", ".jpeg"):
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        params = [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
    elif suffix == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, config.jpeg_quality]
    elif suffix == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    ok, output = cv2.imencode(suffix, image, params)
    if not ok:
        raise ValueError(f"OpenCV could not encode {dst}")
    output.tofile(dst)


def resolve_output_path(
    src_rel: Path,
    dst_root: Path,
    to_format: Optional[str],
) -> Path:
    if to_format is None:
        return dst_root / src_rel
    suffix = "." + to_format.lower().lstrip(".")
    return dst_root / src_rel.with_suffix(suffix)
