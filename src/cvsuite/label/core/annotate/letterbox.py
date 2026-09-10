from typing import Tuple
import cv2
import numpy as np


def letterbox(img: np.ndarray, new_shape: int | Tuple[int, int], color=(114, 114, 114)):
    """Resize image with unchanged aspect ratio using padding (YOLO-style).

    Args:
        img: BGR image.
        new_shape: Target size (int for square or (height, width)).
        color: Padding color.

    Returns:
        canvas: Padded image.
        ratio: Scaling ratio applied to the original image.
        pad: Tuple of (pad_w, pad_h) applied to left/top.
    """
    shape = img.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    target_h, target_w = new_shape

    # Compute scale
    r = min(target_h / shape[0], target_w / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w, h)

    # Compute padding to reach target
    dw = (target_w - new_unpad[0]) / 2  # width padding
    dh = (target_h - new_unpad[1]) / 2  # height padding

    # Resize
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    canvas = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return canvas, r, (dw, dh)
