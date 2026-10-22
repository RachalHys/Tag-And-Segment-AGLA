"""Explicit RGB view construction and CLIP geometry for TAS v7.

FG-U blurs the FULL image and composites the result ONLY inside foreground.
Construction never resizes, crops, normalizes, or substitutes an empty mask.
Legacy E1 raw-before-processor masking is a distinct geometry contract.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ViewConfig:
    version: str = "tas-v7-fgu-v1"
    geometry: str = "processor_canvas"
    sigma_ratio: float = 0.04
    gaussian_truncate: float = 3.0
    border: str = "reflect101"
    color_space: str = "RGB-sRGB-encoded-uint8"
    mask_interpolation: str = "nearest"
    empty_mask_policy: str = "mathematical"

    def __post_init__(self):
        if self.geometry not in {"processor_canvas", "legacy_raw"}:
            raise ValueError("Unknown geometry contract")
        if not math.isfinite(self.sigma_ratio) or self.sigma_ratio < 0:
            raise ValueError("sigma_ratio must be finite and nonnegative")
        if not math.isfinite(self.gaussian_truncate) or self.gaussian_truncate <= 0:
            raise ValueError("gaussian_truncate must be positive")
        if self.border not in {"reflect101", "reflect", "replicate"}:
            raise ValueError("Supported borders: reflect101, reflect, replicate")
        if (
            self.mask_interpolation != "nearest"
            or self.empty_mask_policy != "mathematical"
        ):
            raise ValueError(
                "Changed mask semantics require a separately implemented view version"
            )
        if self.color_space != "RGB-sRGB-encoded-uint8":
            raise ValueError("Only explicit RGB uint8 blur is implemented")

    def to_dict(self):
        return asdict(self)


def binary_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Validate a binary HxW mask instead of silently thresholding soft masks."""
    array = np.asarray(mask)
    if array.shape != tuple(shape):
        raise ValueError(f"Mask {array.shape} does not match image canvas {shape}")
    if array.dtype != np.bool_ and not np.isin(array, [0, 1]).all():
        raise ValueError("Union masks must contain only boolean/0/1 values")
    return array.astype(bool, copy=False)


def gaussian_full_image(
    image: Image.Image,
    sigma: float,
    *,
    truncate: float = 3.0,
    border: str = "reflect101",
) -> Image.Image:
    if image.mode != "RGB":
        raise ValueError(
            "Image must be explicitly converted to RGB before view construction"
        )
    if not math.isfinite(sigma) or sigma < 0 or truncate <= 0:
        raise ValueError("Invalid Gaussian parameters")
    if sigma == 0:
        return image.copy()
    import cv2

    borders = {
        "reflect101": cv2.BORDER_REFLECT_101,
        "reflect": cv2.BORDER_REFLECT,
        "replicate": cv2.BORDER_REPLICATE,
    }
    if border not in borders:
        raise ValueError("Unknown Gaussian border")
    radius = max(1, math.ceil(truncate * sigma))
    blurred = cv2.GaussianBlur(
        np.asarray(image),
        (2 * radius + 1, 2 * radius + 1),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=borders[border],
    )
    return Image.fromarray(blurred, mode="RGB")


def create_views(
    image: Image.Image,
    union_mask: np.ndarray,
    sigma_ratio: float = 0.04,
    *,
    config: ViewConfig | None = None,
) -> dict[str, Image.Image]:
    """Construct ORIG/OBJ-H/BH/FG-U on the caller's already-locked canvas.

    With M=0: object is RGB black, BH/FG-U equal input.  With M=1:
    object equals input, BH is black, FG-U is the full Gaussian.  E1's legacy
    no-detection fallback must be applied separately via legacy_object_view.
    """
    if image.mode != "RGB":
        raise ValueError("create_views requires RGB input")
    cfg = config if config is not None else ViewConfig(sigma_ratio=sigma_ratio)
    pixels = np.asarray(image)
    mask = binary_mask(union_mask, pixels.shape[:2])
    object_pixels, background_pixels = pixels.copy(), pixels.copy()
    object_pixels[~mask] = 0
    background_pixels[mask] = 0
    foreground_blur = pixels.copy()
    if mask.any() and cfg.sigma_ratio > 0:
        full_blur = np.asarray(
            gaussian_full_image(
                image,
                cfg.sigma_ratio * min(pixels.shape[:2]),
                truncate=cfg.gaussian_truncate,
                border=cfg.border,
            )
        )
        foreground_blur[mask] = full_blur[mask]
    return {
        "ORIG": image.copy(),
        "OBJ-H": Image.fromarray(object_pixels, mode="RGB"),
        "BH": Image.fromarray(background_pixels, mode="RGB"),
        "FG-U": Image.fromarray(foreground_blur, mode="RGB"),
    }


def legacy_object_view(
    image: Image.Image, union_mask: np.ndarray, *, fallback: bool = False
) -> Image.Image:
    """Mask RAW pixels before legacy CLIP processing; retain explicit fallback.

    fallback=True means the original pipeline returned raw input (empty label
    inventory or no detector boxes).  An empty segmentation alone is not proof
    that fallback occurred, and remains black when fallback=False.
    """
    if image.mode != "RGB":
        raise ValueError("Legacy input must be RGB")
    mask = binary_mask(union_mask, (image.height, image.width))
    if fallback:
        return image.copy()
    pixels = np.array(image, copy=True)
    pixels[~mask] = 0
    return Image.fromarray(pixels, mode="RGB")


def _processor_setting(processor: Any, name: str, default=None):
    return (
        processor.get(name, default)
        if isinstance(processor, dict)
        else getattr(processor, name, default)
    )


def processor_geometry_canvas(
    image: Image.Image, union_mask: np.ndarray, processor: Any
) -> tuple[Image.Image, np.ndarray, dict]:
    """Apply HF CLIP's resize-shortest / center-crop geometry to RGB AND mask.

    Mirrors Transformers 4.35.2 image_transforms get_resize_output_image_size
    and center_crop (floor integer dimensions/offsets).  RGB uses the supplied
    processor's Pillow resample; masks use nearest.  Rescale/normalize happen
    later with processor do_resize=False and do_center_crop=False.
    Unknown processor geometry is rejected instead of approximated.
    """
    if image.mode != "RGB":
        raise ValueError("Geometry requires RGB input")
    mask = binary_mask(union_mask, (image.height, image.width))
    rgb = image.copy()
    mask_image = Image.fromarray(mask.astype(np.uint8), mode="L")
    original_size = [rgb.height, rgb.width]
    resize_to = None
    resample = int(_processor_setting(processor, "resample", Image.Resampling.BICUBIC))
    if _processor_setting(processor, "do_resize", True):
        size = _processor_setting(processor, "size")
        if not isinstance(size, dict) or set(size) != {"shortest_edge"}:
            raise ValueError(
                "Expected CLIP size={'shortest_edge': N}; unsupported geometry must be explicit"
            )
        edge = int(size["shortest_edge"])
        if edge <= 0:
            raise ValueError("Processor shortest edge must be positive")
        width, height = rgb.size
        if width <= height:
            width, height = edge, int(edge * height / width)
        else:
            width, height = int(edge * width / height), edge
        resize_to = [height, width]
        rgb = rgb.resize((width, height), resample=resample)
        mask_image = mask_image.resize(
            (width, height), resample=Image.Resampling.NEAREST
        )
    crop_box = None
    if _processor_setting(processor, "do_center_crop", True):
        size = _processor_setting(processor, "crop_size")
        if not isinstance(size, dict) or set(size) != {"height", "width"}:
            raise ValueError("Expected explicit processor crop_size height/width")
        crop_h, crop_w = int(size["height"]), int(size["width"])
        if crop_h <= 0 or crop_w <= 0:
            raise ValueError("Crop dimensions must be positive")
        # PIL crop pads with zero; offsets reproduce HF's symmetric padding.
        # HF places the odd extra padded pixel on bottom/right.  A negative
        # floor crop offset would place it on top/left, so construct padding.
        if rgb.width < crop_w or rgb.height < crop_h:
            new_w, new_h = max(crop_w, rgb.width), max(crop_h, rgb.height)
            offset = ((new_w - rgb.width) // 2, (new_h - rgb.height) // 2)
            padded_rgb, padded_mask = (
                Image.new("RGB", (new_w, new_h)),
                Image.new("L", (new_w, new_h)),
            )
            padded_rgb.paste(rgb, offset)
            padded_mask.paste(mask_image, offset)
            rgb, mask_image = padded_rgb, padded_mask
        left, top = (rgb.width - crop_w) // 2, (rgb.height - crop_h) // 2
        crop_box = [left, top, left + crop_w, top + crop_h]
        rgb, mask_image = rgb.crop(tuple(crop_box)), mask_image.crop(tuple(crop_box))
    metadata = {
        "contract": "processor_canvas",
        "implementation": "hf-clip-4.35.2-geometry-v1",
        "original_hw": original_size,
        "resize_hw": resize_to,
        "crop_box_xyxy": crop_box,
        "canvas_hw": [rgb.height, rgb.width],
        "image_resample": resample,
        "mask_resample": "nearest",
        "normalization_applied": False,
    }
    return rgb, np.asarray(mask_image).astype(bool), metadata


def area_matched_random_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Exact-area compact region at a random location, independent of labels.

    This controls intervention area, not shape/semantic content.  It is not a
    distribution of object-like masks.  The caller records the per-image seed.
    """
    original = np.asarray(mask)
    if original.ndim != 2:
        raise ValueError("Mask must be two dimensional")
    original = binary_mask(original, original.shape)
    h, w = original.shape
    count = int(original.sum())
    result = np.zeros_like(original)
    if not count:
        return result
    if count == h * w:
        return np.ones_like(original)
    block_w = min(w, max(math.ceil(count / h), int(round(math.sqrt(count)))))
    full_rows, tail = divmod(count, block_w)
    block_h = full_rows + bool(tail)
    top = int(rng.integers(0, h - block_h + 1))
    left = int(rng.integers(0, w - block_w + 1))
    result[top : top + full_rows, left : left + block_w] = True
    if tail:
        result[top + full_rows, left : left + tail] = True
    return result
