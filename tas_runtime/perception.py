"""The original BLIP inventory and YOLO/MobileSAM pipeline with reusable metadata.

Question text, cleanup, detector/SAM defaults and original fallback semantics
remain explicit. Inventory labels only select regions; they never enter LLaVA's
caption prompt. Perception caches are independent of the fusion policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .artifacts import sha256_file


@dataclass
class PerceptionResult:
    mask: np.ndarray
    instances: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    metadata: dict

    def arrays(self):
        return {
            "mask": self.mask.astype(bool),
            "instances": self.instances.astype(bool),
            "boxes": self.boxes.astype(np.float32),
            "scores": self.scores.astype(np.float32),
        }

    @classmethod
    def from_cache(cls, value):
        arrays, metadata = value
        return cls(
            arrays["mask"],
            arrays["instances"],
            arrays["boxes"],
            arrays["scores"],
            metadata,
        )


class PerceptionRuntime:
    def __init__(self, paths, *, device="cuda:0", confidence=0.2):
        import torch
        import clip
        from transformers import BlipProcessor, BlipForQuestionAnswering
        from ultralytics import YOLOWorld
        from mobile_sam import sam_model_registry, SamPredictor
        from eval_tas.inventory_detector import BLIPInventoryDetector

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Perception runs on GPU at the existing precisions; CPU preparation only downloads its assets."
            )
        self.device, self.confidence = device, confidence
        # Reuse the original four questions and text-cleanup implementation.
        self.inventory = BLIPInventoryDetector.__new__(BLIPInventoryDetector)
        self.inventory.device = device
        self.inventory.processor = BlipProcessor.from_pretrained(
            paths["blip"], local_files_only=True
        )
        self.inventory.model = (
            BlipForQuestionAnswering.from_pretrained(
                paths["blip"], torch_dtype=torch.float16, local_files_only=True
            )
            .to(device)
            .eval()
        )
        self.yolo = YOLOWorld(str(paths["yolo"])).to(device)
        # The pinned WorldModel caches this exact CLIP instance across class updates.
        # Supplying the local checkpoint avoids a hidden first-prediction download.
        self.yolo.model.clip_model = clip.load(
            str(paths["clip_text"]), device=device, jit=False
        )[0].eval()
        sam = sam_model_registry["vit_t"](checkpoint=str(paths["mobile_sam"])).to(
            device=device
        )
        self.sam = SamPredictor(
            sam.eval()
        )  # original SAM FP32, no implicit half conversion

    def run(self, image, *, inventory_override=None, boxes_override=None):
        import torch

        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        inventory = (
            self.inventory.get_inventory(image)
            if inventory_override is None
            else list(inventory_override)
        )
        metadata = {
            "inventory": inventory,
            "inventory_source": "BLIP"
            if inventory_override is None
            else "annotation_diagnostic",
            "labels_enter_caption_prompt": False,
            "legacy_fallback": False,
            "confidence": self.confidence,
            "expansion_ratio": 0.0,
        }
        if boxes_override is None and not inventory:
            metadata.update(legacy_fallback=True, fallback_reason="empty_inventory")
            return PerceptionResult(
                np.zeros((height, width), bool),
                np.zeros((0, height, width), bool),
                np.empty((0, 4), np.float32),
                np.empty(0, np.float32),
                metadata,
            )
        if boxes_override is None:
            classes = inventory + ["object"]
            self.yolo.set_classes(classes)
            result = self.yolo.predict(image, conf=self.confidence, verbose=False)[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            metadata["proposed_labels"] = [classes[index] for index in class_ids]
            metadata["yolo_settings"] = {
                key: getattr(self.yolo.predictor.args, key, None)
                for key in (
                    "imgsz",
                    "iou",
                    "max_det",
                    "agnostic_nms",
                    "half",
                    "rect",
                    "augment",
                )
            }
        else:
            boxes = np.asarray(boxes_override, dtype=np.float32).reshape(-1, 4)
            scores = np.ones(len(boxes), np.float32)
            metadata["proposed_labels"] = ["annotation_box"] * len(boxes)
        if len(boxes) == 0:
            metadata.update(legacy_fallback=True, fallback_reason="no_boxes")
            return PerceptionResult(
                np.zeros((height, width), bool),
                np.zeros((0, height, width), bool),
                boxes.reshape(0, 4),
                scores,
                metadata,
            )
        self.sam.set_image(rgb)
        instances = []
        qualities = []
        with torch.inference_mode():
            for box in boxes:
                masks, iou, _ = self.sam.predict(
                    box=box[None, :], multimask_output=False
                )
                instances.append(masks[0].astype(bool))
                qualities.append(float(iou[0]))
        masks = np.stack(instances)
        metadata["sam_predicted_iou"] = qualities
        metadata["mask_fraction"] = float(masks.any(axis=0).mean())
        metadata["mask_empty"] = not bool(masks.any())
        metadata["mask_full"] = bool(masks.any(axis=0).all())
        return PerceptionResult(masks.any(axis=0), masks, boxes, scores, metadata)

    def close(self):
        import gc
        import torch

        self.inventory = self.yolo = self.sam = None
        gc.collect()
        torch.cuda.empty_cache()  # one stage-boundary release, never per image


def perception_identity(asset_identity, confidence=0.2):
    import importlib.metadata
    from eval_tas import inventory_detector

    return {
        "schema": "tas-perception-v1",
        "models": asset_identity,
        "inventory_source_sha256": sha256_file(inventory_detector.__file__),
        "implementation_sha256": sha256_file(__file__),
        "confidence": confidence,
        "expansion_ratio": 0.0,
        "blip_dtype": "float16",
        "sam_dtype": "float32",
        "yolo_dtype": "original_default_float32",
        "labels_enter_caption_prompt": False,
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "ultralytics",
                "timm",
                "opencv-python",
            )
        },
    }


def perception_key(image_path, identity):
    return {"image_sha256": sha256_file(image_path), "perception": identity}
