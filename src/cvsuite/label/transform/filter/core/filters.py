from __future__ import annotations
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from cvsuite.common.core import Record

from .config import FilterConfig
from .features import FeatureExtractor
from .records import build_batch, apply_sources_to_record
from .types import Features
from .utils import nms_xyxy, cross_class_suppress_xyxy

Array = np.ndarray
SourceRef = Tuple[str, Optional[int], Optional[int]]


def _slice_batch(
    boxes: Array,
    scores: Array,
    labels: Array,
    masks: Optional[List[Array]],
    sources: List[SourceRef],
    keep: Array,
) -> Tuple[Array, Array, Array, Optional[List[Array]], List[SourceRef]]:
    if boxes.size == 0 or not np.any(keep):
        return (
            np.zeros((0, 4), dtype=float),
            np.zeros((0,), dtype=float),
            np.array([], dtype=object),
            [] if masks is not None else None,
            [],
        )
    if keep.dtype != bool:
        keep = keep.astype(bool)
    boxes2 = boxes[keep]
    scores2 = scores[keep]
    labels2 = labels[keep]
    masks2 = [masks[i] for i, k in enumerate(keep) if k] if masks is not None else None
    sources2 = [sources[i] for i, k in enumerate(keep) if k] if sources else []
    return boxes2, scores2, labels2, masks2, sources2


def _conf_keep_mask(
    scores: Array,
    sources: List[SourceRef],
    det_conf: Optional[float],
    seg_conf: Optional[float],
) -> Array:
    if scores.size == 0:
        return np.zeros((0,), dtype=bool)
    if det_conf is None and seg_conf is None:
        return np.ones((scores.shape[0],), dtype=bool)
    keep = np.ones((scores.shape[0],), dtype=bool)
    for i, src in enumerate(sources):
        if i >= scores.shape[0]:
            break
        if src[0] == "box":
            if det_conf is not None:
                keep[i] &= scores[i] >= float(det_conf)
        elif src[0] == "poly":
            if seg_conf is not None:
                keep[i] &= scores[i] >= float(seg_conf)
    return keep


class ThresholdFilter:
    """Axis-aligned threshold filter with shared postprocess."""

    def __init__(self, cfg: FilterConfig) -> None:
        self.cfg = cfg
        self.fx = FeatureExtractor()

    def keep_mask(self, feats: Features, classes: Optional[Sequence[str]] = None) -> Array:
        N = feats.score.shape[0]
        if N == 0:
            return np.zeros((0,), dtype=bool)

        all_keep = np.zeros((N,), dtype=bool)
        all_classes = list(classes) if classes is not None else list(np.unique(feats.labels))

        has_mask = None
        if feats.masks is not None:
            hm = np.zeros((N,), dtype=bool)
            for i, m in enumerate(feats.masks[:N]):
                if m is None:
                    hm[i] = False
                else:
                    arr = np.asarray(m)
                    hm[i] = bool(arr.size > 0 and np.any(arr))
            has_mask = hm

        g_require_mask = bool(self.cfg.global_cfg.require_mask)

        for cls in all_classes:
            cls_mask = (feats.labels == cls)
            if not np.any(cls_mask):
                continue
            k = self._axis_aligned_keep(str(cls), feats)

            cc = self.cfg.class_cfg(str(cls))
            require_mask = bool(cc.require_mask or g_require_mask)
            if require_mask and has_mask is not None:
                k = k & has_mask
            all_keep |= (cls_mask & k)

        return all_keep

    def postprocess(
        self,
        boxes: Array,
        scores: Array,
        labels: Array,
        masks: Optional[List[Array]],
        sources: List[SourceRef],
        nms_iou: float,
        topk_per_class: Optional[int],
        cross_class_iou: Optional[float] = None,
    ) -> Tuple[Array, Array, Array, Optional[List[Array]], List[SourceRef]]:
        if boxes.size == 0:
            return boxes, scores, labels, masks, sources

        keep_after: List[int] = []
        uniq = np.unique(labels)
        for c in uniq:
            m = (labels == c)
            ids = np.where(m)[0]
            sub_keep = nms_xyxy(boxes[m], scores[m], float(nms_iou))
            keep_after.extend(list(ids[sub_keep]))

        keep_mask = np.zeros((labels.shape[0],), dtype=bool)
        if keep_after:
            keep_mask[np.array(keep_after, dtype=int)] = True

        boxes, scores, labels, masks, sources = _slice_batch(boxes, scores, labels, masks, sources, keep_mask)

        if topk_per_class is not None and topk_per_class > 0 and boxes.shape[0] > 0:
            keep_after2: List[int] = []
            uniq2 = np.unique(labels)
            for c in uniq2:
                m = (labels == c)
                ids = np.where(m)[0]
                ords = np.argsort(scores[m])[::-1][: int(topk_per_class)]
                keep_after2.extend(list(ids[ords]))
            keep_mask2 = np.zeros((labels.shape[0],), dtype=bool)
            if keep_after2:
                keep_mask2[np.array(keep_after2, dtype=int)] = True
            boxes, scores, labels, masks, sources = _slice_batch(boxes, scores, labels, masks, sources, keep_mask2)

        if cross_class_iou is not None and cross_class_iou > 0.0 and boxes.shape[0] > 0:
            keep_idx = cross_class_suppress_xyxy(boxes, scores, labels, float(cross_class_iou))
            keep_mask3 = np.zeros((labels.shape[0],), dtype=bool)
            keep_mask3[keep_idx] = True
            boxes, scores, labels, masks, sources = _slice_batch(boxes, scores, labels, masks, sources, keep_mask3)

        return boxes, scores, labels, masks, sources

    def filter_record(self, rec: Record, classes: List[str], rgb: Optional[Array] = None) -> Record:
        batch = build_batch(rec, classes=classes, rgb=rgb)
        boxes = batch["boxes"]  # type: ignore[assignment]
        scores = batch["scores"]  # type: ignore[assignment]
        labels = batch["labels"]  # type: ignore[assignment]
        masks = batch["masks"]  # type: ignore[assignment]
        sources = batch["sources"]  # type: ignore[assignment]

        if boxes.size == 0:
            return replace(rec, boxes=[], polys=[])  # type: ignore[name-defined]

        keep_conf = self._confidence_keep_mask(scores, sources, labels)
        boxes, scores, labels, masks, sources = _slice_batch(boxes, scores, labels, masks, sources, keep_conf)
        if boxes.size == 0:
            return replace(rec, boxes=[], polys=[])  # type: ignore[name-defined]

        feats = self.fx.compute(
            H=int(batch["H"]),  # type: ignore[arg-type]
            W=int(batch["W"]),  # type: ignore[arg-type]
            boxes=boxes,
            scores=scores,
            labels=labels,
            masks=masks,
            rgb=batch.get("rgb"),  # type: ignore[arg-type]
        )
        keep = self.keep_mask(feats, classes=classes)
        boxes, scores, labels, masks, sources = _slice_batch(boxes, scores, labels, masks, sources, keep)
        giou = float(self.cfg.global_cfg.nms_iou)
        topkpc = self.cfg.global_cfg.topk_per_class
        cc_iou = self.cfg.global_cfg.cross_class_iou
        cc_iou = float(cc_iou) if (cc_iou is not None) else None
        _, _, _, _, kept_sources = self.postprocess(
            boxes, scores, labels, masks, sources, nms_iou=giou, topk_per_class=topkpc, cross_class_iou=cc_iou
        )
        return apply_sources_to_record(rec, kept_sources)

    def _confidence_keep_mask(self, scores: Array, sources: List[SourceRef], labels: Array) -> Array:
        keep = _conf_keep_mask(scores, sources, self.cfg.global_cfg.det_conf, self.cfg.global_cfg.seg_conf)
        if scores.size == 0 or labels.size == 0:
            return keep

        for i, src in enumerate(sources):
            if i >= scores.shape[0] or i >= labels.shape[0]:
                break
            cls_cfg = self.cfg.class_cfg(str(labels[i]))
            per_class_thr = cls_cfg.det_conf if src[0] == "box" else cls_cfg.seg_conf
            if per_class_thr is not None:
                keep[i] = scores[i] >= float(per_class_thr)
        return keep

    def _axis_aligned_keep(self, cls: str, feats: Features) -> Array:
        cc = self.cfg.class_cfg(cls)
        n = feats.score.shape[0]
        keep = np.ones((n,), dtype=bool)

        thr = cc.post_conf
        if thr is not None:
            keep &= feats.score >= float(thr)

        mn = cc.min_box_area_frac
        if mn is not None:
            keep &= feats.box_area_frac >= float(mn)
        mx = cc.max_box_area_frac
        if mx is not None:
            keep &= feats.box_area_frac <= float(mx)

        mn = cc.min_short_side_frac
        if mn is not None:
            keep &= feats.short_side_frac >= float(mn)

        mn = cc.min_ar
        if mn is not None:
            keep &= feats.aspect_ratio >= float(mn)
        mx = cc.max_ar
        if mx is not None:
            keep &= feats.aspect_ratio <= float(mx)

        mn = cc.mask_min_per_box
        if mn is not None and feats.mask_per_box is not None:
            keep &= feats.mask_per_box >= float(mn)

        mn = cc.mask_min_circ
        if mn is not None and feats.mask_circularity is not None:
            keep &= feats.mask_circularity >= float(mn)

        mn = cc.mask_min_solidity
        if mn is not None and feats.mask_solidity is not None:
            keep &= feats.mask_solidity >= float(mn)

        mx = cc.mask_max_holes_frac
        if mx is not None and feats.mask_holes_frac is not None:
            keep &= feats.mask_holes_frac <= float(mx)

        mn = cc.mask_min_ar
        if mn is not None and feats.mask_ar is not None:
            keep &= feats.mask_ar >= float(mn)
        mx = cc.mask_max_ar
        if mx is not None and feats.mask_ar is not None:
            keep &= feats.mask_ar <= float(mx)

        mn = cc.white_min_frac
        if mn is not None and feats.white_frac is not None:
            keep &= feats.white_frac >= float(mn)

        mx = cc.black_max_frac
        if mx is not None and feats.black_frac is not None:
            keep &= feats.black_frac <= float(mx)

        mn = cc.yellow_min_frac
        if mn is not None and feats.yellow_frac is not None:
            keep &= feats.yellow_frac >= float(mn)

        mn = cc.red_min_frac
        if mn is not None and feats.red_frac is not None:
            keep &= feats.red_frac >= float(mn)

        mn = cc.green_min_frac
        if mn is not None and feats.green_frac is not None:
            keep &= feats.green_frac >= float(mn)

        mn = cc.blue_min_frac
        if mn is not None and feats.blue_frac is not None:
            keep &= feats.blue_frac >= float(mn)

        return keep


class LogisticFilter(ThresholdFilter):
    """Logistic-gate filter: tries per-class logistic gate then falls back to thresholds."""

    def keep_mask(self, feats: Features, classes: Optional[Sequence[str]] = None) -> Array:
        N = feats.score.shape[0]
        if N == 0:
            return np.zeros((0,), dtype=bool)

        all_keep = np.zeros((N,), dtype=bool)
        all_classes = list(classes) if classes is not None else list(np.unique(feats.labels))

        has_mask = None
        if feats.masks is not None:
            hm = np.zeros((N,), dtype=bool)
            for i, m in enumerate(feats.masks[:N]):
                if m is None:
                    hm[i] = False
                else:
                    arr = np.asarray(m)
                    hm[i] = bool(arr.size > 0 and np.any(arr))
            has_mask = hm

        g_require_mask = bool(self.cfg.global_cfg.require_mask)

        for cls in all_classes:
            cls_mask = (feats.labels == cls)
            if not np.any(cls_mask):
                continue

            gate = self._resolve_logistic_gate(str(cls))
            if gate is not None:
                k = self._logistic_keep(gate, feats)
                if k is None:
                    k = self._axis_aligned_keep(str(cls), feats)
            else:
                k = self._axis_aligned_keep(str(cls), feats)

            cc = self.cfg.class_cfg(str(cls))
            require_mask = bool(cc.require_mask or g_require_mask)
            if require_mask and has_mask is not None:
                k = k & has_mask
            all_keep |= (cls_mask & k)
        return all_keep

    def _resolve_logistic_gate(self, cls: str) -> Optional[Dict[str, Any]]:
        cc = self.cfg.class_cfg(cls).logistic_gate
        if cc is None:
            return None
        return {
            "features": list(cc.features),
            "mean": None if cc.mean is None else np.asarray(cc.mean, dtype=float),
            "std": None if cc.std is None else np.asarray(cc.std, dtype=float),
            "weights": np.asarray(cc.weights, dtype=float),
            "bias": float(cc.bias),
            "threshold_type": str(cc.threshold_type),
            "threshold_value": float(cc.threshold_value),
        }

    def _build_view(self, feats: Features, names: List[str]) -> Optional[Array]:
        cols: List[Array] = []
        for n in names:
            v = getattr(feats, n, None)
            if v is None:
                return None
            cols.append(np.asarray(v, dtype=float).reshape(-1, 1))
        return np.concatenate(cols, axis=1) if cols else None

    def _apply_preprocess(self, X: Array, mean: Optional[Array], std: Optional[Array]) -> Array:
        Z = X
        if mean is not None and mean.size == X.shape[1]:
            Z = Z - mean.reshape(1, -1)
        if std is not None and std.size == X.shape[1]:
            Z = Z / (std.reshape(1, -1) + 1e-12)
        return Z

    def _logistic_keep(self, gate: Dict[str, Any], feats: Features) -> Optional[Array]:
        X = self._build_view(feats, list(gate["features"]))
        if X is None or X.shape[0] == 0:
            return None
        mean = gate.get("mean", None)
        std = gate.get("std", None)
        W = np.asarray(gate["weights"], dtype=float).reshape(-1)
        b = float(gate.get("bias", 0.0))
        ttype = str(gate.get("threshold_type", "logit"))
        tval = float(gate.get("threshold_value", 0.0))
        Z = self._apply_preprocess(X, mean, std)
        z = Z @ W.reshape(-1, 1) + b
        if ttype == "prob":
            thr = np.log(tval / (1.0 - tval + 1e-12) + 1e-12)
        else:
            thr = tval
        return (z.reshape(-1) >= thr)
