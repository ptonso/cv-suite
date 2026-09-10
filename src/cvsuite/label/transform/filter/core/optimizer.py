"""Logistic gate learning for VisionDataset pairs (permissive vs GT)."""

from __future__ import annotations
import copy
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import numpy as np
import yaml

from cvsuite.common.core import VisionDataset

from .config import filter_config_from_dict
from .filters import ThresholdFilter, _conf_keep_mask, _slice_batch
from .features import FeatureExtractor
from .records import (
    build_batch,
    gt_boxes_labels_from_record,
    warn_on_stem_mismatch,
    load_rgb_for_rec,
)
from .utils_optimizer import UtilsOptimizer

Array = np.ndarray


@dataclass
class OptimizerConfig:
    out_cfg:                 str
    infer_cfg:               str = ""
    match_iou:             float = 0.50
    epochs:                  int = 500
    lr:                    float = 0.05
    l2:                    float = 1e-3
    alpha:                 float = 0.05
    beta:                  float = 1.0
    conf_metric:            str = "ap"
    max_prune_iters:         int = 25
    min_features:            int = 1
    always_keep:             str = "score,box_area_frac"
    seed:                    int = 42
    val_split:            float = 0.2
    device:                  str = "cpu"
    use_rgb:                bool = False
    tune_nms_iou:           bool = False
    nms_grid:                str = "0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60"
    tune_cross_class_iou:   bool = False
    cross_grid:              str = "0.10,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55"


@dataclass
class SplitResult:
    train: int
    val: int
    val_split: float
    mode: str = "random"


@dataclass
class SplitStats:
    train_gt: Dict[str, int] = field(default_factory=dict)
    val_gt: Dict[str, int] = field(default_factory=dict)
    train_pred: Dict[str, int] = field(default_factory=dict)
    val_pred: Dict[str, int] = field(default_factory=dict)


@dataclass
class ConfTuneResult:
    det_conf: Optional[float]
    seg_conf: Optional[float]
    metric_label: str
    score: float
    stats: Dict[str, int]


@dataclass
class IouTuneResult:
    best_iou: float
    metric_label: str
    score: float
    stats: Dict[str, int]


@dataclass
class PerClassTrainResult:
    cls: str
    n: int
    num_features: int
    threshold: float
    f1: float
    tp: int
    fp: int
    fn: int
    features: List[str]


@dataclass
class MetricResult:
    P: float
    R: float
    F1: float
    TP: int
    FP: int
    FN: int


@dataclass
class OptimizerResults:
    split: Optional[SplitResult] = None
    split_stats: Optional[SplitStats] = None
    conf_tune: Optional[ConfTuneResult] = None
    nms_tune: Optional[IouTuneResult] = None
    cross_class_tune: Optional[IouTuneResult] = None
    classes: List[str] = field(default_factory=list)
    per_class: List[PerClassTrainResult] = field(default_factory=list)
    micro_train: Optional[MetricResult] = None
    val: Optional[MetricResult] = None
    out_config: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LogisticOptimizer:
    """Learn per-class logistic gates and optionally tune IoUs."""

    def __init__(self, cfg: OptimizerConfig) -> None:
        self.cfg = cfg
        self.fx = FeatureExtractor()
        self.utils = UtilsOptimizer(cfg=cfg, fx=self.fx)

    def run(
        self,
        gt_ds: VisionDataset,
        permissive_ds: VisionDataset,
        results: Optional[OptimizerResults] = None,
    ) -> int:
        res = results
        base_cfg_dict: Dict[str, Any] = {}
        if self.cfg.infer_cfg:
            base_cfg_dict = yaml.safe_load(Path(self.cfg.infer_cfg).read_text(encoding="utf-8")) or {}

        common, gt_map, perm_map = warn_on_stem_mismatch(gt_ds, permissive_ds)
        if not common:
            raise RuntimeError("No overlapping images between permissive dataset and GT dataset.")

        gt_classes = gt_ds.classes if gt_ds.classes else permissive_ds.classes
        feature_order = [
            "aspect_ratio",
            "black_frac",
            "blue_frac",
            "box_area_frac",
            "green_frac",
            "mask_ar",
            "mask_circularity",
            "mask_holes_frac",
            "mask_per_box",
            "mask_solidity",
            "red_frac",
            "score",
            "short_side_frac",
            "white_frac",
            "yellow_frac",
        ]

        samples: List[Tuple[Dict[str, Any], Array, List[str]]] = []
        for stem in common:
            pred_rec = perm_map[stem]
            gt_rec = gt_map[stem]

            rgb = load_rgb_for_rec(pred_rec) if self.cfg.use_rgb else None
            batch = build_batch(pred_rec, classes=permissive_ds.classes, rgb=rgb)
            batch["split"] = gt_rec.split or pred_rec.split
            gt_boxes, gt_labels = gt_boxes_labels_from_record(gt_rec, classes=gt_classes)
            samples.append((batch, gt_boxes, gt_labels))

        train_samples, val_samples, split_mode = self.utils._split_samples(
            samples=samples,
            val_split=float(self.cfg.val_split),
            seed=int(self.cfg.seed),
        )
        if val_samples:
            split_ratio = len(val_samples) / max(len(train_samples) + len(val_samples), 1)
            mode_suffix = "" if split_mode == "random" else f" mode={split_mode}"
            print(f"[split] train={len(train_samples)} val={len(val_samples)} val_split={split_ratio:.2f}{mode_suffix}")
            if res is not None:
                res.split = SplitResult(
                    train=len(train_samples),
                    val=len(val_samples),
                    val_split=float(split_ratio),
                    mode=str(split_mode),
                )
        train_gt, train_pred = self.utils._label_counts(train_samples)
        val_gt, val_pred = self.utils._label_counts(val_samples)
        if res is not None:
            res.split_stats = SplitStats(
                train_gt=train_gt,
                val_gt=val_gt,
                train_pred=train_pred,
                val_pred=val_pred,
            )
        if val_samples:
            missing = sorted(set(val_gt) - set(train_gt))
            if missing:
                print(f"[warn] val contains classes not seen in train: {', '.join(missing)}")

        out_cfg = copy.deepcopy(base_cfg_dict)
        if "global" not in out_cfg and "global_cfg" in out_cfg:
            out_cfg["global"] = copy.deepcopy(out_cfg.get("global_cfg") or {})
        else:
            out_cfg.setdefault("global", {})
        out_cfg["global"]["use_rgb"] = bool(self.cfg.use_rgb)

        det_conf, seg_conf, conf_stats, conf_score = self.utils._tune_conf_thresholds(
            cached_samples=train_samples,
            match_iou_thr=float(self.cfg.match_iou),
            beta=float(self.cfg.beta),
            metric=str(self.cfg.conf_metric),
        )
        if det_conf is not None:
            out_cfg["global"]["det_conf"] = float(det_conf)
        else:
            out_cfg["global"].pop("det_conf", None)
        if seg_conf is not None:
            out_cfg["global"]["seg_conf"] = float(seg_conf)
        else:
            out_cfg["global"].pop("seg_conf", None)
        if conf_stats is not None:
            det_str = "none" if det_conf is None else f"{det_conf:.2f}"
            seg_str = "none" if seg_conf is None else f"{seg_conf:.2f}"
            metric = str(self.cfg.conf_metric).lower()
            if metric in ("ap", "auc", "prauc"):
                metric_label = "microAP"
            else:
                metric_label = f"microF{float(self.cfg.beta):.1f}"
            print(f"[tune] conf det={det_str} seg={seg_str} -> {metric_label}={conf_score:.3f} "
                  f"(TP={conf_stats['TP']} FP={conf_stats['FP']} FN={conf_stats['FN']})")
            if res is not None:
                res.conf_tune = ConfTuneResult(
                    det_conf=None if det_conf is None else float(det_conf),
                    seg_conf=None if seg_conf is None else float(seg_conf),
                    metric_label=metric_label,
                    score=float(conf_score),
                    stats=dict(conf_stats),
                )

        cfg_for_iou = filter_config_from_dict(out_cfg)
        topk = cfg_for_iou.global_cfg.topk_per_class
        pf_post = ThresholdFilter(cfg_for_iou)

        kept_per_image = []
        if self.cfg.tune_nms_iou or self.cfg.tune_cross_class_iou:
            kept_per_image = self.utils._cache_after_conf(
                cached_samples=train_samples,
                det_conf=det_conf,
                seg_conf=seg_conf,
            )

        if self.cfg.tune_nms_iou:
            grid = [float(x) for x in (self.cfg.nms_grid.split(",") if self.cfg.nms_grid else []) if str(x).strip()]
            best_iou, stats, best_f1 = self.utils._tune_nms_iou(
                kept_per_image=kept_per_image,
                match_iou_thr=float(self.cfg.match_iou),
                grid=grid,
                topk=topk,
                pf=pf_post,
                default_iou=float(cfg_for_iou.global_cfg.nms_iou),
            )
            out_cfg["global"]["nms_iou"] = float(best_iou)
            print(f"[tune] best nms_iou={best_iou:.2f} -> microF1={best_f1:.3f} "
                  f"(TP={stats['TP']} FP={stats['FP']} FN={stats['FN']})")
            if res is not None:
                res.nms_tune = IouTuneResult(
                    best_iou=float(best_iou),
                    metric_label="microF1",
                    score=float(best_f1),
                    stats=dict(stats),
                )

        if self.cfg.tune_cross_class_iou:
            cgrid = [float(x) for x in (self.cfg.cross_grid.split(",") if self.cfg.cross_grid else []) if str(x).strip()]
            giou_default = float(out_cfg["global"].get("nms_iou", cfg_for_iou.global_cfg.nms_iou))
            best_cc, stats_cc, best_f1_cc = self.utils._tune_cross_class_iou(
                kept_per_image=kept_per_image,
                match_iou_thr=float(self.cfg.match_iou),
                grid=cgrid,
                topk=topk,
                giou=giou_default,
                pf=pf_post,
                default_iou=float(cfg_for_iou.global_cfg.cross_class_iou or 0.0),
            )
            out_cfg["global"]["cross_class_iou"] = float(best_cc)
            print(f"[tune] best cross_class_iou={best_cc:.2f} -> microF1={best_f1_cc:.3f} "
                  f"(TP={stats_cc['TP']} FP={stats_cc['FP']} FN={stats_cc['FN']})")
            if res is not None:
                res.cross_class_tune = IouTuneResult(
                    best_iou=float(best_cc),
                    metric_label="microF1",
                    score=float(best_f1_cc),
                    stats=dict(stats_cc),
                )

        per_class_X: DefaultDict[str, List[Array]] = self.utils._dd_list()
        per_class_y: DefaultDict[str, List[int]] = self.utils._dd_list()
        per_class_names: DefaultDict[str, List[str]] = self.utils._dd_list()
        per_class_names["_ORDER_"] = feature_order

        for batch, gt_boxes, gt_labels in train_samples:
            boxes      = batch["boxes"]
            scores     = batch["scores"]
            labels_arr = batch["labels"]
            masks      = batch["masks"]
            sources    = batch["sources"]
            if boxes.size == 0:
                continue
            keep_conf = _conf_keep_mask(scores, sources, det_conf, seg_conf)
            boxes, scores, labels_arr, masks, sources = _slice_batch(boxes, scores, labels_arr, masks, sources, keep_conf)
            if boxes.size == 0:
                continue
            feats = self.fx.compute(
                H=int(batch["H"]),
                W=int(batch["W"]),
                boxes=boxes,
                scores=scores,
                labels=labels_arr,
                masks=masks,
                rgb=batch.get("rgb"),
            )
            is_tp = self.utils._match_tp_fp(
                boxes,
                labels_arr,
                gt_boxes,
                gt_labels,
                iou_thr=float(self.cfg.match_iou),
            )

            X_cols: List[Array] = []
            for fn in feature_order:
                v = getattr(feats, fn, None)
                X_cols.append(np.zeros_like(feats.score) if v is None else np.asarray(v, dtype=float))
            X = np.stack(X_cols, axis=1) if X_cols else np.zeros((feats.score.shape[0], 0), dtype=float)

            labels = [str(l) for l in labels_arr.tolist()]
            for i, cls in enumerate(labels):
                per_class_X[cls].append(X[i])
                per_class_y[cls].append(1 if is_tp[i] else 0)

        classes = [c for c in per_class_X.keys() if c != "_ORDER_"]
        classes_sorted = sorted(classes)
        print(f"[scan] classes: {', '.join(classes_sorted)}")
        if res is not None:
            res.classes = classes_sorted

        per_cls_cfg = dict(out_cfg.get("per_class", {}) or {})
        micro_tp = micro_fp = micro_fn = 0
        always_keep = [s.strip() for s in (self.cfg.always_keep or "").split(",") if s.strip()]

        for cls in sorted(classes):
            Xc = np.stack(per_class_X[cls], axis=0) if per_class_X[cls] else np.zeros((0, len(per_class_names["_ORDER_"])))
            yc = np.asarray(per_class_y[cls], dtype=int) if per_class_y[cls] else np.zeros((0,), dtype=int)

            n = Xc.shape[0]
            if n == 0:
                continue

            Z, mean, std = self.utils._standardize(Xc)
            names = list(per_class_names["_ORDER_"])

            if yc.min() == yc.max() or Z.shape[0] < 4:
                kept_names = []
                keep_idx = np.zeros((0,), dtype=int)
                b = 0.0
                W = np.zeros((0,), dtype=float)
                tp = int(yc.sum())
                fn = int((yc == 1).size - tp)
                fp = 0
                F1 = self.utils._eval_micro_f1(tp, fp, fn)
                ttype, tval = "logit", float("-inf")
                mean_k = np.zeros((0,), dtype=float)
                std_k = np.zeros((0,), dtype=float)
            else:
                beta, cov, kept_names, Zk, keep_idx = self.utils._prune_loop(
                    Z=Z,
                    y=yc,
                    names=names,
                    alpha=float(self.cfg.alpha),
                    l2=float(self.cfg.l2),
                    max_prune_iters=int(self.cfg.max_prune_iters),
                    min_features=int(self.cfg.min_features),
                    always_keep=always_keep,
                )
                W = beta[:-1].reshape(-1)
                b = float(beta[-1])
                z = np.full((Z.shape[0],), b, dtype=float) if Zk.size == 0 else (Zk @ W) + b
                ttype, tval, F1, tp, fp, fn = self.utils._search_best_thr(z, yc)
                mean_k = mean[keep_idx]
                std_k = std[keep_idx]

            per_cls_cfg.setdefault(cls, {})
            if kept_names:
                per_cls_cfg[cls]["logistic_gate"] = {
                    "features": kept_names,
                    "mean": mean_k.tolist(),
                    "std": std_k.tolist(),
                    "weights": W.astype(float).tolist(),
                    "bias": float(b),
                    "threshold_type": ttype,
                    "threshold_value": float(tval),
                }
            else:
                per_cls_cfg[cls].pop("logistic_gate", None)

            micro_tp += tp
            micro_fp += fp
            micro_fn += fn
            thr_str = f"{tval:.2f}"
            print(f"[train] {cls:<18} n={n:<4d} f={len(kept_names):<2d} thr={thr_str} "
                  f"F1={F1:.3f} tp={tp:3d} fp={fp:3d} fn={fn:3d}")
            print(f"         features: {', '.join(kept_names)}")
            if res is not None:
                res.per_class.append(
                    PerClassTrainResult(
                        cls=str(cls),
                        n=int(n),
                        num_features=int(len(kept_names)),
                        threshold=float(tval),
                        f1=float(F1),
                        tp=int(tp),
                        fp=int(fp),
                        fn=int(fn),
                        features=list(kept_names),
                    )
                )

        micro_p = micro_tp / max(micro_tp + micro_fp, 1)
        micro_r = micro_tp / max(micro_tp + micro_fn, 1)
        micro_f1 = 0.0 if (micro_p + micro_r) <= 0 else 2 * micro_p * micro_r / (micro_p + micro_r)
        print(f"micro (train): P={micro_p:.3f} R={micro_r:.3f} F1={micro_f1:.3f} "
              f"TP={micro_tp} FP={micro_fp} FN={micro_fn}")
        if res is not None:
            res.micro_train = MetricResult(
                P=float(micro_p),
                R=float(micro_r),
                F1=float(micro_f1),
                TP=int(micro_tp),
                FP=int(micro_fp),
                FN=int(micro_fn),
            )

        out_cfg["per_class"] = per_cls_cfg

        if val_samples:
            v_stats = self.utils._evaluate_samples(
                cached_samples=val_samples,
                cfg_dict=out_cfg,
                match_iou_thr=float(self.cfg.match_iou),
            )
            if v_stats is not None:
                print(f"[val] P={v_stats['P']:.3f} R={v_stats['R']:.3f} F1={v_stats['F1']:.3f} "
                      f"TP={v_stats['TP']} FP={v_stats['FP']} FN={v_stats['FN']}")
                if res is not None:
                    res.val = MetricResult(
                        P=float(v_stats["P"]),
                        R=float(v_stats["R"]),
                        F1=float(v_stats["F1"]),
                        TP=int(v_stats["TP"]),
                        FP=int(v_stats["FP"]),
                        FN=int(v_stats["FN"]),
                    )

        Path(self.cfg.out_cfg).parent.mkdir(parents=True, exist_ok=True)
        Path(self.cfg.out_cfg).write_text(
            yaml.safe_dump(out_cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print(f"[write] saved optimized gates -> {self.cfg.out_cfg}")
        if res is not None:
            res.out_config = str(self.cfg.out_cfg)
        return 0
