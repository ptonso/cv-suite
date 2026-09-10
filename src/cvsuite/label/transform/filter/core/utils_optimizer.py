"""Optimizer utilities."""

from __future__ import annotations

import math
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .config import filter_config_from_dict
from .filters import LogisticFilter, ThresholdFilter, _conf_keep_mask, _slice_batch
from .utils import box_iou

if TYPE_CHECKING:
    from .features import FeatureExtractor
    from .optimizer import OptimizerConfig

Array = np.ndarray


class UtilsOptimizer:
    """Helper utilities for LogisticOptimizer."""

    def __init__(self, cfg: OptimizerConfig, fx: FeatureExtractor) -> None:
        self.cfg = cfg
        self.fx = fx

    @staticmethod
    def _dd_list() -> DefaultDict[str, List[Any]]:
        from collections import defaultdict
        return defaultdict(list)

    @staticmethod
    def _split_samples(
        samples: List[Tuple[Dict[str, Any], Array, List[str]]],
        val_split: float,
        seed: int,
    ) -> Tuple[
        List[Tuple[Dict[str, Any], Array, List[str]]],
        List[Tuple[Dict[str, Any], Array, List[str]]],
        str,
    ]:
        n = len(samples)
        if val_split <= 0.0 or n < 2:
            return samples, [], "none"
        val_names = {"val", "valid", "validation", "test"}
        train_names = {"train", "training"}
        splits: List[str] = []
        has_val = False
        has_train = False
        for batch, _, _ in samples:
            split_raw = str(batch.get("split") or "").strip().lower()
            split = split_raw or "train"
            splits.append(split)
            if split in val_names:
                has_val = True
            if split in train_names:
                has_train = True
        if has_val and has_train:
            train = [s for s, sp in zip(samples, splits) if sp not in val_names]
            val = [s for s, sp in zip(samples, splits) if sp in val_names]
            if train and val:
                return train, val, "explicit"
        n_val = int(round(n * float(val_split)))
        if n_val <= 0:
            return samples, [], "none"
        if n_val >= n:
            n_val = n - 1
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        train = [samples[int(i)] for i in train_idx]
        val = [samples[int(i)] for i in val_idx]
        return train, val, "random"

    @staticmethod
    def _label_counts(
        samples: List[Tuple[Dict[str, Any], Array, List[str]]],
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        from collections import defaultdict
        gt_counts: DefaultDict[str, int] = defaultdict(int)
        pred_counts: DefaultDict[str, int] = defaultdict(int)
        for batch, _, gt_labels in samples:
            for lab in gt_labels:
                gt_counts[str(lab)] += 1
            labels = batch.get("labels")
            if isinstance(labels, np.ndarray):
                for lab in labels.tolist():
                    pred_counts[str(lab)] += 1
        return dict(gt_counts), dict(pred_counts)

    def _match_tp_fp(self, boxes: Array, labels: Array, gt_boxes: Array, gt_labels: List[str], iou_thr: float) -> Array:
        N = boxes.shape[0]
        out = np.zeros((N,), dtype=bool)
        if N == 0 or gt_boxes.shape[0] == 0:
            return out
        pred_labels = [str(x) for x in labels.tolist()]
        gt_labels_str = [str(x) for x in gt_labels]
        for cls in sorted(set(pred_labels)):
            p_idx = [i for i, lab in enumerate(pred_labels) if lab == cls]
            g_idx = [j for j, lab in enumerate(gt_labels_str) if lab == cls]
            if not p_idx or not g_idx:
                continue
            P = boxes[p_idx, :]
            G = gt_boxes[g_idx, :]
            iou_mat = np.zeros((P.shape[0], G.shape[0]), dtype=float)
            for i in range(P.shape[0]):
                for j in range(G.shape[0]):
                    iou_mat[i, j] = box_iou(P[i], G[j])
            used_p: set[int] = set()
            used_g: set[int] = set()
            while True:
                max_iou = -1.0
                best_pair = (-1, -1)
                for i in range(P.shape[0]):
                    if i in used_p:
                        continue
                    for j in range(G.shape[0]):
                        if j in used_g:
                            continue
                        val = iou_mat[i, j]
                        if val > max_iou:
                            max_iou = val
                            best_pair = (i, j)
                if max_iou < float(iou_thr):
                    break
                i, j = best_pair
                used_p.add(i)
                used_g.add(j)
                out[p_idx[i]] = True
        return out

    def _greedy_counts(self, boxes: Array, labels: Array, gt_boxes: Array, gt_labels: List[str], thr: float) -> Tuple[int, int, int]:
        if boxes.size == 0 and gt_boxes.size == 0:
            return 0, 0, 0
        tp_total = fp_total = fn_total = 0
        pred_labels = [str(x) for x in labels.tolist()]
        classes = sorted(set(pred_labels) | set([str(g) for g in gt_labels]))
        for c in classes:
            p_idx = [i for i, l in enumerate(pred_labels) if l == c]
            g_idx = [j for j, l in enumerate(gt_labels) if str(l) == c]
            if not p_idx and not g_idx:
                continue
            if not p_idx:
                fn_total += len(g_idx)
                continue
            if not g_idx:
                fp_total += len(p_idx)
                continue
            P = boxes[p_idx, :]
            G = gt_boxes[g_idx, :]
            iou_mat = np.zeros((P.shape[0], G.shape[0]), dtype=float)
            for i in range(P.shape[0]):
                for j in range(G.shape[0]):
                    iou_mat[i, j] = box_iou(P[i], G[j])
            used_p: set[int] = set()
            used_g: set[int] = set()
            while True:
                max_iou = -1.0
                best_pair = (-1, -1)
                for i in range(P.shape[0]):
                    if i in used_p:
                        continue
                    for j in range(G.shape[0]):
                        if j in used_g:
                            continue
                        val = iou_mat[i, j]
                        if val > max_iou:
                            max_iou = val
                            best_pair = (i, j)
                if max_iou < thr:
                    break
                i, j = best_pair
                used_p.add(i)
                used_g.add(j)
            tp = len(used_p)
            fp = P.shape[0] - tp
            fn = G.shape[0] - tp
            tp_total += tp
            fp_total += fp
            fn_total += fn
        return tp_total, fp_total, fn_total

    @staticmethod
    def _eval_micro_f1(tp: int, fp: int, fn: int) -> float:
        P = tp / max(tp + fp, 1)
        R = tp / max(tp + fn, 1)
        return 0.0 if (P + R) == 0 else (2 * P * R) / (P + R)

    @staticmethod
    def _fbeta(tp: int, fp: int, fn: int, beta: float) -> float:
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        if p == 0.0 and r == 0.0:
            return 0.0
        b2 = float(beta) * float(beta)
        return (1 + b2) * p * r / max(b2 * p + r, 1e-12)

    def _fbeta_at_thr(self, z: Array, y: Array, thr: float, beta: float) -> Tuple[float, int, int, int]:
        yhat = (z >= thr)
        tp = int(np.sum(yhat & (y == 1)))
        fp = int(np.sum(yhat & (y == 0)))
        fn = int(np.sum((~yhat) & (y == 1)))
        return self._fbeta(tp, fp, fn, beta), tp, fp, fn

    @staticmethod
    def _standardize(X: Array) -> Tuple[Array, Array, Array]:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        Z = (X - mean) / std
        return Z, mean.reshape(-1), std.reshape(-1)

    @staticmethod
    def _sigmoid(z: Array) -> Array:
        return 1.0 / (1.0 + np.exp(-z))

    def _search_best_thr(self, z: Array, y: Array) -> Tuple[str, float, float, int, int, int]:
        cand_logit = np.percentile(z, np.linspace(10, 90, 17)) if z.size else np.array([0.0])
        best = ("logit", 0.0, 0.0, 0, 0, 0)
        for t in cand_logit:
            F1, tp, fp, fn = self._fbeta_at_thr(z, y, float(t), beta=float(self.cfg.beta))
            if F1 > best[2]:
                best = ("logit", float(t), float(F1), tp, fp, fn)
        p = self._sigmoid(z)
        cand_prob = np.linspace(0.1, 0.9, 17)
        for pthr in cand_prob:
            logt = math.log(pthr / max(1.0 - pthr, 1e-12))
            F1, tp, fp, fn = self._fbeta_at_thr(z, y, float(logt), beta=float(self.cfg.beta))
            if F1 > best[2]:
                best = ("prob", float(pthr), float(F1), tp, fp, fn)
        return best

    @staticmethod
    def _normal_cdf(x: Array) -> Array:
        xv = np.asarray(x, dtype=float)
        flat = xv.ravel()
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        out = np.empty_like(flat)
        for i in range(flat.size):
            out[i] = 0.5 * (1.0 + math.erf(flat[i] * inv_sqrt2))
        return out.reshape(xv.shape)

    def _irls_fit(self, Z: Array, y: Array, l2: float, max_iter: int = 50, tol: float = 1e-6) -> Tuple[Array, Array]:
        if Z.size == 0 or y.size == 0:
            return np.zeros((Z.shape[1] + 1,), dtype=float), np.ones((Z.shape[1] + 1,), dtype=float)
        Xa = np.concatenate([Z, np.ones((Z.shape[0], 1), dtype=float)], axis=1)
        d = Xa.shape[1] - 1
        beta = np.zeros((d + 1,), dtype=float)
        H = None
        for _ in range(max_iter):
            z = Xa @ beta
            p = self._sigmoid(np.clip(z, -40.0, 40.0))
            w = p * (1.0 - p)
            W = w.reshape(-1, 1)
            XTWX = Xa.T @ (Xa * W)
            reg = np.diag(np.concatenate([np.full((d,), float(l2)), np.array([0.0])]))
            H = XTWX + reg + 1e-6 * np.eye(d + 1)
            g = Xa.T @ (p - y)
            step = np.linalg.solve(H, g)
            beta_new = beta - step
            if np.linalg.norm(step, ord=2) < tol:
                beta = beta_new
                break
            beta = beta_new
        if H is None:
            H = np.eye(beta.size, dtype=float)
        try:
            cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(H)
        return beta, np.diag(cov)

    def _wald_pvals(self, beta: Array, cov_diag: Array) -> Array:
        se = np.sqrt(np.maximum(cov_diag, 1e-12))
        z = beta / se
        p_upper = 1.0 - self._normal_cdf(np.abs(z))
        return 2.0 * p_upper

    def _prune_loop(
        self,
        Z: Array,
        y: Array,
        names: List[str],
        alpha: float,
        l2: float,
        max_prune_iters: int,
        min_features: int,
        always_keep: Iterable[str],
    ) -> Tuple[np.ndarray, float, List[str], np.ndarray, np.ndarray]:
        keep = list(range(len(names)))
        ak = set([s.strip() for s in always_keep if s.strip()])
        for _ in range(max(1, int(max_prune_iters))):
            if len(keep) == 0:
                break
            Zk = Z[:, keep] if Z.ndim == 2 else Z
            beta, cov_diag = self._irls_fit(Zk, y, l2=float(l2))
            pvals = self._wald_pvals(beta, cov_diag)
            p_no_bias = pvals[:-1]
            if p_no_bias.size == 0:
                break
            viol = []
            for j_local, j_global in enumerate(keep):
                name = names[j_global]
                if name in ak:
                    continue
                viol.append((p_no_bias[j_local], j_local, j_global, name))
            viol = [v for v in viol if v[0] > float(alpha)]
            if not viol or len(keep) <= int(min_features):
                break
            worst = max(viol, key=lambda t: t[0])
            j_local, j_global = int(worst[1]), int(worst[2])
            keep.remove(j_global)
        final_Z = Z[:, keep] if len(keep) > 0 else Z[:, :0]
        if final_Z.shape[1] == 0:
            beta = np.zeros((1,), dtype=float)
            cov = np.ones((1,), dtype=float)
            kept_names = []
        else:
            beta, cov = self._irls_fit(final_Z, y, l2=float(l2))
            kept_names = [names[j] for j in keep]
        return beta, cov, kept_names, final_Z, np.asarray(keep, dtype=int)

    @staticmethod
    def _conf_grid() -> List[float]:
        return [float(x) for x in np.linspace(0.0, 1.0, 21)]

    @staticmethod
    def _scores_degenerate(scores: List[float], span_eps: float = 1e-4) -> bool:
        if len(scores) < 2:
            return True
        arr = np.asarray(scores, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            return True
        return float(arr.max() - arr.min()) <= float(span_eps)

    @staticmethod
    def _prefer_higher(
        det_conf: Optional[float],
        seg_conf: Optional[float],
        best_det: Optional[float],
        best_seg: Optional[float],
    ) -> bool:
        def _norm(v: Optional[float]) -> float:
            return float(v) if v is not None else -1.0

        return (_norm(det_conf), _norm(seg_conf)) > (_norm(best_det), _norm(best_seg))

    def _tune_conf_thresholds(
        self,
        cached_samples: List[Tuple[Dict[str, Any], Array, List[str]]],
        match_iou_thr: float,
        beta: float,
        metric: str,
    ) -> Tuple[Optional[float], Optional[float], Optional[Dict[str, int]], float]:
        if not cached_samples:
            return None, None, {"TP": 0, "FP": 0, "FN": 0}, 0.0

        det_scores: List[float] = []
        seg_scores: List[float] = []
        for batch, _, _ in cached_samples:
            scores = batch["scores"]  # type: ignore[assignment]
            sources = batch["sources"]  # type: ignore[assignment]
            for i, src in enumerate(sources):
                if i >= scores.shape[0]:
                    break
                if src[0] == "box":
                    det_scores.append(float(scores[i]))
                elif src[0] == "poly":
                    seg_scores.append(float(scores[i]))

        det_degenerate = self._scores_degenerate(det_scores) if det_scores else True
        seg_degenerate = self._scores_degenerate(seg_scores) if seg_scores else True
        det_grid: List[Optional[float]] = self._conf_grid() if det_scores and not det_degenerate else [None]
        seg_grid: List[Optional[float]] = self._conf_grid() if seg_scores and not seg_degenerate else [None]

        metric_key = str(metric).strip().lower()
        if metric_key in ("ap", "auc", "prauc"):
            ap_cache = self._build_ap_cache(cached_samples, match_iou_thr=float(match_iou_thr))
            best_det: Optional[float] = None
            best_seg: Optional[float] = None
            best_score = -1.0
            for det_thr in det_grid:
                for seg_thr in seg_grid:
                    score = self._ap_for_thresholds(
                        ap_cache=ap_cache,
                        det_conf=det_thr,
                        seg_conf=seg_thr,
                    )
                    if score > best_score or (
                        abs(score - best_score) < 1e-12
                        and self._prefer_higher(det_thr, seg_thr, best_det, best_seg)
                    ):
                        best_score = score
                        best_det = det_thr
                        best_seg = seg_thr
            stats = self._conf_counts(
                cached_samples=cached_samples,
                det_conf=best_det,
                seg_conf=best_seg,
                match_iou_thr=float(match_iou_thr),
            )
            return best_det, best_seg, stats, best_score

        best_det: Optional[float] = None
        best_seg: Optional[float] = None
        best_stats = {"TP": 0, "FP": 0, "FN": 0}
        best_f = -1.0

        for det_thr in det_grid:
            for seg_thr in seg_grid:
                tp_sum = fp_sum = fn_sum = 0
                for batch, gt_boxes, gt_labels in cached_samples:
                    boxes = batch["boxes"]  # type: ignore[assignment]
                    scores = batch["scores"]  # type: ignore[assignment]
                    labels = batch["labels"]  # type: ignore[assignment]
                    masks = batch["masks"]  # type: ignore[assignment]
                    sources = batch["sources"]  # type: ignore[assignment]
                    if boxes.size == 0:
                        tp, fp, fn = 0, 0, len(gt_labels)
                    else:
                        km = _conf_keep_mask(scores, sources, det_thr, seg_thr)
                        k_boxes, k_scores, k_labels, k_masks, k_sources = _slice_batch(
                            boxes, scores, labels, masks, sources, km
                        )
                        if k_boxes.size == 0:
                            tp, fp, fn = 0, 0, len(gt_labels)
                        else:
                            tp, fp, fn = self._greedy_counts(
                                k_boxes, k_labels, gt_boxes, gt_labels, thr=float(match_iou_thr)
                            )
                    tp_sum += tp
                    fp_sum += fp
                    fn_sum += fn
                fbeta = self._fbeta(tp_sum, fp_sum, fn_sum, beta=float(beta))
                if fbeta > best_f or (
                    abs(fbeta - best_f) < 1e-12
                    and self._prefer_higher(det_thr, seg_thr, best_det, best_seg)
                ):
                    best_f = fbeta
                    best_det = det_thr
                    best_seg = seg_thr
                    best_stats = {"TP": tp_sum, "FP": fp_sum, "FN": fn_sum}

        return best_det, best_seg, best_stats, best_f

    def _conf_counts(
        self,
        cached_samples: List[Tuple[Dict[str, Any], Array, List[str]]],
        det_conf: Optional[float],
        seg_conf: Optional[float],
        match_iou_thr: float,
    ) -> Dict[str, int]:
        tp_sum = fp_sum = fn_sum = 0
        for batch, gt_boxes, gt_labels in cached_samples:
            boxes = batch["boxes"]  # type: ignore[assignment]
            scores = batch["scores"]  # type: ignore[assignment]
            labels = batch["labels"]  # type: ignore[assignment]
            masks = batch["masks"]  # type: ignore[assignment]
            sources = batch["sources"]  # type: ignore[assignment]
            if boxes.size == 0:
                tp, fp, fn = 0, 0, len(gt_labels)
            else:
                km = _conf_keep_mask(scores, sources, det_conf, seg_conf)
                k_boxes, k_scores, k_labels, k_masks, k_sources = _slice_batch(
                    boxes, scores, labels, masks, sources, km
                )
                if k_boxes.size == 0:
                    tp, fp, fn = 0, 0, len(gt_labels)
                else:
                    tp, fp, fn = self._greedy_counts(
                        k_boxes, k_labels, gt_boxes, gt_labels, thr=float(match_iou_thr)
                    )
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
        return {"TP": tp_sum, "FP": fp_sum, "FN": fn_sum}

    def _build_ap_cache(
        self,
        cached_samples: List[Tuple[Dict[str, Any], Array, List[str]]],
        match_iou_thr: float,
    ) -> Tuple[
        List[Tuple[float, str, int, str, List[Tuple[int, float]]]],
        Dict[Tuple[int, str], int],
        int,
    ]:
        preds: List[Tuple[float, str, int, str, List[Tuple[int, float]]]] = []
        gt_counts: Dict[Tuple[int, str], int] = {}
        total_gt = 0

        for img_idx, (batch, gt_boxes, gt_labels) in enumerate(cached_samples):
            gt_by_cls: Dict[str, List[Array]] = {}
            for j, lab in enumerate(gt_labels):
                gt_by_cls.setdefault(str(lab), []).append(gt_boxes[j])

            for cls, boxes in gt_by_cls.items():
                gt_counts[(img_idx, cls)] = len(boxes)
                total_gt += len(boxes)

            boxes = batch["boxes"]  # type: ignore[assignment]
            scores = batch["scores"]  # type: ignore[assignment]
            labels = batch["labels"]  # type: ignore[assignment]
            sources = batch["sources"]  # type: ignore[assignment]
            if boxes.size == 0:
                continue
            for i in range(boxes.shape[0]):
                cls = str(labels[i])
                gt_list = gt_by_cls.get(cls, [])
                cand: List[Tuple[int, float]] = []
                for g_idx, g_box in enumerate(gt_list):
                    iou = box_iou(boxes[i], g_box)
                    if iou >= float(match_iou_thr):
                        cand.append((g_idx, float(iou)))
                cand.sort(key=lambda t: t[1], reverse=True)
                src_kind = sources[i][0] if sources and i < len(sources) else "box"
                preds.append((float(scores[i]), str(src_kind), img_idx, cls, cand))

        return preds, gt_counts, total_gt

    def _ap_for_thresholds(
        self,
        ap_cache: Tuple[
            List[Tuple[float, str, int, str, List[Tuple[int, float]]]],
            Dict[Tuple[int, str], int],
            int,
        ],
        det_conf: Optional[float],
        seg_conf: Optional[float],
    ) -> float:
        preds, gt_counts, total_gt = ap_cache
        if total_gt <= 0:
            return 0.0
        if not preds:
            return 0.0

        kept: List[Tuple[float, str, int, str, List[Tuple[int, float]]]] = []
        for score, src_kind, img_idx, cls, cand in preds:
            if src_kind == "box":
                if det_conf is not None and score < float(det_conf):
                    continue
            elif src_kind == "poly":
                if seg_conf is not None and score < float(seg_conf):
                    continue
            kept.append((score, src_kind, img_idx, cls, cand))

        if not kept:
            return 0.0

        kept.sort(key=lambda t: t[0], reverse=True)
        matched = {k: np.zeros(v, dtype=bool) for k, v in gt_counts.items()}
        tp = fp = 0
        tps: List[int] = []
        fps: List[int] = []

        for score, src_kind, img_idx, cls, cand in kept:
            key = (img_idx, cls)
            if key not in matched or matched[key].size == 0:
                fp += 1
            else:
                hit = False
                used = matched[key]
                for g_idx, _ in cand:
                    if g_idx < used.size and not used[g_idx]:
                        used[g_idx] = True
                        hit = True
                        break
                if hit:
                    tp += 1
                else:
                    fp += 1
            tps.append(tp)
            fps.append(fp)

        tp_arr = np.asarray(tps, dtype=float)
        fp_arr = np.asarray(fps, dtype=float)
        precision = tp_arr / np.maximum(tp_arr + fp_arr, 1.0)
        recall = tp_arr / max(float(total_gt), 1.0)

        mrec = np.concatenate([[0.0], recall, [1.0]])
        mpre = np.concatenate([[0.0], precision, [0.0]])
        for i in range(mpre.size - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
        return ap

    def _cache_after_conf(
        self,
        cached_samples: List[Tuple[Dict[str, Any], Array, List[str]]],
        det_conf: Optional[float],
        seg_conf: Optional[float],
    ) -> List[Tuple[Tuple[Array, Array, Array, Optional[List[Array]], List[Tuple[str, Optional[int], Optional[int]]]], Array, List[str]]]:
        kept_per_image: List[Tuple[Tuple[Array, Array, Array, Optional[List[Array]], List[Tuple[str, Optional[int], Optional[int]]]], Array, List[str]]] = []
        for batch, gt_boxes, gt_labels in cached_samples:
            boxes = batch["boxes"]  # type: ignore[assignment]
            scores = batch["scores"]  # type: ignore[assignment]
            labels = batch["labels"]  # type: ignore[assignment]
            masks = batch["masks"]  # type: ignore[assignment]
            sources = batch["sources"]  # type: ignore[assignment]
            if boxes.size == 0:
                kept = (
                    np.zeros((0, 4), dtype=float),
                    np.zeros((0,), dtype=float),
                    np.array([], dtype=object),
                    [] if masks is not None else None,
                    [],
                )
            else:
                km = _conf_keep_mask(scores, sources, det_conf, seg_conf)
                kept = _slice_batch(boxes, scores, labels, masks, sources, km)
            kept_per_image.append((kept, gt_boxes, gt_labels))
        return kept_per_image

    def _evaluate_samples(
        self,
        cached_samples: List[Tuple[Dict[str, Any], Array, List[str]]],
        cfg_dict: Dict[str, Any],
        match_iou_thr: float,
    ) -> Optional[Dict[str, float]]:
        if not cached_samples:
            return None
        cfg = filter_config_from_dict(cfg_dict)
        flt = LogisticFilter(cfg)
        topk = cfg.global_cfg.topk_per_class
        nms_iou = float(cfg.global_cfg.nms_iou)
        cc_iou = cfg.global_cfg.cross_class_iou
        cc_iou = float(cc_iou) if cc_iou is not None else None

        tp_sum = fp_sum = fn_sum = 0
        for batch, gt_boxes, gt_labels in cached_samples:
            boxes = batch["boxes"]  # type: ignore[assignment]
            scores = batch["scores"]  # type: ignore[assignment]
            labels = batch["labels"]  # type: ignore[assignment]
            masks = batch["masks"]  # type: ignore[assignment]
            sources = batch["sources"]  # type: ignore[assignment]
            if boxes.size == 0:
                tp, fp, fn = 0, 0, len(gt_labels)
            else:
                km = _conf_keep_mask(scores, sources, cfg.global_cfg.det_conf, cfg.global_cfg.seg_conf)
                boxes, scores, labels, masks, sources = _slice_batch(boxes, scores, labels, masks, sources, km)
                if boxes.size == 0:
                    tp, fp, fn = 0, 0, len(gt_labels)
                else:
                    feats = self.fx.compute(
                        H=int(batch["H"]),  # type: ignore[arg-type]
                        W=int(batch["W"]),  # type: ignore[arg-type]
                        boxes=boxes,
                        scores=scores,
                        labels=labels,
                        masks=masks,
                        rgb=batch.get("rgb"),  # type: ignore[arg-type]
                    )
                    keep = flt.keep_mask(feats)
                    boxes, scores, labels, masks, sources = _slice_batch(boxes, scores, labels, masks, sources, keep)
                    if boxes.size == 0:
                        tp, fp, fn = 0, 0, len(gt_labels)
                    else:
                        boxes, scores, labels, masks, sources = flt.postprocess(
                            boxes,
                            scores,
                            labels,
                            masks,
                            sources,
                            nms_iou=nms_iou,
                            topk_per_class=topk,
                            cross_class_iou=cc_iou,
                        )
                        tp, fp, fn = self._greedy_counts(
                            boxes, labels, gt_boxes, gt_labels, thr=float(match_iou_thr)
                        )
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn

        p = tp_sum / max(tp_sum + fp_sum, 1)
        r = tp_sum / max(tp_sum + fn_sum, 1)
        f1 = 0.0 if (p + r) <= 0 else (2 * p * r) / (p + r)
        return {
            "P": float(p),
            "R": float(r),
            "F1": float(f1),
            "TP": int(tp_sum),
            "FP": int(fp_sum),
            "FN": int(fn_sum),
        }

    def _tune_nms_iou(
        self,
        kept_per_image: List[Tuple[Tuple[Array, Array, Array, Optional[List[Array]], List[Tuple[str, Optional[int], Optional[int]]]], Array, List[str]]],
        match_iou_thr: float,
        grid: List[float],
        topk: Optional[int],
        pf: ThresholdFilter,
        default_iou: float = 0.6,
    ) -> Tuple[float, Dict[str, int], float]:
        if not kept_per_image:
            return default_iou, {"TP": 0, "FP": 0, "FN": 0}, 0.0

        vals = sorted(set(float(x) for x in grid if np.isfinite(x))) or \
               [0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80]

        best_iou, best_f1 = default_iou, -1.0
        best_stats = {"TP": 0, "FP": 0, "FN": 0}

        for cand in vals:
            tp_sum = fp_sum = fn_sum = 0
            for kept, gt_boxes, gt_labels in kept_per_image:
                k_boxes, k_scores, k_labels, k_masks, k_sources = kept
                if k_boxes.size == 0:
                    tp, fp, fn = 0, 0, len(gt_labels)
                else:
                    kept2 = pf.postprocess(
                        k_boxes, k_scores, k_labels, k_masks, k_sources, nms_iou=float(cand), topk_per_class=topk, cross_class_iou=None
                    )
                    tp, fp, fn = self._greedy_counts(kept2[0], kept2[2], gt_boxes, gt_labels, thr=float(match_iou_thr))
                tp_sum += tp
                fp_sum += fp
                fn_sum += fn
            f1 = self._eval_micro_f1(tp_sum, fp_sum, fn_sum)
            if f1 > best_f1:
                best_f1 = f1
                best_iou = float(cand)
                best_stats = {"TP": tp_sum, "FP": fp_sum, "FN": fn_sum}

        return best_iou, best_stats, best_f1

    def _tune_cross_class_iou(
        self,
        kept_per_image: List[Tuple[Tuple[Array, Array, Array, Optional[List[Array]], List[Tuple[str, Optional[int], Optional[int]]]], Array, List[str]]],
        match_iou_thr: float,
        grid: List[float],
        giou: float,
        topk: Optional[int],
        pf: ThresholdFilter,
        default_iou: float = 0.0,
    ) -> Tuple[float, Dict[str, int], float]:
        if not kept_per_image:
            return default_iou, {"TP": 0, "FP": 0, "FN": 0}, 0.0

        vals = sorted(set(float(x) for x in grid if np.isfinite(x))) or \
               [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]

        best_iou, best_f1 = default_iou, -1.0
        best_stats = {"TP": 0, "FP": 0, "FN": 0}

        for cand in vals:
            tp_sum = fp_sum = fn_sum = 0
            for kept, gt_boxes, gt_labels in kept_per_image:
                k_boxes, k_scores, k_labels, k_masks, k_sources = kept
                if k_boxes.size == 0:
                    tp, fp, fn = 0, 0, len(gt_labels)
                else:
                    kept2 = pf.postprocess(
                        k_boxes,
                        k_scores,
                        k_labels,
                        k_masks,
                        k_sources,
                        nms_iou=giou,
                        topk_per_class=topk,
                        cross_class_iou=float(cand),
                    )
                    tp, fp, fn = self._greedy_counts(kept2[0], kept2[2], gt_boxes, gt_labels, thr=float(match_iou_thr))
                tp_sum += tp
                fp_sum += fp
                fn_sum += fn
            f1 = self._eval_micro_f1(tp_sum, fp_sum, fn_sum)
            if f1 > best_f1:
                best_f1 = f1
                best_iou = float(cand)
                best_stats = {"TP": tp_sum, "FP": fp_sum, "FN": fn_sum}

        return best_iou, best_stats, best_f1
