"""Train logistic-gate filtering for permissive datasets and apply the learned config.

Filtering phases (apply step):
- Per-class logistic gate (fallback to axis-aligned thresholds if no gate).
- Global IoU pass: per-class NMS using global.nms_iou, then optional global.topk_per_class.
- Cross-class IoU pass: suppress overlaps across classes using global.cross_class_iou.

Optimization phase (training):
- Use --gt-src to tune confidence thresholds, optionally tune IoUs, then fit logistic gates.
- Write the optimized config to --out-config and apply it to the current dataset.

Logging:
- Use --log-dir to mirror terminal output to <dir>/filter-train-logistic.log.
- Training results are saved to <target dir>/filter_results.yaml when available.
"""

import argparse
import contextlib
from pathlib import Path
import sys
from typing import List, TextIO

import yaml

from cvsuite.label.core import router
from cvsuite.common.core import Record
from cvsuite.common.core import VisionDataset

from ..core import LogisticFilter, load_filter_config, load_rgb_for_rec
from ..core.optimizer import LogisticOptimizer, OptimizerConfig, OptimizerResults


class _TeeStream:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams
        self._primary = streams[0]

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)

    @property
    def encoding(self) -> str | None:
        return getattr(self._primary, "encoding", None)

    def __getattr__(self, name: str) -> object:
        return getattr(self._primary, name)


def attach(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--gt-src",
        type=Path,
        required=True,
        help="Ground-truth dataset to optimize from (format auto-detected).",
    )
    p.add_argument(
        "--config",
        type=Path,
        help="Optional base filter YAML to extend during optimization.",
    )
    p.add_argument(
        "--out-config",
        type=Path,
        required=True,
        help="Output YAML with learned gates.",
    )
    p.add_argument("--match-iou", type=float, default=0.5, help="IoU for GT matching during optimization.")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--l2", type=float, default=1e-3, help="L2 regularization strength for logistic fitting.")
    p.add_argument("--alpha", type=float, default=0.05, help="Wald p-value cutoff for pruning.")
    p.add_argument("--max-prune-iters", type=int, default=25, help="Max prune iterations for feature selection.")
    p.add_argument("--min-features", type=int, default=1, help="Minimum number of features to keep per class.")
    p.add_argument(
        "--always-keep",
        type=str,
        default="score,box_area_frac",
        help="Comma-separated feature names to never prune during optimization.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--beta", type=float, default=2.0, help="F-beta to maximize when choosing threshold.")
    p.add_argument(
        "--conf-metric",
        type=str,
        default="fbeta",
        choices=["fbeta", "ap"],
        help="Metric to optimize det/seg confidence thresholds (ap or fbeta).",
    )
    p.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of overlapping samples held out for validation (0 disables validation).",
    )
    p.add_argument("--tune-nms-iou", action="store_true", help="Grid-search global.nms_iou for best micro F-beta.")
    p.add_argument("--nms-grid", type=str, default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    p.add_argument("--tune-cross-class-iou", action="store_true", help="Grid-search global.cross_class_iou for best micro F-beta.")
    p.add_argument("--cross-grid", type=str, default="0.10,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55")
    p.add_argument("--with-rgb", action="store_true", help="Load images to enable color-based features.")
    p.add_argument(
        "--log-dir",
        type=Path,
        help="Optional directory for filter-train-logistic.log (mirrors terminal output).",
    )


def run(ds: VisionDataset, args: argparse.Namespace) -> VisionDataset:
    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "filter-train-logistic.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            tee_out = _TeeStream(sys.stdout, log_file)
            tee_err = _TeeStream(sys.stderr, log_file)
            with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                return _run_impl(ds, args)
    return _run_impl(ds, args)


def _run_impl(ds: VisionDataset, args: argparse.Namespace) -> VisionDataset:
    gt_ds = router.ingest(src=Path(args.gt_src), from_hint="auto")
    opt_cfg = OptimizerConfig(
        infer_cfg=str(args.config) if args.config else "",
        out_cfg=str(args.out_config),
        match_iou=float(args.match_iou),
        epochs=int(args.epochs),
        lr=float(args.lr),
        l2=float(args.l2),
        alpha=float(args.alpha),
        max_prune_iters=int(args.max_prune_iters),
        min_features=int(args.min_features),
        always_keep=str(args.always_keep),
        seed=int(args.seed),
        device=str(args.device),
        use_rgb=bool(args.with_rgb),
        beta=float(args.beta),
        conf_metric=str(args.conf_metric),
        val_split=float(args.val_split),
        tune_nms_iou=bool(args.tune_nms_iou),
        nms_grid=str(args.nms_grid),
        tune_cross_class_iou=bool(args.tune_cross_class_iou),
        cross_grid=str(args.cross_grid),
    )
    opt = LogisticOptimizer(opt_cfg)
    results = OptimizerResults()
    opt.run(gt_ds=gt_ds, permissive_ds=ds, results=results)
    _write_results(_resolve_results_dir(args), results)
    cfg_path = args.out_config

    cfg = load_filter_config(cfg_path)
    flt = LogisticFilter(cfg)
    load_rgb = bool(args.with_rgb)
    if load_rgb and not cfg.global_cfg.use_rgb:
        print("[warn] --with-rgb requested but config.use_rgb is false; running without RGB features.")
        load_rgb = False
    filtered = _filter_records(ds, flt, load_rgb=load_rgb)
    return VisionDataset(records=filtered, classes=ds.classes, task=ds.task, root=ds.root, meta=ds.meta)


def _resolve_results_dir(args: argparse.Namespace) -> Path:
    target_dir = getattr(args, "target_dir", None)
    if target_dir:
        return Path(target_dir)
    return Path(args.out_config).parent


def _write_results(out_dir: Path, results: OptimizerResults) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "filter_results.yaml"
    out_path.write_text(
        yaml.safe_dump(results.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _filter_records(ds: VisionDataset, flt: LogisticFilter, load_rgb: bool) -> List[Record]:
    out: List[Record] = []
    for rec in ds.records:
        rgb = load_rgb_for_rec(rec) if load_rgb else None
        out.append(flt.filter_record(rec, classes=ds.classes, rgb=rgb))
    return out
