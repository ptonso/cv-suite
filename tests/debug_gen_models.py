#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tests" / "output"
SUMMARY_FIELDS = ("task_type", "model_id", "device", "precision", "success")

RUN_MATRIX = (
    ("cpu", "fp32"),
    ("auto", "fp32"),
    ("cuda", "nf4"),
)

FIXABLE_FAILURE_MARKERS = (
    "could not be honored",
    "falling back to fp32",
    "failed health check",
    "missing source image",
    "not implemented for model",
    "not supported for this model",
    "precision 'nf4' requires cuda",
    "unsupported precision",
    "unhealthy venv",
)


@dataclass(frozen=True)
class GenJob:
    task_type: str
    model_id: str
    device: str
    precision: str

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.task_type, self.model_id, self.device, self.precision)

    @property
    def stem(self) -> str:
        return f"{self.device}-{self.precision}"


def _project_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    paths = [str(SRC_ROOT), str(REPO_ROOT)]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def resolve_repo_python() -> str:
    candidates = (
        REPO_ROOT / "venv" / "bin" / "python",
        REPO_ROOT / ".venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _import_registry():
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from cvsuite.common.fm.providers.registry import iter_models_for_family

    return iter_models_for_family


def discover_models(tasks: Sequence[str], selected_models: set[str] | None) -> list[tuple[str, str]]:
    iter_models_for_family = _import_registry()
    discovered: list[tuple[str, str]] = []
    for task_type in tasks:
        for row in iter_models_for_family(task_type):
            if selected_models and row.model_name not in selected_models:
                continue
            discovered.append((task_type, row.model_name))
    if selected_models:
        found = {model for _task, model in discovered}
        missing = sorted(selected_models - found)
        if missing:
            raise SystemExit(f"No selected gen create/edit model found for: {', '.join(missing)}")
    return discovered


def ensure_dummy_inputs(output_root: Path) -> tuple[Path, str, str]:
    inputs_dir = output_root / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    source_image = inputs_dir / "edit_source.png"
    if not source_image.exists():
        image = Image.new("RGB", (256, 256), (34, 92, 145))
        draw = ImageDraw.Draw(image)
        draw.rectangle((48, 48, 208, 208), fill=(235, 198, 84))
        draw.ellipse((88, 88, 168, 168), fill=(220, 80, 72))
        draw.text((24, 224), "cvsuite debug input", fill=(255, 255, 255))
        image.save(source_image)
    create_prompt = "a simple product photo of a red cube on a white table"
    edit_prompt = "turn the central shape into a glossy green object"
    return source_image, create_prompt, edit_prompt


def load_successes(summary_path: Path) -> set[tuple[str, str, str, str]]:
    successes: set[tuple[str, str, str, str]] = set()
    if not summary_path.exists():
        return successes
    with summary_path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("success", "")).strip().lower() != "true":
                continue
            successes.add(
                (
                    str(row.get("task_type", "")),
                    str(row.get("model_id", "")),
                    str(row.get("device", "")),
                    str(row.get("precision", "")),
                )
            )
    return successes


def _load_summary_rows(summary_path: Path) -> list[dict[str, str]]:
    if not summary_path.exists():
        return []
    with summary_path.open("r", newline="", encoding="utf-8") as fh:
        return [
            {
                field: str(row.get(field, ""))
                for field in SUMMARY_FIELDS
            }
            for row in csv.DictReader(fh)
        ]


def append_summary(summary_path: Path, job: GenJob, success: bool) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in _load_summary_rows(summary_path)
        if (
            row.get("task_type"),
            row.get("model_id"),
            row.get("device"),
            row.get("precision"),
        ) != job.key
    ]
    rows.append(
        {
            "task_type": job.task_type,
            "model_id": job.model_id,
            "device": job.device,
            "precision": job.precision,
            "success": "true" if success else "false",
        }
    )
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def command_for_job(
    job: GenJob,
    *,
    python_bin: str,
    dst: Path,
    edit_source: Path,
    create_prompt: str,
    edit_prompt: str,
    width: int,
    height: int,
    steps: int,
    backend: str | None,
) -> list[str]:
    prompt = create_prompt if job.task_type == "create" else edit_prompt
    cmd = [
        python_bin,
        "-m",
        "cvsuite.cli",
        "gen",
        job.task_type,
    ]
    if job.task_type == "edit":
        cmd.extend([str(edit_source), "--from", "images"])
    cmd.extend(
        [
            "--provider",
            job.model_id,
            "--prompt",
            prompt,
            "--device",
            job.device,
            "--precision",
            job.precision,
            "--model-arg",
            f"num_inference_steps={steps}",
            "--no-resume",
        ]
    )
    if width is not None:
        cmd.extend(["--width", str(width)])
    if height is not None:
        cmd.extend(["--height", str(height)])
    if backend:
        cmd.extend(["--model-arg", f"backend={backend}"])
    cmd.extend(["to-dst", str(dst)])
    return cmd


def log_command(log_path: Path, cmd: Sequence[str], result: subprocess.CompletedProcess[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(cmd) + "\n\n")
        fh.write(f"returncode={result.returncode}\n\n")
        fh.write("=== stdout ===\n")
        fh.write(result.stdout or "")
        if result.stdout and not result.stdout.endswith("\n"):
            fh.write("\n")
        fh.write("\n=== stderr ===\n")
        fh.write(result.stderr or "")
        if result.stderr and not result.stderr.endswith("\n"):
            fh.write("\n")


def has_fixable_warning(result: subprocess.CompletedProcess[str]) -> bool:
    text = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
    return any(marker in text for marker in FIXABLE_FAILURE_MARKERS)


def run_job(
    job: GenJob,
    *,
    args: argparse.Namespace,
    python_bin: str,
    edit_source: Path,
    create_prompt: str,
    edit_prompt: str,
) -> bool:
    image_path = args.output_root / "images" / job.task_type / job.model_id / f"{job.stem}.png"
    log_path = args.output_root / "logs" / job.task_type / job.model_id / f"{job.stem}.log"
    if image_path.exists():
        image_path.unlink()
    cmd = command_for_job(
        job,
        python_bin=python_bin,
        dst=image_path,
        edit_source=edit_source,
        create_prompt=create_prompt,
        edit_prompt=edit_prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        backend=args.backend,
    )
    print(f"[gen-debug] running {job.task_type}/{job.model_id} device={job.device} precision={job.precision}")
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=_project_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    log_command(log_path, cmd, result)
    success = result.returncode == 0 and image_path.exists() and not has_fixable_warning(result)
    if result.returncode == 0 and not image_path.exists():
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n=== harness failure ===\nGenerated image not found: {image_path}\n")
    if result.returncode == 0 and has_fixable_warning(result):
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write("\n=== harness failure ===\nFixable warning/fallback marker detected in output.\n")
    append_summary(args.summary_path, job, success)
    print(f"[gen-debug] {'ok' if success else 'failed'} log={log_path}")
    return success


def purge_model(model_id: str, output_root: Path, *, python_bin: str) -> bool:
    log_path = output_root / "logs" / "purge" / f"{model_id}.log"
    cmd = [
        python_bin,
        "-m",
        "cvsuite.cli",
        "common",
        "cache-purge",
        model_id,
        "--part",
        "all",
        "--yes",
    ]
    print(f"[gen-debug] purging cache for {model_id}")
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=_project_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    log_command(log_path, cmd, result)
    if result.returncode != 0:
        print(f"[gen-debug] cache purge failed for {model_id}; see {log_path}", file=sys.stderr)
        return False
    return True


def iter_jobs(model_rows: Iterable[tuple[str, str]]) -> Iterable[GenJob]:
    for task_type, model_id in model_rows:
        for device, precision in RUN_MATRIX:
            yield GenJob(task_type=task_type, model_id=model_id, device=device, precision=precision)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Long-running runtime debugger for cvsuite gen create/edit providers.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root for generated images, logs, and CSV.")
    parser.add_argument("--models", nargs="+", default=None, help="Optional provider subset, e.g. flux qwen_image_edit.")
    parser.add_argument("--tasks", nargs="+", choices=["create", "edit"], default=["create", "edit"], help="Task families to run.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed runtime job.")
    parser.add_argument("--skip-existing-success", action="store_true", help="Skip rows already marked success=true in the summary CSV.")
    parser.add_argument("--no-purge", action="store_true", help="Do not purge model cache after all passes for that model succeed.")
    parser.add_argument("--width", type=int, default=None, help="Optional output width override for debug runs.")
    parser.add_argument("--height", type=int, default=None, help="Optional output height override for debug runs.")
    parser.add_argument("--steps", type=int, default=1, help="Requested num_inference_steps model arg for debug runs.")
    parser.add_argument(
        "--backend",
        default=None,
        help="Optional backend override forwarded as --model-arg backend=...; useful for prompt-card smoke checks.",
    )
    args = parser.parse_args(argv)
    args.output_root = args.output_root.resolve()
    args.summary_path = args.output_root / "gen_debug_summary.csv"
    if args.width is not None and args.width < 1:
        raise SystemExit("--width must be >= 1.")
    if args.height is not None and args.height < 1:
        raise SystemExit("--height must be >= 1.")
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1.")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    python_bin = resolve_repo_python()
    print(f"[gen-debug] using python interpreter: {python_bin}")
    edit_source, create_prompt, edit_prompt = ensure_dummy_inputs(args.output_root)
    selected_models = set(args.models or []) or None
    model_rows = discover_models(args.tasks, selected_models)
    successes = load_successes(args.summary_path) if args.skip_existing_success else set()

    failed = False
    for task_type, model_id in model_rows:
        model_success = True
        model_ran_job = False
        for job in iter_jobs([(task_type, model_id)]):
            if args.skip_existing_success and job.key in successes:
                print(
                    f"[gen-debug] skipping existing success "
                    f"{job.task_type}/{job.model_id} device={job.device} precision={job.precision}"
                )
                continue
            model_ran_job = True
            ok = run_job(
                job,
                args=args,
                python_bin=python_bin,
                edit_source=edit_source,
                create_prompt=create_prompt,
                edit_prompt=edit_prompt,
            )
            model_success = model_success and ok
            if not ok:
                failed = True
                if not args.keep_going:
                    return 1
        if model_success and model_ran_job and not args.no_purge:
            if not purge_model(model_id, args.output_root, python_bin=python_bin):
                failed = True
                if not args.keep_going:
                    return 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
