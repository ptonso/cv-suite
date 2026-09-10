from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("Pillow is required to generate test images. Install it with `pip install pillow`.") from exc


REPO_ROOT = Path(__file__).resolve().parent.parent
CLI = REPO_ROOT / "cli.py"
TEST_ROOT = Path(__file__).resolve().parent
RAW_DIR = TEST_ROOT / "raw_input"
LAYOUT_DIR = TEST_ROOT / "layout_output"
CONTENT_DIR = TEST_ROOT / "content_output"


def cleanup(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def make_image(path: Path, size: tuple[int, int], color: str, fmt: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    save_args = {}
    if fmt is not None:
        save_args["format"] = fmt
    img.save(path, **save_args)


def build_raw_dataset() -> None:
    cleanup([RAW_DIR, LAYOUT_DIR, CONTENT_DIR])
    print(f"Creating test dataset under {RAW_DIR}...")

    make_image(RAW_DIR / "animals" / "cat_small.jpg", (320, 240), "red", "JPEG")
    make_image(RAW_DIR / "animals" / "cat_small_dup.jpg", (320, 240), "red", "JPEG")
    make_image(RAW_DIR / "animals" / "dog_wide.png", (1024, 512), "green", "PNG")
    make_image(RAW_DIR / "portraits" / "person_tall.webp", (512, 1024), "blue", "WEBP")
    make_image(RAW_DIR / "misc" / "palette.gif", (128, 128), "yellow", "GIF")
    make_image(RAW_DIR / "misc" / "grayscale.bmp", (400, 200), "gray", "BMP")

    note_path = RAW_DIR / "notes" / "readme.txt"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("Non-image sentinel file for layout non-image handling.\n")

    archive_src = RAW_DIR / "_archive_src"
    make_image(archive_src / "zip_a.png", (64, 64), "orange", "PNG")
    make_image(archive_src / "nested" / "zip_b.tiff", (200, 120), "purple", "TIFF")
    archive_path = RAW_DIR / "archives" / "images.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w") as zf:
        for file_path in archive_src.rglob("*"):
            zf.write(file_path, arcname=file_path.relative_to(archive_src))
    shutil.rmtree(archive_src)
    print("Sample archive created with a couple of images inside.")


def run_cli(label: str, cmd: list[str]) -> None:
    print(f"\nRunning {label} command:")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"{label} command failed with exit code {result.returncode}.")


def run_layout_step() -> None:
    cmd = [
        sys.executable,
        str(CLI),
        "layout",
        str(RAW_DIR),
        str(LAYOUT_DIR),
        "--dedup",
        "--rename-seq",
        "--non-image",
        "collect",
        "--flatten",
        "enc-prefix",
        "-v",
    ]
    run_cli("layout", cmd)
    print(f"Layout output ready at {LAYOUT_DIR}.")
    input("Inspect the layout output, then press Enter to continue to content...\n")


def run_content_step() -> None:
    cmd = [
        sys.executable,
        str(CLI),
        "content",
        str(LAYOUT_DIR),
        str(CONTENT_DIR),
        "--orient",
        "--resize-mode",
        "square",
        "--size",
        "512",
        "--square-mode",
        "pad",
        "--grayscale",
        "--to-format",
        "png",
        "--non-image",
        "collect",
        "-v",
    ]
    run_cli("content", cmd)
    print(f"Content output ready at {CONTENT_DIR}.")
    input("Inspect the content output, then press Enter to clean up the generated files...\n")


def main() -> None:
    if not CLI.exists():
        raise SystemExit(f"Cannot find CLI entrypoint at {CLI}.")

    build_raw_dataset()
    run_layout_step()
    run_content_step()

    cleanup([RAW_DIR, LAYOUT_DIR, CONTENT_DIR])
    print("Temporary test artifacts removed.")


if __name__ == "__main__":
    main()
