from __future__ import annotations

import argparse
from pathlib import Path

SAM3_DEPRECATED_TIMM_IMPORT = "timm.models.layers"
SAM3_TIMM_IMPORT_PATCHES = {
    Path("model/memory.py"): (
        "try:\n"
        "    from timm.layers import DropPath\n"
        "except ModuleNotFoundError:\n"
        "    # compatibility for older timm versions\n"
        "    from timm.models.layers import DropPath\n",
        "from timm.layers import DropPath\n",
    ),
    Path("model/video_tracking_multiplex.py"): (
        "from timm.models.layers import trunc_normal_\n",
        "from timm.layers import trunc_normal_\n",
    ),
    Path("model/vitdet.py"): (
        "try:\n"
        "    from timm.layers import DropPath, trunc_normal_\n"
        "except ModuleNotFoundError:\n"
        "    # compatibility for older timm versions\n"
        "    from timm.models.layers import DropPath, trunc_normal_\n",
        "from timm.layers import DropPath, trunc_normal_\n",
    ),
    Path("model/sam3_tracker_base.py"): (
        "try:\n"
        "    from timm.layers import trunc_normal_\n"
        "except ModuleNotFoundError:\n"
        "    # compatibility for older timm versions\n"
        "    from timm.models.layers import trunc_normal_\n",
        "from timm.layers import trunc_normal_\n",
    ),
}
SAM3_TIMM_IMPORT_PATCH_TARGETS = tuple(str(path) for path in SAM3_TIMM_IMPORT_PATCHES)


def _resolve_sam3_package_root(sam3_root: Path | str) -> Path:
    root = Path(sam3_root)
    candidates = (root, root / "sam3")
    for candidate in candidates:
        if all((candidate / rel_path).exists() for rel_path in SAM3_TIMM_IMPORT_PATCHES):
            return candidate
    missing = ", ".join(str(root / rel_path) for rel_path in SAM3_TIMM_IMPORT_PATCHES)
    raise FileNotFoundError(f"SAM3 patch targets are missing under {root}. Looked for: {missing}")


def find_deprecated_timm_import_paths(sam3_root: Path | str) -> list[Path]:
    root = _resolve_sam3_package_root(sam3_root)
    stale: list[Path] = []
    for rel_path in SAM3_TIMM_IMPORT_PATCHES:
        path = root / rel_path
        if SAM3_DEPRECATED_TIMM_IMPORT in path.read_text(encoding="utf-8"):
            stale.append(rel_path)
    return stale


def patch_sam3_timm_imports(sam3_root: Path | str) -> list[Path]:
    root = _resolve_sam3_package_root(sam3_root)
    modified: list[Path] = []
    for rel_path, (old_text, new_text) in SAM3_TIMM_IMPORT_PATCHES.items():
        path = root / rel_path
        text = path.read_text(encoding="utf-8")
        if new_text in text and old_text not in text:
            continue
        if old_text not in text:
            raise RuntimeError(
                f"Unable to patch {path}: expected deprecated timm import block was not found."
            )
        path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        modified.append(rel_path)
    return modified


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Patch the managed SAM3 source tree to use timm.layers imports.")
    parser.add_argument("sam3_root", help="Path to the cloned SAM3 package root.")
    args = parser.parse_args(argv)
    modified = patch_sam3_timm_imports(args.sam3_root)
    summary = ", ".join(str(path) for path in modified) if modified else "already patched"
    print(f"[sam3-patch] {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
