from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from .config import LinkMode
from .fs import materialize_files
from .logging import Logger


ActionKind = Literal["copy", "move", "link", "drop"]


@dataclass
class Action:
    kind: ActionKind
    src: Path | None
    dst: Path | None
    reason: str | None = None


def apply_actions(
    actions: Sequence[Action],
    link_mode: LinkMode,
    dry_run: bool,
    logger: Logger,
) -> None:
    def _log(action: Action) -> None:
        if action.kind == "drop":
            logger.info(f"[DROP] {action.src} ({action.reason})")
        elif action.kind in ("copy", "move", "link"):
            logger.info(f"[{action.kind.upper()}] {action.src} -> {action.dst}")

    if dry_run:
        for action in actions:
            _log(action)
        logger.info(f"Planned actions: {len(actions)}")
        return

    copy_pairs: list[tuple[Path, Path]] = []
    move_pairs: list[tuple[Path, Path]] = []
    link_pairs: list[tuple[Path, Path]] = []

    for action in actions:
        if action.kind == "drop":
            _log(action)
            continue
        if action.src is None or action.dst is None:
            continue
        _log(action)
        if action.kind == "copy":
            copy_pairs.append((action.src, action.dst))
        elif action.kind == "move":
            move_pairs.append((action.src, action.dst))
        elif action.kind == "link":
            link_pairs.append((action.src, action.dst))

    if copy_pairs and link_mode == "copy":
        materialize_files(copy_pairs, "copy")
    if move_pairs and link_mode == "move":
        materialize_files(move_pairs, "move")
    if link_pairs and link_mode == "hard":
        materialize_files(link_pairs, "hard")
