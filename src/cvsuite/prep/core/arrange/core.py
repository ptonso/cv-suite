from ..actions import apply_actions
from ..config import ArrangeConfig
from ..fs import require_image_source
from ..logging import Logger
from ..manifest import emit_prep_manifest

from .ops import (
    apply_exact_dedup,
    apply_perceptual_dedup,
    apply_flatten,
    apply_rename_seq,
    build_existing_hashes,
    build_existing_phashes,
    build_actions,
    build_entries,
    next_sequence_id,
    route_non_images,
)


def run_arrange_core(config: ArrangeConfig) -> None:
    require_image_source(config.src, allow_video=True, allow_archive=config.unzip)
    logger = Logger(verbose=config.verbose)
    existing_hashes = build_existing_hashes(config.dst, logger) if config.dedup_mode == "exact" else None
    existing_phashes = build_existing_phashes(config.dst, logger) if config.dedup_mode == "perceptual" else None
    entries = build_entries(config, logger)
    apply_flatten(entries, config.flatten)
    skipped_duplicates = 0
    if config.dedup_mode == "exact":
        skipped_duplicates = apply_exact_dedup(entries, logger, existing_hashes)
    elif config.dedup_mode == "perceptual":
        skipped_duplicates = apply_perceptual_dedup(entries, logger, config.perceptual_threshold, existing_phashes)
    route_non_images(entries, config.non_image)
    if config.rename_seq:
        apply_rename_seq(entries, config.flatten, next_sequence_id(config.dst, config.flatten))
    actions = build_actions(config, entries)
    apply_actions(actions, config.link_mode, config.dry_run, logger)
    if config.manifest and not config.dry_run:
        emit_prep_manifest(config, "arrange", stats={"skipped_duplicates": skipped_duplicates})
