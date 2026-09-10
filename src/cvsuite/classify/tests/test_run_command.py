from __future__ import annotations

import os
from collections import Counter
from argparse import Namespace
import argparse
from pathlib import Path

import pytest
from PIL import Image

from cvsuite.classify.core import io as classify_io
from cvsuite.classify.transform.commands import infer as infer_command
from cvsuite.classify.output.commands import to_class_dir
from cvsuite.classify.transform.commands.infer import run as run_infer
from cvsuite.classify.transform.commands.sample import run as run_sample
from cvsuite.common.core import Classification, ImageRecord, Record
from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import prompt_utils


def _write_image(path: Path, *, color: tuple[int, int, int] = (255, 255, 255)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)
    return path


def _record(path: Path, *, label: str | None = None, score: float | None = None, split: str = "train") -> Record:
    classification = None if label is None else Classification(label=label, score=score)
    return Record(
        image=ImageRecord(path=path, width=16, height=16),
        split=split,
        classification=classification,
    )


def _record_with_probs(
    path: Path,
    *,
    label: str | None,
    score: float | None,
    probs: dict[str, float],
    split: str = "train",
    meta: dict[str, object] | None = None,
) -> Record:
    classification = Classification(
        label=label,
        score=score,
        probs=dict(probs),
        meta={} if meta is None else dict(meta),
    )
    return Record(
        image=ImageRecord(path=path, width=16, height=16),
        split=split,
        classification=classification,
    )


def _count_labels(dataset: VisionDataset) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in dataset.records:
        label = None if record.classification is None else record.classification.label
        if label:
            counts[str(label)] += 1
    return counts


def test_infer_transform_populates_fm_request(monkeypatch, tmp_path: Path) -> None:
    dataset = VisionDataset(records=[Record(image=ImageRecord(path=tmp_path / "a.jpg", width=10, height=10))], root=tmp_path)
    captured = {}

    class _Runner:
        def run(self, ds: VisionDataset, *, no_resume: bool = False) -> VisionDataset:
            captured["request"] = ds.fm_request
            captured["no_resume"] = no_resume
            return ds

    monkeypatch.setattr("cvsuite.classify.transform.core.infer.FMRunner.from_dataset", lambda ds: _Runner())

    args = Namespace(
        provider="clip",
        prompt="cat,dog",
        template_prompt=f"a photo with a {prompt_utils.CLASS_TEMPLATE_TOKEN}",
        device="cpu",
        batch=2,
        precision="fp32",
        config=None,
        no_resume=True,
    )
    result = run_infer(dataset, args)
    assert result is dataset
    assert captured["request"] is not None
    assert captured["request"].provider == "clip"
    assert captured["request"].model_args == {"template_prompt": f"a photo with a {prompt_utils.CLASS_TEMPLATE_TOKEN}"}
    assert captured["no_resume"] is True


def test_infer_transform_clears_active_multi_class_ingest_mode(monkeypatch, tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[Record(image=ImageRecord(path=tmp_path / "a.jpg", width=10, height=10))],
        root=tmp_path,
        meta={classify_io.CLASSIFY_INGEST_MODE_META: classify_io.MULTI_CLASS_FMT},
    )
    captured = {}

    class _Runner:
        def run(self, ds: VisionDataset, *, no_resume: bool = False) -> VisionDataset:
            captured["ingest_mode"] = ds.meta.get(classify_io.CLASSIFY_INGEST_MODE_META)
            return ds

    monkeypatch.setattr("cvsuite.classify.transform.core.infer.FMRunner.from_dataset", lambda ds: _Runner())

    run_infer(
        dataset,
        Namespace(
            provider="clip",
            prompt="cat,dog",
            template_prompt=prompt_utils.DEFAULT_CLASS_TEMPLATE_PROMPT,
            device="cpu",
            batch=1,
            precision="fp32",
            config=None,
            no_resume=False,
        ),
    )

    assert captured["ingest_mode"] is None


def test_sample_preserve_mode_scales_class_counts_from_max_n(tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[
            _record(tmp_path / "cat_1.jpg", label="cat"),
            _record(tmp_path / "cat_2.jpg", label="cat"),
            _record(tmp_path / "dog_1.jpg", label="dog"),
            _record(tmp_path / "dog_2.jpg", label="dog"),
            _record(tmp_path / "bird_1.jpg", label="bird"),
            _record(tmp_path / "bird_2.jpg", label="bird"),
            _record(tmp_path / "bird_3.jpg", label="bird"),
        ],
        classes=["bird", "cat", "dog"],
    )

    sampled = run_sample(
        dataset,
        Namespace(
            mode="preserve",
            max_n_per_class=2,
            max_frac_per_class=None,
            with_replacement=False,
            seed=3,
        ),
    )

    assert len(sampled.records) == 4
    assert _count_labels(sampled) == Counter({"bird": 2, "cat": 1, "dog": 1})
    assert sampled.meta["sample"]["targets"] == {"bird": 2, "cat": 1, "dog": 1}

def test_sample_preserve_mode_supports_fraction_targets(tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[
            _record(tmp_path / "cat_1.jpg", label="cat"),
            _record(tmp_path / "cat_2.jpg", label="cat"),
            _record(tmp_path / "dog_1.jpg", label="dog"),
            _record(tmp_path / "dog_2.jpg", label="dog"),
            _record(tmp_path / "bird_1.jpg", label="bird"),
            _record(tmp_path / "bird_2.jpg", label="bird"),
            _record(tmp_path / "bird_3.jpg", label="bird"),
        ],
        classes=["cat", "dog", "bird"],
    )

    sampled = run_sample(
        dataset,
        Namespace(
            mode="preserve",
            max_n_per_class=None,
            max_frac_per_class=0.5,
            with_replacement=False,
            seed=0,
        ),
    )

    assert len(sampled.records) == 3
    assert _count_labels(sampled) == Counter({"cat": 1, "dog": 1, "bird": 1})


def test_sample_balance_mode_supports_fraction_targets(tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[
            _record(tmp_path / "cat_1.jpg", label="cat"),
            _record(tmp_path / "cat_2.jpg", label="cat"),
            _record(tmp_path / "dog_1.jpg", label="dog"),
            _record(tmp_path / "dog_2.jpg", label="dog"),
            _record(tmp_path / "dog_3.jpg", label="dog"),
            _record(tmp_path / "bird_1.jpg", label="bird"),
            _record(tmp_path / "bird_2.jpg", label="bird"),
            _record(tmp_path / "bird_3.jpg", label="bird"),
            _record(tmp_path / "bird_4.jpg", label="bird"),
        ],
        classes=["cat", "dog", "bird"],
    )

    sampled = run_sample(
        dataset,
        Namespace(
            mode="balance",
            max_n_per_class=None,
            max_frac_per_class=0.5,
            with_replacement=False,
            seed=1,
        ),
    )

    assert len(sampled.records) == 3
    assert _count_labels(sampled) == Counter({"cat": 1, "dog": 1, "bird": 1})


def test_sample_balance_mode_with_replacement_allows_duplicate_sampling(tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[
            _record(tmp_path / "cat_1.jpg", label="cat"),
            _record(tmp_path / "dog_1.jpg", label="dog"),
            _record(tmp_path / "dog_2.jpg", label="dog"),
        ],
        classes=["cat", "dog"],
    )

    sampled = run_sample(
        dataset,
        Namespace(
            mode="balance",
            max_n_per_class=2,
            max_frac_per_class=None,
            with_replacement=True,
            seed=0,
        ),
    )

    cat_paths = [record.image.path for record in sampled.records if record.classification and record.classification.label == "cat"]
    assert len(sampled.records) == 4
    assert _count_labels(sampled) == Counter({"cat": 2, "dog": 2})
    assert len(set(cat_paths)) == 1


def test_sample_requires_labeled_records(tmp_path: Path) -> None:
    dataset = VisionDataset(records=[_record(tmp_path / "a.jpg"), _record(tmp_path / "b.jpg")], classes=[])

    with pytest.raises(SystemExit, match="cvsuite prep sample"):
        run_sample(
            dataset,
            Namespace(
                mode="preserve",
                max_n_per_class=1,
                max_frac_per_class=None,
                with_replacement=False,
                seed=0,
            ),
        )


def test_sample_rejects_annotated_datasets(tmp_path: Path) -> None:
    src = tmp_path / "yolo"
    src.mkdir()
    (src / "data.yaml").write_text("train: train/images\nnames: [cat]\n", encoding="utf-8")
    dataset = VisionDataset(records=[_record(tmp_path / "img.jpg", label="cat")], classes=["cat"], root=src)

    with pytest.raises(SystemExit, match="cvsuite label sample"):
        run_sample(
            dataset,
            Namespace(
                mode="preserve",
                max_n_per_class=1,
                max_frac_per_class=None,
                with_replacement=False,
                seed=0,
                source_path=src,
            ),
        )


def test_sample_rejects_active_multi_class_ingest_dataset(tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[_record(tmp_path / "cat.jpg", label="cat")],
        classes=["cat"],
        meta={classify_io.CLASSIFY_INGEST_MODE_META: classify_io.MULTI_CLASS_FMT},
    )

    with pytest.raises(SystemExit, match="--from multi-class"):
        run_sample(
            dataset,
            Namespace(
                mode="preserve",
                max_n_per_class=1,
                max_frac_per_class=None,
                with_replacement=False,
                seed=0,
            ),
        )


def test_sample_fails_without_replacement_when_target_exceeds_class_size(tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[
            _record(tmp_path / "cat_1.jpg", label="cat"),
            _record(tmp_path / "dog_1.jpg", label="dog"),
            _record(tmp_path / "dog_2.jpg", label="dog"),
        ],
        classes=["cat", "dog"],
    )

    with pytest.raises(SystemExit, match="with-replacement"):
        run_sample(
            dataset,
            Namespace(
                mode="balance",
                max_n_per_class=2,
                max_frac_per_class=None,
                with_replacement=False,
                seed=0,
            ),
        )


def test_ingest_multi_class_collapses_duplicate_basenames(tmp_path: Path) -> None:
    src = tmp_path / "src"
    shared_cat = _write_image(src / "cat" / "shared.jpg")
    _write_image(src / "dog" / "shared.jpg")
    nested = _write_image(src / "lights" / "canopy_light" / "nested.jpg")

    dataset = classify_io.ingest(src, from_hint="multi-class")

    assert dataset.meta["format"] == "multi-class"
    assert dataset.meta[classify_io.CLASSIFY_INGEST_MODE_META] == "multi-class"
    assert dataset.classes == ["cat", "dog", "lights/canopy_light"]
    assert len(dataset.records) == 2

    by_name = {str(record.rel_image_path): record for record in dataset.records}
    shared = by_name["shared.jpg"]
    assert shared.image.path == shared_cat
    assert shared.split == "train"
    assert shared.attributes[classify_io.CLASSIFY_MULTI_CLASS_LABELS_ATTR] == ["cat", "dog"]
    assert shared.classification is not None
    assert shared.classification.label == "cat"
    assert shared.classification.score == 1.0
    assert shared.classification.probs == {"cat": 1.0, "dog": 1.0}

    nested_record = by_name["nested.jpg"]
    assert nested_record.image.path == nested
    assert nested_record.attributes[classify_io.CLASSIFY_MULTI_CLASS_LABELS_ATTR] == ["lights/canopy_light"]


def test_ingest_multi_class_rejects_split_prefixed_tree(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write_image(src / "train" / "cat" / "shared.jpg")

    with pytest.raises(SystemExit, match="--from unstructured"):
        classify_io.ingest(src, from_hint="multi-class")


def test_ingest_multi_class_rejects_conflicting_duplicate_files(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write_image(src / "cat" / "shared.jpg", color=(255, 255, 255))
    _write_image(src / "dog" / "shared.jpg", color=(0, 255, 0))

    with pytest.raises(SystemExit, match="different image contents"):
        classify_io.ingest(src, from_hint="multi-class")


def test_to_class_dir_writes_flat_output_when_split_fracs_are_zero(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "src" / "cat.jpg")
    dataset = VisionDataset(records=[_record(image_path, label="cat", split="val")], root=tmp_path / "src")
    dst = tmp_path / "out"

    to_class_dir.run(
        dataset,
        Namespace(
            dst=dst,
            threshold=None,
            hardlink=False,
            unlabeled_name="unlabeled",
            val_frac=0.0,
            test_frac=0.0,
            seed=0,
            randomized=False,
        ),
    )

    assert (dst / "cat" / "cat.jpg").exists()
    assert not (dst / "val").exists()
    assert not (dst / "train").exists()


def test_to_class_dir_preserves_existing_splits_when_requested(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "src" / "cat.jpg")
    dataset = VisionDataset(records=[_record(image_path, label="cat", split="val")], root=tmp_path / "src")
    dst = tmp_path / "out"

    to_class_dir.run(
        dataset,
        Namespace(
            dst=dst,
            threshold=None,
            hardlink=False,
            unlabeled_name="unlabeled",
            val_frac=0.0,
            test_frac=0.0,
            seed=0,
            randomized=False,
            preserve_splits=True,
        ),
    )

    assert (dst / "val" / "cat" / "cat.jpg").exists()
    assert not (dst / "cat").exists()


def test_to_class_dir_preserves_nested_class_paths(tmp_path: Path) -> None:
    first = _write_image(tmp_path / "src" / "first.jpg")
    second = _write_image(tmp_path / "src" / "second.jpg")
    dataset = VisionDataset(
        records=[
            _record(first, label="lights/canopy_light"),
            _record(second, label="lights/led_lights"),
        ],
        root=tmp_path / "src",
    )
    dst = tmp_path / "out"

    to_class_dir.run(
        dataset,
        Namespace(
            dst=dst,
            threshold=None,
            hardlink=False,
            unlabeled_name="unlabeled",
            val_frac=0.0,
            test_frac=0.0,
            seed=0,
            randomized=False,
        ),
    )

    assert (dst / "lights" / "canopy_light" / "first.jpg").exists()
    assert (dst / "lights" / "led_lights" / "second.jpg").exists()
    assert not (dst / "lights_canopy_light").exists()
    assert not (dst / "lights_led_lights").exists()


def test_to_class_dir_assigns_stratified_output_splits(tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[
            _record(_write_image(tmp_path / "src" / "cat_1.jpg"), label="cat", split="test"),
            _record(_write_image(tmp_path / "src" / "cat_2.jpg"), label="cat", split="test"),
            _record(_write_image(tmp_path / "src" / "dog_1.jpg"), label="dog", split="test"),
            _record(_write_image(tmp_path / "src" / "dog_2.jpg"), label="dog", split="test"),
        ],
        root=tmp_path / "src",
    )
    dst = tmp_path / "out"

    to_class_dir.run(
        dataset,
        Namespace(
            dst=dst,
            threshold=None,
            hardlink=False,
            unlabeled_name="unlabeled",
            val_frac=0.5,
            test_frac=0.0,
            seed=0,
            randomized=False,
        ),
    )

    assert len(list((dst / "val" / "cat").glob("*.jpg"))) == 1
    assert len(list((dst / "train" / "cat").glob("*.jpg"))) == 1
    assert len(list((dst / "val" / "dog").glob("*.jpg"))) == 1
    assert len(list((dst / "train" / "dog").glob("*.jpg"))) == 1
    assert not (dst / "test").exists()


def test_to_class_dir_randomized_split_uses_global_counts(tmp_path: Path) -> None:
    records = [_record(_write_image(tmp_path / "src" / "cat_1.jpg"), label="cat")]
    records.extend(
        _record(_write_image(tmp_path / "src" / f"dog_{idx}.jpg"), label="dog")
        for idx in range(1, 6)
    )
    dataset = VisionDataset(records=records, root=tmp_path / "src")
    dst = tmp_path / "out"

    to_class_dir.run(
        dataset,
        Namespace(
            dst=dst,
            threshold=None,
            hardlink=False,
            unlabeled_name="unlabeled",
            val_frac=0.5,
            test_frac=0.0,
            seed=0,
            randomized=True,
        ),
    )

    assert len(list((dst / "val").rglob("*.jpg"))) == 3
    assert len(list((dst / "train").rglob("*.jpg"))) == 3


def test_to_class_dir_routes_low_score_predictions_to_unlabeled(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "src" / "cat.jpg")
    dataset = VisionDataset(records=[_record(image_path, label="cat", score=0.2)], root=tmp_path / "src")
    dst = tmp_path / "out"

    to_class_dir.run(
        dataset,
        Namespace(
            dst=dst,
            threshold=0.5,
            hardlink=False,
            unlabeled_name="unlabeled",
            val_frac=0.0,
            test_frac=0.0,
            seed=0,
            randomized=False,
        ),
    )

    assert (dst / "unlabeled" / "cat.jpg").exists()
    assert not (dst / "cat").exists()


def test_to_class_dir_avoids_filename_collisions(tmp_path: Path) -> None:
    left = _write_image(tmp_path / "left" / "same.jpg", color=(255, 0, 0))
    right = _write_image(tmp_path / "right" / "same.jpg", color=(0, 255, 0))
    dataset = VisionDataset(records=[_record(left, label="cat"), _record(right, label="cat")], root=tmp_path)
    dst = tmp_path / "out"

    to_class_dir.run(
        dataset,
        Namespace(
            dst=dst,
            threshold=None,
            hardlink=False,
            unlabeled_name="unlabeled",
            val_frac=0.0,
            test_frac=0.0,
            seed=0,
            randomized=False,
        ),
    )

    assert (dst / "cat" / "same.jpg").exists()
    assert (dst / "cat" / "same_1.jpg").exists()


def test_to_class_dir_supports_hardlinks(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "src" / "cat.jpg")
    dataset = VisionDataset(records=[_record(image_path, label="cat")], root=tmp_path / "src")
    dst = tmp_path / "out"

    to_class_dir.run(
        dataset,
        Namespace(
            dst=dst,
            threshold=None,
            hardlink=True,
            unlabeled_name="unlabeled",
            val_frac=0.0,
            test_frac=0.0,
            seed=0,
            randomized=False,
        ),
    )

    out_path = dst / "cat" / "cat.jpg"
    assert out_path.exists()
    assert os.stat(image_path).st_ino == os.stat(out_path).st_ino


def test_to_class_dir_multi_class_writes_to_all_thresholded_labels(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "src" / "scene.jpg")
    dataset = VisionDataset(
        records=[
            _record_with_probs(
                image_path,
                label="cat",
                score=0.8,
                probs={"cat": 0.8, "dog": 0.65, "bird": 0.2},
            )
        ],
        root=tmp_path / "src",
    )
    dst = tmp_path / "out"

    to_class_dir.run(
        dataset,
        Namespace(
            dst=dst,
            threshold=0.5,
            multi_class=True,
            hardlink=False,
            unlabeled_name="unlabeled",
            val_frac=0.0,
            test_frac=0.0,
            seed=0,
            randomized=False,
        ),
    )

    assert (dst / "cat" / "scene.jpg").exists()
    assert (dst / "dog" / "scene.jpg").exists()
    assert not (dst / "bird").exists()
    assert not (dst / "unlabeled").exists()


def test_to_class_dir_multi_class_routes_to_unlabeled_when_none_pass_threshold(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "src" / "scene.jpg")
    dataset = VisionDataset(
        records=[
            _record_with_probs(
                image_path,
                label=None,
                score=0.45,
                probs={"cat": 0.45, "dog": 0.4},
            )
        ],
        root=tmp_path / "src",
    )
    dst = tmp_path / "out"

    to_class_dir.run(
        dataset,
        Namespace(
            dst=dst,
            threshold=0.5,
            multi_class=True,
            hardlink=False,
            unlabeled_name="unlabeled",
            val_frac=0.0,
            test_frac=0.0,
            seed=0,
            randomized=False,
        ),
    )

    assert (dst / "unlabeled" / "scene.jpg").exists()
    assert not (dst / "cat").exists()
    assert not (dst / "dog").exists()


def test_to_class_dir_multi_class_supports_hardlinks_in_multiple_output_folders(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "src" / "scene.jpg")
    dataset = VisionDataset(
        records=[
            _record_with_probs(
                image_path,
                label="cat",
                score=0.9,
                probs={"cat": 0.9, "dog": 0.7},
            )
        ],
        root=tmp_path / "src",
    )
    dst = tmp_path / "out"

    to_class_dir.run(
        dataset,
        Namespace(
            dst=dst,
            threshold=0.5,
            multi_class=True,
            hardlink=True,
            unlabeled_name="unlabeled",
            val_frac=0.0,
            test_frac=0.0,
            seed=0,
            randomized=False,
        ),
    )

    cat_path = dst / "cat" / "scene.jpg"
    dog_path = dst / "dog" / "scene.jpg"
    assert cat_path.exists()
    assert dog_path.exists()
    assert os.stat(image_path).st_ino == os.stat(cat_path).st_ino
    assert os.stat(image_path).st_ino == os.stat(dog_path).st_ino


def test_to_class_dir_multi_class_without_threshold_falls_back_to_single_label(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "src" / "scene.jpg")
    dataset = VisionDataset(
        records=[
            _record_with_probs(
                image_path,
                label="cat",
                score=0.8,
                probs={"cat": 0.8, "dog": 0.7},
            )
        ],
        root=tmp_path / "src",
    )
    dst = tmp_path / "out"

    to_class_dir.run(
        dataset,
        Namespace(
            dst=dst,
            threshold=None,
            multi_class=True,
            hardlink=False,
            unlabeled_name="unlabeled",
            val_frac=0.0,
            test_frac=0.0,
            seed=0,
            randomized=False,
        ),
    )

    assert (dst / "cat" / "scene.jpg").exists()
    assert not (dst / "dog").exists()


def test_to_class_dir_multi_class_uses_record_threshold_metadata_when_output_threshold_is_missing(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "src" / "scene.jpg")
    dataset = VisionDataset(
        records=[
            _record_with_probs(
                image_path,
                label="cat",
                score=0.8,
                probs={"cat": 0.8, "dog": 0.7, "bird": 0.2},
                meta={"threshold": 0.5},
            )
        ],
        root=tmp_path / "src",
    )
    dst = tmp_path / "out"

    to_class_dir.run(
        dataset,
        Namespace(
            dst=dst,
            threshold=None,
            multi_class=True,
            hardlink=False,
            unlabeled_name="unlabeled",
            val_frac=0.0,
            test_frac=0.0,
            seed=0,
            randomized=False,
        ),
    )

    assert (dst / "cat" / "scene.jpg").exists()
    assert (dst / "dog" / "scene.jpg").exists()
    assert not (dst / "bird").exists()


def test_to_class_dir_single_label_winner_ignores_its_own_threshold_once_any_class_passes(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "src" / "scene.jpg")
    dataset = VisionDataset(
        records=[
            _record_with_probs(
                image_path,
                label="first",
                score=0.49,
                probs={"first": 0.49, "second": 0.30},
            )
        ],
        root=tmp_path / "src",
    )
    dst = tmp_path / "out"

    to_class_dir.run(
        dataset,
        Namespace(
            dst=dst,
            threshold=0.05,
            class_thresholds=None,
            class_threshold=["first=0.50", "second=0.10"],
            multi_class=False,
            hardlink=False,
            unlabeled_name="unlabeled",
            val_frac=0.0,
            test_frac=0.0,
            seed=0,
            randomized=False,
        ),
    )

    assert (dst / "first" / "scene.jpg").exists()
    assert not (dst / "second").exists()
    assert not (dst / "unlabeled").exists()


def test_to_class_dir_multi_class_cli_override_beats_yaml_threshold(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "src" / "scene.jpg")
    thresholds_path = tmp_path / "thresholds.yaml"
    thresholds_path.write_text("cat: 0.60\ndog: 0.50\n", encoding="utf-8")
    dataset = VisionDataset(
        records=[
            _record_with_probs(
                image_path,
                label="cat",
                score=0.7,
                probs={"cat": 0.7, "dog": 0.6},
            )
        ],
        root=tmp_path / "src",
    )
    dst = tmp_path / "out"

    to_class_dir.run(
        dataset,
        Namespace(
            dst=dst,
            threshold=0.10,
            class_thresholds=thresholds_path,
            class_threshold=["dog=0.80"],
            multi_class=True,
            hardlink=False,
            unlabeled_name="unlabeled",
            val_frac=0.0,
            test_frac=0.0,
            seed=0,
            randomized=False,
        ),
    )

    assert (dst / "cat" / "scene.jpg").exists()
    assert not (dst / "dog").exists()


def test_to_class_dir_auto_enables_multi_class_for_multi_class_ingest(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write_image(src / "cat" / "shared.jpg")
    _write_image(src / "dog" / "shared.jpg")
    dataset = classify_io.ingest(src, from_hint="multi-class")
    dst = tmp_path / "out"

    to_class_dir.run(
        dataset,
        Namespace(
            dst=dst,
            threshold=None,
            class_thresholds=None,
            class_threshold=[],
            multi_class=False,
            hardlink=False,
            unlabeled_name="unlabeled",
            val_frac=0.0,
            test_frac=0.0,
            seed=0,
            randomized=False,
            preserve_splits=False,
        ),
    )

    assert (dst / "cat" / "shared.jpg").exists()
    assert (dst / "dog" / "shared.jpg").exists()


def test_to_class_dir_multi_class_ingest_keeps_all_copies_in_the_same_split(tmp_path: Path) -> None:
    src = tmp_path / "src"
    for name in ("left.jpg", "right.jpg"):
        _write_image(src / "cat" / name)
        _write_image(src / "dog" / name)
    dataset = classify_io.ingest(src, from_hint="multi-class")
    dst = tmp_path / "out"

    to_class_dir.run(
        dataset,
        Namespace(
            dst=dst,
            threshold=None,
            class_thresholds=None,
            class_threshold=[],
            multi_class=False,
            hardlink=False,
            unlabeled_name="unlabeled",
            val_frac=0.5,
            test_frac=0.0,
            seed=0,
            randomized=False,
            preserve_splits=False,
        ),
    )

    for name in ("left.jpg", "right.jpg"):
        matching_splits = [
            split for split in ("train", "val", "test")
            if (dst / split / "cat" / name).exists() or (dst / split / "dog" / name).exists()
        ]
        assert len(matching_splits) == 1
        split = matching_splits[0]
        assert (dst / split / "cat" / name).exists()
        assert (dst / split / "dog" / name).exists()

    assert len(list((dst / "val").rglob("*.jpg"))) == 2
    assert len(list((dst / "train").rglob("*.jpg"))) == 2


def test_infer_help_documents_prompt_yaml_shapes() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    infer_command.attach(parser)
    help_text = parser.format_help()

    assert "Local providers:" in help_text
    assert "  clip: default-model-id=laion/CLIP-ViT-H-14-laion2B-s32B-b79K params=986M" in help_text
    assert "  siglip2: default-model-id=google/siglip2-so400m-patch16-512 params=400M" in help_text
    assert "Prompt YAML examples:" in help_text
    assert "- partition_a" in help_text
    assert "partition_a:" in help_text
    assert "--template-prompt" in help_text
    assert prompt_utils.CLASS_TEMPLATE_TOKEN in help_text
    assert "a photo with a <class>" in help_text


def test_to_class_dir_help_documents_class_threshold_yaml_shape() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    to_class_dir.attach(parser)
    help_text = parser.format_help()

    assert "Class-threshold YAML example:" in help_text
    assert "partition_a: 0.10" in help_text
    assert "class=value" in help_text
    assert "--from multi-class" in help_text
