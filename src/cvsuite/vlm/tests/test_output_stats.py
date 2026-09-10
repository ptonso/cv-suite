from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import yaml

from cvsuite.common.core import VisionDataset
from cvsuite.vlm import cli as vlm_cli


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), color=(255, 0, 0)).save(path)


def _fake_runner_factory(captured: dict):
    class _Runner:
        def run(self, dataset: VisionDataset, *, no_resume: bool = False) -> VisionDataset:
            captured["request"] = dataset.fm_request
            for record in dataset.records:
                for qa in record.vqas:
                    qa.answer = f"ans:{qa.question}"
                    qa.model = "qwen"
            return dataset

    return _Runner()


def test_vlm_to_shards_writes_stats_yaml(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "flat"
    _make_image(src / "0001.jpg")
    (src / "0001.json").write_text(
        json.dumps({"sample_key": "0001", "qa_pairs": [{"question_id": 1, "question": "What is visible?"}]}),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr("cvsuite.vlm.transform.core.runtime.FMRunner.from_dataset", lambda dataset: _fake_runner_factory(captured))

    dst = tmp_path / "sharded"
    exit_code = vlm_cli.main(
        [str(src), "vqa", "--provider", "qwen", "to-shards", str(dst), "--target-shard-size-mb", "1", "--max-samples-per-shard", "1", "--with-stats"]
    )

    assert exit_code == 0
    stats = yaml.safe_load((dst / "stats.yaml").read_text(encoding="utf-8"))
    assert stats["command"] == "to-shards"
    assert stats["summary"]["shards_written"] == 1
    assert stats["summary"]["qa_pairs_total"] == 1
