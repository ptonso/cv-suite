from pathlib import Path

from cvsuite.common.core import VisionDataset


def dump_dataset(ds: VisionDataset, path: Path) -> Path:
    return ds.to_json(path)


def load_dataset(path: Path) -> VisionDataset:
    return VisionDataset.from_json(path)
