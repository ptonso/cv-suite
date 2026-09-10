from enum import Enum

class Task(str, Enum):
    cls = "cls"
    multi_cls = "multi-cls"
    det = "det"
    seg = "inst-seg"
    pose = "pose"

    @classmethod
    def from_raw(cls, value: object) -> "Task":
        raw = str(value or "").strip().lower()
        aliases = {
            "cls": cls.cls,
            "classification": cls.cls,
            "multi-cls": cls.multi_cls,
            "multi_cls": cls.multi_cls,
            "multi-class": cls.multi_cls,
            "multiclass": cls.multi_cls,
            "multi-label": cls.multi_cls,
            "det": cls.det,
            "detect": cls.det,
            "detection": cls.det,
            "seg": cls.seg,
            "inst-seg": cls.seg,
            "inst_seg": cls.seg,
            "segment": cls.seg,
            "segmentation": cls.seg,
            "pose": cls.pose,
            "keypoint": cls.pose,
            "keypoints": cls.pose,
        }
        if raw not in aliases:
            raise ValueError(f"Unsupported task id: {value!r}")
        return aliases[raw]

class Split(str, Enum):
    train = "train"
    val = "val"
    test = "test"


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
