"""Capture real OCR/VQA outputs from docs/example_data and embed them in README.md.

Run after `generate_examples.sh` has produced docs/example_data/images/ocr.jpg
and docs/example_data/answers/dog.json. Writes an SVG "card" rendering of each
answer's JSON (there's no CLI export for recognized OCR text yet, so this
re-runs the OCR transform in-process to get it) next to its source file, and
README.md references those SVGs with a plain markdown image embed. Only the
generated assets change on a rerun -- README.md's own text never needs
touching, since it just embeds whatever the asset currently contains.
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

from cvsuite.label.core import router as label_router
from cvsuite.label.transform.ocr.commands import run as ocr_run

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "example_data"

FONT_SIZE = 14
LINE_HEIGHT = 20
CHAR_WIDTH = 8.4
PAD_X = 16
PAD_Y = 14


def _run_ocr_answer() -> Path:
    dataset = label_router.ingest(DATA / "images" / "ocr.jpg")
    parser = argparse.ArgumentParser()
    ocr_run.attach(parser)
    args = parser.parse_args(["--provider", "paddleocr"])
    dataset = ocr_run.run(dataset, args)

    entries = [
        {"text": box.text, "score": box.score}
        for record in dataset.records
        for box in record.boxes
        if box.kind == "ocr" and box.text
    ]
    out_path = DATA / "ocr-out" / "answer.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


def _svg_card(json_text: str, svg_path: Path) -> None:
    lines = json_text.splitlines()
    width = PAD_X * 2 + max((len(line) for line in lines), default=0) * CHAR_WIDTH
    height = PAD_Y * 2 + LINE_HEIGHT * len(lines)

    text_lines = "\n".join(
        f'<text x="{PAD_X}" y="{PAD_Y + LINE_HEIGHT * i + FONT_SIZE}">{escape(line)}</text>'
        for i, line in enumerate(lines)
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}">
  <rect width="100%" height="100%" rx="6" fill="#f6f8fa" stroke="#d0d7de"/>
  <g font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="{FONT_SIZE}" fill="#1f2328" xml:space="preserve">
{text_lines}
  </g>
</svg>
"""
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg, encoding="utf-8")


def main() -> None:
    ocr_answer_path = _run_ocr_answer()
    ocr_entries = json.loads(ocr_answer_path.read_text(encoding="utf-8"))
    _svg_card(json.dumps(ocr_entries, indent=2, ensure_ascii=False), DATA / "ocr-out" / "answer.svg")

    vqa_answer_path = DATA / "answers" / "dog.json"
    vqa_payload = json.loads(vqa_answer_path.read_text(encoding="utf-8"))
    _svg_card(json.dumps(vqa_payload["qa_pairs"][0], indent=2, ensure_ascii=False), DATA / "answers" / "answer.svg")


if __name__ == "__main__":
    main()
