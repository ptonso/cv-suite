from __future__ import annotations

import json

import pytest

from cvsuite.common.fm.core import prompt_utils


def test_build_prompt_map_applies_default_template_to_label_lists() -> None:
    prompt_map = prompt_utils.build_prompt_map(json.dumps(["cat", "dog"]))

    assert prompt_map == {
        "cat": ["a photo of a cat"],
        "dog": ["a photo of a dog"],
    }


def test_build_prompt_map_applies_custom_template_to_label_lists() -> None:
    prompt_map = prompt_utils.build_prompt_map(
        "cat,dog",
        template_prompt=f"a photo with a {prompt_utils.CLASS_TEMPLATE_TOKEN}",
    )

    assert prompt_map == {
        "cat": ["a photo with a cat"],
        "dog": ["a photo with a dog"],
    }


def test_build_prompt_map_keeps_explicit_prompt_maps_verbatim() -> None:
    prompt_map = prompt_utils.build_prompt_map(
        json.dumps(
            {
                "cat": ["feline object", "house cat"],
                "dog": "canine object",
            }
        )
    )

    assert prompt_map == {
        "cat": ["feline object", "house cat"],
        "dog": ["canine object"],
    }


def test_build_prompt_map_requires_class_placeholder_in_template() -> None:
    with pytest.raises(ValueError, match="<class>"):
        prompt_utils.build_prompt_map("cat,dog", template_prompt="a photo with an animal")
