from __future__ import annotations

import ast
from pathlib import Path

from cvsuite.common.fm.providers import registry


def test_every_discovered_model_has_help_metadata() -> None:
    missing: list[str] = []
    for family in registry.MODEL_PACKAGE_NAMES:
        for model in registry.iter_models_for_family(family, include_descriptions=True):
            if not model.default_model_id:
                missing.append(f"{family}.{model.model_name}")

    assert missing == []


def test_available_model_help_is_key_value_form() -> None:
    help_text = registry.format_available_models("create")

    assert "Local providers:" in help_text
    assert "flux: default-model-id=black-forest-labs/FLUX.1-dev params=12B" in help_text
    assert "\n  -" not in help_text


def test_model_names_are_filename_stems_and_no_wrapper_model_id_attrs() -> None:
    for family in registry.MODEL_PACKAGE_NAMES:
        for model in registry.iter_models_for_family(family, include_descriptions=True):
            if model.provider_class == "local":
                assert model.model_name == model.module.rsplit(".", 1)[-1]
            path = Path(registry._module_source_path(model.module) or "")
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for stmt in tree.body:
                if not isinstance(stmt, ast.ClassDef) or not stmt.name.endswith("Model"):
                    continue
                for class_stmt in stmt.body:
                    targets = []
                    if isinstance(class_stmt, ast.Assign):
                        targets = [target.id for target in class_stmt.targets if isinstance(target, ast.Name)]
                    elif isinstance(class_stmt, ast.AnnAssign) and isinstance(class_stmt.target, ast.Name):
                        targets = [class_stmt.target.id]
                    assert "model_id" not in targets, f"{model.module}.{stmt.name} declares model_id"
