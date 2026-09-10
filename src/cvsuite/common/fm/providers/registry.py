from __future__ import annotations

import ast
import importlib
import importlib.util
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

LOCAL_FAMILY_NAMES = ("classify", "create", "edit", "ground", "ocr", "vlm")
PROVIDER_PACKAGE_ROOT = "cvsuite.common.fm.providers"


@dataclass(frozen=True)
class ProviderSpec:
    provider_name: str
    module: str
    family: str
    provider_class: str = "local"
    description: str = ""
    default_model_id: str = ""
    params: str = "unknown"
    aliases: tuple[str, ...] = ()
    allowed_model_ids: tuple[str, ...] = ()

    @property
    def model_name(self) -> str:
        return self.provider_name


def _discover_local_provider_modules() -> dict[tuple[str, str], str]:
    modules: dict[tuple[str, str], str] = {}
    for family in LOCAL_FAMILY_NAMES:
        package_name = f"{PROVIDER_PACKAGE_ROOT}.{family}"
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            continue
        for module_info in pkgutil.iter_modules(package.__path__):
            if module_info.ispkg or module_info.name.startswith("_"):
                continue
            modules[(family, module_info.name)] = f"{package_name}.{module_info.name}"
    return dict(sorted(modules.items()))


LOCAL_PROVIDER_MODULES = _discover_local_provider_modules()


API_PROVIDER_SPECS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        provider_name="openrouter",
        family="create",
        provider_class="api",
        module="cvsuite.common.fm.providers.api.openrouter_gen",
        description="OpenRouter text-to-image provider.",
        default_model_id="google/gemini-2.5-flash-image",
        params="api",
        allowed_model_ids=(
            "google/gemini-2.5-flash-image",
            "google/gemini-3.1-flash-image-preview",
            "black-forest-labs/flux.2-pro",
            "black-forest-labs/flux.2-flex",
            "sourceful/riverflow-v2-fast",
            "sourceful/riverflow-v2-pro",
            "sourceful/riverflow-v2-standard-preview",
        ),
    ),
    ProviderSpec(
        provider_name="openrouter",
        family="edit",
        provider_class="api",
        module="cvsuite.common.fm.providers.api.openrouter_gen",
        description="OpenRouter image-edit provider.",
        default_model_id="google/gemini-2.5-flash-image",
        params="api",
        allowed_model_ids=(
            "google/gemini-2.5-flash-image",
            "sourceful/riverflow-v2-fast",
            "sourceful/riverflow-v2-pro",
            "sourceful/riverflow-v2-standard-preview",
        ),
    ),
    ProviderSpec(
        provider_name="openrouter",
        family="vlm",
        provider_class="api",
        module="cvsuite.common.fm.providers.api.openrouter_vlm",
        description="OpenRouter multimodal VLM provider.",
        default_model_id="qwen/qwen2.5-vl-72b-instruct",
        params="api",
        allowed_model_ids=(
            "qwen/qwen2.5-vl-72b-instruct",
            "qwen/qwen2.5-vl-32b-instruct",
            "google/gemini-2.5-flash",
            "google/gemini-2.5-pro",
            "openai/gpt-4.1-mini",
        ),
    ),
)


def _module_source_path(module_name: str) -> Path | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return None
    path = Path(spec.origin)
    return path if path.is_file() else None


def _string_value(node: ast.AST, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _string_tuple_value(node: ast.AST, constants: dict[str, str]) -> tuple[str, ...]:
    if not isinstance(node, (ast.Tuple, ast.List)):
        return ()
    values: list[str] = []
    for item in node.elts:
        value = _string_value(item, constants)
        if value is not None:
            values.append(value)
    return tuple(values)


def _source_local_provider_info(module_name: str) -> tuple[str, str, str, tuple[str, ...], tuple[str, ...]]:
    path = _module_source_path(module_name)
    if path is None:
        return "", "", "unknown", (), ()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return "", "", "unknown", (), ()

    constants: dict[str, str] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            value = _string_value(stmt.value, constants)
            if value is None:
                continue
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = value

    description = ""
    default_model_id = constants.get("DEFAULT_MODEL_ID", "")
    params = constants.get("PARAMS", "unknown")
    aliases: tuple[str, ...] = ()
    allowed_model_ids: tuple[str, ...] = ()
    aliases_node = next(
        (
            stmt.value
            for stmt in tree.body
            if isinstance(stmt, ast.Assign)
            for target in stmt.targets
            if isinstance(target, ast.Name) and target.id == "MODEL_ALIASES"
        ),
        None,
    )
    if aliases_node is not None:
        aliases = _string_tuple_value(aliases_node, constants)
    allowed_model_ids_node = next(
        (
            stmt.value
            for stmt in tree.body
            if isinstance(stmt, ast.Assign)
            for target in stmt.targets
            if isinstance(target, ast.Name) and target.id == "ALLOWED_MODEL_IDS"
        ),
        None,
    )
    if allowed_model_ids_node is not None:
        allowed_model_ids = _string_tuple_value(allowed_model_ids_node, constants)
    for stmt in tree.body:
        if not isinstance(stmt, ast.ClassDef):
            continue
        for class_stmt in stmt.body:
            if isinstance(class_stmt, ast.Assign):
                value = _string_value(class_stmt.value, constants)
                for target in class_stmt.targets:
                    if isinstance(target, ast.Name) and value is not None and target.id == "description":
                        description = value
    return description, default_model_id, params, aliases, allowed_model_ids


def _local_provider_spec(family: str, provider_name: str, module_name: str, *, include_descriptions: bool) -> ProviderSpec:
    description = ""
    default_model_id = ""
    params = "unknown"
    aliases: tuple[str, ...] = ()
    allowed_model_ids: tuple[str, ...] = ()
    if include_descriptions:
        description, default_model_id, params, aliases, allowed_model_ids = _source_local_provider_info(module_name)
    return ProviderSpec(
        provider_name=provider_name,
        module=module_name,
        family=family,
        provider_class="local",
        description=description,
        default_model_id=default_model_id,
        params=params,
        aliases=aliases,
        allowed_model_ids=allowed_model_ids,
    )


def iter_providers_for_family(family: str, *, include_descriptions: bool = False) -> Iterator[ProviderSpec]:
    for (candidate_family, provider_name), module_name in sorted(LOCAL_PROVIDER_MODULES.items()):
        if candidate_family != family:
            continue
        yield _local_provider_spec(candidate_family, provider_name, module_name, include_descriptions=include_descriptions)
    for spec in API_PROVIDER_SPECS:
        if spec.family == family:
            yield spec


def provider_registry_items() -> Iterator[tuple[tuple[str, str], ProviderSpec]]:
    for family in LOCAL_FAMILY_NAMES:
        for spec in iter_providers_for_family(family, include_descriptions=True):
            yield (spec.family, spec.provider_name), spec


def resolve_provider_spec(provider_name: str, *, family: str) -> ProviderSpec:
    normalized = str(provider_name or "").strip()
    if not normalized:
        raise ValueError("Provider name must be non-empty.")
    for spec in iter_providers_for_family(family, include_descriptions=True):
        if normalized == spec.provider_name or normalized in spec.aliases:
            return spec
    available = ", ".join(sorted({spec.provider_name for spec in iter_providers_for_family(family, include_descriptions=True)}))
    raise ValueError(f"Unknown FM provider {normalized!r} for family {family!r}. Available providers: {available}.")


def resolve_provider_module(provider_name: str, *, family: str) -> str:
    return resolve_provider_spec(provider_name, family=family).module


def provider_choices_for_family(family: str) -> list[str]:
    return [spec.provider_name for spec in iter_providers_for_family(family, include_descriptions=True)]


def default_model_id_for_provider(provider_name: str, *, family: str) -> str:
    return resolve_provider_spec(provider_name, family=family).default_model_id


def allowed_model_ids_for_provider(provider_name: str, *, family: str) -> tuple[str, ...]:
    return resolve_provider_spec(provider_name, family=family).allowed_model_ids


def format_available_providers(
    family: str,
    *,
    provider_names: tuple[str, ...] | None = None,
    prefer_aliases: bool = False,
) -> str:
    rows = list(iter_providers_for_family(family, include_descriptions=True))
    if provider_names is not None:
        allowed = set(provider_names)
        rows = [row for row in rows if row.provider_name in allowed or any(alias in allowed for alias in row.aliases)]
    if not rows:
        return "Available providers: none discovered."

    sections: dict[str, list[ProviderSpec]] = {"local": [], "api": []}
    for row in rows:
        sections.setdefault(row.provider_class, []).append(row)

    lines: list[str] = []
    for provider_class, title in (("local", "Local providers:"), ("api", "API providers:")):
        section_rows = sections.get(provider_class, [])
        if not section_rows:
            continue
        lines.append(title)
        for row in section_rows:
            default_model_id = row.default_model_id or row.provider_name
            display_id = row.provider_name
            if prefer_aliases and row.aliases:
                if provider_names is None:
                    display_id = row.aliases[0]
                else:
                    display_id = next((alias for alias in row.aliases if alias in set(provider_names)), row.aliases[0])
            lines.append(f"  {display_id}: default-model-id={default_model_id} params={row.params}")
    return "\n".join(lines)


# Compatibility aliases during the rename.
ModelInfo = ProviderSpec
MODEL_PACKAGE_NAMES = LOCAL_FAMILY_NAMES
MODEL_PACKAGE_ROOT = PROVIDER_PACKAGE_ROOT
MODEL_MODULES = {spec.provider_name: spec.module for (_key, spec) in provider_registry_items() if spec.provider_class == "local"}


def resolve_model_module(model_name: str) -> str:
    for family in LOCAL_FAMILY_NAMES:
        try:
            return resolve_provider_module(model_name, family=family)
        except ValueError:
            continue
    available = ", ".join(sorted(MODEL_MODULES))
    raise ValueError(f"Unknown FM provider {model_name!r}. Available providers: {available}.")


def model_registry_items() -> Iterator[tuple[str, str]]:
    for provider_name, module_name in sorted(MODEL_MODULES.items()):
        yield provider_name, module_name


def iter_models_for_family(family: str, *, include_descriptions: bool = False) -> Iterator[ProviderSpec]:
    yield from iter_providers_for_family(family, include_descriptions=include_descriptions)


def format_available_models(
    family: str,
    *,
    model_names: tuple[str, ...] | None = None,
    prefer_aliases: bool = False,
) -> str:
    return format_available_providers(family, provider_names=model_names, prefer_aliases=prefer_aliases)
