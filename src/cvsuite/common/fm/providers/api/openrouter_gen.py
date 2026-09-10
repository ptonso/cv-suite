from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from cvsuite.common.core import FMRequest
from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.base import BatchResult, RuntimeContext
from cvsuite.common.fm.providers.bases.create import CreateGenerationJob
from cvsuite.common.fm.providers.bases.edit import EditGenerationJob
from cvsuite.common.fm.providers.bases.gen import (
    BaseGenerationModel,
    GEN_OUTPUT_KEY_ATTR,
    GeneratedImage,
    generation_record_triplet,
    materialize_generated_output,
)

from .openrouter_common import (
    assistant_image_data_urls_from_response,
    decode_data_url_image,
    image_path_to_data_url,
    post_openrouter_chat,
    require_openrouter_api_key,
    usage_from_response,
)


_ASPECT_RATIOS = {
    (1, 1): "1:1",
    (2, 3): "2:3",
    (3, 2): "3:2",
    (3, 4): "3:4",
    (4, 3): "4:3",
    (4, 5): "4:5",
    (5, 4): "5:4",
    (9, 16): "9:16",
    (16, 9): "16:9",
    (21, 9): "21:9",
}


@dataclass
class OpenRouterGenerationRuntime:
    mode: str
    model_id: str
    api_key: str
    outputs_dir: Path
    usage: list[dict[str, Any]]


@dataclass(frozen=True)
class OpenRouterGenerationOptions:
    model_id: str | None = None
    width: int | None = None
    height: int | None = None


class OpenRouterGenerationProvider(BaseGenerationModel[OpenRouterGenerationOptions, Any, OpenRouterGenerationRuntime]):
    model_name = "openrouter"
    options_cls = OpenRouterGenerationOptions

    def validate_auto_device_request(self, request: FMRequest, *, require_supported: bool = True) -> None:
        return None

    def build_jobs(self, dataset: VisionDataset, ctx: RuntimeContext[OpenRouterGenerationOptions]) -> list[Any]:
        mode = self.validate_generation_request(dataset, ctx.request)
        jobs: list[Any] = []
        for record_idx, record in enumerate(dataset.records):
            prompt, prompt_index, output_key = generation_record_triplet(record_idx, record)
            if mode == "create":
                jobs.append(
                    CreateGenerationJob(
                        record_idx=record_idx,
                        prompt=prompt,
                        prompt_index=prompt_index,
                        output_key=output_key,
                    )
                )
                continue
            source_image_path = utils.resolve_path(record.image.path, dataset.root, ctx.caller_cwd, allow_root_basename=True)
            jobs.append(
                EditGenerationJob(
                    record_idx=record_idx,
                    prompt=prompt,
                    prompt_index=prompt_index,
                    output_key=output_key,
                    source_image_path=source_image_path,
                )
            )
        return jobs

    def job_id(self, job: Any, dataset: VisionDataset, ctx: RuntimeContext[OpenRouterGenerationOptions]) -> str:
        return f"record:{job.record_idx}"

    def load_runtime(
        self,
        dataset: VisionDataset,
        ctx: RuntimeContext[OpenRouterGenerationOptions],
    ) -> OpenRouterGenerationRuntime:
        outputs_dir = ctx.work_dir / "images"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        model_id = str(ctx.options.model_id or "").strip()
        if not model_id:
            raise RuntimeError("OpenRouter generation provider requires `model_id` in request.model_args.")
        return OpenRouterGenerationRuntime(
            mode=str(dataset.meta.get("gen_mode") or "create"),
            model_id=model_id,
            api_key=require_openrouter_api_key(),
            outputs_dir=outputs_dir,
            usage=[],
        )

    @staticmethod
    def _aspect_ratio_for_request(model_id: str, width: int | None, height: int | None) -> str | None:
        if width is None and height is None:
            return None
        if width is None or height is None:
            raise RuntimeError("OpenRouter generation requires both width and height when requesting a mapped aspect ratio.")
        if not model_id.startswith("google/gemini-"):
            raise RuntimeError(
                f"Provider `openrouter` does not support deterministic width/height mapping for model-id {model_id!r}."
            )
        from math import gcd

        common = gcd(width, height)
        ratio = (width // common, height // common)
        if ratio not in _ASPECT_RATIOS:
            raise RuntimeError(
                f"Unsupported width/height pair for OpenRouter Gemini image generation: width={width} height={height}."
            )
        return _ASPECT_RATIOS[ratio]

    @staticmethod
    def _modalities_for_model(model_id: str) -> list[str]:
        if model_id.startswith(("black-forest-labs/", "sourceful/")):
            return ["image"]
        return ["image", "text"]

    def _request_image(
        self,
        runtime: OpenRouterGenerationRuntime,
        job: Any,
        ctx: RuntimeContext[OpenRouterGenerationOptions],
    ) -> Image.Image:
        width = ctx.options.width
        height = ctx.options.height
        aspect_ratio = self._aspect_ratio_for_request(runtime.model_id, width, height)
        user_content: Any
        if runtime.mode == "edit":
            user_content = [
                {"type": "text", "text": job.prompt},
                {"type": "image_url", "image_url": {"url": image_path_to_data_url(job.source_image_path)}},
            ]
        else:
            user_content = job.prompt

        payload: dict[str, Any] = {
            "model": runtime.model_id,
            "messages": [{"role": "user", "content": user_content}],
            "modalities": self._modalities_for_model(runtime.model_id),
        }
        if aspect_ratio is not None:
            payload["image_config"] = {"aspect_ratio": aspect_ratio}
        response = post_openrouter_chat(api_key=runtime.api_key, payload=payload)
        runtime.usage.append(usage_from_response(response))
        data_urls = assistant_image_data_urls_from_response(response)
        if not data_urls:
            raise RuntimeError(f"OpenRouter provider {runtime.model_id!r} did not return an image payload.")
        return decode_data_url_image(data_urls[0])

    def process_batch(
        self,
        dataset: VisionDataset,
        batch: Sequence[Any],
        runtime: OpenRouterGenerationRuntime,
        ctx: RuntimeContext[OpenRouterGenerationOptions],
    ) -> BatchResult:
        dataset.root = runtime.outputs_dir
        modified: list[int] = []
        for job in batch:
            generated = GeneratedImage(image=self._request_image(runtime, job, ctx))
            dst = runtime.outputs_dir / f"{job.output_key}.png"
            width, height = materialize_generated_output(generated, dst)
            record = dataset.records[job.record_idx]
            record.image.path = Path(dst.name)
            record.image.width = int(width)
            record.image.height = int(height)
            utils.append_once(record.attributes.setdefault("fm_tasks", []), "gen")
            modified.append(job.record_idx)
        return BatchResult(modified_record_indices=modified)

    def finalize_dataset(
        self,
        dataset: VisionDataset,
        runtime: OpenRouterGenerationRuntime,
        ctx: RuntimeContext[OpenRouterGenerationOptions],
    ) -> None:
        dataset.root = runtime.outputs_dir

    def build_fm_meta(
        self,
        dataset: VisionDataset,
        runtime: OpenRouterGenerationRuntime,
        ctx: RuntimeContext[OpenRouterGenerationOptions],
    ) -> dict[str, object]:
        return {
            "task": "gen",
            "provider": self.model_name,
            "provider_class": "api",
            "mode": runtime.mode,
            "model_id": runtime.model_id,
            "batch_size": ctx.batch_size,
            "usage": list(runtime.usage),
        }


MODEL = OpenRouterGenerationProvider()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
