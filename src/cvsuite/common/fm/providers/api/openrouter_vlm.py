from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from cvsuite.common.core import FMRequest
from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.providers.base import BatchResult, RuntimeContext
from cvsuite.common.fm.providers.bases.vlm import BaseVLMModel, VLMJob

from .openrouter_common import (
    assistant_text_from_response,
    image_path_to_data_url,
    post_openrouter_chat,
    require_openrouter_api_key,
    usage_from_response,
)


@dataclass
class OpenRouterVLMRuntime:
    api_key: str
    model_id: str
    usage: list[dict[str, Any]]


@dataclass(frozen=True)
class OpenRouterVLMOptions:
    model_id: str | None = None


class OpenRouterVLMProvider(BaseVLMModel[OpenRouterVLMOptions, OpenRouterVLMRuntime]):
    model_name = "openrouter"
    description = "OpenRouter multimodal VLM provider."
    options_cls = OpenRouterVLMOptions

    def validate_auto_device_request(self, request: FMRequest, *, require_supported: bool = True) -> None:
        return None

    def load_runtime(self, dataset: VisionDataset, ctx: RuntimeContext[OpenRouterVLMOptions]) -> OpenRouterVLMRuntime:
        model_id = str(ctx.options.model_id or "").strip()
        if not model_id:
            raise RuntimeError("OpenRouter VLM provider requires `model_id` in request.model_args.")
        return OpenRouterVLMRuntime(
            api_key=require_openrouter_api_key(),
            model_id=model_id,
            usage=[],
        )

    def process_batch(
        self,
        dataset: VisionDataset,
        batch: Sequence[VLMJob],
        runtime: OpenRouterVLMRuntime,
        ctx: RuntimeContext[OpenRouterVLMOptions],
    ) -> BatchResult:
        ready_jobs, _images, warnings = self.load_batch_images(dataset, batch)
        if not ready_jobs:
            return BatchResult(modified_record_indices=[], warnings=warnings)

        answers: list[str] = []
        for job in ready_jobs:
            payload = {
                "model": runtime.model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": job.question},
                            {"type": "image_url", "image_url": {"url": image_path_to_data_url(job.image_path)}},
                        ],
                    }
                ],
            }
            response = post_openrouter_chat(api_key=runtime.api_key, payload=payload)
            runtime.usage.append(usage_from_response(response))
            answers.append(assistant_text_from_response(response))
        return BatchResult(
            modified_record_indices=self.append_answers(dataset, ready_jobs, answers),
            warnings=warnings,
        )

    def build_fm_meta(
        self,
        dataset: VisionDataset,
        runtime: OpenRouterVLMRuntime,
        ctx: RuntimeContext[OpenRouterVLMOptions],
    ) -> dict[str, object]:
        return {
            "task": "vlm",
            "provider": self.model_name,
            "provider_class": "api",
            "model_id": runtime.model_id,
            "batch_size": ctx.batch_size,
            "questions": list(self.question_meta),
            "usage": list(runtime.usage),
        }


MODEL = OpenRouterVLMProvider()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
