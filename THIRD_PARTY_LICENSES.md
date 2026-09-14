# Third-Party Licenses

`cv-suite`'s own code is MIT-licensed (see [LICENSE](LICENSE)). That license
covers the adapter code in `src/cvsuite/common/fm/providers/` — it does
**not** extend to the models those adapters call.

Each provider under `common/fm/providers/setup_venv/*.sh` is optional: it
installs a separate model (code + weights) into its own isolated venv only
when you run `cvsuite setup <provider>`. cv-suite never vendors or
redistributes that code — but by installing it you agree to *its* license,
not cv-suite's. This file tracks those terms per provider.

**Verify before you rely on this table.** Model licenses change between
releases and some entries below could not be confirmed with confidence.
Anything marked "Unverified" should be checked against the upstream repo or
Hugging Face model card before commercial or redistributed use.

## Permissive (Apache-2.0 / MIT / BSD)

| Script | Package / source | Default checkpoint | License | Notes |
|---|---|---|---|---|
| `gdino.sh` | IDEA-Research Grounding DINO | `gdino` | Apache-2.0 | |
| `gsam.sh` | Grounding DINO + `facebookresearch/sam2` | `gsam` | Apache-2.0 | Combines two Apache-2.0 projects |
| `clip.sh` | OpenCLIP | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | MIT | |
| `blip.sh` | Salesforce LAVIS | `Salesforce/blip-vqa-base` | BSD-3-Clause | |
| `siglip2.sh` | Google SigLIP2 | `google/siglip2-so400m-patch16-512` | Apache-2.0 | |
| `paddleocr.sh` | PaddlePaddle/PaddleOCR | `paddleocr` | Apache-2.0 | |
| `llava.sh` | LLaVA-OneVision | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | Apache-2.0 | Base LM (Qwen2-0.5B) also Apache-2.0 |
| `qwen.sh` | Qwen2.5-VL | `Qwen/Qwen2.5-VL-3B-Instruct` | Apache-2.0 | Confirm per size — some Qwen sizes use the Tongyi Qianwen license instead |
| `qwen_image.sh` | Qwen-Image | `Qwen/Qwen-Image` | Apache-2.0 | |
| `qwen_image_edit.sh` | Qwen-Image-Edit | `Qwen/Qwen-Image-Edit` | Apache-2.0 | |

## Restricted / non-commercial / custom terms

| Script | Package / source | Default checkpoint | License | Restriction |
|---|---|---|---|---|
| `yolo_e.sh` | Ultralytics (YOLOE) | `yolo_e` | **AGPL-3.0** | Copyleft; closed-source commercial use requires an Ultralytics Enterprise license |
| `stable_diffusion.sh` | Stability AI SDXL | `stabilityai/stable-diffusion-xl-base-1.0` | CreativeML Open RAIL++-M | Use-based restrictions (no illegal/harmful generation), not pure attribution |
| `flux.sh` | Black Forest Labs FLUX.1-dev | `black-forest-labs/FLUX.1-dev` | FLUX.1-dev Non-Commercial License | **Non-commercial only** — commercial use needs a BFL license |
| `paligemma.sh` | Google PaliGemma | `google/paligemma-3b-ft-ocrvqa-448` | Gemma license | Includes a Prohibited Use Policy; not OSI-approved open source |
| `cogvlm.sh` | Zhipu/Z.ai CogVLM2 | `zai-org/cogvlm2-llama3-chat-19B` | CogVLM2 custom license | Llama-style community license with use/scale restrictions |
| `minicpm_v.sh` | OpenBMB MiniCPM-V | `openbmb/MiniCPM-V-4` | MiniCPM General Model License | Free for most use; commercial use above a threshold requires registration |
| `locate_anything.sh` | NVIDIA LocateAnything | `nvidia/LocateAnything-3B` | NVIDIA Open Model License | Custom terms, not a standard OSS license — read before commercial use |
| `instruct_pix2pix.sh` | `timbrooks/instruct-pix2pix` | same | MIT (code) / inherits SD1.5 weights | Base weights fine-tuned from Stable Diffusion 1.5 — RAIL-style restrictions likely carry over |

## Needs verification (could not confirm from available info)

| Script | Package / source | Default checkpoint | Notes |
|---|---|---|---|
| `sam3.sh` | `facebookresearch/sam3` | `sam3` | Recent release; predecessors (SAM, SAM2) are Apache-2.0 but confirm this one directly |
| `flux2_klein.sh` | Black Forest Labs FLUX.2-klein | `black-forest-labs/FLUX.2-klein-4B` | Newer BFL release — check for the same non-commercial pattern as FLUX.1-dev |
| `sana.sh` | Efficient-Large-Model SANA | `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers` | Likely non-commercial (CC-BY-NC family) — confirm |
| `rex_omni.sh` | IDEA-Research Rex-Omni | `IDEA-Research/Rex-Omni` | IDEA-Research mixes Apache-2.0 and non-commercial research licenses across projects |
| `llmdet.sh` | iSEE-Laboratory LLMDet | `iSEE-Laboratory/llmdet_base` | |
| `internvl.sh` | OpenGVLab InternVL | `OpenGVLab/InternVL2_5-1B` | Code generally MIT; confirm base-LM license if not the Apache-2.0 Qwen2.5 variant |
| `ovis_u1_3b.sh` | AIDC-AI Ovis-U1 | `AIDC-AI/Ovis-U1-3B` | |
| `step1x_edit.sh` | StepFun Step1X-Edit | `stepfun-ai/Step1X-Edit-v1p1-diffusers` | |
| `deep_orientation.sh` | `DuarteBarbosa/deep-image-orientation-detection` | same | Community checkpoint, no clear license found |
| `dim_edit.sh` | `stdKonjac/DIM-4.6B-Edit` (via `showlab/DIM`) | same | Community checkpoint, no clear license found |

## What this means for you as a cv-suite user

- Using cv-suite's core (dataset prep, labeling, CLI) carries only the MIT
  license.
- Running `cvsuite setup <provider>` pulls in that provider's own code and
  weights under its own terms — cv-suite does not, and cannot, relicense
  them.
- If you ship a product that calls `yolo_e`, `stable_diffusion`, `flux`,
  `paligemma`, `cogvlm`, or `minicpm_v`, check the restriction column above
  before commercial or closed-source release.
- This table is a starting point, not legal advice — verify current terms
  directly on the upstream repo or Hugging Face model card before relying on
  it.
