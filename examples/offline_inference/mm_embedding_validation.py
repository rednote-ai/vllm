"""
Validate multimodal embedding return with concurrent streaming requests.

Anchor model: Qwen3-VL (also works with Qwen2-VL / Qwen2.5-VL).

This script:
  1. Initializes AsyncLLM with enable_return_mm_embedding, chunked prefill,
     and prefix caching enabled.
  2. Sends many concurrent streaming requests, each containing an image.
  3. Validates that every finished request carries the correct mm_embedding:
       - Tensor properties (type, device, ndim, non-empty, non-zero).
       - Same image  → identical embedding (bit-exact or within tolerance).
       - Different images → clearly different embeddings (low cosine similarity).
       - Shape consistency across all requests using the same-sized images.
       - shape[0] (num_vision_tokens) matches the expected value computed
         from the Qwen3-VL processor / vision config.

Usage:
    python examples/offline_inference/mm_embedding_validation.py
    python examples/offline_inference/mm_embedding_validation.py \
        --model Qwen/Qwen3-VL-4B-Instruct \
        --num-images 4 --num-requests 40

Requirements:
    - A GPU with enough memory for the chosen model.
    - PIL / Pillow.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"

# Qwen3-VL / Qwen2.5-VL / Qwen2-VL prompt template
PROMPT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{question}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

QUESTIONS = [
    "Describe what you see in this image.",
    "What is the dominant color?",
    "Describe the pattern.",
    "What can you tell me about this picture?",
    "Summarize this image briefly.",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_test_images(num_images: int, size: int = 336) -> list[Image.Image]:
    """Create *visually distinct* test images using numpy for speed."""
    rng = np.random.RandomState(42)
    images: list[Image.Image] = []
    for i in range(num_images):
        # Each image gets a unique gradient + random noise so that the
        # vision encoder produces clearly distinguishable embeddings.
        base = np.zeros((size, size, 3), dtype=np.uint8)

        # Unique directional gradient per image
        xs = np.linspace(0, 255, size, dtype=np.float32)
        ys = np.linspace(0, 255, size, dtype=np.float32)
        xg, yg = np.meshgrid(xs, ys)
        angle = i * (360.0 / num_images)
        rad = np.deg2rad(angle)
        grad = (xg * np.cos(rad) + yg * np.sin(rad)) % 256
        channel = i % 3
        base[:, :, channel] = grad.astype(np.uint8)

        # Add per-image noise so even close angles differ
        noise = rng.randint(0, 60, (size, size, 3), dtype=np.uint8)
        base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        images.append(Image.fromarray(base))
    return images


def compute_expected_image_tokens(
    model_name: str,
    image: Image.Image,
    min_pixels: int,
    max_pixels: int,
) -> int:
    """
    Compute the expected number of vision tokens for a single image using
    the model's processor / vision config.

    This mirrors the logic in Qwen3VLProcessingInfo._get_vision_info():
      1. smart_resize the image to (resized_h, resized_w) respecting
         factor = patch_size * merge_size and the min/max pixel budget.
      2. grid_h = resized_h / patch_size, grid_w = resized_w / patch_size
      3. num_patches = grid_h * grid_w  (grid_t = 1 for images)
      4. num_vision_tokens = num_patches / merge_size^2
    """
    from transformers import AutoConfig
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        smart_resize as image_smart_resize,
    )

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    vision_config = config.vision_config
    patch_size = vision_config.patch_size
    merge_size = vision_config.spatial_merge_size

    factor = patch_size * merge_size
    width, height = image.size  # PIL: (w, h)

    resized_height, resized_width = image_smart_resize(
        height=height,
        width=width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size
    num_patches = grid_h * grid_w  # grid_t = 1 for a single image
    num_vision_tokens = num_patches // (merge_size ** 2)
    return num_vision_tokens


# ---------------------------------------------------------------------------
# Single request coroutine
# ---------------------------------------------------------------------------


async def stream_request(
    engine: AsyncLLM,
    request_id: str,
    image: Image.Image,
    prompt: str,
    sampling_params: SamplingParams,
) -> tuple[str, torch.Tensor | None, str]:
    """Send one streaming request; return (request_id, mm_embedding, text)."""
    mm_embedding: torch.Tensor | None = None
    generated_text = ""

    async for output in engine.generate(
        prompt={
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        },
        sampling_params=sampling_params,
        request_id=request_id,
    ):
        generated_text = output.outputs[0].text
        # mm_embedding is attached to the *final* (finished) output
        if output.outputs[0].mm_embedding is not None:
            mm_embedding = output.outputs[0].mm_embedding

    return request_id, mm_embedding, generated_text


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_embeddings(
    results: list[tuple[str, torch.Tensor | None, str]],
    image_assignments: dict[str, int],
    expected_token_counts: dict[int, int] | None = None,
) -> bool:
    """Run all validation checks.  Returns True iff everything passes."""
    passed = True

    # Group results by image index
    by_image: dict[int, list[tuple[str, torch.Tensor]]] = defaultdict(list)
    none_ids: list[str] = []

    for req_id, emb, _text in results:
        img_idx = image_assignments[req_id]
        if emb is None:
            none_ids.append(req_id)
        else:
            by_image[img_idx].append((req_id, emb))

    if none_ids:
        print(
            f"\n  [FAIL] {len(none_ids)}/{len(results)} requests "
            f"returned mm_embedding=None: {none_ids}"
        )
        passed = False

    if not by_image:
        print("\n  [FAIL] No embeddings collected at all!")
        return False

    # ---- Check 1: basic tensor properties ----
    print("\n  --- Check 1: tensor properties ---")
    for img_idx, entries in sorted(by_image.items()):
        for req_id, emb in entries:
            ok = True
            if not isinstance(emb, torch.Tensor):
                print(f"  [FAIL] {req_id}: not a torch.Tensor")
                ok = False
            elif emb.device != torch.device("cpu"):
                print(f"  [FAIL] {req_id}: device={emb.device}, expected cpu")
                ok = False
            elif emb.ndim != 2:
                print(f"  [FAIL] {req_id}: ndim={emb.ndim}, expected 2")
                ok = False
            elif emb.numel() == 0:
                print(f"  [FAIL] {req_id}: empty tensor")
                ok = False
            elif torch.all(emb == 0):
                print(f"  [FAIL] {req_id}: all zeros")
                ok = False
            if not ok:
                passed = False
    if passed:
        sample = next(iter(by_image.values()))[0][1]
        print(
            f"  All embeddings OK: shape={list(sample.shape)}, "
            f"dtype={sample.dtype}, device=cpu"
        )

    # ---- Check 2: same image → same embedding ----
    print("\n  --- Check 2: same image → same embedding ---")
    for img_idx in sorted(by_image):
        entries = by_image[img_idx]
        if len(entries) < 2:
            print(f"  Image {img_idx}: only 1 request, nothing to compare")
            continue
        ref_id, ref_emb = entries[0]
        mismatches = 0
        for req_id, emb in entries[1:]:
            if ref_emb.shape != emb.shape:
                print(
                    f"  [FAIL] Image {img_idx}: shape mismatch "
                    f"{req_id}{list(emb.shape)} vs "
                    f"{ref_id}{list(ref_emb.shape)}"
                )
                mismatches += 1
                continue
            if not torch.allclose(ref_emb, emb, atol=1e-3, rtol=1e-3):
                max_diff = (ref_emb - emb).abs().max().item()
                print(
                    f"  [FAIL] Image {img_idx}: {req_id} differs from "
                    f"{ref_id}, max_diff={max_diff:.6f}"
                )
                mismatches += 1
        if mismatches:
            passed = False
        else:
            print(f"  Image {img_idx}: {len(entries)} requests all match")

    # ---- Check 3: different images → different embeddings ----
    print("\n  --- Check 3: different images → different embeddings ---")
    indices = sorted(by_image)
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            a_idx, b_idx = indices[i], indices[j]
            a_emb = by_image[a_idx][0][1].flatten().unsqueeze(0).float()
            b_emb = by_image[b_idx][0][1].flatten().unsqueeze(0).float()
            cos = torch.nn.functional.cosine_similarity(a_emb, b_emb).item()
            if cos > 0.999:
                print(
                    f"  [FAIL] Image {a_idx} vs {b_idx}: "
                    f"cos_sim={cos:.6f} (too similar)"
                )
                passed = False
            else:
                print(f"  Image {a_idx} vs {b_idx}: cos_sim={cos:.4f}")

    # ---- Check 4: shape consistency across all requests ----
    print("\n  --- Check 4: shape consistency ---")
    shapes = {emb.shape for entries in by_image.values() for _, emb in entries}
    if len(shapes) == 1:
        print(f"  All embeddings share shape {list(shapes.pop())}")
    else:
        print(f"  [FAIL] Multiple shapes found: {shapes}")
        passed = False

    # ---- Check 5: shape[0] == expected image token count from processor ----
    if expected_token_counts:
        print("\n  --- Check 5: shape[0] matches processor image token count ---")
        for img_idx in sorted(by_image):
            expected = expected_token_counts.get(img_idx)
            if expected is None:
                continue
            entries = by_image[img_idx]
            for req_id, emb in entries:
                actual = emb.shape[0]
                if actual != expected:
                    print(
                        f"  [FAIL] Image {img_idx} ({req_id}): "
                        f"shape[0]={actual}, expected {expected} "
                        f"(from processor)"
                    )
                    passed = False
            # Only print once per image if all match
            if all(emb.shape[0] == expected for _, emb in entries):
                print(
                    f"  Image {img_idx}: shape[0]={expected} == "
                    f"expected_image_tokens ({len(entries)} requests)"
                )
    else:
        print("\n  --- Check 5: skipped (no expected token counts) ---")

    # ---- Check 6: embedding statistics sanity ----
    print("\n  --- Check 6: embedding statistics sanity ---")
    for img_idx in sorted(by_image):
        emb = by_image[img_idx][0][1].float()
        mean_val = emb.mean().item()
        std_val = emb.std().item()
        min_val = emb.min().item()
        max_val = emb.max().item()
        # Check for NaN / Inf
        if torch.isnan(emb).any():
            print(f"  [FAIL] Image {img_idx}: embedding contains NaN")
            passed = False
        elif torch.isinf(emb).any():
            print(f"  [FAIL] Image {img_idx}: embedding contains Inf")
            passed = False
        elif std_val < 1e-6:
            print(
                f"  [FAIL] Image {img_idx}: std={std_val:.6e} "
                f"(nearly constant)"
            )
            passed = False
        else:
            print(
                f"  Image {img_idx}: mean={mean_val:.4f}, "
                f"std={std_val:.4f}, min={min_val:.4f}, max={max_val:.4f}"
            )

    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(args: argparse.Namespace) -> None:
    print("=" * 72)
    print("  MM Embedding Return — Concurrent Streaming Validation")
    print(f"  Anchor model: {args.model}")
    print("=" * 72)

    min_pixels = args.min_pixels
    max_pixels = args.max_pixels

    # ---- 1. Engine init ----
    print(f"\n[1/5] Initializing AsyncLLM")
    print(f"  model               = {args.model}")
    print(f"  num_images          = {args.num_images}")
    print(f"  num_requests        = {args.num_requests}")
    print(f"  min_pixels          = {min_pixels}")
    print(f"  max_pixels          = {max_pixels}")
    print(f"  chunked_prefill     = True")
    print(f"  prefix_caching      = True")
    print(f"  return_mm_embedding = True")

    engine_args = AsyncEngineArgs(
        model=args.model,
        enable_return_mm_embedding=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        max_num_seqs=args.max_num_seqs,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        },
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    # ---- 2. Build requests ----
    print(
        f"\n[2/5] Preparing {args.num_requests} requests "
        f"with {args.num_images} distinct images"
    )
    test_images = create_test_images(args.num_images)
    rng = random.Random(args.seed)

    image_assignments: dict[str, int] = {}
    request_cfgs: list[tuple[str, Image.Image, str]] = []
    for i in range(args.num_requests):
        req_id = f"req-{i:04d}"
        img_idx = rng.randint(0, args.num_images - 1)
        question = rng.choice(QUESTIONS)
        prompt = PROMPT_TEMPLATE.format(question=question)
        image_assignments[req_id] = img_idx
        request_cfgs.append((req_id, test_images[img_idx], prompt))

    dist = Counter(image_assignments.values())
    print(f"  distribution: {dict(sorted(dist.items()))}")

    # ---- 3. Compute expected image token counts via processor ----
    print(f"\n[3/5] Computing expected image token counts via processor …")
    expected_token_counts: dict[int, int] = {}
    for img_idx, img in enumerate(test_images):
        expected = compute_expected_image_tokens(
            model_name=args.model,
            image=img,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        expected_token_counts[img_idx] = expected
        print(
            f"  Image {img_idx}: size={img.size}, "
            f"expected_vision_tokens={expected}"
        )

    # ---- 4. Fire concurrent streaming requests ----
    print(
        f"\n[4/5] Sending {args.num_requests} concurrent streaming requests …"
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
    )

    t0 = time.perf_counter()
    tasks = [
        stream_request(engine, rid, img, prompt, sampling_params)
        for rid, img, prompt in request_cfgs
    ]
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0
    print(
        f"  done in {elapsed:.2f}s ({args.num_requests / elapsed:.1f} req/s)"
    )

    # ---- 5. Validate ----
    print(f"\n[5/5] Validating mm_embeddings …")
    ok = validate_embeddings(results, image_assignments, expected_token_counts)

    print("\n" + "=" * 72)
    if ok:
        print("  RESULT: ALL CHECKS PASSED")
    else:
        print("  RESULT: SOME CHECKS FAILED")
    print("=" * 72)

    engine.shutdown()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate mm_embedding return with async streaming"
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--num-images", type=int, default=4, help="Number of distinct images"
    )
    p.add_argument(
        "--num-requests", type=int, default=40, help="Total concurrent requests"
    )
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument(
        "--min-pixels", type=int, default=28 * 28,
        help="Minimum pixel budget for image resizing",
    )
    p.add_argument(
        "--max-pixels", type=int, default=1280 * 28 * 28,
        help="Maximum pixel budget for image resizing",
    )
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
