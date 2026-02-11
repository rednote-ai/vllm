# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test for returning multimodal embeddings in LLM outputs.

This test verifies that when enable_return_mm_embedding=True, the multimodal
embeddings are captured and returned in the CompletionOutput.
"""

import pytest

# Skip if multimodal models are not available
pytest.importorskip("PIL")

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset


@pytest.mark.skip(reason="Requires a multimodal model and GPU")
def test_mm_embedding_return_with_llava():
    """
    Test that multimodal embeddings are returned when enabled.
    
    This test uses LLaVA as an example multimodal model.
    """
    # Initialize LLM with enable_return_mm_embedding=True
    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        enable_return_mm_embedding=True,
        max_model_len=2048,
        limit_mm_per_prompt={"image": 1},
    )
    
    # Prepare a multimodal prompt
    image = ImageAsset("cherry_blossom").pil_image
    prompt = "<image>\nWhat is shown in this image?"
    
    # Generate with multimodal input
    sampling_params = SamplingParams(temperature=0.0, max_tokens=50)
    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        },
        sampling_params=sampling_params,
    )
    
    # Verify that mm_embedding is present in the output
    assert len(outputs) == 1
    output = outputs[0]
    assert len(output.outputs) == 1
    completion_output = output.outputs[0]
    
    # Check that mm_embedding is not None
    assert completion_output.mm_embedding is not None
    
    # Verify it's a torch tensor on CPU
    import torch
    assert isinstance(completion_output.mm_embedding, torch.Tensor)
    assert completion_output.mm_embedding.device == torch.device("cpu")
    
    # Verify the shape is reasonable (should have hidden_size dimension)
    assert len(completion_output.mm_embedding.shape) == 2
    assert completion_output.mm_embedding.shape[1] > 0  # hidden_size > 0
    
    print(f"✓ MM embedding shape: {completion_output.mm_embedding.shape}")
    print(f"✓ MM embedding device: {completion_output.mm_embedding.device}")


@pytest.mark.skip(reason="Requires a multimodal model and GPU")
def test_mm_embedding_not_returned_when_disabled():
    """
    Test that multimodal embeddings are NOT returned when disabled (default).
    """
    # Initialize LLM without enable_return_mm_embedding
    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        max_model_len=2048,
        limit_mm_per_prompt={"image": 1},
    )
    
    # Prepare a multimodal prompt
    image = ImageAsset("cherry_blossom").pil_image
    prompt = "<image>\nWhat is shown in this image?"
    
    # Generate with multimodal input
    sampling_params = SamplingParams(temperature=0.0, max_tokens=50)
    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        },
        sampling_params=sampling_params,
    )
    
    # Verify that mm_embedding is None
    assert len(outputs) == 1
    output = outputs[0]
    assert len(output.outputs) == 1
    completion_output = output.outputs[0]
    
    # Check that mm_embedding is None when not enabled
    assert completion_output.mm_embedding is None
    
    print("✓ MM embedding is None when disabled")


@pytest.mark.skip(reason="Requires a multimodal model and GPU")
def test_mm_embedding_with_text_only_input():
    """
    Test that mm_embedding is None for text-only inputs even when enabled.
    """
    # Initialize LLM with enable_return_mm_embedding=True
    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        enable_return_mm_embedding=True,
        max_model_len=2048,
    )
    
    # Text-only prompt (no image)
    prompt = "What is the capital of France?"
    
    # Generate with text-only input
    sampling_params = SamplingParams(temperature=0.0, max_tokens=50)
    outputs = llm.generate(prompt, sampling_params=sampling_params)
    
    # Verify that mm_embedding is None for text-only input
    assert len(outputs) == 1
    output = outputs[0]
    assert len(output.outputs) == 1
    completion_output = output.outputs[0]
    
    # Should be None because there's no multimodal input
    assert completion_output.mm_embedding is None
    
    print("✓ MM embedding is None for text-only input")


if __name__ == "__main__":
    # These tests require a GPU and multimodal model
    # Run with: pytest tests/entrypoints/llm/test_mm_embedding_return.py -v -s
    print("Note: These tests are skipped by default.")
    print("To run them, remove the @pytest.mark.skip decorators and ensure you have:")
    print("  1. A GPU available")
    print("  2. The llava-hf/llava-1.5-7b-hf model downloaded")
    print("  3. PIL/Pillow installed")
