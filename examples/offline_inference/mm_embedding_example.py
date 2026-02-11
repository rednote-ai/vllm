"""
Example script demonstrating how to return multimodal embeddings from vLLM.

This example shows how to:
1. Enable multimodal embedding return
2. Process multimodal inputs
3. Access the returned embeddings

Requirements:
- A multimodal model (e.g., LLaVA)
- PIL/Pillow for image processing
- A GPU with sufficient memory
"""

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset


def main():
    # Initialize LLM with enable_return_mm_embedding=True
    print("Initializing LLM with multimodal embedding return enabled...")
    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        enable_return_mm_embedding=True,
        max_model_len=2048,
        limit_mm_per_prompt={"image": 1},
    )
    
    # Prepare a multimodal prompt with an image
    print("\nPreparing multimodal input...")
    image = ImageAsset("cherry_blossom").pil_image
    prompt = "<image>\nWhat is shown in this image? Describe it in detail."
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=100,
    )
    
    # Generate with multimodal input
    print("Generating response...")
    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        },
        sampling_params=sampling_params,
    )
    
    # Process the outputs
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    for output in outputs:
        print(f"\nRequest ID: {output.request_id}")
        print(f"Prompt: {output.prompt}")
        
        for i, completion in enumerate(output.outputs):
            print(f"\n--- Completion {i} ---")
            print(f"Generated text: {completion.text}")
            
            # Access the multimodal embedding
            if completion.mm_embedding is not None:
                print(f"\n✓ Multimodal Embedding Retrieved:")
                print(f"  - Shape: {completion.mm_embedding.shape}")
                print(f"  - Device: {completion.mm_embedding.device}")
                print(f"  - Dtype: {completion.mm_embedding.dtype}")
                print(f"  - Min value: {completion.mm_embedding.min().item():.4f}")
                print(f"  - Max value: {completion.mm_embedding.max().item():.4f}")
                print(f"  - Mean value: {completion.mm_embedding.mean().item():.4f}")
                
                # Example: Save the embedding to a file
                # import torch
                # torch.save(completion.mm_embedding, "mm_embedding.pt")
                # print("  - Saved to: mm_embedding.pt")
            else:
                print("\n✗ No multimodal embedding (text-only input)")
    
    print("\n" + "=" * 80)
    
    # Example 2: Text-only input (should not have mm_embedding)
    print("\n\nExample 2: Text-only input")
    print("-" * 80)
    
    text_only_prompt = "What is the capital of France?"
    outputs = llm.generate(text_only_prompt, sampling_params=sampling_params)
    
    for output in outputs:
        for completion in output.outputs:
            print(f"Generated text: {completion.text}")
            print(f"MM Embedding: {completion.mm_embedding}")
            assert completion.mm_embedding is None, "Text-only input should not have mm_embedding"
    
    print("\n✓ All examples completed successfully!")


if __name__ == "__main__":
    main()
