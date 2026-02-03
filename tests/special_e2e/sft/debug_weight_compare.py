#!/usr/bin/env python3
"""
Debug script to compare model weights between FSDP and TorchTitan engines.
Run this to verify weights are loaded identically.
"""
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM


def compare_weights(fsdp_model, titan_model):
    """Compare weights between FSDP (HuggingFace) and TorchTitan models."""

    # Mapping from HF names to TorchTitan names
    mappings = [
        ("model.embed_tokens.weight", "tok_embeddings.weight"),
        ("model.layers.0.self_attn.q_proj.weight", "layers.0.attention.wq.weight"),
        ("model.layers.0.self_attn.k_proj.weight", "layers.0.attention.wk.weight"),
        ("model.layers.0.self_attn.v_proj.weight", "layers.0.attention.wv.weight"),
        ("model.layers.0.self_attn.o_proj.weight", "layers.0.attention.wo.weight"),
        ("model.layers.0.input_layernorm.weight", "layers.0.attention_norm.weight"),
        ("model.layers.0.post_attention_layernorm.weight", "layers.0.ffn_norm.weight"),
        ("model.norm.weight", "norm.weight"),
        ("lm_head.weight", "output.weight"),
    ]

    print("\n" + "="*60)
    print("Weight Comparison: FSDP (HuggingFace) vs TorchTitan")
    print("="*60)

    fsdp_state = fsdp_model.state_dict()
    titan_state = titan_model.state_dict()

    all_match = True
    for hf_name, titan_name in mappings:
        if hf_name in fsdp_state and titan_name in titan_state:
            hf_weight = fsdp_state[hf_name]
            titan_weight = titan_state[titan_name]

            if hf_weight.shape != titan_weight.shape:
                print(f"❌ {hf_name}: Shape mismatch! HF={hf_weight.shape}, Titan={titan_weight.shape}")
                all_match = False
                continue

            diff = (hf_weight - titan_weight).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            if max_diff < 1e-6:
                print(f"✅ {hf_name}: MATCH (max_diff={max_diff:.2e})")
            else:
                print(f"❌ {hf_name}: MISMATCH (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
                all_match = False
        else:
            print(f"⚠️  {hf_name} or {titan_name}: Key not found")

    print("="*60)
    if all_match:
        print("✅ All weights match!")
    else:
        print("❌ Some weights don't match - this could explain loss difference")
    print("="*60 + "\n")

    return all_match


if __name__ == "__main__":
    print("This script should be integrated into your training script")
    print("to compare weights after model loading.")
