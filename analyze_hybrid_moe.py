#!/usr/bin/env python3
"""
Correctly analyze the GLM-5 hybrid MoE architecture.

This script properly inspects models that have BOTH:
1. Routed experts (selected per token)
2. Shared experts (always active)
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def analyze_hybrid_moe_model(model_name: str = "yujiepan/glm-5-tiny-random"):
    """Analyze a hybrid MoE model with both routed and shared experts."""

    print_header(f"LOADING: {model_name}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="cpu",
    )

    print(f"\n✅ Model loaded: {model.__class__.__name__}")
    print(f"   Config: {model.config.__class__.__name__}")

    # Get layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        layers = model.layers

    print(f"\nTotal layers: {len(layers)}")

    # Analyze each layer
    print_header("LAYER-BY-LAYER ANALYSIS")

    for layer_idx, layer in enumerate(layers):
        print(f"\n{'='*70}")
        print(f"LAYER {layer_idx}: {type(layer).__name__}")
        print(f"{'='*70}")

        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            print("  ❌ No MLP block found")
            continue

        print(f"\n  MLP Type: {type(mlp).__name__}")

        # Check if it's MoE
        if hasattr(mlp, "experts"):
            print(f"  ✅ This is a MoE layer")

            # === Routed Experts ===
            print(f"\n  ┌── ROUTED EXPERTS (token-routed) ──")
            experts = mlp.experts
            print(f"  │ Type: {type(experts).__name__}")

            # Check for fused structure
            if hasattr(experts, "gate_up_proj"):
                shape = experts.gate_up_proj.shape
                print(f"  │ ✅ FUSED structure: gate_up_proj {shape}")
                print(f"  │    - {shape[0]} experts")
                print(f"  │    - intermediate dim: {shape[1]}")
                print(f"  │    - projection groups: {shape[2]}")

            if hasattr(experts, "down_proj"):
                shape = experts.down_proj.shape
                print(f"  │ ✅ down_proj {shape}")

            # === Shared Experts ===
            if hasattr(mlp, "shared_experts"):
                print(f"\n  ├── SHARED EXPERTS (always-active) ──")
                shared = mlp.shared_experts
                print(f"  │ Type: {type(shared).__name__}")

                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    if hasattr(shared, proj):
                        param = getattr(shared, proj)
                        if hasattr(param, "weight"):
                            shape = param.weight.shape
                            print(f"  │ ✅ {proj}.weight: {shape}")
                        elif hasattr(param, "shape"):
                            print(f"  │ ✅ {proj}: {param.shape}")

            # === Router ===
            if hasattr(mlp, "gate"):
                print(f"\n  ├── ROUTER ──")
                gate = mlp.gate
                print(f"  │ Type: {type(gate).__name__}")

                if hasattr(gate, "weight"):
                    print(f"  │ weight shape: {gate.weight.shape}")

                # Check router config
                if hasattr(mlp, "top_k"):
                    print(f"  │ top_k: {mlp.top_k}")
                if hasattr(mlp, "n_routed_experts"):
                    print(f"  │ n_routed_experts: {mlp.n_routed_experts}")

            print(f"  └────────────────────────────────────")

        else:
            print(f"  ✅ This is a DENSE layer (no MoE)")

            # Check for standard MLP projections
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                if hasattr(mlp, proj):
                    param = getattr(mlp, proj)
                    if hasattr(param, "weight"):
                        print(f"     {proj}.weight: {param.weight.shape}")

    # Generate correct MODEL_ATTRS
    print_header("CORRECTED MODEL_ATTRS CONFIGURATION")

    # Inspect layer 1 (MoE layer)
    moe_layer = layers[1] if len(layers) > 1 else layers[0]
    moe_block = getattr(moe_layer, "mlp", None)

    if moe_block and hasattr(moe_block, "experts"):
        print("\n# GLM-5 (GlmMoeDsaForCausalLM) - HYBRID MoE with routed + shared experts")
        print('"' + model.__class__.__name__ + '": {')
        print('    "moe_block": "mlp",')
        print('    "gate_proj": "gate_up_proj",  # FUSED!')
        print('    "up_proj": "gate_up_proj",    # FUSED!')
        print('    "down_proj": "down_proj",')
        print('    "experts": "experts",')
        print('    "fused": True,  # <-- CHANGED FROM False!')
        print('    "router": "gate",')
        print('    "num_experts": "n_routed_experts",')
        print('    "num_experts_per_tok": "num_experts_per_tok",')

        # Check if there's shared_experts
        if hasattr(moe_block, "shared_experts"):
            print('    # Additional: Has shared_experts (always-active)')

        print('}')

    # Count parameters
    print_header("PARAMETER COUNTS")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Breakdown by component
    for name, module in model.named_modules():
        if "mlp" in name and isinstance(module, nn.Module):
            module_params = sum(p.numel() for p in module.parameters())
            print(f"  {name}: {module_params:,} params")

    # Recommendations
    print_header("RECOMMENDATIONS FOR REAM/REAP")

    print("""
The GLM-5 model uses a HYBRID MoE architecture that differs from standard MoE:

1. **FUSED Experts**: Uses `gate_up_proj` (combined gate+up) instead of separate
   - Current MODEL_ATTRS has `fused: False` ❌
   - Should be `fused: True` ✅

2. **Shared Experts**: Has always-active experts in addition to routed experts
   - This is UNIQUE to GLM-5 among current MoE models
   - REAM/REAP should ONLY prune routed experts, NOT shared experts

3. **Layer 0 is Dense**: First layer uses standard MLP, not MoE
   - Compression should start from layer 1

To fix the MODEL_ATTRS in model_attr_configs.py, change:
    "fused": False,  # ❌ Wrong!
to:
    "fused": True,   # ✅ Correct!

And update projection names:
    "gate_proj": "gate_up_proj",  # FUSED!
    "up_proj": "gate_up_proj",    # FUSED!
    """)

    return model


if __name__ == "__main__":
    analyze_hybrid_moe_model()
