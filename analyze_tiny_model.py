#!/usr/bin/env python3
"""
Analyze the tiny GLM-5 random model structure.

This script loads the yujiepan/glm-5-tiny-random model and inspects its
architecture to understand how it differs from the main GLM-5 model.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List
from functools import reduce

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def get_attr_recursive(obj: nn.Module, path: str) -> Any:
    """Get an attribute using dot notation path."""
    try:
        return reduce(getattr, path.split("."), obj)
    except AttributeError:
        return None


def analyze_module(module: nn.Module, indent: int = 0, max_depth: int = 4) -> Dict[str, Any]:
    """Recursively analyze a module's structure."""
    prefix = "  " * indent
    info = {"type": type(module).__name__, "children": {}, "params": 0}

    # Count parameters
    try:
        info["params"] = sum(p.numel() for p in module.parameters())
    except Exception:
        pass

    # Recursively analyze children if within depth limit
    if indent < max_depth:
        for name, child in list(module.named_children())[:20]:  # Limit children
            info["children"][name] = analyze_module(child, indent + 1, max_depth)

    return info


def print_module_info(info: Dict[str, Any], indent: int = 0):
    """Print module analysis info."""
    prefix = "  " * indent
    print(f"{prefix}{info['type']}")
    if info["params"] > 0:
        print(f"{prefix}  (params: {info['params']:,})")
    for name, child in list(info["children"].items())[:10]:
        print(f"{prefix}{name}: ", end="")
        print_module_info(child, 0)


def analyze_model_structure(model: nn.Module):
    """Analyze the complete model structure."""
    print_section("MODEL STRUCTURE ANALYSIS")

    # Model class and config
    print(f"\nModel Class: {model.__class__.__name__}")
    print(f"Config Class: {model.config.__class__.__name__}")

    # Get layers
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers

    if layers is None:
        print("\n❌ Could not find layers in model!")
        return

    print(f"\nNumber of layers: {len(layers)}")

    # Analyze each layer
    for layer_idx, layer in enumerate(layers):
        print_subsection(f"Layer {layer_idx}")

        print(f"\nLayer type: {type(layer).__name__}")

        # Print all direct attributes
        print("\nDirect attributes:")
        for attr in sorted(dir(layer)):
            if not attr.startswith("_"):
                obj = getattr(layer, attr)
                if not callable(obj):
                    obj_type = type(obj).__name__
                    if isinstance(obj, nn.Module):
                        params = sum(p.numel() for p in obj.parameters())
                        print(f"  {attr}: {obj_type} ({params:,} params)")
                    else:
                        print(f"  {attr}: {obj_type}")

        # Analyze MLP/MoE block
        print_subsection(f"Layer {layer_idx} - MLP/MoE Block Analysis")

        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            print(f"\n✅ Found 'mlp' attribute")
            print(f"   Type: {type(mlp).__name__}")

            # Print MLP attributes
            print("\n  MLP attributes:")
            for attr in sorted(dir(mlp)):
                if not attr.startswith("_"):
                    obj = getattr(mlp, attr)
                    if not callable(obj):
                        obj_type = type(obj).__name__
                        if isinstance(obj, nn.Module):
                            params = sum(p.numel() for p in obj.parameters())
                            print(f"    {attr}: {obj_type} ({params:,} params)")
                        else:
                            print(f"    {attr}: {obj_type}")

            # Check for experts
            experts = getattr(mlp, "experts", None)
            if experts is not None:
                print(f"\n  ✅ Found 'experts' in MLP")
                print(f"     Type: {type(experts).__name__}")

                if isinstance(experts, nn.ModuleList):
                    print(f"     Number of experts: {len(experts)}")
                    if len(experts) > 0:
                        print(f"     First expert type: {type(experts[0]).__name__}")
                        # Print first expert structure
                        print("\n     First expert structure:")
                        for attr in sorted(dir(experts[0])):
                            if not attr.startswith("_"):
                                obj = getattr(experts[0], attr)
                                if not callable(obj) and isinstance(obj, nn.Module):
                                    params = sum(p.numel() for p in obj.parameters())
                                    print(f"        {attr}: {type(obj).__name__} ({params:,} params)")
                elif isinstance(experts, nn.Module):
                    print(f"     Experts is a Module (fused experts?)")
                    for attr in ["gate_up_proj", "gate_proj", "up_proj", "down_proj"]:
                        if hasattr(experts, attr):
                            obj = getattr(experts, attr)
                            if isinstance(obj, nn.Parameter):
                                print(f"        {attr}: Parameter with shape {obj.shape}")
            else:
                print(f"\n  ❌ No 'experts' attribute in MLP")

            # Check for gate/router
            gate = getattr(mlp, "gate", None)
            if gate is not None:
                print(f"\n  ✅ Found 'gate' in MLP")
                print(f"     Type: {type(gate).__name__}")
                if isinstance(gate, nn.Linear):
                    print(f"     in_features: {gate.in_features}, out_features: {gate.out_features}")
            else:
                print(f"\n  ❌ No 'gate' attribute in MLP")

            # Check for indexer (specific to some layers)
            indexer = getattr(mlp, "indexer", None)
            if indexer is not None:
                print(f"\n  ✅ Found 'indexer' in MLP")
                print(f"     Type: {type(indexer).__name__}")
                if hasattr(indexer, "__len__"):
                    try:
                        print(f"     Length: {len(indexer)}")
                    except:
                        pass

        else:
            print(f"\n❌ No 'mlp' attribute in layer")

        # Only analyze first 3 layers in detail
        if layer_idx >= 2:
            print("\n... (remaining layers have similar structure)")


def analyze_config_differences(config):
    """Analyze the model config to understand expected structure."""
    print_section("CONFIG ANALYSIS")

    print("\nConfig attributes related to MoE:")
    moe_attrs = [
        "num_experts", "n_routed_experts", "moe_num_experts", "num_local_experts",
        "num_experts_per_tok", "top_k", "moe_k", "num_selected_experts",
        "num_key_value_heads", "num_attention_heads",
    ]

    for attr in sorted(dir(config)):
        if not attr.startswith("_") and any(moe_attr in attr for moe_attr in moe_attrs):
            value = getattr(config, attr)
            print(f"  {attr}: {value}")


def compare_with_expected_structure(model: nn.Module):
    """Compare the model structure with expected MODEL_ATTRS configuration."""
    print_section("COMPARISON WITH EXPECTED GLM-5 STRUCTURE")

    # Expected MODEL_ATTRS for GlmMoeDsaForCausalLM
    expected = {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    }

    print("\nExpected MODEL_ATTRS configuration:")
    for key, value in expected.items():
        print(f"  {key}: {value}")

    # Get layers
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers

    if layers is None or len(layers) == 0:
        print("\n❌ No layers found!")
        return

    # Check first MoE layer
    print_subsection("Actual Structure Verification (Layer 0)")

    layer = layers[0]
    moe_block = getattr(layer, "mlp", None)

    if moe_block is None:
        print("\n❌ No 'mlp' block found!")
        return

    print("\n✅ Found 'mlp' block")

    # Check experts
    print("\nChecking 'experts' attribute:")
    experts = getattr(moe_block, "experts", None)
    if experts is not None:
        print(f"  ✅ Found 'experts': {type(experts).__name__}")
    else:
        print(f"  ❌ Missing 'experts' attribute")
        # Check for indexer instead
        indexer = getattr(moe_block, "indexer", None)
        if indexer is not None:
            print(f"  ℹ️  Found 'indexer' instead: {type(indexer).__name__}")

    # Check router
    print("\nChecking 'gate' (router) attribute:")
    gate = getattr(moe_block, "gate", None)
    if gate is not None:
        print(f"  ✅ Found 'gate': {type(gate).__name__}")
    else:
        print(f"  ❌ Missing 'gate' attribute")

    # Check projections
    for proj_name in ["gate_proj", "up_proj", "down_proj"]:
        print(f"\nChecking '{proj_name}':")
        found = False
        if experts is not None and hasattr(experts, "__iter__"):
            try:
                for expert in experts:
                    if hasattr(expert, proj_name):
                        print(f"  ✅ Found in experts: {type(getattr(expert, proj_name)).__name__}")
                        found = True
                        break
            except:
                pass
        if not found:
            print(f"  ❌ Not found in experts")


def print_parameter_shapes(model: nn.Module):
    """Print all parameter names and shapes."""
    print_section("PARAMETER SHAPES")

    print("\nAll model parameters:")
    for name, param in list(model.named_parameters())[:100]:  # Limit output
        print(f"  {name}: {param.shape}")

    if len(list(model.named_parameters())) > 100:
        print(f"\n... and {len(list(model.named_parameters())) - 100} more parameters")


def main():
    """Main entry point."""
    model_name = "yujiepan/glm-5-tiny-random"

    print_section(f"LOADING MODEL: {model_name}")

    # Load config first
    print("\nLoading config...")
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        print(f"✅ Config loaded: {config.__class__.__name__}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return

    # Load model
    print("\nLoading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cpu",  # Load on CPU to avoid GPU issues
        )
        print(f"✅ Model loaded: {model.__class__.__name__}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run analyses
    analyze_config_differences(config)
    analyze_model_structure(model)
    compare_with_expected_structure(model)
    print_parameter_shapes(model)

    print_section("ANALYSIS COMPLETE")
    print("\nThis analysis shows how the tiny random model differs from")
    print("the expected GLM-5 structure, which explains the verification errors.")


if __name__ == "__main__":
    main()
