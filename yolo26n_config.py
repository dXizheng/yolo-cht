"""
YOLO26n Configuration

Configuration options for YOLO26n CHT + QAT model.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class ReplaceMode(Enum):
    """Layer replacement mode for CHT."""
    BACKBONE = "backbone"           # Only backbone layers
    BACKBONE_NECK = "backbone_neck"  # Backbone + neck layers
    ALL = "all"                      # All layers (backbone + neck + head)


@dataclass
class YOLO26nConfig:
    """Configuration for YOLO26n CHT + QAT model."""

    # Model paths
    model_path: str = "yolo26n.pt"

    # CHT (sparsity) configuration
    sparsity: float = 0.9                    # Target sparsity (0.9 = 90%)
    replace_mode: ReplaceMode = ReplaceMode.BACKBONE_NECK  # Which layers to replace
    skip_first_n_convs: int = 2              # Number of early convs to skip from CHT
    regrow_method: str = "L3n"               # Regrow method for CHT
    shared_mask_sw: bool = True              # Share mask across sliding windows
    soft: bool = True                         # Use soft removal/regrowth
    link_update_ratio: float = 0.1           # Fraction of active connections to update

    # QAT (quantization) configuration
    qat_enabled: bool = True                 # Enable quantization-aware training
    quantization_dtype: str = "int8"        # Quantization type: "int8", "fp8", or None

    # Training configuration
    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640
    device: str = "cuda"

    # Other options
    verbose: bool = True
    debug: bool = False


def create_cht_config(
    sparsity: float = 0.9,
    start_sparsity: float = 0.0,
    evolve_duration: int = 0,
    evolve_speed: float = 0.1,
    regrow_method: str = "L3n",
    shared_mask_sw: bool = True,
    soft: bool = True,
    link_update_ratio: float = 0.1,
    **kwargs
):
    """
    Create CHT configuration for YOLO26n model.

    Args:
        sparsity: Target sparsity (0.9 = 90%)
        start_sparsity: Starting sparsity for gradual pruning
        evolve_duration: Number of evolve() calls to reach target sparsity
        evolve_speed: Speed of sigmoid transition
        regrow_method: Method for regrowing connections ("L3n" for CHT)
        shared_mask_sw: Whether to share mask across sliding windows
        soft: Use soft removal/regrowth
        link_update_ratio: Fraction of active connections to update per evolve

    Returns:
        CHTConfig object
    """
    # Import from reference implementation
    from cht_layer_unshared import CHTConfig

    use_ss = evolve_duration > 0

    return CHTConfig(
        sparsity=sparsity,
        mlp_sparsity=kwargs.get('mlp_sparsity', 0.5),
        link_update_ratio=link_update_ratio,
        remove_method=kwargs.get('remove_method', 'wm'),
        regrow_method=regrow_method,
        shared_mask_sw=shared_mask_sw,
        shared_mask_zone=kwargs.get('shared_mask_zone', False),
        zone_sz=kwargs.get('zone_sz', 0),
        avg_remove=kwargs.get('avg_remove', True),
        avg_regrow=kwargs.get('avg_regrow', True),
        soft=soft,
        use_opt4=kwargs.get('use_opt4', True),
        delta=kwargs.get('delta', 0.3),
        delta_max=kwargs.get('delta_max', 0.5),
        delta_d=kwargs.get('delta_d', 0.01),
        delta_remove=kwargs.get('delta_remove', 0.5),
        ch_method=kwargs.get('ch_method', 'CH3'),
        use_hidden=kwargs.get('use_hidden', False),
        l3n_batch_sz=kwargs.get('l3n_batch_sz', 32),
        evolve_es=kwargs.get('evolve_es', False),
        use_manual=kwargs.get('use_manual', False),
        use_ss=use_ss,
        ss_sparsity_initial=start_sparsity,
        ss_k=evolve_speed,
        ss_duration=evolve_duration,
        dropout=kwargs.get('dropout', 0.0),
        sparsity_schedule=kwargs.get('sparsity_schedule', 'step'),
        sparsity_warmup_epochs=kwargs.get('sparsity_warmup_epochs', 5),
        sparsity_step_epochs=kwargs.get('sparsity_step_epochs', 5),
        sparsity_step_size=kwargs.get('sparsity_step_size', 0.1),
        debug=kwargs.get('debug', False)
    )


def create_qat_config(quantization_dtype: str = "int8"):
    """
    Create QAT configuration for YOLO26n model.

    Args:
        quantization_dtype: Quantization type ('int8', 'fp8', or None)

    Returns:
        torch.quantization.QConfig or None
    """
    import torch
    import torch.quantization as tq

    if quantization_dtype is None or quantization_dtype == 'none':
        return None

    if quantization_dtype == 'int8':
        return torch.quantization.QConfig(
            activation=tq.FakeQuantize.with_args(
                observer=tq.MovingAverageMinMaxObserver,
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_symmetric,
                reduce_range=False
            ),
            weight=tq.FakeQuantize.with_args(
                observer=tq.MovingAverageMinMaxObserver,
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric,
                reduce_range=False
            )
        )
    elif quantization_dtype == 'fp8':
        # FP8 simulation - quantize weights only, not activations
        # Note: PyTorch's native FP8 FakeQuantize has issues (returns zeros),
        # so we use INT8 weights with per-channel quantization as a fallback
        # that provides similar benefits (weight quantization without activation quantization)
        return torch.quantization.QConfig(
            activation=None,  # Disable activation quantization
            weight=tq.FakeQuantize.with_args(
                observer=tq.MovingAveragePerChannelMinMaxObserver,
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric
            )
        )
    else:
        return None
