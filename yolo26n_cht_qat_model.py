"""
YOLO26n CHT + QAT Model

Load pretrained YOLO26n weights and apply:
1. CHT layers (for sparsity)
2. QAT modules (for quantization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as tq
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import copy

# Import CHT layer from reference implementation
from cht_layer_unshared import Conv2d_CHT, CHTConfig
from yolo26n_config import ReplaceMode, create_cht_config, create_qat_config


class FP8FakeQuantize(nn.Module):
    """
    Custom FP8 fake quantization that actually works.
    Uses torch.float8_e4m3fn dtype for quantization.
    """

    def __init__(self, observer='minmax', per_channel=True):
        super().__init__()
        self.per_channel = per_channel
        self.fp8_dtype = torch.float8_e4m3fn

        # Register buffers for scale
        if per_channel:
            self.register_buffer('scale', torch.ones(1))
            self.register_buffer('num_channels', torch.tensor(1))
        else:
            self.register_buffer('scale', torch.ones(1))

    def forward(self, x):
        # Calculate scale based on input range
        if self.per_channel:
            # Per-channel: scale for each output channel
            if x.dim() == 4:
                # (N, C, H, W) -> (C,)
                scale = x.abs().flatten(2).max(dim=2)[0].clamp(min=1e-8)
                # Expand to match weight shape if needed
                if scale.numel() == 1:
                    scale = scale.expand(x.shape[1])
            else:
                scale = x.abs().max().clamp(min=1e-8)
        else:
            # Per-tensor
            scale = x.abs().max().clamp(min=1e-8)

        # Quantize to FP8
        x_scaled = x / scale
        x_fp8 = x_scaled.to(self.fp8_dtype)
        x_dequant = x_fp8.float()

        # Handle per-channel scale
        if self.per_channel and x.dim() == 4 and x_dequant.shape[1] != scale.numel():
            # Reshape for convolution: (N, C, H, W) -> (C, N*H*W)
            orig_shape = x_dequant.shape
            x_dequant = x_dequant.permute(1, 0, 2, 3).reshape(x_dequant.shape[1], -1)
            x_dequant = x_dequant * scale.view(-1, 1)
            x_dequant = x_dequant.reshape(x_dequant.shape[0], orig_shape[1], orig_shape[2], orig_shape[3])
            x_dequant = x_dequant.permute(1, 0, 2, 3)
        else:
            x_dequant = x_dequant * scale

        return x_dequant

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value


class FP8WeightFakeQuantize(nn.Module):
    """
    Custom FP8 weight fake quantization for Conv2d weights.
    Handles per-channel quantization correctly for weights.
    """

    def __init__(self, observer='minmax'):
        super().__init__()
        self.fp8_dtype = torch.float8_e4m3fn
        self.register_buffer('scale', torch.ones(1))

    def forward(self, weight):
        """
        Quantize weights to FP8.
        weight shape: (out_channels, in_channels, kH, kW)
        """
        # Per-channel: scale for each output channel
        # Flatten spatial dims: (out_channels, in_channels * kH * kW)
        weight_flat = weight.flatten(1)

        # Scale per output channel
        scale = weight_flat.abs().max(dim=1)[0].clamp(min=1e-8)

        # Quantize
        weight_scaled = weight_flat / scale.unsqueeze(1)
        weight_fp8 = weight_scaled.to(self.fp8_dtype)
        weight_dequant = weight_fp8.float()

        # Reshape back
        weight_dequant = weight_dequant.reshape(weight.shape)

        # Store scale for forward
        self.scale = scale

        return weight_dequant * scale.unsqueeze(1).unsqueeze(2).unsqueeze(3)

# Monkey-patch Conv2d_CHT to add out_channels property for Ultralytics compatibility
# Ultralytics expects this attribute when fusing Conv+BN layers
if not hasattr(Conv2d_CHT, 'out_channels'):
    @property
    def out_channels(self):
        return self.c_out

    @property
    def in_channels(self):
        return self.c_in

    Conv2d_CHT.out_channels = out_channels
    Conv2d_CHT.in_channels = in_channels


# Attention-like module families seen in this YOLO variant.
ATTENTION_TYPES = ['C2PSA', 'C2f', 'Attention', 'PSABlock', 'C3k2']

# BF16 should be reserved for the most numerically sensitive attention paths,
# not broad hybrid blocks such as C3k2/C2f that contain many quantizable convs.
BF16_WRAP_TYPES = ['C2PSA', 'Attention', 'PSABlock']


def _contains_attention(module: nn.Module) -> bool:
    """Check if module contains attention mechanism (recursively)."""
    for name, child in module.named_modules():
        for attn_type in ATTENTION_TYPES:
            if attn_type in type(child).__name__:
                return True
    return False


def _is_attention_module(module: nn.Module) -> bool:
    """Check if module itself is an attention type."""
    module_type = type(module).__name__
    for attn_type in ATTENTION_TYPES:
        if attn_type in module_type:
            return True
    return False


def _should_bf16_wrap(module: nn.Module) -> bool:
    """Return True when the module should execute under BF16 autocast."""
    module_type = type(module).__name__
    for attn_type in BF16_WRAP_TYPES:
        if attn_type in module_type:
            return True
    return False


class BF16AttentionWrapper(nn.Module):
    """Run protected attention modules under CUDA BF16 autocast."""

    def __init__(self, module: nn.Module):
        super().__init__()
        # Store wrapped module - use non-underscore prefix so PyTorch registers it as a child module
        # This ensures train()/eval() propagate correctly to the wrapped module
        self.wrapped_module = module

        # Copy essential attributes for Ultralytics compatibility
        module_type = type(module).__name__
        self.type = module_type
        self.f = getattr(module, 'f', -1)  # from layer index
        self.i = getattr(module, 'i', -1)  # layer index - MUST preserve for _predict_once to save outputs
        self.nf = getattr(module, 'nf', 1)  # number of outputs
        self.m = module  # the actual module (for some operations)
        self.save = getattr(module, 'save', [])  # save list for _predict_once

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        input_dtype = x.dtype
        target_device = x.device

        for tensor in list(self.wrapped_module.parameters()) + list(self.wrapped_module.buffers()):
            target_device = tensor.device
            break

        if x.device != target_device:
            x = x.to(device=target_device)

        if x.is_cuda:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = self.wrapped_module(x)
        else:
            output = self.wrapped_module(x)

        if output.dtype != input_dtype:
            output = output.to(input_dtype)
        return output

    def train(self, mode=True):
        """Override train() to propagate training mode to wrapped module."""
        super().train(mode)
        if hasattr(self, 'wrapped_module'):
            self.wrapped_module.train(mode)
        return self

    def eval(self):
        """Override eval() to propagate eval mode to wrapped module."""
        super().eval()
        if hasattr(self, 'wrapped_module'):
            self.wrapped_module.eval()
        return self

    @property
    def module(self):
        """Return wrapped module for compatibility."""
        return self.wrapped_module


class QATConv2d(nn.Module):
    """
    Quantization-aware Conv2d layer with INT8 or FP8 quantization (without CHT).
    Wraps a regular Conv2d with fake quantization for QAT.
    """

    def __init__(
        self,
        conv_layer: nn.Conv2d,
        qconfig: Optional[torch.quantization.QConfig] = None,
        quantization_dtype: str = "int8"
    ):
        super().__init__()

        self.conv_layer = conv_layer
        self.quantization_dtype = quantization_dtype
        self._use_qat = True

        # Check if using custom FP8
        use_custom_fp8 = quantization_dtype == "fp8"

        if use_custom_fp8:
            # Use custom FP8 quantization
            self.weight_fq = FP8WeightFakeQuantize()
            self.act_fq = None  # No activation quantization for FP8
        elif qconfig is None:
            # Default QAT config if not provided
            qconfig = torch.quantization.QConfig(
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
            self.qconfig = qconfig

            # Create fake quant modules
            if self.qconfig.weight is not None:
                self.weight_fq = self.qconfig.weight()
                self.weight_fq.enable_observer()
            if self.qconfig.activation is not None:
                self.act_fq = self.qconfig.activation()
                self.act_fq.enable_observer()
        else:
            self.qconfig = qconfig
            # Create fake quant modules
            if self.qconfig.weight is not None:
                self.weight_fq = self.qconfig.weight()
                self.weight_fq.enable_observer()
            if self.qconfig.activation is not None:
                self.act_fq = self.qconfig.activation()
                self.act_fq.enable_observer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get input device
        input_device = x.device

        # Quantize input (skip for FP8)
        if hasattr(self, 'act_fq') and self.act_fq is not None and self.training:
            x = self.act_fq(x)

        # Quantize weight - move to same device as input
        weight = self.conv_layer.weight.to(device=input_device)
        if hasattr(self, 'weight_fq') and self.weight_fq is not None and self.training:
            weight = self.weight_fq(weight)

        # Move bias to same device if it exists
        bias = self.conv_layer.bias.to(device=input_device) if self.conv_layer.bias is not None else None

        # Apply convolution
        out = self.conv_layer._conv_forward(x, weight, bias)

        # Quantize output (skip for FP8)
        if hasattr(self, 'act_fq') and self.act_fq is not None and self.training:
            out = self.act_fq(out)

        return out

    def to(self, *args, **kwargs):
        """Move module to device and ensure fake quantize buffers are also moved."""
        super().to(*args, **kwargs)
        # Move fake quantize modules to the same device as the module
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        if self.act_fq is not None:
            self.act_fq = self.act_fq.to(device)
        if self.weight_fq is not None:
            self.weight_fq = self.weight_fq.to(device)
        return self


class QATConv2d_CHT(nn.Module):
    """
    Quantization-aware CHT Conv2d layer with INT8 or FP8 quantization.
    """

    def __init__(
        self,
        cht_conv_layer: nn.Module,
        qconfig: Optional[torch.quantization.QConfig] = None,
        quantization_dtype: str = "int8"
    ):
        super().__init__()

        self.cht_layer = cht_conv_layer
        self.in_cht = cht_conv_layer.c_in
        self.out_cht = cht_conv_layer.c_out
        self.kernel_size = cht_conv_layer.kernel_size
        self.padding = cht_conv_layer.padding
        self.stride = cht_conv_layer.stride
        self.use_bias = cht_conv_layer.use_bias
        self.quantization_dtype = quantization_dtype

        # Ultralytics compatibility: expose weight and other Conv2d attributes
        self.in_channels = cht_conv_layer.c_in
        self.out_channels = cht_conv_layer.c_out

        # Check if using custom FP8
        use_custom_fp8 = quantization_dtype == "fp8"

        if use_custom_fp8:
            # Use custom FP8 quantization
            self.weight_fq = FP8WeightFakeQuantize()
            self.act_fq = None  # No activation quantization for FP8
        elif qconfig is None:
            # Default QAT config if not provided
            qconfig = torch.quantization.QConfig(
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
            self.qconfig = qconfig

            # Create fake quant modules
            if self.qconfig.weight is not None:
                self.weight_fq = self.qconfig.weight()
                self.weight_fq.enable_observer()
            else:
                self.weight_fq = None

            if self.qconfig.activation is not None:
                self.act_fq = self.qconfig.activation()
                self.act_fq.enable_observer()
            else:
                self.act_fq = None
        else:
            self.qconfig = qconfig

            # Create fake quant modules
            if self.qconfig.weight is not None:
                self.weight_fq = self.qconfig.weight()
                self.weight_fq.enable_observer()
            else:
                self.weight_fq = None

            if self.qconfig.activation is not None:
                self.act_fq = self.qconfig.activation()
                self.act_fq.enable_observer()
            else:
                self.act_fq = None

        # Flag to simulate quantization during inference
        self.simulate_quant = False

    @property
    def simulate_quant(self):
        """Return simulate_quant flag."""
        return self._simulate_quant

    @simulate_quant.setter
    def simulate_quant(self, value: bool):
        """Set simulate_quant flag and propagate to child QAT layers."""
        self._simulate_quant = value
        # Also propagate to CHT layer if it has this attribute
        if hasattr(self.cht_layer, 'simulate_quant'):
            self.cht_layer.simulate_quant = value

    @property
    def weight(self):
        """Return the underlying CHT layer weight for Ultralytics compatibility."""
        return self.cht_layer.weight

    @property
    def bias(self):
        """Return the underlying CHT layer bias."""
        return self.cht_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fake quantization."""
        # Handle mixed precision: convert weight to match input dtype and device
        input_dtype = x.dtype
        input_device = x.device

        # Quantize input during training OR when simulate_quant is enabled
        if self.act_fq is not None and (self.training or self.simulate_quant):
            x = self.act_fq(x)

        # Get weight with mask applied
        # Convert mask to float for proper gradient computation
        if hasattr(self.cht_layer, 'mask') and self.cht_layer.mask is not None:
            weight = self.cht_layer.weight * self.cht_layer.mask.float()
        else:
            weight = self.cht_layer.weight

        # Move weight to the same device as input (handles GPU training)
        weight = weight.to(device=input_device)

        # Convert weight to match input dtype for mixed precision support
        # This allows BF16 input (from BF16AttentionWrapper) to work with FP32 weights
        if input_dtype == torch.float16 and weight.dtype != torch.float16:
            weight = weight.to(torch.float16)

        # Quantize weight during training OR when simulate_quant is enabled
        if self.weight_fq is not None and (self.training or self.simulate_quant):
            weight = self.weight_fq(weight)

        # Perform convolution
        if self.use_bias:
            bias = self.cht_layer.bias
            # Convert bias to match input device and dtype
            bias = bias.to(device=input_device)
            if input_dtype == torch.float16 and bias.dtype != torch.float16:
                bias = bias.to(torch.float16)
        else:
            bias = None

        out = F.conv2d(
            x, weight, bias,
            stride=self.stride,
            padding=self.padding
        )

        # Quantize output during training OR when simulate_quant is enabled
        if self.act_fq is not None and (self.training or self.simulate_quant):
            out = self.act_fq(out)

        return out

    def to(self, *args, **kwargs):
        """Move module to device and ensure fake quantize buffers are also moved."""
        super().to(*args, **kwargs)
        # Move fake quantize modules to the same device as the module
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        if self.act_fq is not None:
            self.act_fq = self.act_fq.to(device)
        if self.weight_fq is not None:
            self.weight_fq = self.weight_fq.to(device)
        return self


class YOLO26nCHTQATModel(nn.Module):
    """
    YOLO26n model with both CHT sparsity and QAT quantization.

    This model loads pretrained YOLO26n weights and replaces Conv2d layers
    with CHT layers for sparsity, optionally wrapped with QAT for quantization.
    """

    def __init__(
        self,
        original_model: nn.Module,
        cht_config: CHTConfig,
        qconfig: Optional[torch.quantization.QConfig] = None,
        replace_backbone: bool = True,
        replace_neck: bool = True,
        replace_head: bool = False,
        enable_qat: bool = True,
        skip_first_n_convs: int = 2,
        replace_inside_attention: bool = False,
        quantization_dtype: str = "int8"
    ):
        """
        Initialize YOLO26n CHT-QAT model.

        Args:
            original_model: Original YOLO model
            cht_config: CHT configuration for sparsity
            qconfig: Quantization configuration (if None, uses default INT8)
            replace_backbone: Replace backbone Conv2d with CHT
            replace_neck: Replace neck Conv2d with CHT
            replace_head: Replace head Conv2d with CHT
            enable_qat: Enable quantization-aware training
            skip_first_n_convs: Number of early conv layers to skip from CHT (keep dense)
            replace_inside_attention: Whether to replace Conv2d inside attention modules with CHT
            quantization_dtype: Quantization type ('int8', 'fp8', or None)
        """
        super().__init__()
        self.cht_config = cht_config
        self.qconfig = qconfig
        self.replace_backbone = replace_backbone
        self.replace_neck = replace_neck
        self.replace_head = replace_head
        self.enable_qat = enable_qat
        self.skip_first_n_convs = skip_first_n_convs
        self.replace_inside_attention = replace_inside_attention
        self.quantization_dtype = quantization_dtype

        # Store original model
        self.original_model = original_model

        # Ultralytics compatibility
        self.args = {}
        self.nc = 80
        self.names = {i: str(i) for i in range(self.nc)}
        self.yaml = {}

        # Setup Detect layer
        self._setup_detect_layer()

        # Initialize bf16_attention_modules before building model (populated by _build_model)
        self.bf16_attention_modules: List[nn.Module] = []

        # Build the model with CHT and optionally QAT
        self._model = self._build_model(
            original_model,
            cht_config,
            qconfig,
            replace_backbone,
            replace_neck,
            replace_head,
            enable_qat,
            skip_first_n_convs,
            replace_inside_attention,
            quantization_dtype
        )

        # Track CHT layers
        self.cht_layers: List[Conv2d_CHT] = []
        self.qat_layers: List[QATConv2d_CHT] = []
        self._collect_layers(self._model)

        # Propagate Ultralytics attributes to _model for validation compatibility
        self._propagate_model_attributes()

    def _propagate_model_attributes(self):
        """Propagate Ultralytics model attributes to _model for validation."""
        # Copy essential Ultralytics attributes to _model
        attrs_to_copy = ['nc', 'names', 'stride', 'reg_max', 'nl', 'yaml', 'args']
        for attr in attrs_to_copy:
            if hasattr(self, attr):
                setattr(self._model, attr, getattr(self, attr))

        # Also set on the model's model attribute if it exists
        if hasattr(self._model, 'model'):
            for attr in attrs_to_copy:
                if hasattr(self, attr):
                    setattr(self._model.model, attr, getattr(self, attr))

    def _setup_detect_layer(self):
        """Extract Detect layer info from original model."""
        try:
            if hasattr(self.original_model, 'model'):
                detect = self.original_model.model[-1]
            else:
                detect = self.original_model[-1]

            if detect.__class__.__name__ == 'Detect':
                self.stride = detect.stride
                self.nc = detect.nc
                self.reg_max = detect.reg_max if hasattr(detect, 'reg_max') else 1
                self.nl = detect.nl if hasattr(detect, 'nl') else 4
            else:
                self._set_default_detect()
        except Exception:
            self._set_default_detect()

    def _set_default_detect(self):
        """Set default Detect layer values."""
        self.stride = torch.tensor([8., 16., 32.])
        self.nc = 80
        self.reg_max = 1
        self.nl = 4

    @property
    def model(self):
        """Return the model for Ultralytics compatibility."""
        if hasattr(self._model, 'model'):
            return self._model.model
        return self._model

    def _build_model(
        self,
        model: nn.Module,
        cht_config: CHTConfig,
        qconfig: Optional[torch.quantization.QConfig],
        replace_backbone: bool,
        replace_neck: bool,
        replace_head: bool,
        enable_qat: bool,
        skip_first_n_convs: int = 2,
        replace_inside_attention: bool = False,
        quantization_dtype: str = "int8"
    ) -> nn.Module:
        """Build model with CHT and optionally QAT layers."""
        new_model = copy.deepcopy(model)

        def get_layer_position(name: str) -> str:
            """Determine layer position from name."""
            try:
                if '.' in name:
                    parts = name.split('.')
                    for i in range(len(parts)):
                        try:
                            idx = int(parts[i])
                            # YOLO26n typically has: backbone (0-9), neck (10-16), head (17+)
                            if idx <= 9:
                                return 'backbone'
                            elif idx <= 16:
                                return 'neck'
                            else:
                                return 'head'
                        except ValueError:
                            continue
            except (ValueError, IndexError):
                pass

            if 'backbone' in name.lower():
                return 'backbone'
            elif 'neck' in name.lower():
                return 'neck'
            elif 'head' in name.lower():
                return 'head'

            return 'unknown'

        # Attention-related submodule names that should NOT have their Conv2d replaced
        # These are critical parts of attention mechanisms
        ATTENTION_SUBMODULE_NAMES = ['attn', 'qkv', 'proj', 'pe', 'ffn']

        def _is_inside_attention_submodule(name: str) -> bool:
            """Check if layer name indicates it's inside an attention-related submodule."""
            name_lower = name.lower()
            # Check if the name contains attention module types
            attention_types = ['c2psa', 'c2f', 'psablock', 'c3k2', 'attention']
            if any(att in name_lower for att in attention_types):
                return True
            # Also check individual parts
            parts = name.split('.')
            for part in parts[:-1]:  # Exclude the last part (e.g., 'conv')
                if part in ATTENTION_SUBMODULE_NAMES:
                    return True
            return False

        def _is_critical_attention_conv(name: str) -> bool:
            """Check if this is a critical attention component that should NEVER be replaced.

            These are the core components of attention mechanisms:
            - qkv: query-key-value projection
            - proj: output projection
            - pe: positional encoding
            - ffn: feed-forward network
            Replacing these breaks the attention mechanism.
            """
            name_lower = name.lower()
            # Check for critical attention components
            critical_names = ['qkv', 'proj', 'pe']
            if any(crit in name_lower for crit in critical_names):
                return True
            # Check for ffn patterns (ffn.0, ffn.1 inside attention modules)
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part in ['ffn', 'mlp'] and i + 1 < len(parts):
                    # This is inside an ffn/mlp block within attention
                    return True
            return False

        # Counter for tracking conv layer order (to skip first N)
        conv_counter = [0]

        def create_layer(module: nn.Module, name: str, parent_is_attention: bool = False) -> nn.Module:
            """Create CHT or QAT-CHT layer from Conv2d."""
            # Check if this module itself is an attention type
            is_attention = _is_attention_module(module)

            if isinstance(module, nn.Conv2d):
                position = get_layer_position(name)

                # Check if we should skip this early conv layer
                conv_counter[0] += 1
                skip_this_conv = conv_counter[0] <= skip_first_n_convs

                should_replace = False
                if not skip_this_conv:
                    if position == 'backbone' and replace_backbone:
                        should_replace = True
                    elif position == 'neck' and replace_neck:
                        should_replace = True
                    elif position == 'head' and replace_head:
                        should_replace = True

                # Skip replacing Conv2d inside attention-related submodules
                # These are critical parts of attention (qkv, proj, ffn, etc.)
                # Unless replace_inside_attention is enabled
                if not replace_inside_attention:
                    if parent_is_attention or is_attention or _is_inside_attention_submodule(name):
                        should_replace = False

                # CRITICAL: Never replace core attention components (qkv, proj, pe, ffn)
                # These are essential to the attention mechanism and breaking them destroys model performance
                if _is_critical_attention_conv(name):
                    should_replace = False

                if should_replace:
                    # Apply CHT (sparsity) + QAT (quantization)
                    cht_layer = Conv2d_CHT(
                        c_in=module.in_channels,
                        c_out=module.out_channels,
                        kernel_sz=module.kernel_size[0],
                        cht_config=cht_config,
                        padding=module.padding[0],
                        stride=module.stride[0],
                        bias=module.bias is not None
                    )

                    # Copy pretrained weights
                    with torch.no_grad():
                        cht_layer.weight.copy_(module.weight)
                        if module.bias is not None:
                            cht_layer.bias.copy_(module.bias)

                    # Wrap with QAT (CHT + QAT)
                    if enable_qat:
                        return QATConv2d_CHT(cht_layer, qconfig, quantization_dtype)

                    return cht_layer
                elif enable_qat and isinstance(module, nn.Conv2d):
                    # Apply QAT only (no CHT) - quantization to all layers
                    return QATConv2d(module, qconfig, quantization_dtype)

            return module

        def wrap_attention_modules(m: nn.Module, parent_name: str = '') -> bool:
            """Recursively wrap attention modules with BF16 wrapper. Returns True if module was wrapped."""
            wrapped_any = False
            for name, child in list(m.named_children()):
                full_name = f"{parent_name}.{name}" if parent_name else name

                # Check if this child is an attention module
                if _should_bf16_wrap(child):
                    print(f"  Wrapping attention module {full_name} ({type(child).__name__}) with BF16")
                    # Wrap with BF16 wrapper
                    wrapped = BF16AttentionWrapper(child)
                    m.__setattr__(name, wrapped)
                    self.bf16_attention_modules.append(wrapped)
                    wrapped_any = True
                elif hasattr(child, 'named_children'):
                    # Recurse into child
                    if wrap_attention_modules(child, full_name):
                        wrapped_any = True

            return wrapped_any

        def recursive_build(m: nn.Module, parent_name: str = '', parent_is_attention: bool = False, inside_bf16_wrapper: bool = False):
            """Recursively build model."""
            for name, child in list(m.named_children()):
                full_name = f"{parent_name}.{name}" if parent_name else name

                # Skip processing if this is a wrapped_module (internal of BF16 wrapper)
                if name == 'wrapped_module':
                    continue

                # Check if this child is a BF16-wrapped attention module
                is_bf16_wrapper = isinstance(child, BF16AttentionWrapper)
                # Check if this child is an attention module
                is_attention_child = _is_attention_module(child)

                # Determine if we're currently inside a BF16 wrapper (from parent context)
                currently_inside_wrapper = inside_bf16_wrapper or is_bf16_wrapper

                # Skip replacing if inside BF16 wrapper (already wrapped, don't process internal Conv2d)
                if not currently_inside_wrapper:
                    # First, try to replace this child with CHT
                    new_child = create_layer(child, full_name, parent_is_attention)
                    if new_child is not child:
                        m.__setattr__(name, new_child)

                # Then recurse into children (if it's a container)
                # But skip recursion into BF16AttentionWrapper - it contains its own module tree
                is_inside_attention = parent_is_attention or is_attention_child
                if hasattr(child, 'named_children') and not is_bf16_wrapper:
                    recursive_build(child, full_name, is_inside_attention, currently_inside_wrapper)

        # First, replace Conv2d with CHT/QAT (including inside attention if flag is True)
        recursive_build(new_model)

        # Wrap attention modules with BF16 wrapper
        # This ensures softmax runs in FP16/BF16 for better accuracy
        print("Wrapping attention modules with BF16 wrapper...")
        wrap_attention_modules(new_model)

        return new_model

    def _collect_layers(self, model: nn.Module):
        """Collect CHT and QAT layers."""
        for module in model.modules():
            if isinstance(module, QATConv2d_CHT):
                # CHT + QAT layers - counts as QAT (inner Conv2d_CHT counted separately)
                self.qat_layers.append(module)
            elif isinstance(module, QATConv2d):
                # QAT only (no CHT) - e.g., head layers with quantization but no sparsity
                self.qat_layers.append(module)
            elif isinstance(module, Conv2d_CHT):
                # CHT layers - includes those wrapped in QATConv2d_CHT
                self.cht_layers.append(module)

    def forward(self, x: torch.Tensor, **kwargs):
        """
        Forward pass through the model.

        Args:
            x: Input tensor
            **kwargs: Additional arguments (augment, visualize, embed, etc.)
                     - These are ignored as CHT model doesn't support augmentation

        Returns output in standard Ultralytics format:
        - During training: list of feature tensors with gradients
        - During inference: tuple of (predictions, info_dict)

        Note: Attention modules (C2PSA, C2f, etc.) are NOT replaced with QATConv2d_CHT,
        so they don't receive int8/fp8 quantization. They run in FP32 by default,
        which satisfies the requirement that attention layers should not use int8/fp8.
        """
        # Get raw output from the model
        raw_output = self._model(x)

        # During training, return feature maps (feats) which have gradients
        if self.training:
            if isinstance(raw_output, dict):
                # Return feats list which has gradients
                if 'one2many' in raw_output and 'feats' in raw_output['one2many']:
                    return raw_output['one2many']['feats']
                elif 'one2one' in raw_output and 'feats' in raw_output['one2one']:
                    return raw_output['one2one']['feats']
            elif isinstance(raw_output, (list, tuple)):
                # For training, return as-is - let loss function handle format
                return list(raw_output)

        # For inference, convert to Ultralytics format
        return self._convert_to_ultralytics_format(raw_output)

    def _convert_to_ultralytics_format(self, raw_output):
        """
        Convert model output to standard Ultralytics format.

        Standard Ultralytics format:
        - During inference: tuple of (predictions [batch, N, 6], info_dict)
        - During training: list of detection layer tensors, each shape [batch, channels, H, W]
        """
        # Handle tuple output (standard ultralytics inference format)
        # This is the case when model is in eval mode
        if isinstance(raw_output, tuple):
            # Standard ultralytics format: (predictions [batch, N, 6], info_dict)
            # Return the tuple directly
            return raw_output

        # Handle dict output (current YOLO26n format with 'one2one'/'one2many' keys)
        if isinstance(raw_output, dict):
            if 'one2one' in raw_output:
                pred_dict = raw_output['one2one']
            elif 'one2many' in raw_output:
                pred_dict = raw_output['one2many']
            else:
                pred_dict = raw_output

            if isinstance(pred_dict, dict):
                feats = pred_dict.get('feats', None)

                # If feats is a list of feature tensors, return as standard format
                if feats is not None and isinstance(feats, list):
                    # Standard ultralytics format: list of [batch, channels, H, W]
                    return feats
                else:
                    # Otherwise reconstruct from boxes/scores
                    boxes = pred_dict.get('boxes', None)
                    scores = pred_dict.get('scores', None)

                    if boxes is not None and isinstance(boxes, torch.Tensor):
                        # Reshape from [batch, channels, num_anchors] to [batch, channels, H, W]
                        # Try to find appropriate spatial dimensions
                        num_anchors = boxes.shape[2] if boxes.dim() >= 3 else 1
                        # Estimate H, W (assume square-ish)
                        h = w = int(num_anchors ** 0.5)
                        if h * w != num_anchors:
                            h, w = 1, num_anchors

                        boxes_reshaped = boxes.view(boxes.shape[0], boxes.shape[1], h, w)
                        return [boxes_reshaped]

        # Handle list/tuple output (already in standard YOLO format)
        elif isinstance(raw_output, (list, tuple)):
            # Standard YOLO output: list of [batch, channels, H, W] for each detection layer
            return list(raw_output)

        # Handle single tensor output
        if isinstance(raw_output, torch.Tensor):
            if raw_output.dim() == 4:
                # Already [batch, channels, H, W] - return as list
                return [raw_output]
            elif raw_output.dim() == 3:
                # [batch, channels, num_anchors] - reshape to [batch, channels, H, W]
                b, c, n = raw_output.shape
                h = w = int(n ** 0.5)
                if h * w != n:
                    h, w = 1, n
                return [raw_output.view(b, c, h, w)]

        # Fallback: wrap in list
        if isinstance(raw_output, torch.Tensor):
            return [raw_output]
        elif isinstance(raw_output, (list, tuple)):
            return list(raw_output)
        else:
            return [raw_output]

    def fuse(self, verbose: bool = True):
        """Override fuse to prevent Ultralytics from fusing Conv+BN layers."""
        # Instead of skipping, disable fuse on all submodules
        for module in self._model.modules():
            if hasattr(module, 'fuse'):
                # Replace fuse with a no-op that returns the module itself
                module.fuse = lambda v=True, *args, **kwargs: module
        if verbose:
            print("  Disabled fuse on all submodules (CHT layers)")
        return self

    def evolve(self, current_epoch: int = None, verbose: bool = True) -> Dict:
        """Evolve all CHT layers (sparsity)."""
        stats = {
            'layers_evolved': 0,
            'total_cancellation': 0.0,
            'avg_min_score': 0.0,
            'avg_mean_score': 0.0,
            'layer_stats': []
        }

        for i, layer in enumerate(self.cht_layers):
            if hasattr(layer, 'num_update') and layer.num_update <= 0:
                if verbose:
                    print(f"  Skipping layer {i+1}/{len(self.cht_layers)}: {layer.layer_name}")
                continue

            if verbose:
                print(f"  Evolving layer {i+1}/{len(self.cht_layers)}: {layer.layer_name}")

            cancellation, min_score, mean_score = layer.evolve(current_epoch=current_epoch)

            if cancellation is not None:
                stats['layers_evolved'] += 1
                stats['total_cancellation'] += cancellation if cancellation else 0
                stats['avg_min_score'] += min_score if min_score else 0
                stats['avg_mean_score'] += mean_score if mean_score else 0

                stats['layer_stats'].append({
                    'name': layer.layer_name,
                    'cancellation': cancellation,
                    'min_score': min_score,
                    'mean_score': mean_score
                })

        if stats['layers_evolved'] > 0:
            stats['total_cancellation'] /= stats['layers_evolved']
            stats['avg_min_score'] /= stats['layers_evolved']
            stats['avg_mean_score'] /= stats['layers_evolved']

        return stats

    def get_sparsity(self) -> float:
        """Get average sparsity across all CHT layers."""
        if not self.cht_layers:
            return 0.0

        total_sparsity = 0.0
        for layer in self.cht_layers:
            if hasattr(layer, 'mask') and layer.mask is not None:
                active = layer.mask.sum().item()
                total = layer.mask.numel()
                sparsity = 1.0 - (active / total)
                total_sparsity += sparsity

        return total_sparsity / len(self.cht_layers)

    def get_sparsity_target(self) -> float:
        """Get target sparsity from CHT layers."""
        if not self.cht_layers:
            return 0.0

        for layer in self.cht_layers:
            if hasattr(layer, 'sparsity_target'):
                return layer.sparsity_target

        return 0.0

    def get_num_cht_layers(self) -> int:
        """Get number of CHT layers."""
        return len(self.cht_layers)

    def get_num_qat_layers(self) -> int:
        """Get number of QAT layers (includes CHT+QAT and QAT-only)."""
        return len(self.qat_layers)

    def get_num_bf16_attention_modules(self) -> int:
        """Get number of BF16-wrapped attention modules."""
        return len(self.bf16_attention_modules)

    def get_num_bf16_parameters(self) -> int:
        """Get total number of parameters in BF16 attention modules."""
        total_params = 0
        for wrapper in self.bf16_attention_modules:
            # Count parameters in the wrapped module
            for param in wrapper.wrapped_module.parameters():
                total_params += param.numel()
        return total_params

    def get_fallback_count(self) -> int:
        """Get total fallback count across all CHT layers."""
        if not self.cht_layers:
            return 0

        total_fallback = 0
        for layer in self.cht_layers:
            if hasattr(layer, 'fallback_count'):
                total_fallback += layer.fallback_count

        return total_fallback

    def ensure_on_device(self, device: torch.device):
        """Ensure all parameters are on the specified device."""
        # Normalize device string: 'cuda' -> 'cuda:0'
        device_str = str(device)
        if device_str == 'cuda':
            device_str = 'cuda:0'

        for module in self.modules():
            for param in module.parameters(recurse=True):
                # Compare as strings to handle torch.device objects correctly
                param_device_str = str(param.device)
                # Also normalize param device string
                if param_device_str == 'cuda':
                    param_device_str = 'cuda:0'
                if param_device_str != device_str:
                    param.data = param.data.to(device)
        return self

    def find_wrong_device_modules(self, device: torch.device):
        """Find modules that have parameters on wrong device (for debugging)."""
        wrong_modules = []
        # Normalize device string: 'cuda' -> 'cuda:0'
        device_str = str(device) if isinstance(device, torch.device) else device
        if device_str == 'cuda':
            device_str = 'cuda:0'

        for name, module in self.named_modules():
            module_params = list(module.parameters(recurse=False))
            if module_params:
                module_device = module_params[0].device if module_params else None
                module_device_str = str(module_device) if module_device else None
                if module_device_str and module_device_str != device_str:
                    wrong_modules.append((name, type(module).__name__, module_device))
        return wrong_modules

    def enable_observer(self):
        """Enable observers for collecting quantization statistics."""
        for module in self.modules():
            if hasattr(module, 'act_fq') and module.act_fq is not None:
                module.act_fq.enable_observer()
            if hasattr(module, 'weight_fq') and module.weight_fq is not None:
                module.weight_fq.enable_observer()

    def disable_observer(self):
        """Disable observers."""
        for module in self.modules():
            if hasattr(module, 'act_fq') and module.act_fq is not None:
                module.act_fq.disable_observer()
            if hasattr(module, 'weight_fq') and module.weight_fq is not None:
                module.weight_fq.disable_observer()

    def set_simulate_quant(self, enabled: bool):
        """Enable or disable quantization simulation during inference."""
        for module in self.modules():
            if hasattr(module, 'simulate_quant'):
                module.simulate_quant = enabled

    def train(self, mode=True):
        """Override train() to set inference mode flag on all CHT layers."""
        super().train(mode)
        # Explicitly set inference mode flag on all CHT layers
        for layer in self.cht_layers:
            layer._inference_mode = not mode
        return self

    def eval(self):
        """Override eval() to set inference mode flag on all CHT layers."""
        super().eval()
        # Explicitly set inference mode flag on all CHT layers
        for layer in self.cht_layers:
            layer._inference_mode = True
        return self


def load_yolo26n_cht_qat_model(
    model_path: str = "yolo26n.pt",
    sparsity: float = 0.9,
    replace_mode: ReplaceMode = ReplaceMode.BACKBONE_NECK,
    quantization: str = "int8",
    regrow_method: str = "L3n",
    shared_mask_sw: bool = True,
    soft: bool = True,
    link_update_ratio: float = 0.1,
    skip_first_n_convs: int = 2,
    replace_inside_attention: bool = True,
    **cht_kwargs
) -> YOLO26nCHTQATModel:
    """
    Load YOLO26n model with both sparsity (CHT) and quantization (QAT).

    Args:
        model_path: Path to YOLO26n model weights
        sparsity: Target sparsity (0.9 = 90%)
        replace_mode: Which layers to replace (backbone, backbone_neck, all)
        quantization: Quantization type ('int8', 'fp8', or None)
        regrow_method: Method for regrowing connections ("L3n" for CHT)
        shared_mask_sw: Whether to share mask across sliding windows
        soft: Use soft removal/regrowth
        link_update_ratio: Fraction of active connections to update per evolve
        skip_first_n_convs: Number of early conv layers to skip from CHT (keep dense)
        replace_inside_attention: Whether to replace Conv2d inside attention modules with CHT
        **cht_kwargs: Additional CHT config parameters

    Returns:
        YOLO26nCHTQATModel with CHT + QAT
    """
    print(f"Loading YOLO26n model from {model_path}...")
    print(f"Target sparsity: {sparsity * 100}%")
    print(f"Replace mode: {replace_mode.value}")
    print(f"Skip first {skip_first_n_convs} conv layers from CHT")
    print(f"Replace inside attention: {replace_inside_attention}")
    print(f"Quantization: {quantization}")

    # Determine which layers to replace
    # Use .value to compare to handle different enum instances
    mode_value = replace_mode.value if hasattr(replace_mode, 'value') else str(replace_mode)
    if mode_value == 'backbone':
        replace_backbone = True
        replace_neck = False
        replace_head = False
    elif mode_value == 'backbone_neck':
        replace_backbone = True
        replace_neck = True
        replace_head = False
    else:  # ALL or unknown
        replace_backbone = True
        replace_neck = True
        replace_head = True

    # Load original model
    try:
        from ultralytics import YOLO
        original_model = YOLO(model_path)
        original_model = original_model.model
    except ImportError:
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(state_dict, dict):
            if 'model' in state_dict:
                original_model = state_dict['model']
            else:
                raise ValueError("Cannot determine model structure from state dict")
        else:
            original_model = state_dict

    # Unfreeze all model parameters for training
    # Ultralytics loads pretrained models with requires_grad=False (frozen backbone)
    # For CHT+QAT training, we need to train all layers
    for param in original_model.parameters():
        param.requires_grad = True

    # Create CHT config
    cht_config = create_cht_config(
        sparsity=sparsity,
        regrow_method=regrow_method,
        shared_mask_sw=shared_mask_sw,
        soft=soft,
        link_update_ratio=link_update_ratio,
        **cht_kwargs
    )

    # Create QAT config
    qconfig = create_qat_config(quantization)

    # Enable QAT if quantization is specified
    enable_qat = quantization is not None and quantization != 'none'

    # Create model
    model = YOLO26nCHTQATModel(
        original_model=original_model,
        cht_config=cht_config,
        qconfig=qconfig,
        replace_backbone=replace_backbone,
        replace_neck=replace_neck,
        replace_head=replace_head,
        enable_qat=enable_qat,
        skip_first_n_convs=skip_first_n_convs,
        replace_inside_attention=replace_inside_attention,
        quantization_dtype=quantization
    )

    print(f"Model created with {model.get_num_cht_layers()} CHT layers")
    print(f"BF16 attention modules: {model.get_num_bf16_attention_modules()}")
    print(f"QAT enabled: {enable_qat}")

    return model


def count_model_params(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
