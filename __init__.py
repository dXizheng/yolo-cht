"""
YOLO26n CHT + QAT Model Package

This package provides YOLO26n model with:
- CHT (Compressed Hierarchical Training) layers for sparsity
- QAT (Quantization-Aware Training) for quantization
"""

from .yolo26n_cht_qat_model import (
    YOLO26nCHTQATModel,
    load_yolo26n_cht_qat_model,
    QATConv2d_CHT
)
from .yolo26n_config import (
    ReplaceMode,
    YOLO26nConfig,
    create_cht_config,
    create_qat_config
)

__all__ = [
    'YOLO26nCHTQATModel',
    'load_yolo26n_cht_qat_model',
    'QATConv2d_CHT',
    'ReplaceMode',
    'YOLO26nConfig',
    'create_cht_config',
    'create_qat_config',
]

__version__ = '1.0.0'
