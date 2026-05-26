# Copyright (c) 2026
# 基于因果干预的动态防御模块
# Causal Intervention Dynamic Defense Module

from .defense_engine import (
    CausalDefenseConfig,
    CausalDefenseEngine,
    fuse_loaded_vectors,
    load_and_normalize_vectors,
    load_malicious_vector_files,
)
from .defense_proj import StateProjectionMonitor
from .hooks import MaliciousVectorInjector
from .immune_delta_preserver import ImmuneDeltaPreserver
from .immune_vector_continuation import ImmuneVectorContinuationInjector
from .gradient_analyzer import GradientAnalyzer
from .gradient_probe_defense import InjectionGradientProbeDefense

__all__ = [
    "CausalDefenseConfig",
    "CausalDefenseEngine",
    "MaliciousVectorInjector",
    "ImmuneDeltaPreserver",
    "ImmuneVectorContinuationInjector",
    "StateProjectionMonitor",
    "GradientAnalyzer",
    "InjectionGradientProbeDefense",
    "fuse_loaded_vectors",
    "load_and_normalize_vectors",
    "load_malicious_vector_files",
]
