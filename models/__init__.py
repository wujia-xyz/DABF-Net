"""DABF-Net model factory."""
from .hybrid_model import build_hybrid_model, KBNetEMCADHybrid

# Public-facing aliases
build_dabf_net = build_hybrid_model
DABFNet = KBNetEMCADHybrid

__all__ = [
    "build_hybrid_model",
    "KBNetEMCADHybrid",
    "build_dabf_net",
    "DABFNet",
]
