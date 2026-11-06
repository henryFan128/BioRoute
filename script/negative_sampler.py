"""Backward-compatible public import path for NegativeSampler."""

if __package__:
    from .negative_sampling.core import _NegativeSamplerCore
else:
    from negative_sampling.core import _NegativeSamplerCore


class NegativeSampler(_NegativeSamplerCore):
    """Public sampler class retained for existing imports and pickle files."""
