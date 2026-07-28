"""Fusion strategies — merge results from multiple retrievers."""

from raglan.fusion.round_robin import RoundRobinFusion
from raglan.fusion.rrf import RRFFusion
from raglan.fusion.weighted import WeightedFusion

__all__ = ["RRFFusion", "RoundRobinFusion", "WeightedFusion"]
