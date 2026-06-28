"""Algorithms Interface.

Exposes internal algorithms (velocity, haversine, scoring) behind a consolidated module boundary.
"""

from prj.algorithms.graph.velocity_validator import validate_velocity
from prj.algorithms.spatial.haversine import haversine_distance, spatiotemporal_velocity
from prj.algorithms.ranking.multi_factor_score import calculate_risk_score

__all__ = [
    "validate_velocity",
    "haversine_distance",
    "spatiotemporal_velocity",
    "calculate_risk_score"
]
