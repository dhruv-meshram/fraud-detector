"""Algorithms Interface.

Exposes internal algorithms (velocity, haversine, scoring) behind a consolidated module boundary.
"""

from fraud_detector.algorithms.graph.velocity_validator import validate_velocity
from fraud_detector.algorithms.spatial.haversine import haversine_distance, spatiotemporal_velocity
from fraud_detector.algorithms.ranking.multi_factor_score import calculate_risk_score

__all__ = [
    "validate_velocity",
    "haversine_distance",
    "spatiotemporal_velocity",
    "calculate_risk_score"
]
