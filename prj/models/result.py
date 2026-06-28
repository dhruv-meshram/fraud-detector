from pydantic import BaseModel, Field
from typing import List, Optional

class RiskBreakdown(BaseModel):
    velocity_kmh: float = Field(..., description="Spatiotemporal velocity in km/h")
    distance_km: float = Field(..., description="Distance in km to closest historical cluster centroid")
    device_mismatch: bool = Field(..., description="Whether device hash is unrecognized")

class FraudResult(BaseModel):
    risk_score: float = Field(..., description="Calculated risk score (0 to 100)")
    is_fraudulent: bool = Field(..., description="Whether the login was flagged as fraudulent")
    reasons: List[str] = Field(default_factory=list, description="Reasoning/justification list for the flags")
    status: str = Field(..., description="Detailed classification status (e.g. COLD_START_BYPASS, KNOWN_ZONE, IMPOSSIBLE_VELOCITY, OUTLIER)")
    details: RiskBreakdown = Field(..., description="Broken down metrics for the score evaluation")
