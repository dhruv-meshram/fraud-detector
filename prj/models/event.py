from pydantic import BaseModel, Field
from datetime import datetime, timezone

class LoginEvent(BaseModel):
    user_id: str = Field(..., description="Unique user identifier", json_schema_extra={"example": "123"})
    latitude: float = Field(..., description="Login latitude coordinate", json_schema_extra={"example": 19.11})
    longitude: float = Field(..., description="Login longitude coordinate", json_schema_extra={"example": 72.83})
    timestamp: float = Field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp(),
        description="UNIX timestamp"
    )
    device_hash: str = Field(..., description="Client device unique hash value", json_schema_extra={"example": "abc"})
