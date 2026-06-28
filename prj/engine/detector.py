from typing import Dict, Any, Union
from prj.models.event import LoginEvent
from prj.models.result import FraudResult
from prj.engine.pipeline import DetectionPipeline

class FraudDetector:
    """The main external interface for orchestrating login anomaly and fraud checks."""
    
    def __init__(self, profile_store=None, cache_store=None, db_store=None, alert_producer=None):
        self.pipeline = DetectionPipeline(
            profile_store=profile_store,
            cache_store=cache_store,
            db_store=db_store,
            alert_producer=alert_producer
        )

    def analyze(self, login_data: Union[Dict[str, Any], LoginEvent]) -> FraudResult:
        """Analyzes a login event for potential anomalies and impossible travel."""
        if isinstance(login_data, dict):
            event = LoginEvent(**login_data)
        else:
            event = login_data
            
        return self.pipeline.process(event)
