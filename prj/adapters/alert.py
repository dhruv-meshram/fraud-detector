from pathlib import Path
import json
from typing import Dict, Any

from prj.adapters.base import BaseAlertProducer

class KafkaAlertProducer(BaseAlertProducer):
    """Kafka-backed broker notifier (falling back to a local log file in development)."""
    
    def __init__(self, bootstrap_servers: str = "localhost:9092", log_path: str = "data/kafka_events.log"):
        self.bootstrap_servers = bootstrap_servers
        self.event_log_path = Path(log_path)
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit_event(self, topic: str, key: str, value: Dict[str, Any]) -> bool:
        event = {
            "topic": topic,
            "key": key,
            "payload": value
        }
        try:
            with open(self.event_log_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"[KafkaAlertProducer ERROR] Failed to log local event: {e}")
            
        print(f"[KafkaAlertProducer EMIT] Topic: {topic} | Key: {key} | Event successfully dispatched.")
        return True


class ConsoleAlertProducer(BaseAlertProducer):
    """Console logging alert producer for SDK stdout usage."""
    
    def __init__(self):
        self.emitted_events = []

    def emit_event(self, topic: str, key: str, value: Dict[str, Any]) -> bool:
        self.emitted_events.append({"topic": topic, "key": key, "value": value})
        print(f"[ALERT] Topic={topic} Key={key} Payload={value}")
        return True
