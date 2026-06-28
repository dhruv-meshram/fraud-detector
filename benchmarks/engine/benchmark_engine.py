"""Engine benchmark measuring overall and stage-by-stage latencies."""

import time
import numpy as np
from benchmarks.utils import measure_memory
from prj import FraudDetector
from prj.engine.pipeline import DetectionPipeline
from prj.adapters import InMemoryProfileStore, InMemoryCacheStore, InMemoryDBStore, ConsoleAlertProducer
from prj.models.event import LoginEvent

class InstrumentedDetectionPipeline(DetectionPipeline):
    """Subclass of DetectionPipeline that instruments each stage for precise timing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_timings = {
            "fetch_last_node": [],
            "velocity_check": [],
            "ml_predict": [],
            "device_mismatch": [],
            "scoring": [],
            "alerting": []
        }

    def process(self, event: LoginEvent):
        user_id = event.user_id
        lat = event.latitude
        lon = event.longitude
        ts = event.timestamp
        device = event.device_hash

        # 1. Fetch last verified login node
        t0 = time.perf_counter()
        last_node = self.cache_store.get_last_node(user_id)
        if not last_node:
            last_node = self.db_store.get_last_verified_login(user_id)
            if last_node:
                self.cache_store.set_last_node(user_id, last_node)
        self.stage_timings["fetch_last_node"].append((time.perf_counter() - t0) * 1000)

        # 2. Velocity check
        t0 = time.perf_counter()
        velocity_kmh = 0.0
        if last_node:
            from prj.algorithms import spatiotemporal_velocity, validate_velocity
            velocity_kmh = spatiotemporal_velocity(
                last_node["latitude"], last_node["longitude"], last_node["timestamp"],
                lat, lon, ts
            )
            validate_velocity(last_node, {"latitude": lat, "longitude": lon, "timestamp": ts})
        self.stage_timings["velocity_check"].append((time.perf_counter() - t0) * 1000)

        # 3. ML Inference prediction
        t0 = time.perf_counter()
        evaluation = self.inference_service.predict(event)
        self.stage_timings["ml_predict"].append((time.perf_counter() - t0) * 1000)

        if evaluation["status"] == "NO_PROFILE":
            self.db_store.record_login(user_id, lat, lon, ts, device, is_verified=True)
            self.cache_store.set_last_node(user_id, {"latitude": lat, "longitude": lon, "timestamp": ts, "device_hash": device})
            return
            
        if evaluation["status"] == "KNOWN_ZONE":
            self.db_store.record_login(user_id, lat, lon, ts, device, is_verified=True)
            self.cache_store.set_last_node(user_id, {"latitude": lat, "longitude": lon, "timestamp": ts, "device_hash": device})
            return

        # 4. Device Mismatch check
        t0 = time.perf_counter()
        historic_devices = self.db_store.get_all_device_hashes(user_id)
        device_mismatch = device not in historic_devices if historic_devices else True
        self.stage_timings["device_mismatch"].append((time.perf_counter() - t0) * 1000)

        # 5. Risk Scoring
        t0 = time.perf_counter()
        from prj.algorithms import calculate_risk_score
        closest_dist = evaluation["distance_km"]
        mfa_score = calculate_risk_score(velocity_kmh, closest_dist, device_mismatch)
        self.stage_timings["scoring"].append((time.perf_counter() - t0) * 1000)

        # 6. Alerting
        t0 = time.perf_counter()
        self.alert_producer.emit_event(
            topic="shieldflow.anomalies",
            key=user_id,
            value={"user_id": user_id, "score": mfa_score}
        )
        self.stage_timings["alerting"].append((time.perf_counter() - t0) * 1000)

def run_engine_benchmarks():
    """Benchmarks full detector.analyze() latency and stage details."""
    p_store = InMemoryProfileStore()
    c_store = InMemoryCacheStore()
    d_store = InMemoryDBStore()
    a_producer = ConsoleAlertProducer()
    
    # Instantiate instrumented pipeline detector
    detector = FraudDetector(
        profile_store=p_store,
        cache_store=c_store,
        db_store=d_store,
        alert_producer=a_producer
    )
    detector.pipeline = InstrumentedDetectionPipeline(
        profile_store=p_store,
        cache_store=c_store,
        db_store=d_store,
        alert_producer=a_producer
    )
    
    # Seed data
    user_id = "engine_user"
    p_store.save_profile(user_id, {
        "user_id": user_id,
        "clusters": [{"cluster_id": 0, "centroid_lat": 37.7749, "centroid_lon": -122.4194, "dynamic_radius_km": 15.0}]
    })
    
    # Pre-populate device and node histories to force outlier logic
    d_store.record_login(user_id, 37.7749, -122.4194, 1600000000.0, "dev_known", True)
    c_store.set_last_node(user_id, {"latitude": 37.7749, "longitude": -122.4194, "timestamp": 1600000000.0, "device_hash": "dev_known"})
    
    # Event causing outlier check
    event = {
        "user_id": user_id,
        "latitude": 40.7128, # New York
        "longitude": -74.0060,
        "timestamp": 1600010000.0,
        "device_hash": "dev_unknown"
    }
    
    total_latencies = []
    
    # Execute 1000 runs
    for _ in range(1000):
        t0 = time.perf_counter()
        detector.analyze(event)
        total_latencies.append((time.perf_counter() - t0) * 1000) # to ms
        
    # Compile timings
    pipeline = detector.pipeline
    stage_report = {}
    for stage, times in pipeline.stage_timings.items():
        stage_report[stage] = {
            "avg_ms": round(float(np.mean(times)), 4),
            "p95_ms": round(float(np.percentile(times, 95)), 4)
        }
        
    _, peak_kb = measure_memory(detector.analyze, event)
    
    return {
        "total_latency": {
            "p50": round(float(np.percentile(total_latencies, 50)), 4),
            "p95": round(float(np.percentile(total_latencies, 95)), 4),
            "p99": round(float(np.percentile(total_latencies, 99)), 4),
            "avg": round(float(np.mean(total_latencies)), 4)
        },
        "stages": stage_report,
        "peak_memory_kb": round(peak_kb, 2)
    }

if __name__ == "__main__":
    res = run_engine_benchmarks()
    print("Engine Benchmarks:", res)
