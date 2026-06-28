# ShieldFlow Anomaly & Fraud Detection Engine (SDK)

ShieldFlow is a context-aware, low-latency login anomaly and fraud detection engine designed to identify credentials misuse, session hijacking, and impossible-travel events.

It has been redesigned as a **fully decoupled, reusable Python SDK (`prj`)** that can be integrated into external applications, with customizable storage, cache, database, and broker adapters.

---

## 🏗️ Architecture & SDK Design

ShieldFlow partitions its detection logic into two paths:

1. **Fast Path (Live Inference)**
   - Rules-based validations (Spatiotemporal Velocity) and boundary checks using DBSCAN cluster profiles.
2. **Slow Path (Offline Batch Training)**
   - DBSCAN models trained per user to identify habitual spatial boundaries.

### Adapter Pattern Integration

The SDK is decoupled from infrastructure via abstract base classes in `prj/adapters/base.py`. Developers can plug in their own storage engines:

- **Profile Store (`BaseProfileStore`)**: Retrieves user spatial boundary profiles.
  - *Production*: `PostgreSQLProfileStore` (hydrates from Redis or reads from filesystem fallback).
  - *Headless / Testing*: `InMemoryProfileStore` (completely stateless).
- **Cache Store (`BaseCacheStore`)**: Manages real-time session caching of the last login node.
  - *Production*: `RedisCacheStore`.
  - *Headless / Testing*: `InMemoryCacheStore`.
- **Database Store (`BaseDBStore`)**: Commits confirmed logins and checks historic device logs.
  - *Production*: `PostgresDBStore` (simulated via pandas CSV).
  - *Headless / Testing*: `InMemoryDBStore`.
- **Alert Producer (`BaseAlertProducer`)**: Dispatches asynchronous streaming anomaly alerts.
  - *Production*: `KafkaAlertProducer`.
  - *Headless / Testing*: `ConsoleAlertProducer`.

---

## 📦 Package Structure

```directory
├── prj/                       # Core SDK Package
│   ├── __init__.py            # Exposes FraudDetector public interface
│   ├── engine/                # Core pipeline orchestration
│   │   ├── detector.py        # FraudDetector orchestrator
│   │   └── pipeline.py        # DetectionPipeline
│   ├── models/                # Pydantic v2 domain schemas (LoginEvent, FraudResult)
│   ├── algorithms/            # Stateless mathematical algorithms (Haversine, Graph velocity)
│   ├── ml/                    # DBSCAN ML inference loader
│   └── adapters/              # Infrastructure adapters (Redis, Postgres, Kafka, In-Memory)
├── app/                       # FastAPI controller layer (consumes SDK)
├── tests/                     # Unit and Integration test suite
├── pyproject.toml             # Python packaging configuration
```

---

## 🚀 Installation

You can install the SDK locally in editable mode or from a git repository:

```bash
# Install locally
pip install -e .

# Install directly from git
pip install git+https://github.com/your-org/fraud-detector.git
```

---

## 💡 Usage Examples

### 1. Headless / Testing Setup (Zero External Dependencies)

Perfect for CI/CD pipelines, local testing, or serverless environments where Redis/Postgres/Kafka are not available.

```python
from prj import FraudDetector
from prj.adapters import (
    InMemoryProfileStore,
    InMemoryCacheStore,
    InMemoryDBStore,
    ConsoleAlertProducer
)

# Initialize in-memory adapters
detector = FraudDetector(
    profile_store=InMemoryProfileStore(),
    cache_store=InMemoryCacheStore(),
    db_store=InMemoryDBStore(),
    alert_producer=ConsoleAlertProducer()
)

# Seed database with user's last verified node
detector.pipeline.db_store.record_login(
    user_id="test_user",
    lat=37.7749,
    lon=-122.4194,
    ts=1600000000.0,
    device_hash="dev_iphone",
    is_verified=True
)

# Analyze an impossible travel event (London, 1 hour later)
result = detector.analyze({
    "user_id": "test_user",
    "latitude": 51.5074,
    "longitude": -0.1278,
    "timestamp": 1600003600.0,
    "device_hash": "dev_iphone"
})

print(result.status)        # IMPOSSIBLE_VELOCITY
print(result.risk_score)    # 100.0
print(result.is_fraudulent) # True
```

### 2. Production Setup (Redis, PostgreSQL, Kafka)

```python
from prj import FraudDetector
from prj.adapters import (
    PostgreSQLProfileStore,
    RedisCacheStore,
    PostgresDBStore,
    KafkaAlertProducer
)

detector = FraudDetector(
    profile_store=PostgreSQLProfileStore(),
    cache_store=RedisCacheStore(),
    db_store=PostgresDBStore(),
    alert_producer=KafkaAlertProducer()
)

result = detector.analyze({
    "user_id": "user_123",
    "latitude": 19.1100,
    "longitude": 72.8300,
    "timestamp": 1780000000.0,
    "device_hash": "dev_abc123"
})
```

---

## 🧪 Verification & Testing

ShieldFlow contains two test suites:

### 1. Internal Package Test Suite
Validates internal module changes:
```bash
venv/bin/pytest
```

### 2. External Library Validation Suite
Runs black-box validation of the package as an external consumer:
```bash
venv/bin/python lib-tests/run_all.py
```
This validation generates test summaries and performance metrics in the `lib-tests/reports/` folder.

---

## 📖 Complete Documentation
For full setup instructions, model schemas, background ML pipeline details, and web server configurations, refer to the [documentation.md](documentation.md) file in the root of this project.

