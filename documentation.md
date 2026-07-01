# ShieldFlow Anomaly & Fraud Detection Engine: SDK Documentation

This document serves as the comprehensive user and developer guide for the ShieldFlow Fraud Detection SDK. It covers architectural principles, directory layout, exposed public APIs, setup instructions, validation suites, and execution steps.

---

## 📖 1. Project Overview & Background

ShieldFlow is a context-aware, low-latency login anomaly and fraud detection engine designed to identify credentials misuse, session hijacking, and impossible-travel anomalies. 

The codebase was originally a monolith tied directly to standard FastAPI endpoints, PostgreSQL databases, Redis cache servers, and Kafka clusters. It has been refactored into a **reusable Python SDK package (`fraud_detector`)** that decouples the engine's core logic from infrastructure. Using the **Adapter Pattern**, external consumers can inject custom database, cache, profile, and queue brokers to execute fraud checks in any environment.

---

## 🏗️ 2. Architectural Design

ShieldFlow partitions its check paths to support sub-millisecond live checks while training complex ML models in the background:

### 2.1 Live Inference Path (Fast Path)
Executed in real-time on incoming login events:
1. **Spatiotemporal Velocity Verification**: Evaluates travel speed between the user's current login and their last verified login node. Speed > 900 km/h triggers immediate `IMPOSSIBLE_VELOCITY` flags.
2. **Cold Start Bypass**: If the user has fewer than 10 historic verified logins, rules-based evaluations are bypassed to avoid false positives on new profiles.
3. **Centroid-Distance Classification**: The user's coordinates are matched against precalculated DBSCAN spatial cluster profiles. If the coordinate lies outside the radius of all clusters, it is classified as a spatial `OUTLIER`.
4. **Multi-Factor Risk Score**: Synthesizes spatial outliers, distance offsets, and client device fingerprint mismatches into a final risk percentage.

### 2.2 Offline Batch Training (Slow Path)
Executed asynchronously in the background:
1. **Deduplication**: Drops same-device logins within a 10-second sliding window.
2. **DBSCAN Clustering**: Identifies spatial hubs representing home/office regions, establishing user specific centroid locations and dynamic boundaries.
3. **Profile Generation**: Saves centroids and allowed radii as JSON profiles for the Fast Path.

### 2.3 Adapter-Driven Design
All state reads and writes are managed through abstract base interfaces defined in `fraud_detector/adapters/base.py`:
*   `BaseProfileStore`: Retrieves spatial profiles.
*   `BaseCacheStore`: Caches real-time session coordinates.
*   `BaseDBStore`: Persists login audit events.
*   `BaseAlertProducer`: Dispatches real-time Kafka alerts.

---

## 📂 3. Directory Layout

```directory
├── fraud_detector/           # Reusable SDK Core Package
│   ├── __init__.py           # Main SDK namespace exports
│   ├── engine/               # Pipeline orchestrators
│   │   ├── detector.py       # FraudDetector public API
│   │   └── pipeline.py       # Core processing logic
│   ├── models/               # Pydantic v2 domain schemas
│   ├── algorithms/           # Pure stateless mathematical utilities
│   │   ├── spatial/           # Haversine distance & velocity math
│   │   └── ranking/           # Multi-factor score calculator
│   ├── ml/                   # DBSCAN ML training & inference pipeline
│   ├── adapters/             # Infrastructure adapters (Redis, Postgres, Kafka, In-Memory)
│   ├── configs/              # Deployment resource configuration files
│   ├── deployment/           # Worker Dockerfile and compose definitions
│   ├── data/                 # Raw/processed/external datasets & cache
│   └── scripts/              # CLI scripts and utilities
├── tests/                    # Internal SDK unit and integration test suite
├── benchmarks/               # Performance and load benchmarking suite
├── lib-tests/                # External verification and validation test suite
├── pyproject.toml            # Package metadata & build configuration
└── README.md                 # Project README
```

---

## 💡 4. Public SDK Interface Reference

### 4.1 Exposed Entry Points (`fraud_detector`)
You can import the core orchestrator, standard models, and custom stores directly from the package root:
```python
from fraud_detector import FraudDetector, LoginEvent, FraudResult
```

### 4.2 Initializing the Detector
The constructor accepts customizable storage adapter overrides. If left empty, default adapters are instantiated.

```python
from fraud_detector import FraudDetector
from fraud_detector.adapters import (
    InMemoryProfileStore,
    InMemoryCacheStore,
    InMemoryDBStore,
    ConsoleAlertProducer
)

# Headless / Testing Configuration
detector = FraudDetector(
    profile_store=InMemoryProfileStore(),
    cache_store=InMemoryCacheStore(),
    db_store=InMemoryDBStore(),
    alert_producer=ConsoleAlertProducer()
)
```

### 4.3 Input Model: `LoginEvent`
```python
class LoginEvent(BaseModel):
    user_id: str
    latitude: float
    longitude: float
    timestamp: float
    device_hash: str
```

### 4.4 Output Model: `FraudResult`
```python
class FraudResult(BaseModel):
    risk_score: float         # Risk percentage (0.0 to 100.0)
    is_fraudulent: bool       # Final verdict threshold flag
    reasons: List[str]        # Plain-text reasons
    status: str               # Status code: KNOWN_ZONE, OUTLIER, IMPOSSIBLE_VELOCITY, COLD_START_BYPASS
    details: RiskBreakdown
```

---

## 🚀 5. Getting Started & Installation

### 5.1 Set Up Virtual Environment & Dependencies
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### 5.2 Install the SDK Package
Install the SDK in editable mode so changes are reflected in local applications:
```bash
pip install -e .
```

---

## ⚙️ 6. Running the System Pipelines

### Step 1: Generate Synthetic Dataset
Stream chronological mock events (75,000 logins over 1,000 users):
```bash
python fraud_detector/ml/datasets/synthetic_generator.py
```
*Output:* `fraud_detector/data/raw/synthetic_logins.csv`

### Step 2: Clean & Preprocess Data
Sanitize coordinate dropouts, apply 10s window deduplication, and filter outlier inputs:
```bash
python fraud_detector/ml/preprocessing/run_pipeline.py
```
*Output:* `fraud_detector/data/processed/clean_logins.csv`

### Step 3: Run Model Training & Profile Retraining
Process the preprocessed records and generate DBSCAN cluster profiles for all users:
```bash
python fraud_detector/ml/training/train_pipeline.py
```
*Output:* User JSON profiles generated in `fraud_detector/data/processed/profiles/`

---

## 🧪 7. Executing the Test Suites

### 7.1 Internal Package Test Suite
Runs all tests inside the package development directory:
```bash
venv/bin/pytest
```

### 7.2 External Library Validation Suite
Runs the comprehensive black-box validation suite which simulates third-party package imports:
```bash
venv/bin/python lib-tests/run_all.py
```
This suite automatically generates summary and benchmark metrics:
*   `lib-tests/reports/master_report.json`
*   `lib-tests/reports/master_report.md`
