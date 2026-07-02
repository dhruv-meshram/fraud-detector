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

### 4.1 Geolocation Responsibility Policy
The ShieldFlow SDK is **completely independent of any external geolocation database or service** (such as MaxMind, GeoLite, IPinfo, or ipapi). 

* **Caller Responsibility**: The host application must resolve the client's IP address to geographic coordinates (`latitude` and `longitude`) using its preferred geolocation provider before invoking the SDK.
* **Coordinate Requirements**: The SDK expects valid geographic coordinates. Input coordinates are validated strictly:
  * `latitude` must be a float/int between `-90.0` and `90.0`.
  * `longitude` must be a float/int between `-180.0` and `180.0`.
  * Invalid coordinate values or types will immediately raise a `ValueError`.

#### Example Workflow:
```text
User Login
    ↓
Host Application
    ↓
Resolve IP → Coordinates (using any geolocation provider)
    ↓
Call fraud_detector.check_fraud SDK
    ↓
SDK stores event (with original IP & resolved coordinates)
    ↓
SDK runs spatial fraud detection algorithms
    ↓
SDK returns fraud score (0 to 1)
```

### 4.2 Exposed Entry Points (`fraud_detector`)
You can import the main validation function directly from the package root:
```python
from fraud_detector import check_fraud, FraudDetector
```

### 4.3 Public API: `check_fraud`
```python
def check_fraud(
    db_conn = None,
    user_id: str = None,
    device_fingerprint: str = None,
    ip_address: str = "unknown",
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    event: Optional[Dict[str, Any]] = None
) -> float:
```
* **Parameters**:
  * `db_conn`: SQLAlchemy database connection or engine (defaults to an in-memory SQLite database).
  * `user_id`: Unique identifier for the user.
  * `device_fingerprint`: Client device hash fingerprint.
  * `ip_address`: Original client IP address (stored for auditing and logs).
  * `latitude`: Decimal latitude of the request.
  * `longitude`: Decimal longitude of the request.
  * `event`: Optional dict containing metadata (such as `timestamp`).
* **Returns**:
  * `fraud_score`: A float between `0.0` (legitimate) and `1.0` (high risk).

### 4.4 Model Schemas
If using the object-oriented API directly:
```python
from fraud_detector import FraudDetector
from fraud_detector.models.event import LoginEvent

# Direct LoginEvent definition
class LoginEvent(BaseModel):
    user_id: str
    latitude: float
    longitude: float
    timestamp: float
    device_hash: str
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
