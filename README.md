# ShieldFlow Anomaly & Fraud Detection Engine

ShieldFlow is a context-aware, low-latency login anomaly and fraud detection engine designed to identify credentials misuse, session hijacking, and impossible-travel events. It operates on a bifurcated architecture that separates deterministic check paths from heavy machine learning clustering.

---

## 🏗️ Architecture

ShieldFlow partitions its detection logic into two paths to guarantee sub-millisecond real-time responses:

1. **Fast Path (Live Inference)**
   - Operates in real-time at the edge/gateway layer (FastAPI & Redis).
   - Runs fast rule-based validations, such as Spatiotemporal Velocity (impossible travel checks) and cached boundary checks using centroid-distance lookups.

2. **Slow Path (Offline Batch Training)**
   - Runs asynchronously in the background.
   - Sanitizes historical authenticated logins (Null coordinate drop, 10s deduplication, outlier filtering).
   - Trains unsupervised **DBSCAN** models per user to identify habitual spatial boundaries and centroids, avoiding spherical assumptions and auto-filtering noise.

---

## 📂 Project Structure

```directory
├── algorithms/             # Hardcoded fast-path detection mathematics
│   ├── spatial/            # Haversine distance calculators
│   └── graph/              # Spatiotemporal travel velocity models
├── app/                    # FastAPI main edge application
├── configs/                # Environment-specific configuration files
├── core/                   # Shared DDD domain entities & schemas
├── data/                   # Data directory
│   ├── raw/                # Synthetic raw logins (synthetic_logins.csv)
│   └── processed/          # Clean preprocessed logins (clean_logins.csv)
├── deployment/             # Docker, PostgreSQL, Redis configurations
├── docs/                   # ML and System design documentation
├── ml/                     # Machine learning workflows (Slow Path)
│   ├── datasets/           # Chronological queue-based generators
│   ├── features/           # Feature extraction and Radian converters
│   ├── models/             # PyTorch-accelerated DBSCAN architectures
│   └── preprocessing/      # Cleaning and 10s window deduplication
└── services/               # Internal orchestrators
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- `venv` (Python Virtual Environment package)

### 2. Setup Virtual Environment & Install Dependencies
Initialize the virtual environment and install standard requirements:
```bash
# Create local virtual environment
python3 -m venv venv

# Upgrade package manager
venv/bin/pip install --upgrade pip

# Install dependencies (Pandas, Numpy)
venv/bin/pip install pandas
```

---

## ⚙️ How to Run the Pipeline

### Step 1: Generate Labeled Synthetic Dataset
ShieldFlow uses a queue-based chronologically streamed generation algorithm to create 75,000 login records across 1,000 distinct user profiles:
```bash
python3 ml/datasets/synthetic_generator.py
```
*Output File:* `data/raw/synthetic_logins.csv`

### Step 2: Clean & Preprocess Data
Run the sanitization pipeline to apply coordinate dropouts, sort logs per user ascending, deduplicate same-device logins within a 10s window, and prune impossible travel outliers:
```bash
venv/bin/python ml/preprocessing/run_pipeline.py
```
*Output File:* `data/processed/clean_logins.csv`
