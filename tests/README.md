# ShieldFlow SDK Test Suite Guide

This directory contains the internal unit and integration tests for the ShieldFlow Fraud Detector SDK.

---

## 🚀 1. Running the Test Suite

### 1.1 Run All Tests
To run the full suite of unit and integration tests, use `pytest` within your activated virtual environment:

```bash
# Run all tests using the virtual environment's pytest
venv/bin/pytest
```

### 1.2 Run Specific Test Categories
You can target specific folders to test modules in isolation:

```bash
# Run unit tests only (mathematical algorithms, headless validation)
venv/bin/pytest tests/unit/

# Run integration tests only (database persistence, caching flow)
venv/bin/pytest tests/integration/

# Run ML mathematical tests only (DBSCAN & distance matrices)
venv/bin/pytest tests/ml/
```

### 1.3 Run a Specific Test File
```bash
venv/bin/pytest tests/unit/test_headless_sdk.py
```

---

## 🏗️ 2. Test Architecture & Coverage

The test suite is structured as follows:
*   **`tests/unit/`**: Validates the SDK engine algorithms, including coordinate boundary validation, velocity verification, risk scoring, and headless SDK operations.
*   **`tests/integration/`**: Verifies database-backed persistence, Redis cache hydration, and table auto-initialization side-effects.
*   **`tests/ml/`**: Checks pairwise distance matrix parity and DBSCAN helper functions.

---

## 🔍 3. External Library Validation Suite

In addition to the internal tests in this directory, there is an **External Library Validation Suite** located in the `lib-tests/` directory at the project root. This suite runs black-box execution simulating how a client application imports and integrates the SDK as an installed package.

To run the external validation suite:
```bash
venv/bin/python lib-tests/run_all.py
```
This will run integration benchmarks and output latency reports under `lib-tests/reports/`.
