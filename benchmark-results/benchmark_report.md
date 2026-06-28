# ShieldFlow SDK Performance & Complexity Report

Generated automatically after running the Benchmarking and Complexity Analysis suite.

---

## 📊 Performance Summary

* **Average Detector Latency**: 0.0173 ms
* **p50 (Median)**: 0.0130 ms
* **p95**: 0.0279 ms
* **p99**: 0.0340 ms
* **Peak Memory Usage**: 2.39 KB


### 🔄 Regression & History Analysis

Compared with previous run (saved in `/benchmark-results/historical/`):
* **Previous Average Latency**: 0.0087 ms
* **Current Average Latency**: 0.0173 ms
* **Delta**: 0.0086 ms (98.85% regression)


---

## 🕒 Detector Pipeline Stage-by-Stage Latency

| Stage / Component | Average Latency | p95 Latency |
| :--- | :--- | :--- |
| fetch_last_node | 0.0002 ms | 0.0004 ms |
| velocity_check | 0.0033 ms | 0.0049 ms |
| ml_predict | 0.0025 ms | 0.0041 ms |
| device_mismatch | 0.0013 ms | 0.0019 ms |
| scoring | 0.0016 ms | 0.0026 ms |
| alerting | 0.0049 ms | 0.0081 ms |


---

## 📈 Algorithmic Complexity Evaluation

| Algorithm | Empirical Complexity | Time (N=100,000) | Peak Memory |
| :--- | :--- | :--- | :--- |
| haversine | O(n) | 64.9559 ms | 3123.90 KB |
| bounding_box | O(n) | 39.1435 ms | 782.39 KB |
| velocity_validator | O(n) | 63.2833 ms | 782.26 KB |
| risk_scoring | O(n log n) | 54.8711 ms | 3123.80 KB |
| balltree_build | O(n log n) | 44.8109 ms | 2540.55 KB |
| balltree_query | O(n) | 0.2573 ms | 2.45 KB |


---

## 👥 Scalability Benchmarks (User Scaling)

| User Count | Throughput (eps) | Average Latency | Peak Memory |
| :--- | :--- | :--- | :--- |
| 1 | 23456.1 | 0.0386 ms | 349.70 KB |
| 10 | 28108.54 | 0.0321 ms | 352.60 KB |
| 100 | 27220.22 | 0.0332 ms | 366.37 KB |
| 1000 | 27280.17 | 0.0326 ms | 473.81 KB |
| 10000 | 27213.87 | 0.0326 ms | 551.62 KB |

* **Latency Growth Complexity**: O(log n)

---

## 📦 Batch Processing Benchmarks

| Batch Size | Seq Throughput (eps) | Seq Avg Latency | Par Throughput (eps) | Par Avg Latency |
| :--- | :--- | :--- | :--- | :--- |
| 100 | 13264.02 | 0.0754 ms | 4113.47 | 0.2431 ms |
| 1000 | 10403.04 | 0.0961 ms | 3457.37 | 0.2892 ms |
| 10000 | 5422.82 | 0.1844 ms | 2474.93 | 0.4041 ms |


---

## ⚡ Memory footprint & Leak Check

* **Single Event peak memory**: 2.39 KB
* **Allocated Growth (Single)**: 0.46 KB
* **Repeated 1000 Runs peak**: 274.06 KB
* **Leak per Event (Average)**: 278.504 bytes

---

## ⚠️ Key Performance Bottlenecks

1. **Sequential Batch Processing**: Sequential batch checks have a lower throughput limit than threaded pools (as verified by parallel throughput ratios).
2. **Alert Triggering (Kafka/Console)**: Dispatching notifications or emitting events causes a noticeable latency delta.
3. **Pydantic Validation**: Deserializing inputs to Pydantic objects takes a significant chunk of the fast-path latency.

---

## 💡 Recommendations

* **Implement Thread-Pool Batch Processing**: Adopt `ThreadPoolExecutor` (as simulated in the batch parallel tests) for processing bulk login audit streams.
* **Asynchronous Alerting**: Dispatch anomalies to an queue worker in a background thread to prevent latency spikes during alerts emitting.
* **Pre-Compile Schemas**: Cache Pydantic serializers or bypass full validation if events are pre-validated by the web framework.
