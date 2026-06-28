"""Master test runner for the ShieldFlow SDK validation suite."""

import os
import sys
import json
import time
from pathlib import Path
import pytest

class ResultCollector:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1
                self.errors.append({
                    "nodeid": report.nodeid,
                    "message": str(report.longrepr)
                })
            elif report.skipped:
                self.skipped += 1

def run_suite():
    print("=" * 60)
    print("🚀 STARTING SHIELDFLOW SDK EXTERNAL VALIDATION SUITE")
    print("=" * 60)
    
    start_time = time.perf_counter()
    
    # Run tests using pytest programmatically
    collector = ResultCollector()
    exit_code = pytest.main(["-q", "--tb=short", "lib-tests/"], plugins=[collector])
    
    total_elapsed = time.perf_counter() - start_time
    total_tests = collector.passed + collector.failed + collector.skipped
    
    # Read performance results if available
    perf_data = {}
    perf_report_path = Path("lib-tests/reports/performance_report.json")
    if perf_report_path.exists():
        with open(perf_report_path, "r") as f:
            perf_data = json.load(f)
            
    # Read latency distribution results if available
    latency_data = {}
    latency_report_path = Path("lib-tests/reports/latency_distribution.json")
    if latency_report_path.exists():
        with open(latency_report_path, "r") as f:
            latency_data = json.load(f)

    # Compile master report
    master_report = {
        "summary": {
            "total_tests": total_tests,
            "passed": collector.passed,
            "failed": collector.failed,
            "skipped": collector.skipped,
            "total_elapsed_seconds": round(total_elapsed, 4),
            "exit_code": int(exit_code)
        },
        "performance": perf_data,
        "latency_percentiles": latency_data,
        "failures": collector.errors
    }
    
    # Write JSON report
    reports_dir = Path("lib-tests/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "master_report.json", "w") as f:
        json.dump(master_report, f, indent=2)
        
    # Write Markdown report
    write_markdown_report(master_report, reports_dir / "master_report.md")
    
    # Print console summary
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUITE RUN SUMMARY")
    print("=" * 60)
    print(f"Total Tests Run: {total_tests}")
    print(f"Passed:          {collector.passed}")
    print(f"Failed:          {collector.failed}")
    print(f"Skipped:         {collector.skipped}")
    print(f"Total Time:      {total_elapsed:.2f} seconds")
    
    if latency_data:
        print("-" * 60)
        print(f"Average Latency: {latency_data.get('avg')} ms")
        print(f"p50 Latency:      {latency_data.get('p50')} ms")
        print(f"p95 Latency:      {latency_data.get('p95')} ms")
        print(f"p99 Latency:      {latency_data.get('p99')} ms")
        print(f"Throughput:      {latency_data.get('throughput')} rps")
        
    print("=" * 60)
    
    if collector.failed > 0:
        print("\n❌ SDK VALIDATION FAILED!")
        for err in collector.errors:
            print(f"- {err['nodeid']}: {err['message'].splitlines()[-1] if err['message'] else 'Unknown error'}")
        sys.exit(1)
    else:
        print("\n✅ SDK VALIDATION PASSED SUCCESSFULLY!")
        sys.exit(0)

def write_markdown_report(report, path):
    sum_data = report["summary"]
    lat_data = report["latency_percentiles"]
    perf_data = report["performance"]
    
    md_content = f"""# ShieldFlow SDK Library Validation Report

Generated automatically after running the external black-box validation suite.

---

## 📊 Summary

* **Total Tests Run**: {sum_data['total_tests']}
* **Passed**: {sum_data['passed']}
* **Failed**: {sum_data['failed']}
* **Skipped**: {sum_data['skipped']}
* **Total Time**: {sum_data['total_elapsed_seconds']} seconds
* **Status**: {"❌ FAILED" if sum_data['failed'] > 0 else "✅ PASSED"}

---

## ⚡ Latency & Throughput (In-Memory Setup)

* **Average Latency**: {lat_data.get('avg', 'N/A')} ms
* **p50 (Median)**: {lat_data.get('p50', 'N/A')} ms
* **p95**: {lat_data.get('p95', 'N/A')} ms
* **p99**: {lat_data.get('p99', 'N/A')} ms
* **Throughput**: {lat_data.get('throughput', 'N/A')} rps

---

## 📈 Batch Benchmarks

| Batch Size | Elapsed Time (s) | Throughput (eps) | Avg Latency (ms) |
| :--- | :--- | :--- | :--- |
"""
    for key, val in perf_data.items():
        md_content += f"| {val['size']} | {val['elapsed_seconds']} | {val['throughput_eps']} | {val['average_latency_ms']} |\n"
        
    if report["failures"]:
        md_content += "\n---\n\n## ❌ Failures Detail\n\n"
        for fail in report["failures"]:
            md_content += f"### {fail['nodeid']}\n```\n{fail['message']}\n```\n\n"

    with open(path, "w") as f:
        f.write(md_content)

if __name__ == "__main__":
    run_suite()
