# Zilliz Cloud Load Test Tool

A CLI tool for load testing Zilliz Cloud with configurable QPS and comprehensive metrics collection.

## Features

- Configurable QPS levels (default: 100, 500, 1000 QPS)
- Measures P95 and P99 latencies
- Tracks recall accuracy from Zilliz Cloud responses
- Real-time progress reporting
- Comprehensive results summary

## Installation

```bash
cd /Users/jaybyoun/developer/zilliz-loadtest
go mod download
go build -o zilliz-loadtest
```

## Usage

The tool uses an interactive CLI that prompts for each parameter:

```bash
./zilliz-loadtest
```

You will be prompted to enter:

1. **API Key** (required): Your Zilliz Cloud API key
2. **Database URL** (required): Your Zilliz Cloud database URL
3. **Collection Name** (required): The collection to test
4. **QPS Levels** (required): Comma-separated QPS values to test (e.g., `100,500,1000`)
5. **Level** (optional, default: 5): Integer from 1-10 where:
   - Level 1: Optimizes for latency (faster searches)
   - Level 10: Optimizes for recall (more accurate results)
6. **Duration** (optional, default: 30s): Duration for each QPS test (e.g., `30s`, `1m`)
7. **Vector Dimension** (optional, default: 128): Vector dimension for query vectors

## Example Output

```
Zilliz Cloud Load Test Configuration
====================================

Enter API Key: your_api_key_here
Enter Database URL: https://your-cluster.zillizcloud.com
Enter Collection Name: my_collection

Enter QPS levels to test (comma-separated, e.g., 100,500,1000):
QPS Levels: 100,500,1000
Enter Level (1-10, where 10 optimizes for recall): 7
Enter Duration for each QPS test (e.g., 30s, 1m) [default: 30s]: 30s
Enter Vector Dimension [default: 128]: 128

Starting Zilliz Cloud Load Test
==============================
Database URL: https://your-cluster.zillizcloud.com
Collection: my_collection
Level: 7
Duration per QPS: 30s
QPS Levels: [100 500 1000]

--- Running test at 100 QPS for 30s ---
Completed: 3000 queries in 30.5s (actual QPS: 98.36, errors: 0)

--- Running test at 500 QPS for 30s ---
Completed: 15000 queries in 30.2s (actual QPS: 496.69, errors: 0)

--- Running test at 1000 QPS for 30s ---
Completed: 30000 queries in 30.1s (actual QPS: 996.68, errors: 2)

==============================
Load Test Results Summary
==============================

QPS        | P95 (ms)     | P99 (ms)     | Avg Recall     | Total Queries
-----------+--------------+--------------+----------------+---------------
100        | 45.23        | 67.89        | 0.9500         | 3000
500        | 52.45        | 89.12        | 0.9498         | 15000
1000       | 78.34        | 125.67       | 0.9495         | 30000
```

## Notes

- The tool uses random query vectors for testing. In production, you may want to use actual query vectors from your dataset.
- Vector dimensions default to 128 but can be configured during the interactive prompt.
- **Level Parameter**: The level parameter (1-10) controls the trade-off between recall and latency:
  - Lower levels (1-3): Faster searches, potentially lower recall
  - Medium levels (4-7): Balanced performance
  - Higher levels (8-10): Better recall, potentially higher latency
- **Recall Measurement**: Recall is measured during the test run based on search results. For accurate recall measurement, you would need ground truth data to compare against.
- The SDK method signatures may vary by version. If you encounter compilation errors, you may need to adjust the `Search` method call in `loadtest.go` to match your SDK version.
- Ensure your collection is loaded and contains data before running the load test.
- The tool supports both `client.NewClient` and `client.NewGrpcClient` initialization methods for compatibility with different SDK versions.
