# Zilliz Cloud Load Test Tool

A CLI tool for seeding and load testing Zilliz Cloud with configurable QPS and comprehensive metrics collection.

## Features

- **Database Seeding**: Seed your database with 2 million 768-dimension vectors via the upsert API
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

The tool uses an interactive CLI that starts with a menu:

```bash
./zilliz-loadtest
```

### Menu Options

1. **Seed the database**: Upsert 2 million 768-dimension vectors into your collection
2. **Run a read query load test**: Perform load testing with configurable QPS levels

### Seed Database

When you choose option 1, you will be prompted to enter:

1. **API Key** (required): Your Zilliz Cloud API key
2. **Database URL** (required): Your Zilliz Cloud database URL
3. **Collection Name** (required): The collection to seed
4. **Confirmation**: Confirm before starting the seed operation

The seed operation will:
- Upsert 2,000,000 vectors of 768 dimensions
- Use batches of 50,000 vectors for efficient insertion
- Display progress and insertion rate
- Assume autoID is enabled (no primary key required)
- Assume no scalar fields (only vector data)

### Load Test

When you choose option 2, you will be prompted to enter:

1. **API Key** (required): Your Zilliz Cloud API key
2. **Database URL** (required): Your Zilliz Cloud database URL
3. **Collection Name** (required): The collection to test
4. **Vector Dimension** (required): Vector dimension for query vectors
5. **Metric Type** (required): Distance metric type - one of:
   - `L2`: Euclidean distance (L2 norm)
   - `IP`: Inner Product
   - `COSINE`: Cosine similarity
6. **QPS Levels** (required): Comma-separated QPS values to test (e.g., `100,500,1000`)
7. **Level** (optional, default: 5): Integer from 1-10 where:
   - Level 1: Optimizes for latency (faster searches)
   - Level 10: Optimizes for recall (more accurate results)
8. **Duration** (optional, default: 30s): Duration for each QPS test (e.g., `30s`, `1m`)

## Example Output

### Seed Database Example

```
Zilliz Cloud Load Test Tool
===========================

What would you like to do?
1. Seed the database
2. Run a read query load test

Enter your choice (1 or 2): 1

Database Seed Configuration
===========================

Enter API Key: your_api_key_here
Enter Database URL: https://your-cluster.zillizcloud.com
Enter Collection Name: my_collection
Vector Dimension: 768 (fixed for seed operation)
Total Vectors: 2000000 (fixed for seed operation)

This will upsert 2,000,000 vectors of 768 dimensions into the collection.
Do you want to continue? (yes/no): yes

Starting database seed operation
================================
Collection: my_collection
Vector Dimension: 768
Total Vectors: 2000000
Batch Size: 50,000

Batch 1/40: Inserted 50000 vectors in 2.3s (21739 vectors/sec) [Total: 50000/2000000]
Batch 2/40: Inserted 50000 vectors in 2.1s (23810 vectors/sec) [Total: 100000/2000000]
...
Batch 40/40: Inserted 50000 vectors in 2.2s (22727 vectors/sec) [Total: 2000000/2000000]

================================
Seed operation completed!
Total vectors inserted: 2000000
Total time: 1m 32s
Average rate: 21739 vectors/sec
================================
```

### Load Test Example

```
Zilliz Cloud Load Test Tool
===========================

What would you like to do?
1. Seed the database
2. Run a read query load test

Enter your choice (1 or 2): 2

Load Test Configuration
=======================

Enter API Key: your_api_key_here
Enter Database URL: https://your-cluster.zillizcloud.com
Enter Collection Name: my_collection
Enter Vector Dimension: 768

Metric Type options: L2, IP (Inner Product), COSINE
Enter Metric Type: L2

Enter QPS levels to test (comma-separated, e.g., 100,500,1000):
QPS Levels: 100,500,1000
Enter Level (1-10, where 10 optimizes for recall) [default: 5]: 7
Enter Duration for each QPS test (e.g., 30s, 1m) [default: 30s]: 30s

Starting Zilliz Cloud Load Test
==============================
Database URL: https://your-cluster.zillizcloud.com
Collection: my_collection
Vector Dimension: 768
Metric Type: L2
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

### Database Seeding

- **Seed Operation**: The seed function upserts 2 million vectors of 768 dimensions in batches of 50,000
- **AutoID**: The seed operation assumes autoID is enabled, so no primary key is required
- **No Scalar Fields**: The seed operation only inserts vector data (no scalar fields)
- **Collection Requirements**: Ensure your collection exists and is configured with:
  - A vector field named "vector" with 768 dimensions
  - AutoID enabled for the primary key
  - Appropriate index created (the tool will use AUTOINDEX with level parameter during load tests)

### Load Testing

- The tool uses random query vectors for testing. In production, you may want to use actual query vectors from your dataset.
- **Vector Dimension**: Required parameter that must match the dimension of vectors in your collection.
- **Metric Type**: Required parameter that must match the metric type used when creating the collection index. Common options are L2 (Euclidean distance), IP (Inner Product), and COSINE (Cosine similarity).
- **Level Parameter**: The level parameter (1-10) controls the trade-off between recall and latency:
  - Lower levels (1-3): Faster searches, potentially lower recall
  - Medium levels (4-7): Balanced performance
  - Higher levels (8-10): Better recall, potentially higher latency
- **Recall Measurement**: Recall is automatically calculated by Zilliz Cloud when `enable_recall_calculation` is enabled in the search parameters. The tool uses this feature to get accurate recall rates for each query. See the [Zilliz documentation](https://docs.zilliz.com/docs/tune-recall-rate#tune-recall-rate) for more details.
- Ensure your collection is loaded and contains data before running the load test.

### General

- The SDK method signatures may vary by version. If you encounter compilation errors, you may need to adjust the method calls in `loadtest.go` to match your SDK version.
- The tool supports both `client.NewClient` and `client.NewGrpcClient` initialization methods for compatibility with different SDK versions.
