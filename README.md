# Zilliz Cloud Load Test Tool

A CLI tool for seeding and load testing Zilliz Cloud with configurable QPS and comprehensive metrics collection.

## Features

- **Database Seeding**: Seed your database with 2 million 768-dimension vectors via the insert API
- Configurable QPS levels (default: 100, 500, 1000 QPS)
- Measures P95 and P99 latencies
- Real-time status updates during load tests
- Waits for all in-flight queries to complete before reporting results
- Comprehensive results summary with fired vs completed QPS metrics

## Installation

```bash
go mod download
go build -o zilliz-loadtest
```

## Usage

The tool uses an interactive CLI that starts with a menu:

```bash
./zilliz-loadtest
```

### Menu Options

1. **Seed the database**: Insert 2 million 768-dimension vectors into your collection
2. **Run a read query load test**: Perform load testing with configurable QPS levels

### Seed Database

When you choose option 1, you will be prompted to enter:

1. **API Key** (required): Your Zilliz Cloud API key
2. **Database URL** (required): Your Zilliz Cloud database URL
3. **Collection Name** (required): The collection to seed
4. **Confirmation**: Confirm before starting the seed operation

The seed operation will:
- Insert 2,000,000 vectors of 768 dimensions
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
7. **Duration** (optional, default: 30s): Duration for each QPS test (e.g., `30s`, `1m`)

**Note**: The tool uses the default search level (level 1) which optimizes for latency. This is ideal for pure latency testing.

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

This will insert 2,000,000 vectors of 768 dimensions into the collection.
Do you want to continue? (yes/no): yes

Starting database seed operation
================================
Collection: my_collection
Vector Dimension: 768
Total Vectors: 2000000
Batch Size: 15000

[Progress: 0.0%] Generating batch 1/134 (15000 vectors)...
[Progress: 0.8%] Batch 1/134: Inserted 15000 vectors (Generate: 0.9s, Upload: 1.7s, Total: 2.6s, 5769 vec/s) [ETA: 5m 45s]
[Progress: 1.5%] Generating batch 2/134 (15000 vectors)...
[Progress: 1.5%] Batch 2/134: Inserted 15000 vectors (Generate: 0.8s, Upload: 1.6s, Total: 2.4s, 6250 vec/s) [ETA: 5m 18s]
...
[Progress: 100.0%] Batch 134/134: Inserted 15000 vectors (Generate: 0.9s, Upload: 1.7s, Total: 2.6s, 5769 vec/s) [ETA: 0s]

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
Enter Duration for each QPS test (e.g., 30s, 1m) [default: 30s]: 30s

Starting Zilliz Cloud Load Test
==============================
Database URL: https://your-cluster.zillizcloud.com
Collection: my_collection
Vector Dimension: 768
Metric Type: L2
Duration per QPS: 30s
QPS Levels: [100 500 1000]

--- Running test at 100 QPS for 30s ---
[Status] Elapsed: 5s | Fired: 500 | Completed: 485 | Current QPS: 97.00
[Status] Elapsed: 10s | Fired: 1000 | Completed: 975 | Current QPS: 97.50
[Status] Elapsed: 15s | Fired: 1500 | Completed: 1470 | Current QPS: 98.00
[Status] Elapsed: 20s | Fired: 2000 | Completed: 1965 | Current QPS: 98.25
[Status] Elapsed: 25s | Fired: 2500 | Completed: 2455 | Current QPS: 98.20
[Status] Elapsed: 30s | Fired: 3000 | Completed: 2940 | Current QPS: 98.00

Test duration ended. Waiting for 60 in-flight queries to complete...
  Waiting... 2950 queries completed so far
  Waiting... 2995 queries completed so far
All queries completed
Fired: 3000 queries | Completed: 3000 queries in 30.5s
Fired at: 100 QPS | Completed at: 98.36 QPS | Errors: 0

--- Running test at 500 QPS for 30s ---
[Status] Elapsed: 5s | Fired: 2500 | Completed: 2400 | Current QPS: 480.00
[Status] Elapsed: 10s | Fired: 5000 | Completed: 4850 | Current QPS: 485.00
[Status] Elapsed: 15s | Fired: 7500 | Completed: 7300 | Current QPS: 486.67
[Status] Elapsed: 20s | Fired: 10000 | Completed: 9750 | Current QPS: 487.50
[Status] Elapsed: 25s | Fired: 12500 | Completed: 12200 | Current QPS: 488.00
[Status] Elapsed: 30s | Fired: 15000 | Completed: 14650 | Current QPS: 488.33

Test duration ended. Waiting for 350 in-flight queries to complete...
  Waiting... 14750 queries completed so far
  Waiting... 14900 queries completed so far
All queries completed
Fired: 15000 queries | Completed: 15000 queries in 30.2s
Fired at: 500 QPS | Completed at: 496.69 QPS | Errors: 0

--- Running test at 1000 QPS for 30s ---
[Status] Elapsed: 5s | Fired: 5000 | Completed: 4800 | Current QPS: 960.00
[Status] Elapsed: 10s | Fired: 10000 | Completed: 9600 | Current QPS: 960.00
[Status] Elapsed: 15s | Fired: 15000 | Completed: 14400 | Current QPS: 960.00
[Status] Elapsed: 20s | Fired: 20000 | Completed: 19200 | Current QPS: 960.00
[Status] Elapsed: 25s | Fired: 25000 | Completed: 24000 | Current QPS: 960.00
[Status] Elapsed: 30s | Fired: 30000 | Completed: 28800 | Current QPS: 960.00

Test duration ended. Waiting for 1200 in-flight queries to complete...
  Waiting... 29000 queries completed so far
  Waiting... 29950 queries completed so far
All queries completed
Fired: 30000 queries | Completed: 29998 queries in 30.1s
Fired at: 1000 QPS | Completed at: 996.68 QPS | Errors: 2
Note: 2 queries were still in flight when test ended (100.0% completion rate)

==============================
Load Test Results Summary
==============================

QPS        | P95 (ms)     | P99 (ms)     | Total Queries
-----------+--------------+--------------+---------------
100        | 45.23        | 67.89        | 3000
500        | 52.45        | 89.12        | 15000
1000       | 78.34        | 125.67       | 29998
```

## Notes

### Database Seeding

- **Seed Operation**: The seed function inserts 2 million vectors of 768 dimensions:
  - Batch size: 15,000 vectors per batch (~45MB, well under 64MB gRPC limit)
  - Uses `Insert` API since autoID is enabled and no primary key is required
  - Provides real-time progress updates with ETA and performance metrics
  - Sequential processing to avoid overwhelming the server and reduce write errors
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
- **Search Level**: The tool uses the default search level (level 1), which optimizes for latency. This is ideal for pure latency testing. The level parameter is not configurable in this version.
- **QPS Measurement**: The tool fires queries at the exact target QPS rate, regardless of completion time. This allows you to measure latency at the intended load level. The results show both "Fired at" (target QPS) and "Completed at" (actual completion rate) metrics.
- **Status Updates**: During the test, status updates are printed every 5 seconds showing elapsed time, fired queries, completed queries, and current QPS.
- **In-Flight Queries**: After the test duration ends, the tool waits for all in-flight queries to complete before reporting final results. This ensures accurate latency measurements even if queries take longer than the test duration.
- **Latency Metrics**: P95 and P99 latencies are calculated from all completed queries, providing accurate percentile measurements.
- Ensure your collection is loaded and contains data before running the load test.

### General

- The SDK method signatures may vary by version. If you encounter compilation errors, you may need to adjust the method calls in `loadtest.go` to match your SDK version.
- The tool supports both `client.NewClient` and `client.NewGrpcClient` initialization methods for compatibility with different SDK versions.
