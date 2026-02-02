# Zilliz Cloud Load Test Tool

A comprehensive CLI tool for seeding and load testing Zilliz Cloud with configurable QPS, enhanced metrics collection, pre-flight validation, and result export capabilities.

## Features

- **Database Seeding**: Seed your database with synthetic or real embeddings
  - **Synthetic Vectors**: Random vectors for quick testing
  - **Real Embeddings**: BEIR dataset with Cohere embeddings (8.84M passages) for realistic performance testing
- **Pre-flight Validation**: Automatic collection validation before tests (existence, schema, data, indexes)
- **Enhanced Metrics**: P50, P90, P95, P99 latencies, min/max/avg, error categorization, success rates
- **Configurable QPS Levels**: Test multiple QPS levels in a single run (default: 100, 500, 1000 QPS)
- **Real-time Status Updates**: Progress updates every 5 seconds during load tests
- **Warm-up Period**: Configurable warm-up queries to stabilize connections and caches
- **Error Categorization**: Automatic classification of errors (network, API, timeout, SDK)
- **Result Export**: Export results to JSON and CSV formats for analysis
- **Configuration File Support**: YAML config file with environment variable overrides
- **Configurable Search Parameters**: Customize topK, filter expressions, search level, and output fields
- **Comprehensive Reporting**: Detailed results summary with error breakdowns
- **Command-Line Interface**: Non-interactive mode with flags for automation and scripting
- **Structured Logging**: JSON or text format logging with configurable levels
- **Profiling Support**: CPU and memory profiling for performance analysis
- **Retry Logic**: Automatic retry for transient errors with exponential backoff
- **Input Validation**: Comprehensive validation with clear error messages
- **Test Comparison**: Compare multiple test runs to identify improvements/regressions

## Installation

```bash
go mod download
go build -o zilliz-loadtest ./cmd/zilliz-loadtest
```

## Quick Start

1. **Create a configuration file** (recommended for easier use):
   
   The tool automatically looks for the config file at `./configs/config.yaml` (relative to where you run the command).
   
   **Setup:**
   ```bash
   # Copy the example config to the required location
   cp configs/config.yaml.example configs/config.yaml
   
   # Edit the config file with your API key and database URL
   # Note: configs/config.yaml is in .gitignore for security - it contains sensitive credentials
   ```
   
   You can also specify a custom config path using the `--config` flag:
   ```bash
   ./zilliz-loadtest --config /path/to/custom/config.yaml
   ```

2. **Or use environment variables**:
   ```bash
   export ZILLIZ_API_KEY="your-api-key"
   export ZILLIZ_DB_URL="https://your-cluster.zillizcloud.com"
   ```
   
   Environment variables override config file values.

3. **Run the tool**:
   ```bash
   ./zilliz-loadtest
   ```
   
   If you have a config file, the tool will automatically use all values from it (API key, database URL, collection, vector dimension, metric type, QPS levels, duration, etc.) without prompting.

   Or use command-line flags for non-interactive mode:
   ```bash
   ./zilliz-loadtest --load-test --api-key KEY --database-url URL --collection COLL --qps 100,500 --duration 30s
   ```
   
   You can also specify a custom config path:
   ```bash
   ./zilliz-loadtest --config /path/to/config.yaml
   ```

## Real Dataset Seeding

### Why Use Real Embeddings?

Real embeddings from production datasets provide significantly more accurate performance testing compared to random synthetic vectors:

- **Realistic Distribution**: Real embeddings have meaningful structure and clustering in the latent space, reflecting actual data patterns
- **True Performance**: Random vectors are uniformly distributed and don't capture real-world query behaviors
- **Better Benchmarking**: Testing with real embeddings gives you performance metrics that closely match production workloads
- **Clustering Patterns**: Real data exhibits natural clustering which affects index performance and recall

### BEIR Dataset with Cohere Embeddings

The tool supports seeding with the **Cohere/beir-embed-english-v3** dataset from HuggingFace:

- **Dataset**: MS MARCO passage corpus
- **Total Passages**: 8.84 million passages
- **Dimensions**: 1024 (Cohere Embed V3)
- **Fields**: ID, title (empty for MS MARCO), text, vector
- **Download**: Automatic lazy download with caching to `~/.cache/zilliz-loadtest/datasets/`

**Usage:**

```bash
# Seed with real BEIR embeddings (50,000 vectors)
./zilliz-loadtest --seed --seed-source cohere --seed-vector-count 50000

# Seed with 1 million real embeddings
./zilliz-loadtest --seed --seed-source cohere --seed-vector-count 1000000

# Keep existing collection (don't drop)
./zilliz-loadtest --seed --seed-source cohere --keep-collection

# Use synthetic vectors (default)
./zilliz-loadtest --seed --seed-source synthetic
```

**Configuration file:**

```yaml
# In configs/config.yaml
seed_source: "cohere"        # Use real BEIR embeddings
seed_vector_count: 100000    # Number of vectors to seed
seed_vector_dim: 1024        # Cohere Embed V3 dimension
seed_batch_size: 15000       # Batch size for insertion
```

**Note**: The first run will download the dataset (~2GB per file) and convert it to JSONL format. Subsequent runs use the cached version.

## Configuration

### Configuration File

**Important:** Configuration files contain sensitive credentials (API keys, database URLs) and are automatically excluded from git via `.gitignore`. Never commit your `configs/config.yaml` file to version control.

The tool automatically looks for the configuration file at `./configs/config.yaml` (relative to where you run the command).

**To create your config file:**

```bash
# Copy the example config to the required location
cp configs/config.yaml.example configs/config.yaml

# Edit configs/config.yaml with your credentials
# - Set your api_key
# - Set your database_url
# - Configure other settings as needed
```

**Custom config path:**

You can specify a different config file location using the `--config` flag:

```bash
./zilliz-loadtest --config /path/to/custom/config.yaml
```

**Note:** The example config file (`configs/config.yaml.example`) is safe to commit to git as it contains no real credentials. Your actual `configs/config.yaml` file is automatically ignored by git.

**Example config.yaml:**
```yaml
api_key: "your-api-key-here"
database_url: "https://your-cluster.zillizcloud.com"
default_collection: "my_collection"
default_vector_dim: 768
default_metric_type: "L2"
default_duration: "30s"
default_qps_levels: [100, 500, 1000]
connection_multiplier: 1.5
expected_latency_ms: 75.0
warmup_queries: 100
top_k: 10
search_level: 1
filter_expression: ""
output_fields: ["id"]
```

### Environment Variables

- `ZILLIZ_API_KEY`: Your Zilliz Cloud API key
- `ZILLIZ_DB_URL`: Your Zilliz Cloud database URL

Environment variables override config file values.

### Command-Line Flags

The tool supports both interactive and non-interactive modes. Use flags for automation:

**Seed Operation (Synthetic Vectors):**
```bash
./zilliz-loadtest --seed --api-key KEY --database-url URL --collection COLL \
  --seed-source synthetic --seed-vector-count 2000000 --seed-vector-dim 768
```

**Seed Operation (Real BEIR Embeddings):**
```bash
./zilliz-loadtest --seed --api-key KEY --database-url URL --collection COLL \
  --seed-source cohere --seed-vector-count 100000 --seed-vector-dim 1024
```

**Seed Operation (Keep Existing Collection):**
```bash
./zilliz-loadtest --seed --seed-source cohere --keep-collection
```

**Load Test:**
```bash
./zilliz-loadtest --load-test --api-key KEY --database-url URL --collection COLL \
  --qps 100,500,1000 --duration 30s --vector-dim 768 --metric-type L2 \
  --warmup 100 --output json --output-path results
```

**Profiling:**
```bash
./zilliz-loadtest --load-test ... --cpu-profile cpu.pprof --mem-profile mem.pprof
```

Run `./zilliz-loadtest --help` for complete flag documentation.

### Create a cluster and collection (Zilliz Cloud console)

Create your cluster and collection in the [Zilliz Cloud console](https://cloud.zilliz.com) so they work with this CLI:

1. **Create a cluster**  
   In Zilliz Cloud, create a cluster (Performance-optimized or Capacity-optimized). Note the **public endpoint** (e.g. `https://xxx.aws-us-west-2.vectordb.zillizcloud.com:19530`). Set `database_url` in `configs/config.yaml` to this endpoint (with port).

2. **Create a collection** that matches what the CLI expects:
   - **Primary key**: A field named `id`, type Int64, **AutoID** enabled.
   - **Vector field**: A field named `vector`, type Float Vector, dimension **768** (or match `default_vector_dim` / `seed_vector_dim` in your config).
   - **Index**: Create an index on the `vector` field. Use **AUTOINDEX** on Zilliz Cloud. Set the metric type to **L2**, **IP**, or **COSINE** to match `default_metric_type` in your config.

3. **Config**  
   In `configs/config.yaml` set `api_key`, `database_url`, and `default_collection` to the collection name you created.

After that, you can run the CLI to **seed** (e.g. 1M or 2M vectors) and **load test** against that collection.

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

1. **API Key** (required): Your Zilliz Cloud API key (or use config/env)
2. **Database URL** (required): Your Zilliz Cloud database URL (or use config/env)
3. **Collection Name** (required): The collection to seed
4. **Metric Type** (optional): L2, IP, or COSINE (defaults to config or L2)
5. **Collection Creation**: Choose whether to automatically create the collection or skip (if you've created it manually in the Zilliz UI)
6. **Confirmation**: Confirm before starting the seed operation

The seed operation will:
- **Check cluster readiness**: Verifies the cluster is ready before proceeding (if cluster is still spinning up, you'll get a helpful error message)
- **Automatically create the collection** (if enabled) if it doesn't exist with:
  - AutoID enabled primary key field (`id`)
  - Vector field (`vector`) with the specified dimension
  - Index on the vector field with the specified metric type (default: AUTOINDEX)
- **Skip collection creation** (if enabled): Assumes the collection already exists and only verifies it
- Insert vectors (default: 2,000,000 vectors of 768 dimensions)
- Use batches of 15,000 vectors for efficient insertion
- Display progress and insertion rate
- If collection already exists, it will verify the schema matches and create the index if missing

**Note**: If your cluster is still spinning up, you may see an error message indicating the cluster is not ready. Wait a few minutes for the cluster to become ready, then try again. Alternatively, you can create the collection manually in the Zilliz UI and use the skip option.

### Load Test

When you choose option 2, the tool will:

1. **Load Configuration**: Automatically load from config file or environment variables
2. **Pre-flight Validation**: Validate collection existence, schema, data availability, and indexes
3. **Prompt for Test Parameters**:
   - Collection name (if not in config)
   - Vector dimension (if not in config)
   - Metric type: L2, IP (Inner Product), or COSINE
   - QPS levels (comma-separated, e.g., `100,500,1000`)
   - Duration for each QPS test (e.g., `30s`, `1m`)
   - Connection counts (optional customization)

4. **Run Tests**: Execute load tests at each QPS level with:
   - Warm-up queries (configurable, default: 100)
   - Real-time status updates
   - Comprehensive metrics collection

5. **Display Results**: Show enhanced metrics table with error breakdowns

6. **Export Results**: Optionally export to JSON or CSV

## Enhanced Metrics

The tool now provides comprehensive latency metrics:

- **P50 Latency**: Median latency (50th percentile)
- **P90 Latency**: 90th percentile latency
- **P95 Latency**: 95th percentile latency
- **P99 Latency**: 99th percentile latency
- **Average Latency**: Mean latency across all queries
- **Min/Max Latency**: Minimum and maximum observed latencies
- **Success Rate**: Percentage of successful queries
- **Error Breakdown**: Categorized errors (network, API, timeout, SDK)

## Error Categorization

Errors are automatically categorized into:

- **Network Errors**: Connection issues, network failures
- **API Errors**: Rate limits, authentication, invalid requests
- **Timeout Errors**: Query timeouts, deadline exceeded
- **SDK Errors**: Serialization, protobuf issues
- **Unknown Errors**: Unclassified errors

## Result Export

After completing a load test, you can export results to:

- **JSON Format**: Includes metadata, all test results, and summary statistics
- **CSV Format**: Tabular format for spreadsheet analysis

Export files are automatically timestamped if no custom path is provided.

## Example Output

### Pre-flight Validation

```
============================================================
Pre-flight Validation Results
============================================================
Collection: my_collection

✓ Connection: Healthy
✓ Collection: Exists
✓ Collection: Loaded in memory
✓ Data: 2000000 rows
✓ Vector Field: Exists (dimension: 768)
✓ Index: Exists and built
============================================================
```

### Load Test Results

```
==============================
Load Test Results Summary
==============================

QPS    | P50    | P90    | P95    | P99    | Avg    | Min    | Max    | Errors | Success%
--------+--------+--------+--------+--------+--------+--------+--------+--------+----------
100    | 42.10  | 58.30  | 65.20  | 89.40  | 45.80  | 12.30  | 234.50 | 0      | 100.00
500    | 48.70  | 72.10  | 85.60  | 125.30 | 52.40  | 15.20  | 456.70 | 2      | 99.60
  Error Breakdown: timeout: 1 (50.0%), network: 1 (50.0%)
1000   | 65.40  | 98.20  | 112.50 | 178.90 | 71.20  | 18.90  | 892.10 | 15     | 98.50
  Error Breakdown: timeout: 10 (66.7%), network: 3 (20.0%), api: 2 (13.3%)
```

## Notes

### Database Seeding

- **Seed Operation**: The seed function automatically manages collections and inserts vectors:
  - **Drop Collection by Default**: Collections are dropped before seeding to ensure clean state (use `--keep-collection` to disable)
  - **Seed Source Options**:
    - **Synthetic** (default): Random vectors for quick testing - 768 dimensions by default
    - **Cohere/BEIR**: Real embeddings from MS MARCO corpus - 1024 dimensions, realistic distribution
  - **Automatic Collection Creation**: If the collection doesn't exist, it will be created with:
    - Primary key field `id` with autoID enabled (Int64 for synthetic, VarChar for BEIR)
    - Additional fields for BEIR: `title` (string), `text` (string)
    - Vector field `vector` with the specified dimension
    - Index on the vector field with AUTOINDEX type and the specified metric type (L2, IP, or COSINE)
  - **Schema Verification**: If the collection exists and `--keep-collection` is used:
    - Vector field exists and has the correct dimension
    - Index exists on the vector field (creates it if missing)
    - Metric type matches (warns if mismatch)
  - Batch size: 15,000 vectors per batch (~45MB, well under 64MB gRPC limit)
  - Provides real-time progress updates with ETA and performance metrics
  - Sequential processing to avoid overwhelming the server and reduce write errors
- **Default Values**:
  - Seed source: `synthetic` (use `seed_source: "cohere"` for real embeddings)
  - Vector count: 2,000,000 vectors (configurable via config file)
  - Dimension: 768 for synthetic, 1024 for BEIR/Cohere (configurable via config file)
  - Metric type: L2 (configurable via config file or prompt)
  - Batch size: 15,000 (configurable via config file)

### Load Testing

- **Pre-flight Validation**: The tool automatically validates your collection before starting tests to catch configuration issues early
- **Vector Dimension**: Required parameter that must match the dimension of vectors in your collection
- **Metric Type**: Required parameter that must match the metric type used when creating the collection index. Common options are L2 (Euclidean distance), IP (Inner Product), and COSINE (Cosine similarity)
- **Search Level**: Configurable search level (default: 1) which optimizes for latency
- **QPS Measurement**: The tool fires queries at the exact target QPS rate, regardless of completion time. This allows you to measure latency at the intended load level. The results show both "Fired at" (target QPS) and "Completed at" (actual completion rate) metrics
- **Status Updates**: During the test, status updates are printed every 5 seconds showing elapsed time, fired queries, completed queries, and current QPS
- **Warm-up Period**: Before each test, a configurable number of warm-up queries are executed to stabilize connections and warm caches
- **In-Flight Queries**: After the test duration ends, the tool waits for all in-flight queries to complete before reporting final results. This ensures accurate latency measurements even if queries take longer than the test duration
- **Latency Metrics**: Comprehensive percentile latencies (P50, P90, P95, P99) are calculated from all completed queries, providing accurate percentile measurements
- **Error Handling**: Errors are automatically categorized and reported with breakdowns to help diagnose issues
- Ensure your collection is loaded and contains data before running the load test

### Connection Calculation

The tool automatically calculates the optimal number of connections based on:
- Formula: `connections = (QPS × expected_latency_ms) / 1000 × multiplier`
- Default: 75ms expected latency, 1.5x multiplier
- You can customize connection counts per QPS level if needed

### Best Practices

1. **Run Pre-flight Validation**: Always let the tool validate your collection before testing
2. **Use Warm-up**: Enable warm-up queries (default: 100) to get stable results
3. **Test Multiple QPS Levels**: Test a range of QPS levels to understand performance characteristics
4. **Export Results**: Export results to JSON/CSV for trend analysis and reporting
5. **Monitor Error Rates**: Pay attention to error breakdowns to identify issues early
6. **Use Configuration Files**: Store your settings in config files for consistency

### Troubleshooting

**Collection validation fails:**
- Ensure the collection exists and is accessible
- Verify the vector dimension matches your collection schema
- Check that the collection has data (row count > 0)
- Ensure indexes are built

**Low QPS achievement:**
- Check error breakdowns for network or API issues
- Verify server-side capacity and rate limits
- Consider increasing connection counts (though this may not help if server is the bottleneck)
- Check network bandwidth and latency

**High latency:**
- Review P95/P99 latencies vs average to identify outliers
- Check for server-side queuing (queries with >1s latency are filtered in reports)
- Verify collection is properly indexed
- Consider adjusting search level for latency vs recall tradeoff

**Export issues:**
- Ensure you have write permissions in the output directory
- Check disk space availability

### General

- The SDK method signatures may vary by version. If you encounter compilation errors, you may need to adjust the method calls in `loadtest.go` to match your SDK version
- The tool supports both `client.NewClient` and `client.NewGrpcClient` initialization methods for compatibility with different SDK versions
- Configuration files support YAML format with environment variable overrides
- Results can be exported in JSON (with metadata) or CSV (tabular) formats

## File Structure

```
zilliz-loadtest/
├── cmd/zilliz-loadtest/
│   ├── main.go          # CLI interface and orchestration
│   ├── flags.go         # Command-line flag parsing
│   └── compare.go       # Test result comparison
├── internal/
│   ├── loadtest/        # Core load testing logic
│   │   ├── loadtest.go
│   │   ├── constants.go
│   │   ├── warmup.go
│   │   ├── test_executor.go
│   │   ├── result_processor.go
│   │   ├── seed.go
│   │   └── retry.go
│   ├── datasource/      # Real dataset support (BEIR/Cohere)
│   │   ├── cohere_downloader.go   # HuggingFace dataset downloader
│   │   ├── cohere_reader.go       # Parquet file reader
│   │   └── cohere_jsonl_reader.go # JSONL format reader
│   ├── config/          # Configuration management
│   ├── export/          # Result export functionality
│   ├── validation/      # Pre-flight validation and input validation
│   ├── logger/          # Structured logging
│   ├── profiling/       # CPU/memory profiling
│   └── errors/          # Custom error types
├── scripts/
│   └── convert_parquet.py  # Parquet to JSONL converter
├── configs/
│   └── config.yaml.example  # Example configuration file
├── go.mod               # Go dependencies
└── README.md            # This file
```

## Advanced Features

### Structured Logging

The tool uses structured logging with configurable levels and formats:

- **Log Levels**: DEBUG, INFO, WARN, ERROR
- **Formats**: text (human-readable) or json (machine-readable)
- **Configuration**: Set via config file or environment variables

### Profiling

Enable CPU and memory profiling to identify performance bottlenecks:
```bash
./zilliz-loadtest --load-test ... --cpu-profile cpu.pprof --mem-profile mem.pprof
```

Analyze with `go tool pprof`:
```bash
go tool pprof cpu.pprof
go tool pprof mem.pprof
```

### Test Comparison

Compare multiple test runs to track performance changes:
```go
// Programmatic usage
compare.CompareResultsFromFiles("baseline.json", "comparison.json")
```

### Retry Logic

Automatic retry for transient errors:
- Network errors
- Timeout errors
- Configurable retry count and backoff strategy

## Contributing

This is a template tool for customers. Feel free to customize it for your specific needs. Key areas for customization:

- Search parameters (topK, filters, search level)
- Vector generation strategies
- Custom export formats
- Integration with monitoring systems
