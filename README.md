# Zilliz Cloud Load Test Tool

CLI tool for seeding and load testing Zilliz Cloud vector databases.

## Quick Start

```bash
# Build
go build -o zilliz-loadtest ./cmd/zilliz-loadtest

# Setup config
cp configs/config.yaml.example configs/config.yaml
# Edit configs/config.yaml with your API key and database URL

# Seed with BEIR dataset (recommended)
./zilliz-loadtest --seed --seed-source beir:fiqa

# Run load test
./zilliz-loadtest --load-test --collection beir_fiqa --qps 100,500 --duration 30s
```

## Seed Sources

| Source | Example | Use Case |
|--------|---------|----------|
| `synthetic` | `--seed-source synthetic` | Quick testing with random vectors |
| `beir:<name>` | `--seed-source beir:fiqa` | Real embeddings with human-labeled relevance |

**BEIR datasets:** `fiqa` (57K), `trec-covid` (171K), `quora` (522K), `scifact` (5K)

## Metrics

| Metric | Description |
|--------|-------------|
| **P50/P90/P95/P99** | Latency percentiles (ms) |
| **Math Recall@100** | ANN accuracy vs level-10 search |
| **Biz Recall@100** | Relevant docs found vs human judgments |
| **NDCG@100** | Rank-aware relevance score |

## Configuration

**Config file** (`configs/config.yaml`):
```yaml
api_key: "your-api-key"
database_url: "https://your-cluster.zillizcloud.com"
default_collection: "my_collection"
default_vector_dim: 1024
default_metric_type: "COSINE"
top_k: 100
search_level: 1
```

**Environment variables** (override config):
```bash
export ZILLIZ_API_KEY="your-api-key"
export ZILLIZ_DB_URL="https://your-cluster.zillizcloud.com"
```

## Commands

```bash
# Seed synthetic vectors
./zilliz-loadtest --seed --seed-source synthetic --seed-vector-count 1000000

# Seed BEIR dataset
./zilliz-loadtest --seed --seed-source beir:fiqa

# Load test with recall metrics
./zilliz-loadtest --load-test --collection beir_fiqa --qps 100,500,1000 --duration 30s

# Export results
./zilliz-loadtest --load-test ... --output json --output-path results
```

## Example Output

```
===============================================================================================================
Load Test Results
===============================================================================================================
QPS    | Level | P50(ms) | P90(ms) | P95(ms) | P99(ms) | Avg(ms) | Min(ms) | Max(ms) | Success% | Errors | Math Recall | Biz Recall |    NDCG
-----------------------------------------------------------------------------------------------------------------------
100    |     1 |   45.30 |   62.10 |   68.50 |   92.10 |   48.20 |   13.10 |  245.30 |   100.0% |      0 |      96.50% |     76.20% |  0.5521
500    |     1 |   51.20 |   75.40 |   88.90 |  128.60 |   55.10 |   16.30 |  462.10 |    99.8% |      1 |      95.80% |     76.10% |  0.5518
===============================================================================================================
```

## Flags

Run `./zilliz-loadtest --help` for all options.

Key flags:
- `--seed` / `--load-test`: Operation mode
- `--api-key`, `--database-url`, `--collection`: Connection
- `--qps`: Comma-separated QPS levels (e.g., `100,500,1000`)
- `--duration`: Test duration per level (e.g., `30s`, `1m`)
- `--seed-source`: `synthetic`, `cohere`, or `beir:<dataset>`
- `--search-level`: 1-10, higher = more accurate but slower
