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

**BEIR datasets:** `fiqa` (57K), `trec-covid` (171K), `quora` (522K), `scifact` (5K), `trec-news` (595K), `robust04` (528K), `webis-touche2020` (383K), `arguana` (9K), `nfcorpus` (4K), `scidocs` (26K), `cqadupstack-*` (12 subforums), and more

## Metrics

| Metric | Description |
|--------|-------------|
| **P50/P90/P95/P99** | Latency percentiles (ms) |
| **Math Recall@100** | ANN accuracy vs KNN ground truth |
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

## Example Output (beir_trec_news, 595K vectors)

```
==========================================================================================================
Load Test Results
==========================================================================================================
QPS    | Level |  P50(ms) |  P90(ms) |  P95(ms) |  P99(ms) | Success% | Math Recall | Biz Recall |   NDCG
----------------------------------------------------------------------------------------------------------
50     |     1 |    50.00 |   110.00 |   123.00 |   134.00 |   100.0% |      93.53% |     53.37% | 0.5294
50     |     5 |    49.00 |    56.00 |    59.00 |    63.00 |   100.0% |      99.21% |     54.02% | 0.5346
50     |    10 |    50.00 |    56.00 |    59.00 |    64.00 |   100.0% |     100.00% |     54.02% | 0.5344
100    |     1 |    54.00 |   116.00 |   128.00 |   140.00 |   100.0% |      93.53% |     53.37% | 0.5294
300    |     1 |    50.00 |    57.00 |    60.00 |    75.00 |   100.0% |      93.53% |     53.37% | 0.5294
500    |     1 |    47.00 |    54.00 |    57.00 |    63.00 |   100.0% |      93.53% |     53.37% | 0.5294
700    |     1 |    46.00 |    53.00 |    56.00 |    62.00 |   100.0% |      93.53% |     53.37% | 0.5294
==========================================================================================================

Metrics:
  Level:       Search accuracy level (1-10, higher = more accurate, slower)
  Math Recall: ANN accuracy vs KNN ground truth
  Biz Recall:  Relevant docs found vs human judgments (qrels)
  NDCG:        Normalized Discounted Cumulative Gain (rank-aware relevance)
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
