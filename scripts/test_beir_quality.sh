#!/bin/bash
# Test quality metrics (recall, NDCG) across multiple BEIR datasets.
# For each dataset: seed the collection, then run a load test at low QPS to measure quality.
#
# Usage: ./scripts/test_beir_quality.sh [dataset...]
#   No args = run all small/medium datasets
#   With args = run only specified datasets (e.g., ./scripts/test_beir_quality.sh fiqa scifact)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BINARY="$PROJECT_DIR/bin/zilliz-loadtest"
RESULTS_DIR="$PROJECT_DIR/beir_results"
SUMMARY_FILE="$RESULTS_DIR/summary.txt"

# Build if needed
if [ ! -f "$BINARY" ]; then
    echo "Building zilliz-loadtest..."
    cd "$PROJECT_DIR" && go build -o "$BINARY" ./cmd/zilliz-loadtest
fi

mkdir -p "$RESULTS_DIR"

# Small/medium datasets ordered by corpus size (ascending)
SMALL_DATASETS=(
    nfcorpus        # 3.6K
    scifact         # 5.2K
    arguana         # 8.7K
    cqadupstack-mathematica  # 17K
    cqadupstack-webmasters   # 17K
    cqadupstack-android      # 23K
    scidocs         # 26K
    cqadupstack-programmers  # 32K
    cqadupstack-gis          # 37K
    cqadupstack-physics      # 38K
    cqadupstack-english      # 40K
    cqadupstack-stats        # 42K
    cqadupstack-gaming       # 45K
    cqadupstack-unix         # 47K
    cqadupstack-wordpress    # 49K
    fiqa            # 58K
    cqadupstack-text         # 68K
    trec-covid      # 171K
    webis-touche2020 # 383K
    quora           # 523K
    robust04        # 528K
    trec-news       # 595K
)

# Large datasets (>1M docs) - uncomment to include
# LARGE_DATASETS=(
#     nq              # 2.7M
#     hotpotqa        # 5.2M
#     fever           # 5.4M
#     climate-fever   # 5.4M
# )

# Use specified datasets or default to small/medium
if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("${SMALL_DATASETS[@]}")
fi

echo "=============================================="
echo "BEIR Quality Benchmark"
echo "=============================================="
echo "Datasets to test: ${#DATASETS[@]}"
echo "Results directory: $RESULTS_DIR"
echo ""

# Initialize summary
cat > "$SUMMARY_FILE" <<'HEADER'
==============================================================================================================
BEIR Quality Benchmark Summary
==============================================================================================================
Dataset                  | Corpus  | Level |  P50(ms) |  P90(ms) | Math Recall | Biz Recall |   NDCG | Status
--------------------------------------------------------------------------------------------------------------
HEADER

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

for dataset in "${DATASETS[@]}"; do
    collection="beir_$(echo "$dataset" | tr '-' '_')"
    logfile="$RESULTS_DIR/${dataset}.log"

    echo ""
    echo "=== [$dataset] Seeding collection: $collection ==="

    # Seed
    if "$BINARY" --seed --seed-source "beir:${dataset}" --collection "$collection" > "$logfile" 2>&1; then
        echo "    Seed OK"
    else
        echo "    SEED FAILED (see $logfile)"
        printf "%-24s | %7s | %5s | %8s | %8s | %11s | %10s | %6s | SEED_FAIL\n" \
            "$dataset" "-" "-" "-" "-" "-" "-" "-" >> "$SUMMARY_FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    echo "=== [$dataset] Running quality test ==="

    # Load test at low QPS, short duration, just to measure quality
    test_logfile="$RESULTS_DIR/${dataset}_test.log"
    if "$BINARY" --load-test --collection "$collection" --qps 50 --duration 15s \
        --metric-type COSINE --vector-dim 1024 >> "$test_logfile" 2>&1; then
        echo "    Test OK"

        # Extract the result line from output (the data row after the header line)
        # Table columns: QPS|Level|P50|P90|P95|P99|Avg|Min|Max|Success%|Errors|MathRecall|BizRecall|NDCG
        result_line=$(grep -E '^\s*50\s+\|' "$test_logfile" | head -1 || true)
        if [ -n "$result_line" ]; then
            # Parse fields from the result line (14 pipe-delimited columns)
            level=$(echo "$result_line" | awk -F'|' '{gsub(/[[:space:]]/, "", $2); print $2}')
            p50=$(echo "$result_line" | awk -F'|' '{gsub(/[[:space:]]/, "", $3); print $3}')
            p90=$(echo "$result_line" | awk -F'|' '{gsub(/[[:space:]]/, "", $4); print $4}')
            math_recall=$(echo "$result_line" | awk -F'|' '{gsub(/[[:space:]]/, "", $12); print $12}')
            biz_recall=$(echo "$result_line" | awk -F'|' '{gsub(/[[:space:]]/, "", $13); print $13}')
            ndcg=$(echo "$result_line" | awk -F'|' '{gsub(/[[:space:]]/, "", $14); print $14}')

            printf "%-24s | %7s | %5s | %8s | %8s | %11s | %10s | %6s | OK\n" \
                "$dataset" "-" "$level" "$p50" "$p90" "$math_recall" "$biz_recall" "$ndcg" >> "$SUMMARY_FILE"
            PASS_COUNT=$((PASS_COUNT + 1))
            echo "    Math Recall: $math_recall  Biz Recall: $biz_recall  NDCG: $ndcg"
        else
            printf "%-24s | %7s | %5s | %8s | %8s | %11s | %10s | %6s | NO_DATA\n" \
                "$dataset" "-" "-" "-" "-" "-" "-" "-" >> "$SUMMARY_FILE"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            echo "    WARNING: Could not parse results"
        fi
    else
        echo "    TEST FAILED (see $test_logfile)"
        printf "%-24s | %7s | %5s | %8s | %8s | %11s | %10s | %6s | TEST_FAIL\n" \
            "$dataset" "-" "-" "-" "-" "-" "-" "-" >> "$SUMMARY_FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

# Print summary
echo ""
echo "=============================================="
echo "Results: $PASS_COUNT passed, $FAIL_COUNT failed, $SKIP_COUNT skipped"
echo "=============================================="
echo ""
cat "$SUMMARY_FILE"
echo ""
echo "Full logs in: $RESULTS_DIR/"
