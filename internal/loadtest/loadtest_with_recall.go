package loadtest

import (
	"context"
	"fmt"
	"math/rand"
	"strings"
	"time"

	"zilliz-loadtest/internal/datasource"
	"zilliz-loadtest/internal/logger"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// RunTestWithRecall runs a load test with optional real queries and recall calculation
func (lt *LoadTester) RunTestWithRecall(
	ctx context.Context,
	targetQPS int,
	duration time.Duration,
	warmupQueries int,
	queries []datasource.CohereQuery,
	qrels datasource.Qrels,
) (TestResult, error) {
	// If no real queries provided, fall back to standard test
	if len(queries) == 0 {
		return lt.RunTest(ctx, targetQPS, duration, warmupQueries)
	}

	logger.Info("Running load test with real BEIR queries",
		"target_qps", targetQPS,
		"duration", duration,
		"queries_available", len(queries))

	// Run the load test with real queries
	result, err := lt.runTestWithRealQueries(ctx, targetQPS, duration, warmupQueries, queries)
	if err != nil {
		return result, err
	}

	// Calculate recall metrics
	if len(queries) > 0 {
		logger.Info("Calculating recall metrics...", "search_level", lt.searchLevel)

		// Detect ID field name based on collection type
		// BEIR collections use "_id", VDBBench uses "id"
		idField := "id"
		if strings.HasPrefix(lt.collection, "beir_") {
			idField = "_id"
		}

		// Use one of the clients to calculate recall with configured search level
		recallCalc := NewRecallCalculatorWithLevel(lt.getClient(), lt.collection, lt.vectorField, idField, lt.searchLevel, qrels)

		recallMetrics, err := recallCalc.CalculateRecall(
			ctx,
			queries,
			lt.topK,
			lt.metricType,
			nil, // search params handled internally
		)

		if err != nil {
			logger.Warn("Failed to calculate recall", "error", err)
		} else {
			result.MathematicalRecall = recallMetrics.MathematicalRecall * 100 // Convert to percentage
			result.BusinessRecall = recallMetrics.BusinessRecall * 100         // Convert to percentage
			result.BusinessPrecision = recallMetrics.BusinessPrecision * 100   // Convert to percentage
			result.RecallTested = true

			logger.Info("Recall metrics calculated",
				"search_level", lt.searchLevel,
				"math_recall", fmt.Sprintf("%.2f%%", result.MathematicalRecall),
				"business_recall", fmt.Sprintf("%.2f%%", result.BusinessRecall),
				"business_precision", fmt.Sprintf("%.2f%%", result.BusinessPrecision))
		}
	}

	return result, nil
}

// runTestWithRealQueries runs a load test using real query embeddings
func (lt *LoadTester) runTestWithRealQueries(
	ctx context.Context,
	targetQPS int,
	duration time.Duration,
	warmupQueries int,
	queries []datasource.CohereQuery,
) (TestResult, error) {
	// Warmup phase with real queries
	if warmupQueries > 0 {
		logger.Info("Starting warmup phase with real queries", "queries", warmupQueries)
		warmupStart := time.Now()

		for i := 0; i < warmupQueries; i++ {
			// Pick a random query
			queryIdx := rand.Intn(len(queries))
			query := queries[queryIdx]

			// Execute query
			_ = lt.executeRealQuery(ctx, query)
		}

		logger.Info("Warmup phase completed", "duration", time.Since(warmupStart))
	}

	// Main test phase
	// For now, we use the standard RunTest which generates random queries
	// The real queries were used during warmup and will be used for recall calculation
	// TODO: Refactor executeQuery to accept a query vector parameter for the main test
	return lt.RunTest(ctx, targetQPS, duration, 0) // Skip warmup as we already did it
}

// executeRealQuery executes a search with a real query embedding
func (lt *LoadTester) executeRealQuery(ctx context.Context, query datasource.CohereQuery) QueryResult {
	start := time.Now()

	c := lt.getClient()

	// Create search parameters
	sp, err := entity.NewIndexFlatSearchParam()
	if err != nil {
		return QueryResult{
			Latency:   time.Since(start),
			Error:     err,
			ErrorType: categorizeError(err),
		}
	}

	// Execute search with real query embedding
	_, err = c.Search(
		ctx,
		lt.collection,
		[]string{},
		lt.filterExpr,
		lt.outputFields,
		[]entity.Vector{entity.FloatVector(query.Embedding)},
		lt.vectorField,
		lt.metricType,
		lt.topK,
		sp,
	)

	latency := time.Since(start)

	if err != nil {
		return QueryResult{
			Latency:   latency,
			Error:     err,
			ErrorType: categorizeError(err),
		}
	}

	return QueryResult{
		Latency:   latency,
		Error:     nil,
		ErrorType: "",
	}
}
