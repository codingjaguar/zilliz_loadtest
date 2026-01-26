package loadtest

import (
	"context"
	"fmt"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"zilliz-loadtest/internal/logger"
)

type LoadTester struct {
	clients         []client.Client // Multiple clients for connection pooling
	clientIdx       int64           // Atomic counter for round-robin client selection
	collection      string
	vectorDim       int
	metricType      entity.MetricType
	topK            int
	filterExpr      string
	outputFields    []string
	searchLevel     int
}

// ErrorType categorizes different types of errors
type ErrorType string

const (
	ErrorTypeNetwork ErrorType = "network"
	ErrorTypeAPI     ErrorType = "api"
	ErrorTypeTimeout ErrorType = "timeout"
	ErrorTypeSDK     ErrorType = "sdk"
	ErrorTypeUnknown ErrorType = "unknown"
)

type TestResult struct {
	QPS            int
	P50Latency     float64 // in milliseconds
	P90Latency     float64 // in milliseconds
	P95Latency     float64 // in milliseconds
	P99Latency     float64 // in milliseconds
	MinLatency     float64 // in milliseconds
	MaxLatency     float64 // in milliseconds
	AvgLatency     float64 // in milliseconds
	TotalQueries   int
	Errors         int
	ErrorBreakdown map[ErrorType]int
	SuccessRate    float64 // percentage
}

type QueryResult struct {
	Latency    time.Duration
	Error      error
	ErrorType  ErrorType
}

// CreateZillizClient creates a Zilliz Cloud client with API key authentication
func CreateZillizClient(apiKey, databaseURL string) (client.Client, error) {
	ctx := context.Background()

	if apiKey == "" {
		return nil, fmt.Errorf("API key is required but was empty")
	}
	if databaseURL == "" {
		return nil, fmt.Errorf("database URL is required but was empty")
	}

	// Try the newer client.NewClient approach first
	milvusClient, err := client.NewClient(
		ctx,
		client.Config{
			Address:       databaseURL,
			APIKey:        apiKey,
			EnableTLSAuth: true,
		},
	)

	// If that fails, try the alternative approach with NewGrpcClient
	if err != nil {
		milvusClient, err = client.NewGrpcClient(ctx, databaseURL)
		if err != nil {
			return nil, fmt.Errorf("failed to create client: tried both NewClient and NewGrpcClient, last error: %w. Ensure the database URL is correct and accessible", err)
		}
		// Try SetToken method (common in Zilliz Cloud SDK)
		if tokenClient, ok := milvusClient.(interface{ SetToken(string) error }); ok {
			if err := tokenClient.SetToken(apiKey); err != nil {
				// If SetToken fails, try SetApiKey
				if apiKeyClient, ok := milvusClient.(interface{ SetApiKey(string) }); ok {
					apiKeyClient.SetApiKey(apiKey)
				}
			}
		} else if apiKeyClient, ok := milvusClient.(interface{ SetApiKey(string) }); ok {
			apiKeyClient.SetApiKey(apiKey)
		}
	}

	return milvusClient, nil
}

func NewLoadTester(apiKey, databaseURL, collection string, vectorDim int, metricTypeStr string) (*LoadTester, error) {
	// Default to 10 connections - will be adjusted based on QPS in RunTest
	return NewLoadTesterWithConnections(apiKey, databaseURL, collection, vectorDim, metricTypeStr, 10)
}

// CalculateOptimalConnections calculates the number of connections needed based on target QPS
// Formula: connections = (QPS × expected_latency_ms) / 1000 × multiplier
// This ensures each connection isn't overwhelmed and can handle concurrent requests
//
// We estimate the number of connections based on the target QPS and what we know about
// Zilliz Cloud behavior. This is an educated guess - actual latency may vary based on
// collection size, index type, server load, and network conditions.
//
// Default formula uses:
//   - Expected latency: 75ms (conservative estimate for vector search on Zilliz Cloud)
//   - Multiplier: 1.5x (accounts for connection overhead and headroom)
//
// Default connections for common QPS levels:
//   - 100 QPS: ~12 connections
//   - 500 QPS: ~56 connections
//   - 1000 QPS: ~113 connections
//   - 5000 QPS: ~563 connections
//   - 10000 QPS: ~1125 connections
func CalculateOptimalConnections(targetQPS int) (int, string) {
	// Estimate based on typical Zilliz Cloud vector search latency
	// Formula: (QPS * latency_ms) / 1000 gives concurrent requests needed
	// Then multiply by multiplier to account for connection overhead and headroom
	baseConnections := float64(targetQPS) * ExpectedLatencyMs / 1000.0
	connections := int(baseConnections * ConnectionMultiplier)

	// Enforce reasonable bounds
	if connections < MinConnections {
		connections = MinConnections
	}

	if connections > MaxConnections {
		connections = MaxConnections
	}

	explanation := fmt.Sprintf("Formula: (%d QPS × %.0fms latency) / 1000 × %.1fx = %d connections (default)",
		targetQPS, ExpectedLatencyMs, ConnectionMultiplier, connections)

	return connections, explanation
}

func NewLoadTesterWithConnections(apiKey, databaseURL, collection string, vectorDim int,
	metricTypeStr string, numConnections int) (*LoadTester, error) {
	return NewLoadTesterWithOptions(apiKey, databaseURL, collection, vectorDim, metricTypeStr, numConnections, 10, "", []string{"id"}, 1)
}

func NewLoadTesterWithOptions(apiKey, databaseURL, collection string, vectorDim int,
	metricTypeStr string, numConnections int, topK int, filterExpr string, outputFields []string, searchLevel int) (*LoadTester, error) {
	// Validate inputs
	if apiKey == "" {
		return nil, fmt.Errorf("API key is required but was empty")
	}
	if databaseURL == "" {
		return nil, fmt.Errorf("database URL is required but was empty")
	}
	if collection == "" {
		return nil, fmt.Errorf("collection name is required but was empty")
	}
	if vectorDim <= 0 {
		return nil, fmt.Errorf("vector dimension must be positive, got %d", vectorDim)
	}
	if numConnections < 1 {
		return nil, fmt.Errorf("number of connections must be at least 1, got %d", numConnections)
	}
	if numConnections > 100 {
		return nil, fmt.Errorf("number of connections exceeds maximum of 100, got %d. Consider using connection pooling instead", numConnections)
	}

	// Create multiple clients for connection pooling with retry
	clients := make([]client.Client, numConnections)
	retryConfig := DefaultRetryConfig()
	retryConfig.MaxRetries = 2 // Fewer retries for connection creation
	
	ctx := context.Background()
	for i := 0; i < numConnections; i++ {
		milvusClient, err := RetryClientCreation(ctx, apiKey, databaseURL, retryConfig)
		if err != nil {
			// Clean up already created clients
			for j := 0; j < i; j++ {
				clients[j].Close()
			}
			return nil, fmt.Errorf("failed to create client %d/%d after retries: %w. Check API key, database URL, and network connectivity", i+1, numConnections, err)
		}
		clients[i] = milvusClient
	}

	// Parse metric type
	var metricType entity.MetricType
	switch metricTypeStr {
	case "L2":
		metricType = entity.L2
	case "IP":
		metricType = entity.IP
	case "COSINE":
		metricType = entity.COSINE
	default:
		// Clean up clients
		for _, c := range clients {
			c.Close()
		}
		return nil, fmt.Errorf("unsupported metric type: %s. Supported types are: L2, IP, COSINE", metricTypeStr)
	}

	// Set defaults
	if topK <= 0 {
		topK = 10
	}
	if outputFields == nil || len(outputFields) == 0 {
		outputFields = []string{"id"}
	}
	if searchLevel < 1 {
		searchLevel = 1
	}

	return &LoadTester{
		clients:      clients,
		clientIdx:    0,
		collection:   collection,
		vectorDim:    vectorDim,
		metricType:   metricType,
		topK:         topK,
		filterExpr:   filterExpr,
		outputFields: outputFields,
		searchLevel:  searchLevel,
	}, nil
}

// getClient returns a client using round-robin selection
func (lt *LoadTester) getClient() client.Client {
	idx := int(atomic.AddInt64(&lt.clientIdx, 1) % int64(len(lt.clients)))
	return lt.clients[idx]
}

func (lt *LoadTester) Close() {
	for _, c := range lt.clients {
		if c != nil {
			c.Close()
		}
	}
}

func (lt *LoadTester) RunTest(ctx context.Context, targetQPS int, duration time.Duration, warmupQueries int) (TestResult, error) {
	// Validate inputs
	if targetQPS <= 0 {
		return TestResult{}, fmt.Errorf("target QPS must be positive, got %d", targetQPS)
	}
	if duration <= 0 {
		return TestResult{}, fmt.Errorf("duration must be positive, got %v", duration)
	}
	if duration < 1*time.Second {
		return TestResult{}, fmt.Errorf("duration too short: %v (minimum: 1s). Tests shorter than 1 second may not provide accurate results", duration)
	}

	// Warm-up phase
	lt.runWarmup(ctx, warmupQueries)

	// Setup test execution
	interval := time.Second / time.Duration(targetQPS)
	if interval <= 0 {
		return TestResult{}, fmt.Errorf("calculated interval is invalid for QPS %d", targetQPS)
	}
	testCtx, cancel := context.WithTimeout(ctx, duration)
	defer cancel()

	var wg sync.WaitGroup
	expectedQueries := int(duration.Seconds()) * targetQPS
	resultsChan := make(chan QueryResult, expectedQueries*2)

	ts := &testState{
		resultsChan: resultsChan,
		startTime:   time.Now(),
	}

	queryCtx := ctx // Use parent context, not testCtx

	// Start result collection and status reporting
	ts.startResultCollector()
	ts.startStatusReporter(testCtx)

	// Start firing queries
	lt.startQueryFirer(testCtx, queryCtx, interval, ts, &wg)

	// Wait for test duration
	<-testCtx.Done()

	totalFired := int(atomic.LoadInt64(&ts.queriesFired))
	completedDuringTest := int(atomic.LoadInt64(&ts.queriesCompleted))

	inFlight := totalFired - completedDuringTest
	logger.Info("Test duration ended, waiting for in-flight queries",
		"in_flight", inFlight,
		"total_fired", totalFired,
		"completed", completedDuringTest)

	// Wait for completion
	waitTimeout := calculateWaitTimeout(duration)
	ts.waitForCompletion(&wg, waitTimeout)

	// Close channel and wait for collector
	close(ts.resultsChan)
	time.Sleep(ResultCollectorDelay)

	elapsed := time.Since(ts.startTime)

	// Get final results
	ts.resultsMu.Lock()
	allResults := ts.allResults
	ts.resultsMu.Unlock()

	// Process results
	stats := processQueryResults(allResults)

	// Handle edge case: no results at all
	if len(allResults) == 0 {
		logger.Warn("No query results collected",
			"total_fired", totalFired,
			"message", "This may indicate a configuration issue or all queries were cancelled")
		return TestResult{
			QPS:            targetQPS,
			TotalQueries:   0,
			Errors:         0,
			ErrorBreakdown: make(map[ErrorType]int),
			SuccessRate:    0,
		}, fmt.Errorf("no queries completed - check configuration and network connectivity")
	}

	// Handle edge case: all queries failed
	if len(stats.latencies) == 0 && stats.firstError != nil {
		logger.Error("All queries failed",
			"first_error", stats.firstError.Error(),
			"total_queries", len(allResults),
			"errors", stats.errors)
		// Return result with errors but don't return error - this is a valid test outcome
		return TestResult{
			QPS:            targetQPS,
			TotalQueries:   0,
			Errors:         stats.errors,
			ErrorBreakdown: stats.errorBreakdown,
			SuccessRate:    0,
		}, nil
	}

	// Handle edge case: very few successful queries
	if len(stats.latencies) < 10 && len(allResults) >= 10 {
		logger.Warn("Very few successful queries",
			"successful", len(stats.latencies),
			"total", len(allResults),
			"message", "Results may not be statistically significant")
	}

	min, max, avgLatency, p50, p90, p95, p99 := calculateLatencyStats(stats.latencies)

	// Check and report high latencies
	p95, p99, _ = reportHighLatencyWarning(min, max, avgLatency, p95, p99, stats.latencies)

	// Report summary
	completed := len(stats.latencies)
	reportTestSummary(targetQPS, totalFired, completed, stats.errors, stats.errorBreakdown, elapsed, len(lt.clients))

	successRate := float64(completed) / float64(totalFired) * 100.0

	return TestResult{
		QPS:            targetQPS,
		P50Latency:     p50,
		P90Latency:     p90,
		P95Latency:     p95,
		P99Latency:     p99,
		MinLatency:     min,
		MaxLatency:     max,
		AvgLatency:     avgLatency,
		TotalQueries:   completed,
		Errors:         stats.errors,
		ErrorBreakdown: stats.errorBreakdown,
		SuccessRate:    successRate,
	}, nil
}

// EmptySearchParam implements SearchParam interface with no parameters (defaults to level 1)
type EmptySearchParam struct{}

func (e *EmptySearchParam) Params() map[string]interface{} {
	return make(map[string]interface{})
}

func (e *EmptySearchParam) AddRadius(radius float64) {
	// Not used
}

func (e *EmptySearchParam) AddRangeFilter(rangeFilter float64) {
	// Not used
}

// SearchParamWithLevel implements SearchParam interface with configurable search level
type SearchParamWithLevel struct {
	Level int
}

func (s *SearchParamWithLevel) Params() map[string]interface{} {
	params := make(map[string]interface{})
	if s.Level >= 0 {
		params["level"] = s.Level
	}
	return params
}

func (s *SearchParamWithLevel) AddRadius(radius float64) {
	// Not used
}

func (s *SearchParamWithLevel) AddRangeFilter(rangeFilter float64) {
	// Not used
}

func (lt *LoadTester) executeQuery(ctx context.Context, queryNum int) QueryResult {
	// Generate a random query vector
	queryVector := generateRandomVector(lt.vectorDim)

	// Measure latency for the search operation
	searchStart := time.Now()

	// Create search params with configured search level
	searchParams := &SearchParamWithLevel{
		Level: lt.searchLevel,
	}

	// Execute search with configured parameters
	_, err := lt.getClient().Search(
		ctx,
		lt.collection,
		[]string{},        // partition names (empty for all partitions)
		lt.filterExpr,      // expr (filter expression if configured)
		lt.outputFields,    // output fields
		[]entity.Vector{entity.FloatVector(queryVector)},
		"vector",           // vector field name
		lt.metricType,      // metric type
		lt.topK,            // topK
		searchParams,       // search params with level
	) // opts parameter is optional and not used

	latency := time.Since(searchStart)

	if err != nil {
		return QueryResult{
			Latency:   latency,
			Error:     err,
			ErrorType: categorizeError(err),
		}
	}

	return QueryResult{
		Latency:   latency,
		ErrorType: ErrorTypeUnknown, // No error
	}
}

// extractIDs extracts IDs from a column into a slice for comparison
func extractIDs(idsColumn entity.Column) []interface{} {
	if idsColumn == nil {
		return nil
	}

	var ids []interface{}

	// Try different column types that might be used for IDs
	// IDColumns returns either ColumnInt64 or ColumnVarChar
	switch col := idsColumn.(type) {
	case *entity.ColumnInt64:
		// Use Data() method for better performance
		data := col.Data()
		for _, val := range data {
			ids = append(ids, val)
		}
	case *entity.ColumnVarChar:
		// VarChar is used for string IDs
		data := col.Data()
		for _, val := range data {
			ids = append(ids, val)
		}
	default:
		// Fallback: try to use the generic Column interface methods
		if col.Len() > 0 {
			for i := 0; i < col.Len(); i++ {
				if val, err := col.Get(i); err == nil {
					ids = append(ids, val)
				}
			}
		}
	}
	return ids
}

func generateRandomVector(dim int) []float32 {
	// Generate a pseudo-random vector for testing
	// In production, use actual query vectors from your dataset
	vector := make([]float32, dim)
	for i := range vector {
		// Use a simple pattern that varies per dimension
		vector[i] = float32((i*7+13)%100) / 100.0
	}
	return vector
}

func generateSeedingVector(dim int, seed int64) []float32 {
	// Generate a vector with better distribution for seeding
	// Uses seed to ensure different vectors for each index
	vector := make([]float32, dim)
	for i := range vector {
		// Create a more varied pattern using the seed
		value := float32((int64(i)*7919 + seed*9829) % 10000)
		vector[i] = value / 10000.0
	}
	return vector
}

func calculatePercentile(sortedData []float64, percentile int) float64 {
	if len(sortedData) == 0 {
		return 0
	}
	index := float64(percentile) / 100.0 * float64(len(sortedData))
	upper := int(math.Ceil(index)) - 1
	lower := int(math.Floor(index)) - 1

	if upper < 0 {
		upper = 0
	}
	if lower < 0 {
		lower = 0
	}
	if upper >= len(sortedData) {
		upper = len(sortedData) - 1
	}
	if lower >= len(sortedData) {
		lower = len(sortedData) - 1
	}

	if upper == lower {
		return sortedData[upper]
	}

	weight := index - float64(lower+1)
	return sortedData[lower]*(1-weight) + sortedData[upper]*weight
}

// categorizeError categorizes an error into one of the error types
func categorizeError(err error) ErrorType {
	if err == nil {
		return ErrorTypeUnknown
	}

	errStr := err.Error()
	errStrLower := strings.ToLower(errStr)

	// Check for timeout errors
	if strings.Contains(errStrLower, "timeout") ||
		strings.Contains(errStrLower, "deadline exceeded") ||
		strings.Contains(errStrLower, "context deadline") {
		return ErrorTypeTimeout
	}

	// Check for network errors
	if strings.Contains(errStrLower, "connection") ||
		strings.Contains(errStrLower, "network") ||
		strings.Contains(errStrLower, "refused") ||
		strings.Contains(errStrLower, "unreachable") ||
		strings.Contains(errStrLower, "no such host") {
		return ErrorTypeNetwork
	}

	// Check for API errors (rate limit, authentication, invalid request)
	if strings.Contains(errStrLower, "rate limit") ||
		strings.Contains(errStrLower, "authentication") ||
		strings.Contains(errStrLower, "unauthorized") ||
		strings.Contains(errStrLower, "forbidden") ||
		strings.Contains(errStrLower, "invalid") ||
		strings.Contains(errStrLower, "not found") ||
		strings.Contains(errStrLower, "collection") {
		return ErrorTypeAPI
	}

	// Check for SDK/protobuf errors
	if strings.Contains(errStrLower, "protobuf") ||
		strings.Contains(errStrLower, "marshal") ||
		strings.Contains(errStrLower, "unmarshal") ||
		strings.Contains(errStrLower, "serialization") {
		return ErrorTypeSDK
	}

	return ErrorTypeUnknown
}

// SeedDatabase is now in seed.go
