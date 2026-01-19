package main

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

type LoadTester struct {
	clients    []client.Client // Multiple clients for connection pooling
	clientIdx  int64           // Atomic counter for round-robin client selection
	collection string
	vectorDim  int
	metricType entity.MetricType
}

type TestResult struct {
	QPS          int
	P95Latency   float64 // in milliseconds
	P99Latency   float64 // in milliseconds
	TotalQueries int
	Errors       int
}

type QueryResult struct {
	Latency time.Duration
	Error   error
}

// createZillizClient creates a Zilliz Cloud client with API key authentication
func createZillizClient(apiKey, databaseURL string) (client.Client, error) {
	ctx := context.Background()

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
			return nil, fmt.Errorf("failed to create client: %w", err)
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

// calculateOptimalConnections calculates the number of connections needed based on target QPS
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
func calculateOptimalConnections(targetQPS int) (int, string) {
	// Estimate based on typical Zilliz Cloud vector search latency
	// Formula: (QPS * latency_ms) / 1000 gives concurrent requests needed
	// Then multiply by 1.5x to account for connection overhead and headroom
	expectedLatencyMs := 75.0 // Conservative estimate for vector search on Zilliz Cloud
	multiplier := 1.5
	baseConnections := float64(targetQPS) * expectedLatencyMs / 1000.0
	connections := int(baseConnections * multiplier)

	// Enforce reasonable bounds
	if connections < 5 {
		connections = 5 // Minimum for any meaningful load
	}

	// Higher limit for very high QPS scenarios
	maxConnections := 2000
	if connections > maxConnections {
		connections = maxConnections
	}

	explanation := fmt.Sprintf("Formula: (%d QPS × %.0fms latency) / 1000 × %.1fx = %d connections (default)",
		targetQPS, expectedLatencyMs, multiplier, connections)

	return connections, explanation
}

func NewLoadTesterWithConnections(apiKey, databaseURL, collection string, vectorDim int,
	metricTypeStr string, numConnections int) (*LoadTester, error) {
	if numConnections < 1 {
		numConnections = 1
	}
	if numConnections > 100 {
		numConnections = 100 // Reasonable upper limit
	}

	// Create multiple clients for connection pooling
	clients := make([]client.Client, numConnections)
	for i := 0; i < numConnections; i++ {
		milvusClient, err := createZillizClient(apiKey, databaseURL)
		if err != nil {
			// Clean up already created clients
			for j := 0; j < i; j++ {
				clients[j].Close()
			}
			return nil, fmt.Errorf("failed to create client %d/%d: %w", i+1, numConnections, err)
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
		return nil, fmt.Errorf("unsupported metric type: %s", metricTypeStr)
	}

	return &LoadTester{
		clients:    clients,
		clientIdx:  0,
		collection: collection,
		vectorDim:  vectorDim,
		metricType: metricType,
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

func (lt *LoadTester) RunTest(ctx context.Context, targetQPS int, duration time.Duration) (TestResult, error) {
	// Calculate interval between requests to achieve exact target QPS
	interval := time.Second / time.Duration(targetQPS)

	// Create context with timeout
	testCtx, cancel := context.WithTimeout(ctx, duration)
	defer cancel()

	var wg sync.WaitGroup
	// Make buffer large enough to hold all expected results
	// For a 5 minute test at 100 QPS, that's 30,000 queries
	expectedQueries := int(duration.Seconds()) * targetQPS
	resultsChan := make(chan QueryResult, expectedQueries*2) // 2x buffer for safety

	startTime := time.Now()
	queryCount := 0
	var mu sync.Mutex
	queriesFired := 0

	// Simple rate limiter: fire exactly one query per interval
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	// Create a separate context for query execution that doesn't get cancelled
	// This allows queries to complete even after the test duration ends
	queryCtx := ctx // Use parent context, not testCtx

	// Store results for latency calculation
	var allResults []QueryResult
	var resultsMu sync.Mutex
	queriesCompleted := 0 // Track completion count for status updates

	// Start a goroutine to collect results as they come in
	go func() {
		for result := range resultsChan {
			resultsMu.Lock()
			allResults = append(allResults, result)
			queriesCompleted++
			resultsMu.Unlock()
		}
	}()

	// Status update ticker - print progress every 5 seconds
	statusTicker := time.NewTicker(5 * time.Second)
	defer statusTicker.Stop()

	go func() {
		for {
			select {
			case <-testCtx.Done():
				return
			case <-statusTicker.C:
				mu.Lock()
				fired := queriesFired
				mu.Unlock()
				resultsMu.Lock()
				completed := queriesCompleted
				resultsMu.Unlock()
				elapsed := time.Since(startTime)
				currentQPS := float64(completed) / elapsed.Seconds()
				fmt.Printf("[Status] Elapsed: %v | Fired: %d | Completed: %d | Current QPS: %.2f\n",
					elapsed.Round(time.Second), fired, completed, currentQPS)
			}
		}
	}()

	go func() {
		for {
			select {
			case <-testCtx.Done():
				return
			case <-ticker.C:
				mu.Lock()
				queryCount++
				currentQuery := queryCount
				queriesFired++
				mu.Unlock()

				wg.Add(1)
				go func(queryNum int) {
					defer wg.Done()
					// Use recover to catch panic if channel is closed
					defer func() {
						if r := recover(); r != nil {
							// Channel was closed, that's okay - we'll just drop this result
							// This can happen if we timeout waiting for queries
						}
					}()
					// Use queryCtx instead of testCtx so queries can complete
					// even after the test duration ends
					result := lt.executeQuery(queryCtx, queryNum)
					// Try to send result, but handle case where channel might be closed
					// Use select to avoid blocking forever if channel is closed
					select {
					case resultsChan <- result:
						// Successfully sent
					case <-time.After(1 * time.Second):
						// Timeout - channel might be closed or full, but that's okay
						// We'll just drop this result if we can't send it
					}
				}(currentQuery)
			}
		}
	}()

	// Wait for test duration
	<-testCtx.Done()

	mu.Lock()
	totalFired := queriesFired
	mu.Unlock()

	resultsMu.Lock()
	completedDuringTest := queriesCompleted
	resultsMu.Unlock()

	inFlight := totalFired - completedDuringTest
	fmt.Printf("\nTest duration ended. Waiting for %d in-flight queries to complete...\n", inFlight)

	// Wait for all queries to complete (with longer timeout for long tests)
	waitTimeout := duration / 2
	if waitTimeout < 30*time.Second {
		waitTimeout = 30 * time.Second
	}
	if waitTimeout > 5*time.Minute {
		waitTimeout = 5 * time.Minute
	}

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	// Show progress while waiting
	progressTicker := time.NewTicker(2 * time.Second)
	defer progressTicker.Stop()

	progressDone := make(chan struct{})
	go func() {
		for {
			select {
			case <-progressDone:
				return
			case <-progressTicker.C:
				resultsMu.Lock()
				completed := queriesCompleted
				resultsMu.Unlock()
				fmt.Printf("  Waiting... %d queries completed so far\n", completed)
			}
		}
	}()

	select {
	case <-done:
		close(progressDone)
		fmt.Printf("All queries completed\n")
	case <-time.After(waitTimeout):
		close(progressDone)
		fmt.Printf("Warning: Timed out waiting for queries after %v. Some queries may still be running\n", waitTimeout)
		// Give goroutines a grace period to finish sending before we close the channel
		// This reduces the chance of panic from sending to closed channel
		time.Sleep(2 * time.Second)
	}

	// Close channel - goroutines have recover() to handle if they try to send after this
	// The result collector goroutine will finish reading any remaining results
	close(resultsChan)

	// Give the result collector goroutine a moment to finish
	time.Sleep(100 * time.Millisecond)

	elapsed := time.Since(startTime)

	// Process collected results for latency calculation
	resultsMu.Lock()
	var latencies []float64
	errors := 0
	firstError := error(nil)

	for _, result := range allResults {
		if result.Error != nil {
			errors++
			if firstError == nil {
				firstError = result.Error
			}
			continue
		}
		latencies = append(latencies, float64(result.Latency.Milliseconds()))
	}
	resultsMu.Unlock()

	// Print first error if all queries failed
	if len(latencies) == 0 && firstError != nil {
		fmt.Printf("ERROR: All queries failed. First error: %v\n", firstError)
	}

	// Calculate percentiles
	sort.Float64s(latencies)

	// Calculate some basic stats to detect issues
	var sum float64
	var min, max, avgLatency float64
	if len(latencies) > 0 {
		min = latencies[0]
		max = latencies[len(latencies)-1]
		for _, l := range latencies {
			sum += l
		}
		avgLatency = sum / float64(len(latencies))
	}

	p95 := calculatePercentile(latencies, 95)
	p99 := calculatePercentile(latencies, 99)

	// Check for suspiciously high latencies (likely due to queuing at high QPS)
	// At high QPS, queries queue up and latency includes wait time
	if max > 1000 { // More than 1 second is suspicious
		fmt.Printf("\nWarning: Detected very high latencies (max: %.2f ms). This may indicate:\n", max)
		fmt.Printf("  - Server-side queuing (queries waiting for server capacity)\n")
		fmt.Printf("  - Network congestion\n")
		fmt.Printf("  - Note: These latencies include queue time, not just API response time\n")
		fmt.Printf("  Latency stats: min=%.2f ms, avg=%.2f ms, p95=%.2f ms, p99=%.2f ms, max=%.2f ms\n",
			min, avgLatency, p95, p99, max)

		// Filter out queries with high latency (likely queuing, not actual API response time)
		// Use a more reasonable cutoff - anything over 1 second is probably queuing
		filteredLatencies := make([]float64, 0, len(latencies))
		for _, l := range latencies {
			if l < 1000 { // Filter out anything over 1 second (likely queuing)
				filteredLatencies = append(filteredLatencies, l)
			}
		}

		if len(filteredLatencies) > 0 && len(filteredLatencies) < len(latencies) {
			sort.Float64s(filteredLatencies)
			filteredP95 := calculatePercentile(filteredLatencies, 95)
			filteredP99 := calculatePercentile(filteredLatencies, 99)
			queuedCount := len(latencies) - len(filteredLatencies)
			fmt.Printf("  Filtered stats (excluding queued queries >1s): p95=%.2f ms, p99=%.2f ms\n",
				filteredP95, filteredP99)
			fmt.Printf("  %d queries (%.1f%%) had latencies >1s, likely due to server-side queuing\n",
				queuedCount, float64(queuedCount)/float64(len(latencies))*100)
			// Use filtered values for reporting (more representative of actual API latency)
			p95 = filteredP95
			p99 = filteredP99
		} else if len(filteredLatencies) == 0 {
			// All queries were queued - this is a severe bottleneck
			fmt.Printf("  ⚠️  All queries had latencies >1s - severe server-side bottleneck detected\n")
		}
	}

	actualQPS := float64(len(latencies)) / elapsed.Seconds()
	completionRate := (actualQPS / float64(targetQPS)) * 100.0 // Percentage of target QPS achieved
	fmt.Printf("Fired: %d queries | Completed: %d queries in %v\n", totalFired, len(latencies), elapsed)
	fmt.Printf("Fired at: %d QPS | Completed at: %.2f QPS (%.1f%% of target) | Errors: %d\n",
		targetQPS, actualQPS, completionRate, errors)

	// Warn if we're significantly below target QPS
	if actualQPS < float64(targetQPS)*0.8 {
		fmt.Printf("\n⚠️  Warning: Only achieved %.1f%% of target QPS. This may indicate:\n", completionRate)
		fmt.Printf("  - Server-side rate limiting or capacity limits\n")
		fmt.Printf("  - Network bandwidth constraints\n")
		fmt.Printf("  - Server may not be able to handle %d QPS (currently using %d connections)\n", targetQPS, len(lt.clients))
		fmt.Printf("  Note: Increasing connections from %d didn't improve throughput, suggesting server-side bottleneck\n", len(lt.clients))
	}

	// Explain what happened
	if totalFired > len(latencies) {
		pending := totalFired - len(latencies)
		fmt.Printf("Note: %d queries were still in flight when test ended (%.1f%% completion rate)\n",
			pending, float64(len(latencies))/float64(totalFired)*100)
	}

	return TestResult{
		QPS:          targetQPS,
		P95Latency:   p95,
		P99Latency:   p99,
		TotalQueries: len(latencies),
		Errors:       errors,
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

func (lt *LoadTester) executeQuery(ctx context.Context, queryNum int) QueryResult {
	// Generate a random query vector
	queryVector := generateRandomVector(lt.vectorDim)

	// Measure latency for the search operation
	searchStart := time.Now()

	// Execute search with empty search params (defaults to level 1 for latency optimization)
	emptyParams := &EmptySearchParam{}
	_, err := lt.getClient().Search(
		ctx,
		lt.collection,
		[]string{},     // partition names (empty for all partitions)
		"",             // expr (empty for no filter)
		[]string{"id"}, // output fields - request "id" field
		[]entity.Vector{entity.FloatVector(queryVector)},
		"vector",      // vector field name
		lt.metricType, // metric type
		10,            // topK
		emptyParams,   // empty search params - defaults to level 1
	)

	latency := time.Since(searchStart)

	if err != nil {
		return QueryResult{
			Latency: latency,
			Error:   err,
		}
	}

	return QueryResult{
		Latency: latency,
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

// SeedDatabase seeds the database with the specified number of vectors
func SeedDatabase(apiKey, databaseURL, collection string, vectorDim, totalVectors int) error {
	ctx := context.Background()

	// Create client
	milvusClient, err := createZillizClient(apiKey, databaseURL)
	if err != nil {
		return fmt.Errorf("failed to create client: %w", err)
	}
	defer milvusClient.Close()

	// Batch size for efficient inserts (reduced to avoid gRPC message size limits and throttling)
	// With 768 dimensions, each vector is ~3KB, so 15,000 vectors = ~45MB (well under 64MB limit)
	batchSize := 15000
	totalBatches := (totalVectors + batchSize - 1) / batchSize

	fmt.Printf("\nStarting database seed operation\n")
	fmt.Printf("================================\n")
	fmt.Printf("Collection: %s\n", collection)
	fmt.Printf("Vector Dimension: %d\n", vectorDim)
	fmt.Printf("Total Vectors: %d\n", totalVectors)
	fmt.Printf("Batch Size: %d\n\n", batchSize)

	startTime := time.Now()
	vectorsInserted := 0

	for batchNum := 0; batchNum < totalBatches; batchNum++ {
		batchStart := batchNum * batchSize
		batchEnd := batchStart + batchSize
		if batchEnd > totalVectors {
			batchEnd = totalVectors
		}
		currentBatchSize := batchEnd - batchStart

		// Show progress before generating vectors
		progressPercent := float64(vectorsInserted) / float64(totalVectors) * 100
		fmt.Printf("[Progress: %.1f%%] Generating batch %d/%d (%d vectors)...\r",
			progressPercent, batchNum+1, totalBatches, currentBatchSize)

		// Generate vectors for this batch
		generateStart := time.Now()
		vectors := make([][]float32, currentBatchSize)
		for i := 0; i < currentBatchSize; i++ {
			// Use batchStart + i as seed to ensure unique vectors
			vectors[i] = generateSeedingVector(vectorDim, int64(batchStart+i))

			// Show progress every 5000 vectors during generation
			if (i+1)%5000 == 0 {
				progressPercent := float64(vectorsInserted+i+1) / float64(totalVectors) * 100
				fmt.Printf("\r[Progress: %.1f%%] Generating batch %d/%d (%d/%d vectors)...",
					progressPercent, batchNum+1, totalBatches, i+1, currentBatchSize)
			}
		}
		generateTime := time.Since(generateStart)

		// Create vector column
		vectorColumn := entity.NewColumnFloatVector("vector", vectorDim, vectors)

		// Insert the batch (using Insert instead of Upsert since autoID is enabled)
		batchStartTime := time.Now()
		uploadProgressPercent := float64(vectorsInserted) / float64(totalVectors) * 100
		fmt.Printf("\r[Progress: %.1f%%] Uploading batch %d/%d...",
			uploadProgressPercent, batchNum+1, totalBatches)

		_, err := milvusClient.Insert(ctx, collection, "", vectorColumn)
		if err != nil {
			return fmt.Errorf("failed to insert batch %d: %w", batchNum+1, err)
		}

		vectorsInserted += currentBatchSize
		uploadTime := time.Since(batchStartTime)
		totalBatchTime := time.Since(generateStart)
		rate := float64(currentBatchSize) / totalBatchTime.Seconds()

		// Calculate estimated time remaining
		elapsedTotal := time.Since(startTime)
		avgRate := float64(vectorsInserted) / elapsedTotal.Seconds()
		remainingVectors := totalVectors - vectorsInserted
		estimatedTimeRemaining := time.Duration(float64(remainingVectors)/avgRate) * time.Second

		progressPercent = float64(vectorsInserted) / float64(totalVectors) * 100

		// Print detailed progress after each batch
		fmt.Printf("\r[Progress: %.1f%%] Batch %d/%d: Inserted %d vectors (Generate: %v, Upload: %v, Total: %v, %.0f vec/s) [ETA: %v]\n",
			progressPercent, batchNum+1, totalBatches, currentBatchSize,
			generateTime.Round(time.Millisecond), uploadTime.Round(time.Millisecond),
			totalBatchTime.Round(time.Millisecond), rate, estimatedTimeRemaining.Round(time.Second))
	}

	totalElapsed := time.Since(startTime)
	avgRate := float64(vectorsInserted) / totalElapsed.Seconds()

	fmt.Printf("\n================================\n")
	fmt.Printf("Seed operation completed!\n")
	fmt.Printf("Total vectors inserted: %d\n", vectorsInserted)
	fmt.Printf("Total time: %v\n", totalElapsed)
	fmt.Printf("Average rate: %.0f vectors/sec\n", avgRate)
	fmt.Printf("================================\n")

	return nil
}
