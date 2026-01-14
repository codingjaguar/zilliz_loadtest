package main

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

type LoadTester struct {
	client     client.Client
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
	milvusClient, err := createZillizClient(apiKey, databaseURL)
	if err != nil {
		return nil, err
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
		return nil, fmt.Errorf("unsupported metric type: %s", metricTypeStr)
	}

	return &LoadTester{
		client:     milvusClient,
		collection: collection,
		vectorDim:  vectorDim,
		metricType: metricType,
	}, nil
}

func (lt *LoadTester) Close() {
	if lt.client != nil {
		lt.client.Close()
	}
}

func (lt *LoadTester) RunTest(ctx context.Context, targetQPS int, duration time.Duration) (TestResult, error) {
	// Calculate interval between requests to achieve exact target QPS
	interval := time.Second / time.Duration(targetQPS)

	// Create context with timeout
	testCtx, cancel := context.WithTimeout(ctx, duration)
	defer cancel()

	var wg sync.WaitGroup
	resultsChan := make(chan QueryResult, targetQPS*10) // Buffer for results

	startTime := time.Now()
	queryCount := 0
	var mu sync.Mutex
	queriesFired := 0

	// Simple rate limiter: fire exactly one query per interval
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

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
					result := lt.executeQuery(testCtx, queryNum)
					select {
					case resultsChan <- result:
					case <-testCtx.Done():
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

	// Wait for all queries to complete (with timeout)
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(10 * time.Second):
		fmt.Printf("Warning: Some queries may still be running\n")
	}

	close(resultsChan)
	elapsed := time.Since(startTime)

	// Collect results
	var latencies []float64
	errors := 0
	firstError := error(nil)

	for result := range resultsChan {
		if result.Error != nil {
			errors++
			if firstError == nil {
				firstError = result.Error
			}
			continue
		}
		latencies = append(latencies, float64(result.Latency.Milliseconds()))
	}

	// Print first error if all queries failed
	if len(latencies) == 0 && firstError != nil {
		fmt.Printf("ERROR: All queries failed. First error: %v\n", firstError)
	}

	// Calculate percentiles
	sort.Float64s(latencies)
	p95 := calculatePercentile(latencies, 95)
	p99 := calculatePercentile(latencies, 99)

	actualQPS := float64(len(latencies)) / elapsed.Seconds()
	fmt.Printf("Fired: %d queries | Completed: %d queries in %v\n", totalFired, len(latencies), elapsed)
	fmt.Printf("Fired at: %d QPS | Completed at: %.2f QPS | Errors: %d\n", targetQPS, actualQPS, errors)

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
	_, err := lt.client.Search(
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
