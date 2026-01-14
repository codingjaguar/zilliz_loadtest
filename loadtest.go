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
	level      int
	vectorDim  int
	metricType entity.MetricType
}

type TestResult struct {
	QPS          int
	P95Latency   float64 // in milliseconds
	P99Latency   float64 // in milliseconds
	AvgRecall    float64
	TotalQueries int
	Errors       int
}

type QueryResult struct {
	Latency time.Duration
	Recall  float64
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

func NewLoadTester(apiKey, databaseURL, collection string, level int, vectorDim int, metricTypeStr string) (*LoadTester, error) {
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
		level:      level,
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
	// Calculate interval between requests to achieve target QPS
	interval := time.Second / time.Duration(targetQPS)

	// Create context with timeout
	testCtx, cancel := context.WithTimeout(ctx, duration)
	defer cancel()

	var wg sync.WaitGroup
	resultsChan := make(chan QueryResult, targetQPS*10) // Buffer for results

	startTime := time.Now()
	queryCount := 0
	var mu sync.Mutex

	// Start query workers
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
	var recalls []float64
	errors := 0

	for result := range resultsChan {
		if result.Error != nil {
			errors++
			continue
		}
		latencies = append(latencies, float64(result.Latency.Milliseconds()))
		if result.Recall > 0 {
			recalls = append(recalls, result.Recall)
		}
	}

	// Calculate percentiles
	sort.Float64s(latencies)
	p95 := calculatePercentile(latencies, 95)
	p99 := calculatePercentile(latencies, 99)

	// Calculate average recall
	avgRecall := 0.0
	if len(recalls) > 0 {
		sum := 0.0
		for _, r := range recalls {
			sum += r
		}
		avgRecall = sum / float64(len(recalls))
	}

	actualQPS := float64(len(latencies)) / elapsed.Seconds()
	fmt.Printf("Completed: %d queries in %v (actual QPS: %.2f, errors: %d)\n",
		len(latencies), elapsed, actualQPS, errors)

	return TestResult{
		QPS:          targetQPS,
		P95Latency:   p95,
		P99Latency:   p99,
		AvgRecall:    avgRecall,
		TotalQueries: len(latencies),
		Errors:       errors,
	}, nil
}

// CustomSearchParam implements SearchParam interface with level and enable_recall_calculation
type CustomSearchParam struct {
	level                   int
	enableRecallCalculation bool
}

func (c *CustomSearchParam) Params() map[string]interface{} {
	params := make(map[string]interface{})
	params["level"] = c.level
	params["enable_recall_calculation"] = c.enableRecallCalculation
	return params
}

func (c *CustomSearchParam) AddRadius(radius float64) {
	// Not used for AUTOINDEX
}

func (c *CustomSearchParam) AddRangeFilter(rangeFilter float64) {
	// Not used for AUTOINDEX
}

func (lt *LoadTester) executeQuery(ctx context.Context, queryNum int) QueryResult {
	// Generate a random query vector (dummy vector for testing)
	// In a real scenario, you'd use actual query vectors from your dataset
	queryVector := generateRandomVector(lt.vectorDim)

	// Create search parameters with level and enable_recall_calculation
	// The level parameter (1-10) optimizes for recall vs latency
	// Level 10 optimizes for recall at the expense of latency, level 1 optimizes for latency
	// enable_recall_calculation tells Zilliz to calculate and return the recall rate
	searchParams := &CustomSearchParam{
		level:                   lt.level,
		enableRecallCalculation: true,
	}

	// Measure latency only for the search operation (not including vector generation)
	// Use high-precision timing for accurate measurements
	searchStart := time.Now()
	
	// Execute search
	// When enable_recall_calculation is true, we need to request "recalls" as an output field
	// to get the recall values in the response (even though it's not a collection field)
	searchResults, err := lt.client.Search(
		ctx,
		lt.collection,
		[]string{},        // partition names (empty for all partitions)
		"",                 // expr (empty for no filter)
		[]string{"recalls"}, // output fields - request "recalls" to get recall calculation results
		[]entity.Vector{entity.FloatVector(queryVector)},
		"vector",      // vector field name
		lt.metricType, // metric type
		10,            // topK
		searchParams,
	)

	// Measure latency immediately after Search() returns
	// This includes network round-trip and server processing time
	latency := time.Since(searchStart)

	if err != nil {
		return QueryResult{
			Latency: latency,
			Error:   err,
		}
	}

	// Extract recall from search results
	// When enable_recall_calculation is true, Zilliz returns the recall in the response
	// According to docs: https://docs.zilliz.com/docs/tune-recall-rate#tune-recall-rate
	// The recall is returned as "recalls" (plural) - it may be in Fields or as metadata
	recall := 0.0

	if len(searchResults) > 0 {
		result := searchResults[0]
		
		// Try to get recall from Fields
		if result.Fields != nil {
			// Try "recalls" first (as shown in Python docs)
			recallColumn := result.Fields.GetColumn("recalls")
			if recallColumn == nil {
				// Fallback to "recall" (singular) in case SDK version differs
				recallColumn = result.Fields.GetColumn("recall")
			}

			if recallColumn != nil {
				// The recall column is returned as ColumnDynamic (JSON field)
				// Extract the first recall value (typically one per query, but may have one per result)
				if dynamicColumn, ok := recallColumn.(*entity.ColumnDynamic); ok {
					if dynamicColumn.Len() > 0 {
						// Try to get as double (float64)
						if val, err := dynamicColumn.GetAsDouble(0); err == nil {
							recall = val
						} else {
							// Fallback: try to get as interface{} and convert
							if val, err := dynamicColumn.Get(0); err == nil {
								switch v := val.(type) {
								case float64:
									recall = v
								case float32:
									recall = float64(v)
								case int64:
									recall = float64(v) / 100.0 // Convert from percentage
								case int:
									recall = float64(v) / 100.0 // Convert from percentage
								}
							}
						}
					}
				} else if floatColumn, ok := recallColumn.(*entity.ColumnFloat); ok {
					if floatColumn.Len() > 0 {
						if val, err := floatColumn.ValueByIdx(0); err == nil {
							recall = float64(val)
						}
					}
				} else if doubleColumn, ok := recallColumn.(*entity.ColumnDouble); ok {
					if doubleColumn.Len() > 0 {
						if val, err := doubleColumn.GetAsDouble(0); err == nil {
							recall = val
						}
					}
				} else if int64Column, ok := recallColumn.(*entity.ColumnInt64); ok {
					// Some SDK versions might return as int64 (0-100 scale)
					if int64Column.Len() > 0 {
						if val, err := int64Column.Get(0); err == nil {
							if intVal, ok := val.(int64); ok {
								recall = float64(intVal) / 100.0 // Convert from percentage
							}
						}
					}
				}
			}
		}
	}

	return QueryResult{
		Latency: latency,
		Recall:  recall,
	}
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
