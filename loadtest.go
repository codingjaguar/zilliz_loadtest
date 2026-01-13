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
	start := time.Now()

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

	// Execute search
	searchResults, err := lt.client.Search(
		ctx,
		lt.collection,
		[]string{}, // partition names (empty for all partitions)
		"",         // expr (empty for no filter)
		[]string{}, // output fields (empty for IDs and scores only)
		[]entity.Vector{entity.FloatVector(queryVector)},
		"vector",      // vector field name
		lt.metricType, // metric type
		10,            // topK
		searchParams,
	)

	latency := time.Since(start)

	if err != nil {
		return QueryResult{
			Latency: latency,
			Error:   err,
		}
	}

	// Extract recall from search results
	// When enable_recall_calculation is true, Zilliz returns the recall in the response
	// According to docs: https://docs.zilliz.com/docs/tune-recall-rate#tune-recall-rate
	recall := 0.0

	if len(searchResults) > 0 && searchResults[0].ResultCount > 0 {
		// The recall is returned in the Fields as a column when enable_recall_calculation is true
		// According to the docs, it's returned as "recalls" (plural) in the response
		// Try both "recall" and "recalls" to handle different SDK versions
		fields := searchResults[0].Fields
		if fields != nil {
			// Try "recalls" first (as shown in Python docs)
			recallColumn := fields.GetColumn("recalls")
			if recallColumn == nil {
				// Fallback to "recall" (singular)
				recallColumn = fields.GetColumn("recall")
			}

			if recallColumn != nil {
				// The recall column should contain float values
				// Extract the first recall value (typically one per query)
				if floatColumn, ok := recallColumn.(*entity.ColumnFloat); ok {
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

// SeedDatabase seeds the database with the specified number of vectors using concurrent batching
func SeedDatabase(apiKey, databaseURL, collection string, vectorDim, totalVectors int) error {
	ctx := context.Background()

	// Create client
	milvusClient, err := createZillizClient(apiKey, databaseURL)
	if err != nil {
		return fmt.Errorf("failed to create client: %w", err)
	}
	defer milvusClient.Close()

	// Smaller batch size for better concurrency
	// With 768 dimensions, each vector is ~3KB, so 10,000 vectors = ~30MB (well under 64MB limit)
	batchSize := 10000
	concurrency := 3 // Number of concurrent workers
	
	totalBatches := (totalVectors + batchSize - 1) / batchSize

	fmt.Printf("\nStarting database seed operation\n")
	fmt.Printf("================================\n")
	fmt.Printf("Collection: %s\n", collection)
	fmt.Printf("Vector Dimension: %d\n", vectorDim)
	fmt.Printf("Total Vectors: %d\n", totalVectors)
	fmt.Printf("Batch Size: %d\n", batchSize)
	fmt.Printf("Concurrency: %d workers\n\n", concurrency)

	startTime := time.Now()
	vectorsInserted := int64(0)
	var mu sync.Mutex

	// Batch job structure
	type batchJob struct {
		batchNum       int
		startIdx       int
		endIdx         int
		currentBatchSize int
	}

	// Progress tracking
	type batchResult struct {
		batchNum       int
		vectors        int
		generateTime   time.Duration
		uploadTime     time.Duration
		totalTime      time.Duration
	}

	jobs := make(chan batchJob, totalBatches)
	results := make(chan batchResult, totalBatches)
	errors := make(chan error, 1)

	// Start worker pool
	var wg sync.WaitGroup
	for w := 0; w < concurrency; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for job := range jobs {
				// Generate vectors for this batch
				generateStart := time.Now()
				vectors := make([][]float32, job.currentBatchSize)
				for i := 0; i < job.currentBatchSize; i++ {
					vectors[i] = generateSeedingVector(vectorDim, int64(job.startIdx+i))
				}
				generateTime := time.Since(generateStart)

				// Create vector column
				vectorColumn := entity.NewColumnFloatVector("vector", vectorDim, vectors)

				// Insert batch
				uploadStart := time.Now()
				_, err := milvusClient.Insert(ctx, collection, "", vectorColumn)
				uploadTime := time.Since(uploadStart)
				totalTime := time.Since(generateStart)

				if err != nil {
					select {
					case errors <- fmt.Errorf("failed to insert batch %d: %w", job.batchNum+1, err):
					default:
					}
					return
				}

				// Send result
				results <- batchResult{
					batchNum:     job.batchNum,
					vectors:      job.currentBatchSize,
					generateTime: generateTime,
					uploadTime:   uploadTime,
					totalTime:    totalTime,
				}
			}
		}(w)
	}

	// Send all jobs
	go func() {
		defer close(jobs)
		for batchNum := 0; batchNum < totalBatches; batchNum++ {
			batchStart := batchNum * batchSize
			batchEnd := batchStart + batchSize
			if batchEnd > totalVectors {
				batchEnd = totalVectors
			}
			jobs <- batchJob{
				batchNum:       batchNum,
				startIdx:       batchStart,
				endIdx:         batchEnd,
				currentBatchSize: batchEnd - batchStart,
			}
		}
	}()

	// Close results channel when all workers are done
	go func() {
		wg.Wait()
		close(results)
	}()

	// Process results and update progress
	completedBatches := 0
	batchResults := make(map[int]batchResult) // Track results by batch number for ordered output
	resultsClosed := false

	for completedBatches < totalBatches {
		// First, try to process any results we have in order
		for completedBatches < totalBatches {
			if result, exists := batchResults[completedBatches]; exists {
				mu.Lock()
				vectorsInserted += int64(result.vectors)
				inserted := vectorsInserted
				mu.Unlock()

				// Calculate progress
				progressPercent := float64(inserted) / float64(totalVectors) * 100
				elapsedTotal := time.Since(startTime)
				avgRate := float64(inserted) / elapsedTotal.Seconds()
				remainingVectors := totalVectors - int(inserted)
				estimatedTimeRemaining := time.Duration(float64(remainingVectors)/avgRate) * time.Second
				rate := float64(result.vectors) / result.totalTime.Seconds()

				// Print progress
				fmt.Printf("[Progress: %.1f%%] Batch %d/%d: Inserted %d vectors (Generate: %v, Upload: %v, Total: %v, %.0f vec/s) [ETA: %v]\n",
					progressPercent, result.batchNum+1, totalBatches, result.vectors,
					result.generateTime.Round(time.Millisecond),
					result.uploadTime.Round(time.Millisecond),
					result.totalTime.Round(time.Millisecond),
					rate,
					estimatedTimeRemaining.Round(time.Second))

				// Remove processed result
				delete(batchResults, completedBatches)
				completedBatches++
			} else {
				// Next batch in sequence not ready yet
				break
			}
		}

		// If we've processed all batches, we're done
		if completedBatches >= totalBatches {
			break
		}

		// Wait for more results
		select {
		case err := <-errors:
			return err
		case result, ok := <-results:
			if !ok {
				// Channel closed, but we might still have unprocessed results
				resultsClosed = true
				// Continue processing any remaining results
				continue
			}
			
			// Store result for ordered processing
			batchResults[result.batchNum] = result
		}

		// If results channel is closed and we have no more results to process, break
		if resultsClosed && len(batchResults) == 0 {
			break
		}
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
