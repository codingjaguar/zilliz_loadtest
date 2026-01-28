package loadtest

import (
	"context"
	"fmt"
	"time"

	"zilliz-loadtest/internal/logger"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// SeedDatabase seeds the database with the specified number of vectors
func SeedDatabase(apiKey, databaseURL, collection string, vectorDim, totalVectors int) error {
	return SeedDatabaseWithBatchSize(apiKey, databaseURL, collection, vectorDim, totalVectors, DefaultBatchSize)
}

// SeedDatabaseWithBatchSize seeds the database with a custom batch size
func SeedDatabaseWithBatchSize(apiKey, databaseURL, collection string, vectorDim, totalVectors, batchSize int) error {
	return SeedDatabaseWithBatchSizeAndMetric(apiKey, databaseURL, collection, vectorDim, totalVectors, batchSize, entity.L2, false)
}

// SeedDatabaseWithBatchSizeAndMetric seeds the database with a custom batch size and metric type
// If skipCollectionCreation is true, it will skip automatic collection creation and only verify the collection exists
func SeedDatabaseWithBatchSizeAndMetric(apiKey, databaseURL, collection string, vectorDim, totalVectors, batchSize int, metricType entity.MetricType, skipCollectionCreation bool) error {
	// Validate inputs
	if apiKey == "" {
		return fmt.Errorf("API key is required for seed operation")
	}
	if databaseURL == "" {
		return fmt.Errorf("database URL is required for seed operation")
	}
	if collection == "" {
		return fmt.Errorf("collection name is required for seed operation")
	}
	if vectorDim <= 0 {
		return fmt.Errorf("vector dimension must be positive, got %d", vectorDim)
	}
	if totalVectors <= 0 {
		return fmt.Errorf("total vectors must be positive, got %d", totalVectors)
	}
	if batchSize <= 0 {
		return fmt.Errorf("batch size must be positive, got %d", batchSize)
	}
	if batchSize > 50000 {
		return fmt.Errorf("batch size too large (%d), maximum recommended is 50000 to avoid gRPC message size limits", batchSize)
	}

	ctx := context.Background()

	// Try to create client with retry
	retryConfig := DefaultRetryConfig()
	milvusClient, err := RetryClientCreation(ctx, apiKey, databaseURL, retryConfig)
	if err != nil {
		return fmt.Errorf("failed to create client after retries: %w", err)
	}
	defer milvusClient.Close()

	// Ensure collection exists with correct schema and index
	collectionConfig := CollectionConfig{
		CollectionName: collection,
		VectorDim:      vectorDim,
		MetricType:     metricType,
		IndexType:      "AUTOINDEX", // Default for Zilliz Cloud
		ShardNum:        1,           // Default to 1 shard
	}

	if skipCollectionCreation {
		logger.Info("Skipping automatic collection creation - verifying collection exists", "collection", collection)
	} else {
		logger.Info("Ensuring collection exists with correct schema", "collection", collection)
	}
	if err := EnsureCollectionExists(ctx, milvusClient, collectionConfig, skipCollectionCreation); err != nil {
		return fmt.Errorf("failed to ensure collection exists: %w", err)
	}

	totalBatches := (totalVectors + batchSize - 1) / batchSize

	printSeedHeader(collection, vectorDim, totalVectors, batchSize)

	startTime := time.Now()
	vectorsInserted := 0

	for batchNum := 0; batchNum < totalBatches; batchNum++ {
		batchStart := batchNum * batchSize
		batchEnd := batchStart + batchSize
		if batchEnd > totalVectors {
			batchEnd = totalVectors
		}
		currentBatchSize := batchEnd - batchStart

		if err := processSeedBatch(ctx, milvusClient, collection, vectorDim, batchNum+1, totalBatches, batchStart, currentBatchSize, &vectorsInserted, totalVectors, startTime); err != nil {
			return err
		}
	}

	printSeedSummary(vectorsInserted, time.Since(startTime))
	return nil
}

func printSeedHeader(collection string, vectorDim, totalVectors, batchSize int) {
	logger.Info("Starting database seed operation")
	logger.Info("Seed configuration",
		"collection", collection,
		"vector_dim", vectorDim,
		"total_vectors", totalVectors,
		"batch_size", batchSize)
}

func processSeedBatch(ctx context.Context, milvusClient client.Client, collection string, vectorDim, batchNum, totalBatches, batchStart, currentBatchSize int, vectorsInserted *int, totalVectors int, startTime time.Time) error {
	progressPercent := float64(*vectorsInserted) / float64(totalVectors) * 100
	logger.Debug("Generating batch",
		"progress_percent", progressPercent,
		"batch_num", batchNum,
		"total_batches", totalBatches,
		"batch_size", currentBatchSize)

	generateStart := time.Now()
	vectors := generateBatchVectors(vectorDim, batchStart, currentBatchSize, vectorsInserted, totalVectors, batchNum, totalBatches)
	generateTime := time.Since(generateStart)

	vectorColumn := entity.NewColumnFloatVector("vector", vectorDim, vectors)

	uploadProgressPercent := float64(*vectorsInserted) / float64(totalVectors) * 100
	logger.Debug("Uploading batch",
		"progress_percent", uploadProgressPercent,
		"batch_num", batchNum,
		"total_batches", totalBatches)

	batchStartTime := time.Now()

	// Retry batch insert on transient errors
	retryConfig := DefaultRetryConfig()
	retryConfig.MaxRetries = 2 // Fewer retries for batch operations
	var insertErr error
	err := RetryWithBackoff(ctx, func() error {
		_, err := milvusClient.Insert(ctx, collection, "", vectorColumn)
		if err != nil {
			insertErr = err
			return err
		}
		insertErr = nil
		return nil
	}, retryConfig)

	if err != nil {
		return fmt.Errorf("failed to insert batch %d/%d (%d vectors) after retries: %w. Check collection schema, vector dimension, and network connectivity", batchNum, totalBatches, currentBatchSize, insertErr)
	}
	uploadTime := time.Since(batchStartTime)

	*vectorsInserted += currentBatchSize
	totalBatchTime := time.Since(generateStart)
	rate := float64(currentBatchSize) / totalBatchTime.Seconds()

	elapsedTotal := time.Since(startTime)
	avgRate := float64(*vectorsInserted) / elapsedTotal.Seconds()
	remainingVectors := totalVectors - *vectorsInserted
	estimatedTimeRemaining := time.Duration(float64(remainingVectors)/avgRate) * time.Second

	progressPercent = float64(*vectorsInserted) / float64(totalVectors) * 100

	logger.Info("Batch completed",
		"progress_percent", progressPercent,
		"batch_num", batchNum,
		"total_batches", totalBatches,
		"vectors_inserted", currentBatchSize,
		"generate_time_ms", generateTime.Milliseconds(),
		"upload_time_ms", uploadTime.Milliseconds(),
		"total_time_ms", totalBatchTime.Milliseconds(),
		"rate_vec_per_sec", rate,
		"eta_seconds", estimatedTimeRemaining.Seconds())

	return nil
}

func generateBatchVectors(vectorDim, batchStart, currentBatchSize int, vectorsInserted *int, totalVectors, batchNum, totalBatches int) [][]float32 {
	vectors := make([][]float32, currentBatchSize)
	for i := 0; i < currentBatchSize; i++ {
		vectors[i] = generateSeedingVector(vectorDim, int64(batchStart+i))

		if (i+1)%ProgressUpdateIntervalVectors == 0 {
			progressPercent := float64(*vectorsInserted+i+1) / float64(totalVectors) * 100
			logger.Debug("Generating vectors progress",
				"progress_percent", progressPercent,
				"batch_num", batchNum,
				"total_batches", totalBatches,
				"vectors_generated", i+1,
				"batch_size", currentBatchSize)
		}
	}
	return vectors
}

func printSeedSummary(vectorsInserted int, totalElapsed time.Duration) {
	avgRate := float64(vectorsInserted) / totalElapsed.Seconds()

	logger.Info("Seed operation completed",
		"total_vectors", vectorsInserted,
		"total_time_seconds", totalElapsed.Seconds(),
		"avg_rate_vec_per_sec", avgRate)
}
