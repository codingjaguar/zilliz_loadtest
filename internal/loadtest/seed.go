package loadtest

import (
	"context"
	"fmt"
	"strings"
	"time"

	"zilliz-loadtest/internal/datasource"
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

// SeedDatabaseWithSource seeds the database with specified data source (synthetic or cohere)
func SeedDatabaseWithSource(apiKey, databaseURL, collection string, vectorDim, totalVectors, batchSize int, metricType entity.MetricType, seedSource string, skipCollectionCreation, dropCollectionBeforeSeed bool) error {
	seedSource = strings.ToLower(strings.TrimSpace(seedSource))

	// Drop collection if requested
	if dropCollectionBeforeSeed && !skipCollectionCreation {
		ctx := context.Background()
		c, err := client.NewClient(ctx, client.Config{
			Address: databaseURL,
			APIKey:  apiKey,
		})
		if err != nil {
			return fmt.Errorf("failed to connect to Milvus: %w", err)
		}
		defer c.Close()

		// Check if collection exists
		has, err := c.HasCollection(ctx, collection)
		if err != nil {
			return fmt.Errorf("failed to check collection existence: %w", err)
		}

		if has {
			logger.Info("Dropping existing collection before seeding", "collection", collection)
			err = c.DropCollection(ctx, collection)
			if err != nil {
				return fmt.Errorf("failed to drop collection: %w", err)
			}
			logger.Info("Collection dropped successfully", "collection", collection)
		}
	}

	switch seedSource {
	case "synthetic":
		return SeedDatabaseWithBatchSizeAndMetric(apiKey, databaseURL, collection, vectorDim, totalVectors, batchSize, metricType, skipCollectionCreation)
	case "cohere":
		return SeedDatabaseWithCohere(apiKey, databaseURL, collection, totalVectors, batchSize, metricType, skipCollectionCreation)
	case "cohere-full":
		return SeedDatabaseWithCohereFullCorpus(apiKey, databaseURL, collection, batchSize, metricType, skipCollectionCreation)
	case "bulk-import":
		return SeedDatabaseWithBulkImport(apiKey, databaseURL, collection, metricType, skipCollectionCreation)
	default:
		// Check if it's a VDBBench dataset name (e.g., "vdbbench:cohere_medium_1m")
		if strings.HasPrefix(seedSource, "vdbbench:") {
			datasetName := strings.TrimPrefix(seedSource, "vdbbench:")
			return SeedDatabaseWithVDBBench(apiKey, databaseURL, collection, datasetName, metricType)
		}
		// Check if it's a BEIR dataset name (e.g., "beir:fiqa")
		// BEIR datasets have human-labeled qrels for true business recall
		if strings.HasPrefix(seedSource, "beir:") {
			datasetName := strings.TrimPrefix(seedSource, "beir:")
			return SeedDatabaseWithBEIR(apiKey, databaseURL, collection, datasetName, batchSize)
		}
		return fmt.Errorf("unknown seed source: %s (valid options: synthetic, cohere, cohere-full, bulk-import, vdbbench:<dataset>, beir:<dataset>)", seedSource)
	}
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
		ShardNum:       1,           // Default to 1 shard
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

	// After seeding, proactively load the collection so it is query-ready.
	// Use a bounded timeout so we don't hang forever on very large collections.
	loadCtx, cancel := context.WithTimeout(ctx, 10*time.Minute)
	defer cancel()
	if err := FlushAndLoadCollection(loadCtx, milvusClient, collection); err != nil {
		return err
	}
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

// SeedDatabaseWithCohere seeds the database using Cohere Wikipedia embeddings
func SeedDatabaseWithCohere(apiKey, databaseURL, collection string, totalVectors, batchSize int, metricType entity.MetricType, skipCollectionCreation bool) error {
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

	logger.Info("Seeding with Cohere Wikipedia embeddings", "total_vectors", totalVectors)

	// Initialize downloader and reader
	downloader := datasource.NewCohereDownloader("")
	reader := datasource.NewCohereReader(downloader)

	// Read embeddings from dataset (with lazy download)
	logger.Info("Reading embeddings from Cohere dataset (will download if needed)")
	embeddings, err := reader.ReadEmbeddings(totalVectors)
	if err != nil {
		return fmt.Errorf("failed to read Cohere embeddings: %w", err)
	}

	if len(embeddings) == 0 {
		return fmt.Errorf("no embeddings were read from dataset")
	}

	vectorDim := len(embeddings[0].Embedding)
	logger.Info("Loaded embeddings from Cohere dataset",
		"count", len(embeddings),
		"dimension", vectorDim)

	// Create client with retry
	retryConfig := DefaultRetryConfig()
	milvusClient, err := RetryClientCreation(ctx, apiKey, databaseURL, retryConfig)
	if err != nil {
		return fmt.Errorf("failed to create client after retries: %w", err)
	}
	defer milvusClient.Close()

	// Ensure collection exists with Cohere schema (includes title and text fields)
	collectionConfig := CollectionConfig{
		CollectionName: collection,
		VectorDim:      vectorDim,
		MetricType:     metricType,
		IndexType:      "AUTOINDEX",
		ShardNum:       1,
	}

	if skipCollectionCreation {
		logger.Info("Skipping automatic collection creation - verifying collection exists", "collection", collection)
	} else {
		logger.Info("Ensuring collection exists with Cohere schema", "collection", collection)
	}

	// Create collection with extended schema for Cohere data
	if !skipCollectionCreation {
		if err := createCohereCollection(ctx, milvusClient, collectionConfig); err != nil {
			return fmt.Errorf("failed to create collection: %w", err)
		}
	}

	// Insert embeddings in batches
	totalBatches := (len(embeddings) + batchSize - 1) / batchSize
	printSeedHeader(collection, vectorDim, len(embeddings), batchSize)

	startTime := time.Now()
	vectorsInserted := 0

	for batchNum := 0; batchNum < totalBatches; batchNum++ {
		batchStart := batchNum * batchSize
		batchEnd := batchStart + batchSize
		if batchEnd > len(embeddings) {
			batchEnd = len(embeddings)
		}

		batch := embeddings[batchStart:batchEnd]

		if err := insertCohereBatch(ctx, milvusClient, collection, batch, batchNum+1, totalBatches, &vectorsInserted, len(embeddings), startTime); err != nil {
			return err
		}
	}

	printSeedSummary(vectorsInserted, time.Since(startTime))

	// Load collection
	loadCtx, cancel := context.WithTimeout(ctx, 10*time.Minute)
	defer cancel()
	if err := FlushAndLoadCollection(loadCtx, milvusClient, collection); err != nil {
		return err
	}

	return nil
}

// createCohereCollection creates a collection with schema for Cohere data (includes title, text fields)
func createCohereCollection(ctx context.Context, milvusClient client.Client, config CollectionConfig) error {
	// Check if collection exists
	has, err := milvusClient.HasCollection(ctx, config.CollectionName)
	if err != nil {
		return fmt.Errorf("failed to check collection: %w", err)
	}

	if has {
		logger.Info("Collection already exists", "collection", config.CollectionName)
		return nil
	}

	// Create fields for Cohere metadata using builder pattern
	idField := entity.NewField().
		WithName("id").
		WithDataType(entity.FieldTypeVarChar).
		WithMaxLength(256).
		WithIsPrimaryKey(true).
		WithIsAutoID(false)

	titleField := entity.NewField().
		WithName("title").
		WithDataType(entity.FieldTypeVarChar).
		WithMaxLength(2048)

	textField := entity.NewField().
		WithName("text").
		WithDataType(entity.FieldTypeVarChar).
		WithMaxLength(65535)

	vectorField := entity.NewField().
		WithName("vector").
		WithDataType(entity.FieldTypeFloatVector).
		WithDim(int64(config.VectorDim))

	// Create schema
	schema := &entity.Schema{
		CollectionName: config.CollectionName,
		AutoID:         false,
		Fields:         []*entity.Field{idField, titleField, textField, vectorField},
	}

	logger.Info("Creating collection", "collection", config.CollectionName, "vector_dim", config.VectorDim, "metric_type", config.MetricType, "shard_num", config.ShardNum)

	if err := milvusClient.CreateCollection(ctx, schema, config.ShardNum); err != nil {
		return fmt.Errorf("failed to create collection: %w", err)
	}

	logger.Info("Collection created successfully", "collection", config.CollectionName)

	// Create index
	logger.Info("Creating index on vector field", "collection", config.CollectionName, "index_type", config.IndexType, "metric_type", config.MetricType)

	idx, err := entity.NewIndexAUTOINDEX(config.MetricType)
	if err != nil {
		return fmt.Errorf("failed to create index config: %w", err)
	}

	if err := milvusClient.CreateIndex(ctx, config.CollectionName, "vector", idx, false); err != nil {
		return fmt.Errorf("failed to create index: %w", err)
	}

	logger.Info("Index created successfully", "collection", config.CollectionName)

	// Wait for index build (simplified - just check once after a delay)
	time.Sleep(2 * time.Second)
	state, err := milvusClient.GetIndexState(ctx, config.CollectionName, "vector")
	if err == nil {
		logger.Info("Index build completed", "collection", config.CollectionName, "state", state)
	}

	return nil
}

// SeedDatabaseWithCohereFullCorpus seeds with the entire 8.8M MS MARCO corpus
// Uses streaming to avoid loading all vectors into memory at once
func SeedDatabaseWithCohereFullCorpus(apiKey, databaseURL, collection string, batchSize int, metricType entity.MetricType, skipCollectionCreation bool) error {
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
	if batchSize <= 0 {
		batchSize = DefaultBatchSize
	}
	if batchSize > 50000 {
		return fmt.Errorf("batch size too large (%d), maximum recommended is 50000 to avoid gRPC message size limits", batchSize)
	}

	ctx := context.Background()

	logger.Info("Seeding with full MS MARCO corpus (~8.8M documents)",
		"total_files", datasource.TOTAL_CORPUS_FILES,
		"batch_size", batchSize)

	// Create client with retry
	retryConfig := DefaultRetryConfig()
	milvusClient, err := RetryClientCreation(ctx, apiKey, databaseURL, retryConfig)
	if err != nil {
		return fmt.Errorf("failed to create client after retries: %w", err)
	}
	defer milvusClient.Close()

	// Create collection with Cohere schema (1024 dim for Embed V3)
	vectorDim := 1024 // Cohere Embed V3 dimension
	collectionConfig := CollectionConfig{
		CollectionName: collection,
		VectorDim:      vectorDim,
		MetricType:     metricType,
		IndexType:      "AUTOINDEX",
		ShardNum:       1,
	}

	if !skipCollectionCreation {
		logger.Info("Creating collection with Cohere schema", "collection", collection, "vector_dim", vectorDim)
		if err := createCohereCollection(ctx, milvusClient, collectionConfig); err != nil {
			return fmt.Errorf("failed to create collection: %w", err)
		}
	}

	// Initialize downloader and reader for streaming
	downloader := datasource.NewCohereDownloader("")
	reader := datasource.NewCohereReader(downloader)

	startTime := time.Now()
	vectorsInserted := 0
	batchNum := 0
	totalEstimatedVectors := 8840000 // Approximate total vectors in corpus

	logger.Info("Starting full corpus seed operation",
		"collection", collection,
		"vector_dim", vectorDim,
		"batch_size", batchSize,
		"estimated_total_vectors", totalEstimatedVectors)

	// Stream the entire corpus, inserting batches as they come
	err = reader.StreamFullCorpus(batchSize, func(batch []datasource.CohereEmbedding, fileIndex, totalFiles int) error {
		batchNum++

		// Insert the batch
		if err := insertCohereBatchStreaming(ctx, milvusClient, collection, batch, batchNum, fileIndex, totalFiles, &vectorsInserted, totalEstimatedVectors, startTime); err != nil {
			return err
		}

		return nil
	})

	if err != nil {
		return fmt.Errorf("failed to stream corpus: %w", err)
	}

	printSeedSummary(vectorsInserted, time.Since(startTime))

	// Load collection
	loadCtx, cancel := context.WithTimeout(ctx, 30*time.Minute) // Longer timeout for large corpus
	defer cancel()

	logger.Info("Flushing and loading collection (this may take several minutes for large corpus)")
	if err := FlushAndLoadCollection(loadCtx, milvusClient, collection); err != nil {
		return err
	}

	return nil
}

// insertCohereBatchStreaming inserts a batch during streaming (with file progress tracking)
func insertCohereBatchStreaming(ctx context.Context, milvusClient client.Client, collection string, batch []datasource.CohereEmbedding, batchNum, fileIndex, totalFiles int, vectorsInserted *int, totalVectors int, startTime time.Time) error {
	batchStartTime := time.Now()

	// Prepare column data
	ids := make([]string, len(batch))
	titles := make([]string, len(batch))
	texts := make([]string, len(batch))
	vectors := make([][]float32, len(batch))

	for i, emb := range batch {
		ids[i] = emb.ID
		titles[i] = emb.Title
		texts[i] = emb.Text
		vectors[i] = emb.Embedding
	}

	vectorDim := len(vectors[0])

	// Create columns
	idColumn := entity.NewColumnVarChar("id", ids)
	titleColumn := entity.NewColumnVarChar("title", titles)
	textColumn := entity.NewColumnVarChar("text", texts)
	vectorColumn := entity.NewColumnFloatVector("vector", vectorDim, vectors)

	// Retry batch insert with more retries for bulk operations
	retryConfig := DefaultRetryConfig()
	retryConfig.MaxRetries = 5 // More retries for bulk operations
	retryConfig.InitialDelay = 500 * time.Millisecond
	retryConfig.MaxDelay = 10 * time.Second
	var insertErr error
	attemptNum := 0
	err := RetryWithBackoff(ctx, func() error {
		attemptNum++
		if attemptNum > 1 {
			logger.Info("Retrying batch insert", "batch", batchNum, "attempt", attemptNum, "max_retries", retryConfig.MaxRetries+1)
		}
		_, err := milvusClient.Insert(ctx, collection, "", idColumn, titleColumn, textColumn, vectorColumn)
		if err != nil {
			insertErr = err
			logger.Warn("Batch insert failed", "batch", batchNum, "attempt", attemptNum, "error", err)
			return err
		}
		insertErr = nil
		return nil
	}, retryConfig)

	if err != nil {
		return fmt.Errorf("failed to insert batch %d (file %d/%d, %d vectors) after %d retries: %w", batchNum, fileIndex+1, totalFiles, len(batch), retryConfig.MaxRetries+1, insertErr)
	}

	uploadTime := time.Since(batchStartTime)
	*vectorsInserted += len(batch)
	rate := float64(len(batch)) / uploadTime.Seconds()

	elapsedTotal := time.Since(startTime)
	avgRate := float64(*vectorsInserted) / elapsedTotal.Seconds()
	remainingVectors := totalVectors - *vectorsInserted
	estimatedTimeRemaining := time.Duration(float64(remainingVectors)/avgRate) * time.Second

	progressPercent := float64(*vectorsInserted) / float64(totalVectors) * 100
	fileProgress := float64(fileIndex+1) / float64(totalFiles) * 100

	logger.Info("Batch completed",
		"progress_percent", fmt.Sprintf("%.1f", progressPercent),
		"file_progress", fmt.Sprintf("%d/%d (%.0f%%)", fileIndex+1, totalFiles, fileProgress),
		"batch_num", batchNum,
		"vectors_inserted", len(batch),
		"total_vectors", *vectorsInserted,
		"upload_time_ms", uploadTime.Milliseconds(),
		"rate_vec_per_sec", fmt.Sprintf("%.0f", rate),
		"eta", estimatedTimeRemaining.Round(time.Second))

	// Small delay between batches to avoid overwhelming serverless clusters
	time.Sleep(100 * time.Millisecond)

	return nil
}

// insertCohereBatch inserts a batch of Cohere embeddings
func insertCohereBatch(ctx context.Context, milvusClient client.Client, collection string, batch []datasource.CohereEmbedding, batchNum, totalBatches int, vectorsInserted *int, totalVectors int, startTime time.Time) error {
	progressPercent := float64(*vectorsInserted) / float64(totalVectors) * 100
	logger.Debug("Preparing batch", "progress_percent", progressPercent, "batch_num", batchNum, "total_batches", totalBatches, "batch_size", len(batch))

	batchStartTime := time.Now()

	// Prepare column data
	ids := make([]string, len(batch))
	titles := make([]string, len(batch))
	texts := make([]string, len(batch))
	vectors := make([][]float32, len(batch))

	for i, emb := range batch {
		ids[i] = emb.ID
		titles[i] = emb.Title
		texts[i] = emb.Text
		vectors[i] = emb.Embedding
	}

	vectorDim := len(vectors[0])

	// Create columns
	idColumn := entity.NewColumnVarChar("id", ids)
	titleColumn := entity.NewColumnVarChar("title", titles)
	textColumn := entity.NewColumnVarChar("text", texts)
	vectorColumn := entity.NewColumnFloatVector("vector", vectorDim, vectors)

	// Retry batch insert
	retryConfig := DefaultRetryConfig()
	retryConfig.MaxRetries = 2
	var insertErr error
	err := RetryWithBackoff(ctx, func() error {
		_, err := milvusClient.Insert(ctx, collection, "", idColumn, titleColumn, textColumn, vectorColumn)
		if err != nil {
			insertErr = err
			return err
		}
		insertErr = nil
		return nil
	}, retryConfig)

	if err != nil {
		return fmt.Errorf("failed to insert batch %d/%d (%d vectors) after retries: %w", batchNum, totalBatches, len(batch), insertErr)
	}

	uploadTime := time.Since(batchStartTime)
	*vectorsInserted += len(batch)
	rate := float64(len(batch)) / uploadTime.Seconds()

	elapsedTotal := time.Since(startTime)
	avgRate := float64(*vectorsInserted) / elapsedTotal.Seconds()
	remainingVectors := totalVectors - *vectorsInserted
	estimatedTimeRemaining := time.Duration(float64(remainingVectors)/avgRate) * time.Second

	progressPercent = float64(*vectorsInserted) / float64(totalVectors) * 100

	logger.Info("Batch completed",
		"progress_percent", progressPercent,
		"batch_num", batchNum,
		"total_batches", totalBatches,
		"vectors_inserted", len(batch),
		"upload_time_ms", uploadTime.Milliseconds(),
		"rate_vec_per_sec", rate,
		"eta_seconds", estimatedTimeRemaining.Seconds())

	return nil
}
