package loadtest

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"zilliz-loadtest/internal/logger"
)

// CollectionConfig holds configuration for creating a collection
type CollectionConfig struct {
	CollectionName string
	VectorDim      int
	MetricType     entity.MetricType
	IndexType      string // e.g., "HNSW", "IVF_FLAT", "AUTOINDEX"
	IndexParams    map[string]string
	ShardNum       int32
}

// EnsureCollectionExists checks if collection exists, creates it if it doesn't, and ensures it has the correct schema and index
// If skipCreation is true, it will only verify the collection exists and return an error if it doesn't
func EnsureCollectionExists(ctx context.Context, milvusClient client.Client, config CollectionConfig, skipCreation bool) error {
	// First, check if cluster is ready
	if err := checkClusterReadiness(ctx, milvusClient); err != nil {
		return fmt.Errorf("cluster may still be spinning up or not ready: %w. Please wait a few minutes and try again, or create the collection manually in the Zilliz UI", err)
	}

	// Check if collection exists
	exists, err := milvusClient.HasCollection(ctx, config.CollectionName)
	if err != nil {
		// HasCollection might not be available, try DescribeCollection instead
		_, describeErr := milvusClient.DescribeCollection(ctx, config.CollectionName)
		if describeErr != nil {
			if skipCreation {
				return fmt.Errorf("collection '%s' does not exist. Please create it manually in the Zilliz UI or run without --skip-collection-creation", config.CollectionName)
			}
			// Collection doesn't exist, create it
			logger.Info("Collection does not exist, creating it",
				"collection", config.CollectionName,
				"vector_dim", config.VectorDim,
				"metric_type", config.MetricType)
			return createCollectionWithSchema(ctx, milvusClient, config)
		}
		// Collection exists, verify schema
		return ensureCollectionSchema(ctx, milvusClient, config)
	}

	if !exists {
		if skipCreation {
			return fmt.Errorf("collection '%s' does not exist. Please create it manually in the Zilliz UI or run without --skip-collection-creation", config.CollectionName)
		}
		// Collection doesn't exist, create it
		logger.Info("Collection does not exist, creating it",
			"collection", config.CollectionName,
			"vector_dim", config.VectorDim,
			"metric_type", config.MetricType)
		return createCollectionWithSchema(ctx, milvusClient, config)
	}

	// Collection exists, verify schema and index
	logger.Info("Collection exists, verifying schema and index",
		"collection", config.CollectionName)
	return ensureCollectionSchema(ctx, milvusClient, config)
}

// checkClusterReadiness checks if the cluster is ready by attempting a simple operation
func checkClusterReadiness(ctx context.Context, milvusClient client.Client) error {
	logger.Info("Checking cluster readiness...")
	
	// Try to list collections as a health check
	// This is a lightweight operation that will fail if the cluster isn't ready
	_, err := milvusClient.ListCollections(ctx)
	if err != nil {
		// Check if it's a connection/timeout error that might indicate cluster spin-up
		errMsg := err.Error()
		if strings.Contains(strings.ToLower(errMsg), "connection") ||
			strings.Contains(strings.ToLower(errMsg), "timeout") ||
			strings.Contains(strings.ToLower(errMsg), "unavailable") ||
			strings.Contains(strings.ToLower(errMsg), "refused") ||
			strings.Contains(strings.ToLower(errMsg), "deadline exceeded") {
			return fmt.Errorf("cluster may still be spinning up: %w. Please wait a few minutes for the cluster to become ready, then try again. You can also create the collection manually in the Zilliz UI and use --skip-collection-creation", err)
		}
		// Other errors might be permissions or configuration issues
		logger.Warn("ListCollections failed - this may indicate a permissions or configuration issue", "error", err)
		// Don't fail here - let the actual operation fail with a clearer error
	}
	
	logger.Info("Cluster is ready")
	return nil
}

// createCollectionWithSchema creates a new collection with the specified schema
func createCollectionWithSchema(ctx context.Context, milvusClient client.Client, config CollectionConfig) error {
	// Create schema with autoID enabled
	// Primary key field (autoID)
	idField := entity.NewField().
		WithName("id").
		WithDataType(entity.FieldTypeInt64).
		WithIsPrimaryKey(true).
		WithIsAutoID(true)

	// Vector field
	vectorField := entity.NewField().
		WithName("vector").
		WithDataType(entity.FieldTypeFloatVector).
		WithDim(int64(config.VectorDim))

	// Create schema
	schema := entity.NewSchema().
		WithName(config.CollectionName).
		WithField(idField).
		WithField(vectorField).
		WithDescription("Collection created by zilliz-loadtest tool")

	// Create collection
	shardNum := config.ShardNum
	if shardNum <= 0 {
		shardNum = 1 // Default to 1 shard
	}

	logger.Info("Creating collection",
		"collection", config.CollectionName,
		"vector_dim", config.VectorDim,
		"metric_type", config.MetricType,
		"shard_num", shardNum)

	// Create collection with schema and shard number
	err := milvusClient.CreateCollection(ctx, schema, shardNum)
	if err != nil {
		return fmt.Errorf("failed to create collection: %w", err)
	}

	logger.Info("Collection created successfully", "collection", config.CollectionName)

	// Create index on vector field
	return createVectorIndex(ctx, milvusClient, config)
}

// ensureCollectionSchema verifies the collection schema matches expectations and creates index if needed
func ensureCollectionSchema(ctx context.Context, milvusClient client.Client, config CollectionConfig) error {
	// Describe collection to verify schema
	collection, err := milvusClient.DescribeCollection(ctx, config.CollectionName)
	if err != nil {
		return fmt.Errorf("failed to describe collection: %w", err)
	}

	// Verify vector field exists and has correct dimension
	vectorFieldFound := false
	for _, field := range collection.Schema.Fields {
		if field.Name == "vector" {
			vectorFieldFound = true
			if field.DataType != entity.FieldTypeFloatVector {
				return fmt.Errorf("collection has vector field with wrong data type: %v", field.DataType)
			}
			// Check dimension
			if field.TypeParams != nil {
				if dimStr, ok := field.TypeParams["dim"]; ok {
					dim, err := strconv.Atoi(dimStr)
					if err != nil {
						// Try parsing as string
						var dimInt int
						fmt.Sscanf(dimStr, "%d", &dimInt)
						dim = dimInt
					}
					if dim != config.VectorDim {
						return fmt.Errorf("collection vector dimension mismatch: collection has %d, expected %d", dim, config.VectorDim)
					}
				}
			}
			break
		}
	}

	if !vectorFieldFound {
		return fmt.Errorf("collection does not have a vector field named 'vector'")
	}

	// Check if index exists by trying to describe it
	indexes, err := milvusClient.DescribeIndex(ctx, config.CollectionName, "vector")
	if err != nil {
		// Index doesn't exist, create it
		logger.Info("Index does not exist, creating it", "collection", config.CollectionName)
		return createVectorIndex(ctx, milvusClient, config)
	}

	// Index exists, verify metric type if possible
	if len(indexes) > 0 {
		idx := indexes[0]
		// Try to get metric type from index params
		idxParams := idx.Params()
		if idxParams != nil {
			// Index params might be a map[string]string or similar
			// Try to extract metric_type
			for k, v := range idxParams {
				if strings.ToLower(k) == "metric_type" {
					metricTypeStr := fmt.Sprintf("%v", v)
					expectedMetricType := strings.ToUpper(string(config.MetricType))
					if strings.ToUpper(metricTypeStr) != expectedMetricType {
						logger.Warn("Index metric type mismatch",
							"collection", config.CollectionName,
							"index_metric_type", metricTypeStr,
							"expected_metric_type", expectedMetricType,
							"message", "Index exists but metric type doesn't match. Consider recreating the collection.")
					}
					break
				}
			}
		}
		logger.Info("Index already exists", "collection", config.CollectionName)
	}

	logger.Info("Collection schema and index verified", "collection", config.CollectionName)
	return nil
}

// createVectorIndex creates an index on the vector field
func createVectorIndex(ctx context.Context, milvusClient client.Client, config CollectionConfig) error {
	// Determine index type
	indexType := config.IndexType
	if indexType == "" {
		// Default to AUTOINDEX for Zilliz Cloud
		indexType = "AUTOINDEX"
	}

	// Build index parameters
	indexParams := make(map[string]string)
	if config.IndexParams != nil {
		for k, v := range config.IndexParams {
			indexParams[k] = v
		}
	}

	// Add metric type to index params
	indexParams["metric_type"] = strings.ToUpper(string(config.MetricType))

	// Create index - convert indexType string to entity.IndexType
	var indexTypeEntity entity.IndexType
	switch strings.ToUpper(indexType) {
	case "AUTOINDEX":
		indexTypeEntity = entity.AUTOINDEX
	case "FLAT":
		indexTypeEntity = entity.Flat
	case "IVF_FLAT":
		indexTypeEntity = entity.IvfFlat
	case "IVF_SQ8":
		indexTypeEntity = entity.IvfSQ8
	case "IVF_PQ":
		indexTypeEntity = entity.IvfPQ
	case "HNSW":
		indexTypeEntity = entity.HNSW
	case "SCANN":
		indexTypeEntity = entity.SCANN
	default:
		// Default to AUTOINDEX for Zilliz Cloud
		indexTypeEntity = entity.AUTOINDEX
		indexType = "AUTOINDEX"
	}

	// Create index
	index := entity.NewGenericIndex(
		"vector_idx",
		indexTypeEntity,
		indexParams,
	)

	logger.Info("Creating index on vector field",
		"collection", config.CollectionName,
		"index_type", indexType,
		"metric_type", config.MetricType)

	err := milvusClient.CreateIndex(ctx, config.CollectionName, "vector", index, false)
	if err != nil {
		return fmt.Errorf("failed to create index: %w", err)
	}

	logger.Info("Index created successfully", "collection", config.CollectionName)

	// Wait for index to be built (optional, but recommended)
	// Note: For AUTOINDEX, this might be instant
	maxWaitTime := 60 * time.Second
	checkInterval := 2 * time.Second
	startTime := time.Now()

	for time.Since(startTime) < maxWaitTime {
		state, err := milvusClient.GetIndexState(ctx, config.CollectionName, "vector")
		if err != nil {
			logger.Warn("Could not check index state, assuming it's building", "error", err)
			break
		}

		// Check index state - entity.IndexState is typically an int type
		// IndexStateFinished = 3, IndexStateFailed = 4 (typically)
		// Compare as int values
		if int(state) == 3 {
			logger.Info("Index build completed", "collection", config.CollectionName)
			return nil
		}

		if int(state) == 4 {
			return fmt.Errorf("index build failed")
		}

		logger.Debug("Index is building, waiting...", "state", int(state))
		time.Sleep(checkInterval)
	}

	logger.Info("Index creation initiated (may still be building)", "collection", config.CollectionName)
	return nil
}
