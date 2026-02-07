package loadtest

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"

	"zilliz-loadtest/internal/datasource"
	"zilliz-loadtest/internal/logger"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

const (
	// Zilliz Cloud API endpoint for bulk import
	ZillizCloudAPIEndpoint = "https://api.cloud.zilliz.com"

	// Import job check interval
	ImportJobCheckInterval = 10 * time.Second
)

// VDBBenchDataset represents a VDBBench benchmark dataset
type VDBBenchDataset struct {
	Name       string
	VectorDim  int
	NumRows    int
	MetricType entity.MetricType // L2 for SIFT/GIST, COSINE for Cohere/OpenAI/Glove
}

// BulkImportRequest represents a request to the Zilliz Cloud import API
type BulkImportRequest struct {
	ClusterID      string `json:"clusterId"`
	CollectionName string `json:"collectionName"`
	PartitionName  string `json:"partitionName,omitempty"`
	ObjectURL      string `json:"objectUrl"`
	AccessKey      string `json:"accessKey,omitempty"`
	SecretKey      string `json:"secretKey,omitempty"`
}

// BulkImportResponse represents the response from the import API
type BulkImportResponse struct {
	Code    int    `json:"code"`
	Data    struct {
		JobID string `json:"jobId"`
	} `json:"data"`
	Message string `json:"message"`
}

// ImportJobStatus represents the status of an import job
type ImportJobStatus struct {
	Code    int    `json:"code"`
	Data    struct {
		JobID     string `json:"jobId"`
		State     string `json:"state"`
		Progress  int    `json:"progress"`
		Reason    string `json:"reason"`
		TotalRows int64  `json:"totalRows"`
	} `json:"data"`
	Message string `json:"message"`
}

// ExtractClusterID extracts the cluster ID from a Zilliz Cloud URL
// URL format: https://in05-xxxxxxxxxxxxxxx.serverless.gcp-us-west1.cloud.zilliz.com
func ExtractClusterID(databaseURL string) (string, error) {
	// Remove https:// prefix
	url := strings.TrimPrefix(databaseURL, "https://")
	url = strings.TrimPrefix(url, "http://")

	// Extract the first part before the first dot
	parts := strings.Split(url, ".")
	if len(parts) < 1 {
		return "", fmt.Errorf("invalid Zilliz Cloud URL format: %s", databaseURL)
	}

	clusterID := parts[0]

	// Validate it looks like a cluster ID (e.g., in05-xxxxxxxxxxxxxxx)
	matched, _ := regexp.MatchString(`^in\d+-[a-f0-9]+$`, clusterID)
	if !matched {
		return "", fmt.Errorf("could not extract cluster ID from URL: %s (got: %s)", databaseURL, clusterID)
	}

	return clusterID, nil
}

// SeedDatabaseWithBulkImport seeds the database using Zilliz Cloud's bulk import API
func SeedDatabaseWithBulkImport(apiKey, databaseURL, collection string, metricType entity.MetricType, skipCollectionCreation bool) error {
	ctx := context.Background()

	// Extract cluster ID from URL
	clusterID, err := ExtractClusterID(databaseURL)
	if err != nil {
		return fmt.Errorf("failed to extract cluster ID: %w", err)
	}

	logger.Info("Starting bulk import with Zilliz Cloud API",
		"cluster_id", clusterID,
		"collection", collection,
		"total_files", datasource.TOTAL_CORPUS_FILES)

	// Create client for collection management
	milvusClient, err := CreateZillizClient(apiKey, databaseURL)
	if err != nil {
		return fmt.Errorf("failed to create client: %w", err)
	}
	defer milvusClient.Close()

	// Create collection with schema matching the parquet files
	// Cohere parquet files have: _id (string), title (string), text (string), emb (float array)
	if !skipCollectionCreation {
		if err := createBulkImportCollection(ctx, milvusClient, collection, 1024, metricType); err != nil {
			return fmt.Errorf("failed to create collection: %w", err)
		}
	}

	// Import parquet files from HuggingFace
	startTime := time.Now()
	successfulImports := 0
	failedImports := 0

	for fileIndex := 0; fileIndex < datasource.TOTAL_CORPUS_FILES; fileIndex++ {
		// Construct HuggingFace URL for this parquet file
		parquetURL := fmt.Sprintf("%s/msmarco/corpus/%04d.parquet", datasource.COHERE_BASE_URL, fileIndex)

		logger.Info("Submitting bulk import job",
			"file_index", fileIndex,
			"total_files", datasource.TOTAL_CORPUS_FILES,
			"url", parquetURL)

		jobID, err := submitBulkImportJob(apiKey, clusterID, collection, parquetURL)
		if err != nil {
			logger.Error("Failed to submit import job", "file_index", fileIndex, "error", err)
			failedImports++
			continue
		}

		logger.Info("Import job submitted", "job_id", jobID, "file_index", fileIndex)

		// Wait for job completion
		if err := waitForImportJob(apiKey, clusterID, jobID); err != nil {
			logger.Error("Import job failed", "job_id", jobID, "file_index", fileIndex, "error", err)
			failedImports++
			continue
		}

		successfulImports++
		elapsed := time.Since(startTime)
		avgTimePerFile := elapsed / time.Duration(fileIndex+1)
		remainingFiles := datasource.TOTAL_CORPUS_FILES - fileIndex - 1
		eta := time.Duration(remainingFiles) * avgTimePerFile

		logger.Info("Import job completed",
			"file_index", fileIndex,
			"successful", successfulImports,
			"failed", failedImports,
			"eta", eta.Round(time.Second))
	}

	totalTime := time.Since(startTime)
	logger.Info("Bulk import completed",
		"successful_imports", successfulImports,
		"failed_imports", failedImports,
		"total_time", totalTime.Round(time.Second))

	if failedImports > 0 {
		return fmt.Errorf("%d out of %d import jobs failed", failedImports, datasource.TOTAL_CORPUS_FILES)
	}

	// Load collection
	loadCtx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()

	logger.Info("Loading collection into memory")
	if err := FlushAndLoadCollection(loadCtx, milvusClient, collection); err != nil {
		return fmt.Errorf("failed to load collection: %w", err)
	}

	return nil
}

// createBulkImportCollection creates a collection with schema matching Cohere parquet files
func createBulkImportCollection(ctx context.Context, milvusClient client.Client, collection string, vectorDim int, metricType entity.MetricType) error {
	// Check if collection exists
	has, err := milvusClient.HasCollection(ctx, collection)
	if err != nil {
		return fmt.Errorf("failed to check collection: %w", err)
	}

	if has {
		logger.Info("Dropping existing collection", "collection", collection)
		if err := milvusClient.DropCollection(ctx, collection); err != nil {
			return fmt.Errorf("failed to drop collection: %w", err)
		}
	}

	// Create fields matching parquet schema
	// Parquet columns: _id, title, text, emb
	idField := entity.NewField().
		WithName("_id").
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

	embField := entity.NewField().
		WithName("emb").
		WithDataType(entity.FieldTypeFloatVector).
		WithDim(int64(vectorDim))

	schema := &entity.Schema{
		CollectionName: collection,
		AutoID:         false,
		Fields:         []*entity.Field{idField, titleField, textField, embField},
	}

	logger.Info("Creating collection for bulk import",
		"collection", collection,
		"vector_dim", vectorDim,
		"metric_type", metricType)

	if err := milvusClient.CreateCollection(ctx, schema, 1); err != nil {
		return fmt.Errorf("failed to create collection: %w", err)
	}

	logger.Info("Collection created successfully", "collection", collection)

	// Create index
	logger.Info("Creating index on vector field", "collection", collection)

	idx, err := entity.NewIndexAUTOINDEX(metricType)
	if err != nil {
		return fmt.Errorf("failed to create index config: %w", err)
	}

	if err := milvusClient.CreateIndex(ctx, collection, "emb", idx, false); err != nil {
		return fmt.Errorf("failed to create index: %w", err)
	}

	logger.Info("Index created successfully", "collection", collection)

	return nil
}

// submitBulkImportJob submits a bulk import job to Zilliz Cloud API
func submitBulkImportJob(apiKey, clusterID, collection, objectURL string) (string, error) {
	req := BulkImportRequest{
		ClusterID:      clusterID,
		CollectionName: collection,
		ObjectURL:      objectURL,
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/v2/vectordb/jobs/import/create", ZillizCloudAPIEndpoint)
	httpReq, err := http.NewRequest("POST", url, bytes.NewBuffer(reqBody))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	var importResp BulkImportResponse
	if err := json.Unmarshal(body, &importResp); err != nil {
		return "", fmt.Errorf("failed to parse response: %w (body: %s)", err, string(body))
	}

	if importResp.Code != 0 {
		return "", fmt.Errorf("import API error: code=%d, message=%s", importResp.Code, importResp.Message)
	}

	return importResp.Data.JobID, nil
}

// VDBBenchS3BaseURL is the S3 URI for VDBBench datasets (public bucket)
const VDBBenchS3BaseURL = "s3://assets.zilliz.com/benchmark"

// VDBBenchHTTPSBaseURL is the HTTPS URL for VDBBench datasets (for local downloads)
const VDBBenchHTTPSBaseURL = "https://assets.zilliz.com/benchmark"

// SeedDatabaseWithVDBBench seeds the database using VDBBench dataset via Zilliz Cloud bulk import API
func SeedDatabaseWithVDBBench(apiKey, databaseURL, collection, datasetName string, metricType entity.MetricType) error {
	ctx := context.Background()

	// Dataset configurations with correct metric types from VDBBench
	// Cohere, OpenAI, Glove use COSINE; SIFT, GIST use L2
	datasets := map[string]VDBBenchDataset{
		"cohere_small_100k":  {Name: "cohere_small_100k", VectorDim: 768, NumRows: 100000, MetricType: entity.COSINE},
		"cohere_medium_1m":   {Name: "cohere_medium_1m", VectorDim: 768, NumRows: 1000000, MetricType: entity.COSINE},
		"openai_small_50k":   {Name: "openai_small_50k", VectorDim: 1536, NumRows: 50000, MetricType: entity.COSINE},
		"openai_medium_500k": {Name: "openai_medium_500k", VectorDim: 1536, NumRows: 500000, MetricType: entity.COSINE},
		"sift_small_500k":    {Name: "sift_small_500k", VectorDim: 128, NumRows: 500000, MetricType: entity.L2},
		"sift_medium_5m":     {Name: "sift_medium_5m", VectorDim: 128, NumRows: 5000000, MetricType: entity.L2},
		"gist_small_100k":    {Name: "gist_small_100k", VectorDim: 960, NumRows: 100000, MetricType: entity.L2},
		"gist_medium_1m":     {Name: "gist_medium_1m", VectorDim: 960, NumRows: 1000000, MetricType: entity.L2},
		"glove_small_100k":   {Name: "glove_small_100k", VectorDim: 200, NumRows: 100000, MetricType: entity.COSINE},
		"glove_medium_1m":    {Name: "glove_medium_1m", VectorDim: 200, NumRows: 1000000, MetricType: entity.COSINE},
	}

	dataset, ok := datasets[datasetName]
	if !ok {
		return fmt.Errorf("unknown dataset: %s (available: cohere_small_100k, cohere_medium_1m, openai_small_50k, openai_medium_500k, sift_small_500k, sift_medium_5m, gist_small_100k, gist_medium_1m, glove_small_100k, glove_medium_1m)", datasetName)
	}

	// Use dataset name as collection name for public datasets
	if collection == "" || collection == "loadtest_collection" {
		collection = datasetName
	}

	// Use the dataset's correct metric type (ignore passed metricType for public datasets)
	metricType = dataset.MetricType

	logger.Info("Starting bulk import from public dataset",
		"collection", collection,
		"dataset", datasetName,
		"vector_dim", dataset.VectorDim,
		"metric_type", metricType,
		"expected_rows", dataset.NumRows)

	// Create client for collection management
	milvusClient, err := CreateZillizClient(apiKey, databaseURL)
	if err != nil {
		return fmt.Errorf("failed to create client: %w", err)
	}
	defer milvusClient.Close()

	// Create collection with VDBBench schema (id: int64, emb: vector)
	if err := createVDBBenchCollection(ctx, milvusClient, collection, dataset.VectorDim, metricType); err != nil {
		return fmt.Errorf("failed to create collection: %w", err)
	}

	// Extract cluster ID from database URL
	clusterID, err := ExtractClusterID(databaseURL)
	if err != nil {
		return fmt.Errorf("failed to extract cluster ID from database URL: %w", err)
	}

	// Submit bulk import job using S3 URI
	s3URL := fmt.Sprintf("%s/%s/train.parquet", VDBBenchS3BaseURL, datasetName)
	logger.Info("Submitting bulk import job", "s3_url", s3URL)

	jobID, err := submitBulkImportJob(apiKey, clusterID, collection, s3URL)
	if err != nil {
		return fmt.Errorf("failed to submit bulk import job: %w", err)
	}

	logger.Info("Bulk import job submitted", "job_id", jobID)

	// Wait for import job to complete
	startTime := time.Now()
	if err := waitForImportJob(apiKey, clusterID, jobID); err != nil {
		return fmt.Errorf("import job failed: %w", err)
	}

	elapsed := time.Since(startTime)
	logger.Info("Bulk import completed",
		"duration", elapsed.Round(time.Second),
		"expected_rows", dataset.NumRows)

	// Load collection
	loadCtx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()

	logger.Info("Loading collection into memory")
	if err := FlushAndLoadCollection(loadCtx, milvusClient, collection); err != nil {
		return fmt.Errorf("failed to load collection: %w", err)
	}

	// Verify row count
	stats, err := milvusClient.GetCollectionStatistics(ctx, collection)
	if err == nil {
		if rowCount, ok := stats["row_count"]; ok {
			logger.Info("Collection loaded", "row_count", rowCount)
		}
	}

	return nil
}

// vdbBenchInserter manages batch insertion with automatic reconnection
type vdbBenchInserter struct {
	apiKey      string
	databaseURL string
	collection  string
	client      client.Client
}

func newVDBBenchInserter(apiKey, databaseURL, collection string, c client.Client) *vdbBenchInserter {
	return &vdbBenchInserter{
		apiKey:      apiKey,
		databaseURL: databaseURL,
		collection:  collection,
		client:      c,
	}
}

func (ins *vdbBenchInserter) reconnect() error {
	if ins.client != nil {
		ins.client.Close()
	}
	logger.Info("Reconnecting to Zilliz Cloud")
	c, err := CreateZillizClient(ins.apiKey, ins.databaseURL)
	if err != nil {
		return fmt.Errorf("failed to reconnect: %w", err)
	}
	ins.client = c
	return nil
}

func (ins *vdbBenchInserter) close() {
	if ins.client != nil {
		ins.client.Close()
	}
}

// streamVDBBenchParquet streams data from a VDBBench parquet file to the collection
func streamVDBBenchParquet(ctx context.Context, milvusClient client.Client, apiKey, databaseURL, collection, parquetPath string, vectorDim, batchSize int) error {
	logger.Info("Streaming data from parquet", "path", parquetPath, "batch_size", batchSize)

	ins := newVDBBenchInserter(apiKey, databaseURL, collection, milvusClient)

	totalInserted := 0
	batchNum := 0

	ids := make([]int64, 0, batchSize)
	vectors := make([][]float32, 0, batchSize)

	err := datasource.StreamVDBBenchParquet(parquetPath, func(id int64, emb []float32) error {
		ids = append(ids, id)
		vectors = append(vectors, emb)

		if len(ids) >= batchSize {
			batchNum++
			if err := ins.insertBatch(ctx, ids, vectors, batchNum); err != nil {
				return err
			}
			totalInserted += len(ids)

			if batchNum%10 == 0 {
				logger.Info("Batch progress", "batch", batchNum, "total_inserted", totalInserted)
			}

			ids = ids[:0]
			vectors = vectors[:0]
		}
		return nil
	})

	if err != nil {
		return err
	}

	// Insert remaining
	if len(ids) > 0 {
		batchNum++
		if err := ins.insertBatch(ctx, ids, vectors, batchNum); err != nil {
			return err
		}
		totalInserted += len(ids)
	}

	logger.Info("Streaming completed", "total_inserted", totalInserted, "batches", batchNum)
	return nil
}

// insertBatch inserts a batch with retry and reconnection on TLS errors
func (ins *vdbBenchInserter) insertBatch(ctx context.Context, ids []int64, vectors [][]float32, batchNum int) error {
	idCol := entity.NewColumnInt64("id", ids)
	vecCol := entity.NewColumnFloatVector("emb", len(vectors[0]), vectors)

	maxRetries := 8
	var lastErr error

	for attempt := 1; attempt <= maxRetries; attempt++ {
		_, err := ins.client.Insert(ctx, ins.collection, "", idCol, vecCol)
		if err == nil {
			return nil
		}

		lastErr = err
		logger.Warn("Batch insert failed", "batch", batchNum, "attempt", attempt, "error", err)

		// Reconnect on TLS/connection errors
		errStr := strings.ToLower(err.Error())
		if strings.Contains(errStr, "tls") || strings.Contains(errStr, "unavailable") ||
			strings.Contains(errStr, "connection") || strings.Contains(errStr, "eof") {
			if reconnErr := ins.reconnect(); reconnErr != nil {
				logger.Error("Reconnection failed", "error", reconnErr)
			}
		}

		if attempt < maxRetries {
			delay := time.Duration(attempt) * time.Second
			if delay > 10*time.Second {
				delay = 10 * time.Second
			}
			time.Sleep(delay)
		}
	}

	return fmt.Errorf("failed to insert batch %d after %d retries: %w", batchNum, maxRetries, lastErr)
}

// createVDBBenchCollection creates a collection with VDBBench schema (id: int64, emb: vector)
func createVDBBenchCollection(ctx context.Context, milvusClient client.Client, collection string, vectorDim int, metricType entity.MetricType) error {
	// Check if collection exists
	has, err := milvusClient.HasCollection(ctx, collection)
	if err != nil {
		return fmt.Errorf("failed to check collection: %w", err)
	}

	if has {
		logger.Info("Dropping existing collection", "collection", collection)
		if err := milvusClient.DropCollection(ctx, collection); err != nil {
			return fmt.Errorf("failed to drop collection: %w", err)
		}
	}

	// Create fields matching VDBBench parquet schema: id (int64), emb (vector)
	idField := entity.NewField().
		WithName("id").
		WithDataType(entity.FieldTypeInt64).
		WithIsPrimaryKey(true).
		WithIsAutoID(false)

	embField := entity.NewField().
		WithName("emb").
		WithDataType(entity.FieldTypeFloatVector).
		WithDim(int64(vectorDim))

	schema := &entity.Schema{
		CollectionName: collection,
		AutoID:         false,
		Fields:         []*entity.Field{idField, embField},
	}

	logger.Info("Creating collection for VDBBench import",
		"collection", collection,
		"vector_dim", vectorDim,
		"metric_type", metricType)

	if err := milvusClient.CreateCollection(ctx, schema, 1); err != nil {
		return fmt.Errorf("failed to create collection: %w", err)
	}

	logger.Info("Collection created successfully", "collection", collection)

	// Create index
	logger.Info("Creating index on vector field", "collection", collection)

	idx, err := entity.NewIndexAUTOINDEX(metricType)
	if err != nil {
		return fmt.Errorf("failed to create index config: %w", err)
	}

	if err := milvusClient.CreateIndex(ctx, collection, "emb", idx, false); err != nil {
		return fmt.Errorf("failed to create index: %w", err)
	}

	logger.Info("Index created successfully", "collection", collection)

	return nil
}

// waitForImportJob waits for an import job to complete
func waitForImportJob(apiKey, clusterID, jobID string) error {
	return waitForImportJobWithProgress(apiKey, clusterID, jobID)
}

// waitForImportJobWithProgress waits for an import job to complete with progress logging
func waitForImportJobWithProgress(apiKey, clusterID, jobID string) error {
	url := fmt.Sprintf("%s/v2/vectordb/jobs/import/describe", ZillizCloudAPIEndpoint)
	lastProgress := -1

	for {
		reqBody, _ := json.Marshal(map[string]string{
			"clusterId": clusterID,
			"jobId":     jobID,
		})

		httpReq, err := http.NewRequest("POST", url, bytes.NewBuffer(reqBody))
		if err != nil {
			return fmt.Errorf("failed to create request: %w", err)
		}

		httpReq.Header.Set("Authorization", "Bearer "+apiKey)
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", "application/json")

		client := &http.Client{Timeout: 30 * time.Second}
		resp, err := client.Do(httpReq)
		if err != nil {
			logger.Warn("Failed to check job status, retrying", "error", err)
			time.Sleep(ImportJobCheckInterval)
			continue
		}

		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		var status ImportJobStatus
		if err := json.Unmarshal(body, &status); err != nil {
			logger.Warn("Failed to parse job status, retrying", "error", err)
			time.Sleep(ImportJobCheckInterval)
			continue
		}

		// Log progress when it changes
		if status.Data.Progress != lastProgress {
			logger.Info("Import job progress",
				"job_id", jobID,
				"state", status.Data.State,
				"progress", fmt.Sprintf("%d%%", status.Data.Progress),
				"total_rows", status.Data.TotalRows)
			lastProgress = status.Data.Progress
		}

		switch status.Data.State {
		case "Completed":
			logger.Info("Import job completed",
				"job_id", jobID,
				"total_rows", status.Data.TotalRows)
			return nil
		case "Failed":
			return fmt.Errorf("import job failed: %s", status.Data.Reason)
		case "Pending", "Importing", "InProgress":
			// Still running, wait and check again
			time.Sleep(ImportJobCheckInterval)
		default:
			logger.Warn("Unknown job state", "state", status.Data.State)
			time.Sleep(ImportJobCheckInterval)
		}
	}
}

// SeedDatabaseWithBEIR seeds the database using a BEIR dataset from HuggingFace (with human-labeled qrels)
// This uses streaming insertion since BEIR datasets are on HuggingFace, not Zilliz S3
func SeedDatabaseWithBEIR(apiKey, databaseURL, collection, datasetName string, batchSize int) error {
	ctx := context.Background()

	// Get dataset info
	dataset, err := datasource.GetBEIRDataset(datasetName)
	if err != nil {
		return err
	}

	// Limit batch size for BEIR due to large text fields and 1024-dim vectors
	// gRPC limit is 64MB, so we need smaller batches
	if batchSize > 2000 {
		batchSize = 2000
	}

	// Use dataset name as collection name if not specified
	if collection == "" || collection == "loadtest_collection" {
		collection = "beir_" + datasetName
	}

	logger.Info("Starting BEIR dataset seeding (with human-labeled qrels)",
		"collection", collection,
		"dataset", datasetName,
		"vector_dim", dataset.VectorDim,
		"metric_type", dataset.MetricType,
		"expected_rows", dataset.CorpusSize)

	// Create client for collection management
	milvusClient, err := CreateZillizClient(apiKey, databaseURL)
	if err != nil {
		return fmt.Errorf("failed to create client: %w", err)
	}
	defer milvusClient.Close()

	// Create collection with BEIR schema
	if err := createBEIRCollection(ctx, milvusClient, collection, dataset.VectorDim, dataset.MetricType); err != nil {
		return fmt.Errorf("failed to create collection: %w", err)
	}

	// Initialize BEIR data loader
	loader, err := datasource.NewBEIRDataLoader(datasetName, "")
	if err != nil {
		return err
	}

	// Download corpus if needed
	corpusPath, err := loader.EnsureCorpusFile()
	if err != nil {
		return fmt.Errorf("failed to ensure corpus file: %w", err)
	}

	// Stream data into collection
	if err := streamBEIRCorpus(ctx, milvusClient, apiKey, databaseURL, collection, corpusPath, batchSize); err != nil {
		return fmt.Errorf("failed to stream corpus: %w", err)
	}

	// Load collection
	loadCtx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()

	logger.Info("Loading collection into memory")
	if err := FlushAndLoadCollection(loadCtx, milvusClient, collection); err != nil {
		return fmt.Errorf("failed to load collection: %w", err)
	}

	// Verify row count
	stats, err := milvusClient.GetCollectionStatistics(ctx, collection)
	if err == nil {
		if rowCount, ok := stats["row_count"]; ok {
			logger.Info("Collection loaded", "row_count", rowCount)
		}
	}

	return nil
}

// createBEIRCollection creates a collection with BEIR schema
func createBEIRCollection(ctx context.Context, milvusClient client.Client, collection string, vectorDim int, metricType entity.MetricType) error {
	// Check if collection exists
	has, err := milvusClient.HasCollection(ctx, collection)
	if err != nil {
		return fmt.Errorf("failed to check collection: %w", err)
	}

	if has {
		logger.Info("Dropping existing collection", "collection", collection)
		if err := milvusClient.DropCollection(ctx, collection); err != nil {
			return fmt.Errorf("failed to drop collection: %w", err)
		}
	}

	// Create fields matching BEIR corpus schema: _id, title, text, emb
	idField := entity.NewField().
		WithName("_id").
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

	embField := entity.NewField().
		WithName("emb").
		WithDataType(entity.FieldTypeFloatVector).
		WithDim(int64(vectorDim))

	schema := &entity.Schema{
		CollectionName: collection,
		AutoID:         false,
		Fields:         []*entity.Field{idField, titleField, textField, embField},
	}

	logger.Info("Creating collection for BEIR import",
		"collection", collection,
		"vector_dim", vectorDim,
		"metric_type", metricType)

	if err := milvusClient.CreateCollection(ctx, schema, 1); err != nil {
		return fmt.Errorf("failed to create collection: %w", err)
	}

	logger.Info("Collection created successfully", "collection", collection)

	// Create index
	logger.Info("Creating index on vector field", "collection", collection)

	idx, err := entity.NewIndexAUTOINDEX(metricType)
	if err != nil {
		return fmt.Errorf("failed to create index config: %w", err)
	}

	if err := milvusClient.CreateIndex(ctx, collection, "emb", idx, false); err != nil {
		return fmt.Errorf("failed to create index: %w", err)
	}

	logger.Info("Index created successfully", "collection", collection)

	return nil
}

// beirInserter manages batch insertion with automatic reconnection for BEIR data
type beirInserter struct {
	apiKey      string
	databaseURL string
	collection  string
	client      client.Client
}

func newBEIRInserter(apiKey, databaseURL, collection string, c client.Client) *beirInserter {
	return &beirInserter{
		apiKey:      apiKey,
		databaseURL: databaseURL,
		collection:  collection,
		client:      c,
	}
}

func (ins *beirInserter) reconnect() error {
	if ins.client != nil {
		ins.client.Close()
	}
	logger.Info("Reconnecting to Zilliz Cloud")
	c, err := CreateZillizClient(ins.apiKey, ins.databaseURL)
	if err != nil {
		return fmt.Errorf("failed to reconnect: %w", err)
	}
	ins.client = c
	return nil
}

// insertBatch inserts a batch of BEIR records with retry and reconnection
func (ins *beirInserter) insertBatch(ctx context.Context, ids, titles, texts []string, vectors [][]float32, batchNum int) error {
	idCol := entity.NewColumnVarChar("_id", ids)
	titleCol := entity.NewColumnVarChar("title", titles)
	textCol := entity.NewColumnVarChar("text", texts)
	vecCol := entity.NewColumnFloatVector("emb", len(vectors[0]), vectors)

	maxRetries := 8
	var lastErr error

	for attempt := 1; attempt <= maxRetries; attempt++ {
		_, err := ins.client.Insert(ctx, ins.collection, "", idCol, titleCol, textCol, vecCol)
		if err == nil {
			return nil
		}

		lastErr = err
		logger.Warn("Batch insert failed", "batch", batchNum, "attempt", attempt, "error", err)

		// Reconnect on TLS/connection errors
		errStr := strings.ToLower(err.Error())
		if strings.Contains(errStr, "tls") || strings.Contains(errStr, "unavailable") ||
			strings.Contains(errStr, "connection") || strings.Contains(errStr, "eof") {
			if reconnErr := ins.reconnect(); reconnErr != nil {
				logger.Error("Reconnection failed", "error", reconnErr)
			}
		}

		if attempt < maxRetries {
			delay := time.Duration(attempt) * time.Second
			if delay > 10*time.Second {
				delay = 10 * time.Second
			}
			time.Sleep(delay)
		}
	}

	return fmt.Errorf("failed to insert batch %d after %d retries: %w", batchNum, maxRetries, lastErr)
}

// streamBEIRCorpus streams data from a BEIR corpus parquet file to the collection
func streamBEIRCorpus(ctx context.Context, milvusClient client.Client, apiKey, databaseURL, collection, parquetPath string, batchSize int) error {
	logger.Info("Streaming BEIR corpus", "path", parquetPath, "batch_size", batchSize)

	ins := newBEIRInserter(apiKey, databaseURL, collection, milvusClient)

	totalInserted := 0
	batchNum := 0

	ids := make([]string, 0, batchSize)
	titles := make([]string, 0, batchSize)
	texts := make([]string, 0, batchSize)
	vectors := make([][]float32, 0, batchSize)

	err := datasource.StreamBEIRCorpusParquetPy(parquetPath, func(id, title, text string, emb []float32) error {
		ids = append(ids, id)
		titles = append(titles, title)
		texts = append(texts, text)
		vectors = append(vectors, emb)

		if len(ids) >= batchSize {
			batchNum++
			if err := ins.insertBatch(ctx, ids, titles, texts, vectors, batchNum); err != nil {
				return err
			}
			totalInserted += len(ids)

			if batchNum%10 == 0 {
				logger.Info("BEIR corpus batch progress", "batch", batchNum, "total_inserted", totalInserted)
			}

			ids = ids[:0]
			titles = titles[:0]
			texts = texts[:0]
			vectors = vectors[:0]
		}
		return nil
	})

	if err != nil {
		return err
	}

	// Insert remaining
	if len(ids) > 0 {
		batchNum++
		if err := ins.insertBatch(ctx, ids, titles, texts, vectors, batchNum); err != nil {
			return err
		}
		totalInserted += len(ids)
	}

	logger.Info("BEIR corpus streaming completed", "total_inserted", totalInserted, "batches", batchNum)
	return nil
}
