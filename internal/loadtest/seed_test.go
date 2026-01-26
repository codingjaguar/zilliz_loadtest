package loadtest

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"zilliz-loadtest/internal/mocks"
)

func TestSeedDatabase_Validation(t *testing.T) {
	tests := []struct {
		name        string
		apiKey      string
		databaseURL string
		collection  string
		vectorDim   int
		totalVectors int
		batchSize   int
		wantErr     bool
		errContains string
	}{
		{"empty API key", "", "url", "coll", 768, 1000, 100, true, "API key"},
		{"empty database URL", "key", "", "coll", 768, 1000, 100, true, "database URL"},
		{"empty collection", "key", "url", "", 768, 1000, 100, true, "collection name"},
		{"zero vector dim", "key", "url", "coll", 0, 1000, 100, true, "vector dimension"},
		{"negative vector dim", "key", "url", "coll", -1, 1000, 100, true, "vector dimension"},
		{"zero total vectors", "key", "url", "coll", 768, 0, 100, true, "total vectors"},
		{"negative total vectors", "key", "url", "coll", 768, -1, 100, true, "total vectors"},
		{"zero batch size", "key", "url", "coll", 768, 1000, 0, true, "batch size"},
		{"negative batch size", "key", "url", "coll", 768, 1000, -1, true, "batch size"},
		{"batch size too large", "key", "url", "coll", 768, 1000, 50001, true, "batch size too large"},
		// Note: We don't test "valid inputs" here as it would try to create real client connection
		// Valid input testing should be done in integration tests with proper setup
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// All these tests should fail validation immediately without creating clients
			// Use a short timeout to ensure we don't hang on client creation attempts
			ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
			defer cancel()

			done := make(chan error, 1)
			go func() {
				done <- SeedDatabaseWithBatchSize(tt.apiKey, tt.databaseURL, tt.collection, tt.vectorDim, tt.totalVectors, tt.batchSize)
			}()

			select {
			case err := <-done:
				if (err != nil) != tt.wantErr {
					t.Errorf("SeedDatabaseWithBatchSize() error = %v, wantErr %v", err, tt.wantErr)
					return
				}
				if tt.wantErr && tt.errContains != "" {
					if err == nil || !strings.Contains(err.Error(), tt.errContains) {
						t.Errorf("SeedDatabaseWithBatchSize() error = %v, should contain %q", err, tt.errContains)
					}
				}
			case <-ctx.Done():
				// If we timeout, it means the function tried to create a client
				// This should only happen if validation passed but client creation failed
				// For our test cases, validation should fail first, so timeout indicates a problem
				t.Errorf("Test timed out - SeedDatabaseWithBatchSize() should fail validation immediately for %q", tt.name)
			}
		})
	}
}

func TestGenerateBatchVectors(t *testing.T) {
	vectorDim := 128
	batchStart := 0
	currentBatchSize := 10
	vectorsInserted := 0
	totalVectors := 100
	batchNum := 1
	totalBatches := 10

	vectors := generateBatchVectors(vectorDim, batchStart, currentBatchSize, &vectorsInserted, totalVectors, batchNum, totalBatches)

	if len(vectors) != currentBatchSize {
		t.Errorf("generateBatchVectors() length = %d, want %d", len(vectors), currentBatchSize)
	}

	for i, vector := range vectors {
		if len(vector) != vectorDim {
			t.Errorf("generateBatchVectors() vector[%d] length = %d, want %d", i, len(vector), vectorDim)
		}
		// Verify vector values are in valid range
		for j, val := range vector {
			if val < 0 || val >= 1.0 {
				t.Errorf("generateBatchVectors() vector[%d][%d] = %f, want in range [0, 1)", i, j, val)
			}
		}
	}
}

func TestProcessSeedBatch_InsertError(t *testing.T) {
	ctx := context.Background()
	collection := "test_collection"
	vectorDim := 128
	batchNum := 1
	totalBatches := 10
	batchStart := 0
	currentBatchSize := 10
	vectorsInserted := 0
	totalVectors := 100

	mockClient := &mocks.MockClient{
		InsertFunc: func(ctx context.Context, collectionName string, partitionName string, columns ...entity.Column) (entity.Column, error) {
			return nil, errors.New("insert failed")
		},
	}

	startTime := time.Now()
	err := processSeedBatch(ctx, mockClient, collection, vectorDim, batchNum, totalBatches, batchStart, currentBatchSize, &vectorsInserted, totalVectors, startTime)
	if err == nil {
		t.Error("processSeedBatch() expected error on insert failure")
	}
}

func TestProcessSeedBatch_Success(t *testing.T) {
	ctx := context.Background()
	collection := "test_collection"
	vectorDim := 128
	batchNum := 1
	totalBatches := 10
	batchStart := 0
	currentBatchSize := 10
	vectorsInserted := 0
	totalVectors := 100

	insertCalled := false
	mockClient := &mocks.MockClient{
		InsertFunc: func(ctx context.Context, collectionName string, partitionName string, columns ...entity.Column) (entity.Column, error) {
			insertCalled = true
			if collectionName != collection {
				t.Errorf("processSeedBatch() collection = %q, want %q", collectionName, collection)
			}
			if len(columns) != 1 {
				t.Errorf("processSeedBatch() columns length = %d, want 1", len(columns))
			}
			return nil, nil
		},
	}

	startTime := time.Now()
	err := processSeedBatch(ctx, mockClient, collection, vectorDim, batchNum, totalBatches, batchStart, currentBatchSize, &vectorsInserted, totalVectors, startTime)
	if err != nil {
		t.Errorf("processSeedBatch() error = %v", err)
	}
	if !insertCalled {
		t.Error("processSeedBatch() Insert should be called")
	}
	if vectorsInserted != currentBatchSize {
		t.Errorf("processSeedBatch() vectorsInserted = %d, want %d", vectorsInserted, currentBatchSize)
	}
}
