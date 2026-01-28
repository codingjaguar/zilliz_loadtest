package loadtest

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"zilliz-loadtest/internal/mocks"
)

func TestEnsureCollectionExists_CreateNew(t *testing.T) {
	ctx := context.Background()
	collectionName := "test_collection"
	createCalled := false
	indexCalled := false

	mockClient := &mocks.MockClient{
		HasCollectionFunc: func(ctx context.Context, collectionName string) (bool, error) {
			return false, nil
		},
		CreateCollectionFunc: func(ctx context.Context, schema *entity.Schema, shardsNum int32, opts ...client.CreateCollectionOption) error {
			createCalled = true
			// Verify schema has id and vector fields
			if len(schema.Fields) != 2 {
				t.Errorf("CreateCollection() schema should have 2 fields, got %d", len(schema.Fields))
			}
			return nil
		},
		CreateIndexFunc: func(ctx context.Context, collectionName string, fieldName string, idx entity.Index, async bool, opts ...client.IndexOption) error {
			indexCalled = true
			if fieldName != "vector" {
				t.Errorf("CreateIndex() field name = %q, want vector", fieldName)
			}
			return nil
		},
		GetIndexStateFunc: func(ctx context.Context, collectionName string, fieldName string, opts ...client.IndexOption) (entity.IndexState, error) {
			return 3, nil // IndexStateFinished
		},
	}

	config := CollectionConfig{
		CollectionName: collectionName,
		VectorDim:      768,
		MetricType:     entity.L2,
		IndexType:      "AUTOINDEX",
		ShardNum:       1,
	}

	err := EnsureCollectionExists(ctx, mockClient, config, false)
	if err != nil {
		t.Errorf("EnsureCollectionExists() error = %v", err)
	}
	if !createCalled {
		t.Error("EnsureCollectionExists() should have called CreateCollection")
	}
	if !indexCalled {
		t.Error("EnsureCollectionExists() should have called CreateIndex")
	}
}

func TestEnsureCollectionExists_CollectionExists(t *testing.T) {
	ctx := context.Background()
	collectionName := "existing_collection"

	mockClient := &mocks.MockClient{
		HasCollectionFunc: func(ctx context.Context, collectionName string) (bool, error) {
			return true, nil
		},
		ListCollectionsFunc: func(ctx context.Context) ([]*entity.Collection, error) {
			return []*entity.Collection{}, nil
		},
		DescribeCollectionFunc: func(ctx context.Context, collectionName string) (*entity.Collection, error) {
			// Return a collection with correct schema
			return &entity.Collection{
				Schema: &entity.Schema{
					Fields: []*entity.Field{
						{
							Name:     "id",
							DataType: entity.FieldTypeInt64,
						},
						{
							Name:     "vector",
							DataType: entity.FieldTypeFloatVector,
							TypeParams: map[string]string{
								"dim": "768",
							},
						},
					},
				},
			}, nil
		},
		DescribeIndexFunc: func(ctx context.Context, collectionName string, fieldName string) ([]entity.Index, error) {
			// Return an existing index - use a simple mock
			return []entity.Index{}, nil
		},
	}

	config := CollectionConfig{
		CollectionName: collectionName,
		VectorDim:      768,
		MetricType:     entity.L2,
		IndexType:      "AUTOINDEX",
		ShardNum:       1,
	}

	err := EnsureCollectionExists(ctx, mockClient, config, false)
	if err != nil {
		t.Errorf("EnsureCollectionExists() error = %v", err)
	}
}

func TestEnsureCollectionExists_CollectionExistsNoIndex(t *testing.T) {
	ctx := context.Background()
	collectionName := "existing_collection_no_index"
	indexCalled := false

	mockClient := &mocks.MockClient{
		HasCollectionFunc: func(ctx context.Context, collectionName string) (bool, error) {
			return true, nil
		},
		ListCollectionsFunc: func(ctx context.Context) ([]*entity.Collection, error) {
			return []*entity.Collection{}, nil
		},
		DescribeCollectionFunc: func(ctx context.Context, collectionName string) (*entity.Collection, error) {
			return &entity.Collection{
				Schema: &entity.Schema{
					Fields: []*entity.Field{
						{
							Name:     "id",
							DataType: entity.FieldTypeInt64,
						},
						{
							Name:     "vector",
							DataType: entity.FieldTypeFloatVector,
							TypeParams: map[string]string{
								"dim": "768",
							},
						},
					},
				},
			}, nil
		},
		DescribeIndexFunc: func(ctx context.Context, collectionName string, fieldName string) ([]entity.Index, error) {
			// Index doesn't exist
			return nil, errors.New("index not found")
		},
		CreateIndexFunc: func(ctx context.Context, collectionName string, fieldName string, idx entity.Index, async bool, opts ...client.IndexOption) error {
			indexCalled = true
			return nil
		},
		GetIndexStateFunc: func(ctx context.Context, collectionName string, fieldName string, opts ...client.IndexOption) (entity.IndexState, error) {
			return 3, nil // IndexStateFinished
		},
	}

	config := CollectionConfig{
		CollectionName: collectionName,
		VectorDim:      768,
		MetricType:     entity.L2,
		IndexType:      "AUTOINDEX",
		ShardNum:       1,
	}

	err := EnsureCollectionExists(ctx, mockClient, config, false)
	if err != nil {
		t.Errorf("EnsureCollectionExists() error = %v", err)
	}
	if !indexCalled {
		t.Error("EnsureCollectionExists() should have called CreateIndex when index is missing")
	}
}

func TestEnsureCollectionExists_DimensionMismatch(t *testing.T) {
	ctx := context.Background()
	collectionName := "mismatch_collection"

	mockClient := &mocks.MockClient{
		HasCollectionFunc: func(ctx context.Context, collectionName string) (bool, error) {
			return true, nil
		},
		ListCollectionsFunc: func(ctx context.Context) ([]*entity.Collection, error) {
			return []*entity.Collection{}, nil
		},
		DescribeCollectionFunc: func(ctx context.Context, collectionName string) (*entity.Collection, error) {
			return &entity.Collection{
				Schema: &entity.Schema{
					Fields: []*entity.Field{
						{
							Name:     "vector",
							DataType: entity.FieldTypeFloatVector,
							TypeParams: map[string]string{
								"dim": "512", // Different dimension
							},
						},
					},
				},
			}, nil
		},
	}

	config := CollectionConfig{
		CollectionName: collectionName,
		VectorDim:      768, // Expecting 768
		MetricType:     entity.L2,
		IndexType:      "AUTOINDEX",
		ShardNum:       1,
	}

	err := EnsureCollectionExists(ctx, mockClient, config, false)
	if err == nil {
		t.Error("EnsureCollectionExists() should return error for dimension mismatch")
	}
	if !strings.Contains(err.Error(), "dimension mismatch") {
		t.Errorf("EnsureCollectionExists() error = %v, should contain 'dimension mismatch'", err)
	}
}

