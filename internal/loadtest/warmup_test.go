package loadtest

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"zilliz-loadtest/internal/mocks"
)

func TestRunWarmup_ZeroQueries(t *testing.T) {
	mockClient := &mocks.MockClient{
		SearchFunc: func(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
			t.Error("runWarmup() should not execute queries when warmupQueries is 0")
			return nil, nil
		},
	}

	lt := &LoadTester{
		clients:      []client.Client{mockClient},
		collection:   "test",
		vectorDim:    768,
		metricType:   entity.L2,
		topK:         10,
		outputFields: []string{"id"},
		searchLevel:  1,
	}

	ctx := context.Background()
	lt.runWarmup(ctx, 0)
}

func TestRunWarmup_NegativeQueries(t *testing.T) {
	mockClient := &mocks.MockClient{
		SearchFunc: func(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
			t.Error("runWarmup() should not execute queries when warmupQueries is negative")
			return nil, nil
		},
	}

	lt := &LoadTester{
		clients:      []client.Client{mockClient},
		collection:   "test",
		vectorDim:    768,
		metricType:   entity.L2,
		topK:         10,
		outputFields: []string{"id"},
		searchLevel:  1,
	}

	ctx := context.Background()
	lt.runWarmup(ctx, -1)
}

func TestRunWarmup_Success(t *testing.T) {
	queryCount := 0
	mockClient := &mocks.MockClient{
		SearchFunc: func(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
			queryCount++
			return []client.SearchResult{}, nil
		},
	}

	lt := &LoadTester{
		clients:      []client.Client{mockClient},
		collection:   "test",
		vectorDim:    768,
		metricType:   entity.L2,
		topK:         10,
		outputFields: []string{"id"},
		searchLevel:  1,
	}

	ctx := context.Background()
	warmupQueries := 5
	lt.runWarmup(ctx, warmupQueries)

	if queryCount != warmupQueries {
		t.Errorf("runWarmup() executed %d queries, want %d", queryCount, warmupQueries)
	}
}

func TestRunWarmup_WithErrors(t *testing.T) {
	queryCount := 0
	expectedError := errors.New("warmup error")
	mockClient := &mocks.MockClient{
		SearchFunc: func(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
			queryCount++
			return nil, expectedError
		},
	}

	lt := &LoadTester{
		clients:      []client.Client{mockClient},
		collection:   "test",
		vectorDim:    768,
		metricType:   entity.L2,
		topK:         10,
		outputFields: []string{"id"},
		searchLevel:  1,
	}

	ctx := context.Background()
	warmupQueries := 10
	lt.runWarmup(ctx, warmupQueries)

	// Should still execute all queries even with errors
	if queryCount != warmupQueries {
		t.Errorf("runWarmup() executed %d queries, want %d", queryCount, warmupQueries)
	}
}

func TestRunWarmup_Timeout(t *testing.T) {
	mockClient := &mocks.MockClient{
		SearchFunc: func(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
			// Simulate slow query
			time.Sleep(2 * time.Second)
			return []client.SearchResult{}, nil
		},
	}

	lt := &LoadTester{
		clients:      []client.Client{mockClient},
		collection:   "test",
		vectorDim:    768,
		metricType:   entity.L2,
		topK:         10,
		outputFields: []string{"id"},
		searchLevel:  1,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	// Should respect timeout
	lt.runWarmup(ctx, 5)
	// Test should complete without hanging
}
