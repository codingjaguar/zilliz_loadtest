package loadtest

import (
	"context"
	"testing"
	"time"

	"zilliz-loadtest/internal/mocks"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// Test RunTest integration scenarios
func TestLoadTesterRunTestShort(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	searchCallCount := 0
	mockClient := &mocks.MockClient{
		SearchFunc: func(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
			searchCallCount++
			return []client.SearchResult{}, nil
		},
	}

	lt := createTestLoadTester([]client.Client{mockClient}, "test", 768, entity.L2)

	result, err := lt.RunTest(ctx, 100, 2*time.Second, 0)

	if err != nil {
		t.Errorf("RunTest() error = %v", err)
		return
	}

	if result.QPS != 100 {
		t.Errorf("RunTest() QPS = %v, want 100", result.QPS)
	}

	if result.TotalQueries == 0 && searchCallCount == 0 {
		t.Error("RunTest() should have called Search at least once")
	}
}

func TestLoadTesterRunTestWithWarmup(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	searchCallCount := 0
	mockClient := &mocks.MockClient{
		SearchFunc: func(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
			searchCallCount++
			return []client.SearchResult{}, nil
		},
	}

	lt := createTestLoadTester([]client.Client{mockClient}, "test", 768, entity.L2)

	_, err := lt.RunTest(ctx, 100, 2*time.Second, 5)

	if err != nil {
		t.Errorf("RunTest() error = %v", err)
	}

	if searchCallCount < 5 {
		t.Errorf("RunTest() should have called Search at least 5 times for warmup, got %d", searchCallCount)
	}
}

func TestLoadTesterRunTestWithErrors(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	errorCount := 0
	mockClient := &mocks.MockClient{
		SearchFunc: func(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
			errorCount++
			if errorCount <= 2 {
				return nil, context.DeadlineExceeded
			}
			return []client.SearchResult{}, nil
		},
	}

	lt := createTestLoadTester([]client.Client{mockClient}, "test", 768, entity.L2)

	result, err := lt.RunTest(ctx, 100, 2*time.Second, 0)

	if err != nil {
		t.Errorf("RunTest() error = %v", err)
	}

	if result.Errors == 0 && errorCount >= 2 {
		t.Error("RunTest() should have recorded some errors")
	}

	if result.Errors > 0 && result.ErrorBreakdown[ErrorTypeTimeout] == 0 {
		t.Error("RunTest() should have categorized errors as timeout")
	}
}

func TestLoadTesterRunTestWithNoResults(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	errorCount := 0
	mockClient := &mocks.MockClient{
		SearchFunc: func(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
			errorCount++
			return nil, context.DeadlineExceeded
		},
	}

	lt := createTestLoadTester([]client.Client{mockClient}, "test", 768, entity.L2)

	result, err := lt.RunTest(ctx, 100, 2*time.Second, 0)

	// All queries failed is a valid test outcome, no error should be returned
	if err != nil {
		t.Errorf("RunTest() error = %v (expected nil when all queries fail)", err)
	}

	if result.Errors == 0 {
		t.Error("RunTest() should have recorded errors when all queries fail")
	}

	if result.ErrorBreakdown[ErrorTypeTimeout] != result.Errors {
		t.Errorf("Expected all errors to be timeout, got breakdown: %v", result.ErrorBreakdown)
	}
}

func TestLoadTesterRunTestWithFilteredLatencies(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	callCount := 0
	mockClient := &mocks.MockClient{
		SearchFunc: func(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
			callCount++
			// Use context-aware sleep
			select {
			case <-time.After(1100 * time.Millisecond): // > 1 second
			case <-ctx.Done():
				return nil, ctx.Err()
			}
			return []client.SearchResult{}, nil
		},
	}

	lt := createTestLoadTester([]client.Client{mockClient}, "test", 768, entity.L2)

	result, err := lt.RunTest(ctx, 10, 2*time.Second, 0)

	if err != nil {
		t.Errorf("RunTest() error = %v", err)
	}

	if result.TotalQueries == 0 && callCount > 0 {
		t.Error("RunTest() should have some completed queries")
	}
}
