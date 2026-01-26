package loadtest

import (
	"context"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"zilliz-loadtest/internal/mocks"
)

// Test RunTest integration scenarios
func TestLoadTesterRunTestShort(t *testing.T) {
	searchCallCount := 0
	mockClient := &mocks.MockClient{
		SearchFunc: func(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
			searchCallCount++
			return []client.SearchResult{}, nil
		},
	}

	lt := createTestLoadTester([]client.Client{mockClient}, "test", 768, entity.L2)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	result, err := lt.RunTest(ctx, 100, 100*time.Millisecond, 0)

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
	searchCallCount := 0
	mockClient := &mocks.MockClient{
		SearchFunc: func(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
			searchCallCount++
			return []client.SearchResult{}, nil
		},
	}

	lt := createTestLoadTester([]client.Client{mockClient}, "test", 768, entity.L2)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	_, err := lt.RunTest(ctx, 100, 50*time.Millisecond, 5)

	if err != nil {
		t.Errorf("RunTest() error = %v", err)
	}

	if searchCallCount < 5 {
		t.Errorf("RunTest() should have called Search at least 5 times for warmup, got %d", searchCallCount)
	}
}

func TestLoadTesterRunTestWithErrors(t *testing.T) {
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

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	result, err := lt.RunTest(ctx, 100, 50*time.Millisecond, 0)

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
	errorCount := 0
	mockClient := &mocks.MockClient{
		SearchFunc: func(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
			errorCount++
			return nil, context.DeadlineExceeded
		},
	}

	lt := createTestLoadTester([]client.Client{mockClient}, "test", 768, entity.L2)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	result, err := lt.RunTest(ctx, 100, 50*time.Millisecond, 0)

	if err != nil {
		t.Errorf("RunTest() error = %v", err)
	}

	if result.Errors == 0 {
		t.Error("RunTest() should have recorded errors when all queries fail")
	}

	if result.ErrorBreakdown[ErrorTypeTimeout] != result.Errors {
		t.Errorf("Expected all errors to be timeout, got breakdown: %v", result.ErrorBreakdown)
	}
}

func TestLoadTesterRunTestWithFilteredLatencies(t *testing.T) {
	callCount := 0
	mockClient := &mocks.MockClient{
		SearchFunc: func(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
			callCount++
			time.Sleep(1100 * time.Millisecond) // > 1 second
			return []client.SearchResult{}, nil
		},
	}

	lt := createTestLoadTester([]client.Client{mockClient}, "test", 768, entity.L2)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	result, err := lt.RunTest(ctx, 10, 200*time.Millisecond, 0)

	if err != nil {
		t.Errorf("RunTest() error = %v", err)
	}

	if result.TotalQueries == 0 && callCount > 0 {
		t.Error("RunTest() should have some completed queries")
	}
}
