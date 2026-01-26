package loadtest

import (
	"context"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"zilliz-loadtest/internal/mocks"
)

// createTestLoadTester creates a LoadTester with mock clients for testing
func createTestLoadTester(mockClients []client.Client, collection string, vectorDim int, metricType entity.MetricType) *LoadTester {
	return &LoadTester{
		clients:      mockClients,
		clientIdx:    0,
		collection:   collection,
		vectorDim:    vectorDim,
		metricType:   metricType,
		topK:         10,
		filterExpr:   "",
		outputFields: []string{"id"},
		searchLevel:  1,
	}
}

func TestLoadTesterRunTestShort(t *testing.T) {
	// Create a mock client that succeeds quickly
	searchCallCount := 0
	mockClient := &mocks.MockClient{
		SearchFunc: func(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
			searchCallCount++
			// Simulate very fast response
			return []client.SearchResult{}, nil
		},
	}

	lt := createTestLoadTester([]client.Client{mockClient}, "test", 768, entity.L2)

	// Run a very short test with higher QPS to ensure we get some queries
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	result, err := lt.RunTest(ctx, 100, 100*time.Millisecond, 0) // 100 QPS, 100ms duration, no warmup

	if err != nil {
		t.Errorf("RunTest() error = %v", err)
		return
	}

	if result.QPS != 100 {
		t.Errorf("RunTest() QPS = %v, want 100", result.QPS)
	}

	// With 100 QPS for 100ms, we should get at least a few queries
	// But allow for timing variations
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

	_, err := lt.RunTest(ctx, 100, 50*time.Millisecond, 5) // 5 warmup queries, then 100 QPS for 50ms

	if err != nil {
		t.Errorf("RunTest() error = %v", err)
	}

	// Should have called search at least 5 times (warmup)
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

	result, err := lt.RunTest(ctx, 100, 50*time.Millisecond, 0) // Higher QPS to ensure we hit the error cases

	if err != nil {
		t.Errorf("RunTest() error = %v", err)
	}

	// Should have some errors (at least the first 2 should error)
	if result.Errors == 0 && errorCount >= 2 {
		t.Error("RunTest() should have recorded some errors")
	}

	// If we got errors, they should be categorized as timeout
	if result.Errors > 0 && result.ErrorBreakdown[ErrorTypeTimeout] == 0 {
		t.Error("RunTest() should have categorized errors as timeout")
	}
}

func TestCalculatePercentileEdgeCases(t *testing.T) {
	// Test with single element
	data := []float64{100.0}
	p95 := calculatePercentile(data, 95)
	if p95 != 100.0 {
		t.Errorf("calculatePercentile([100.0], 95) = %v, want 100.0", p95)
	}

	// Test with two elements
	data2 := []float64{50.0, 100.0}
	p50 := calculatePercentile(data2, 50)
	if p50 < 50.0 || p50 > 100.0 {
		t.Errorf("calculatePercentile([50.0, 100.0], 50) = %v, want between 50.0 and 100.0", p50)
	}

	// Test P100 (max)
	data3 := []float64{10.0, 20.0, 30.0, 40.0, 50.0}
	p100 := calculatePercentile(data3, 100)
	if p100 != 50.0 {
		t.Errorf("calculatePercentile([10,20,30,40,50], 100) = %v, want 50.0", p100)
	}

	// Test P0 (min)
	p0 := calculatePercentile(data3, 0)
	if p0 != 10.0 {
		t.Errorf("calculatePercentile([10,20,30,40,50], 0) = %v, want 10.0", p0)
	}
}
