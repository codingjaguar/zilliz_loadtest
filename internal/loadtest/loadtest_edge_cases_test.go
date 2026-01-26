package loadtest

import (
	"zilliz-loadtest/internal/mocks"
	"context"
	"errors"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func TestLoadTesterRunTestWithNoResults(t *testing.T) {
	// Test with a client that always errors immediately
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

	// Should have errors
	if result.Errors == 0 {
		t.Error("RunTest() should have recorded errors when all queries fail")
	}

	// All errors should be timeout
	if result.ErrorBreakdown[ErrorTypeTimeout] != result.Errors {
		t.Errorf("Expected all errors to be timeout, got breakdown: %v", result.ErrorBreakdown)
	}
}

func TestLoadTesterRunTestWithFilteredLatencies(t *testing.T) {
	// Test scenario where we have high latencies that get filtered
	callCount := 0
	mockClient := &mocks.MockClient{
		SearchFunc: func(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
			callCount++
			// Simulate slow responses that will be filtered
			time.Sleep(1100 * time.Millisecond) // > 1 second
			return []client.SearchResult{}, nil
		},
	}

	lt := createTestLoadTester([]client.Client{mockClient}, "test", 768, entity.L2)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	result, err := lt.RunTest(ctx, 10, 200*time.Millisecond, 0) // Low QPS to allow slow responses

	if err != nil {
		t.Errorf("RunTest() error = %v", err)
	}

	// Should have completed some queries (even if slow)
	if result.TotalQueries == 0 && callCount > 0 {
		t.Error("RunTest() should have some completed queries")
	}
}

func TestCalculatePercentileBoundaryCases(t *testing.T) {
	// Test P1 (1st percentile)
	data := []float64{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0}
	p1 := calculatePercentile(data, 1)
	if p1 < 10.0 || p1 > 20.0 {
		t.Errorf("calculatePercentile([10..100], 1) = %v, want between 10.0 and 20.0", p1)
	}

	// Test P25
	p25 := calculatePercentile(data, 25)
	if p25 < 20.0 || p25 > 40.0 {
		t.Errorf("calculatePercentile([10..100], 25) = %v, want between 20.0 and 40.0", p25)
	}

	// Test P75
	p75 := calculatePercentile(data, 75)
	if p75 < 60.0 || p75 > 90.0 {
		t.Errorf("calculatePercentile([10..100], 75) = %v, want between 60.0 and 90.0", p75)
	}
}

func TestCalculateOptimalConnectionsEdgeCases(t *testing.T) {
	// Test with QPS = 1
	conn, _ := CalculateOptimalConnections(1)
	if conn < 5 {
		t.Errorf("CalculateOptimalConnections(1) = %d, want at least 5 (minimum)", conn)
	}

	// Test with very high QPS
	conn, _ = CalculateOptimalConnections(50000)
	if conn > 2000 {
		t.Errorf("CalculateOptimalConnections(50000) = %d, want at most 2000 (maximum)", conn)
	}
	if conn != 2000 {
		t.Errorf("CalculateOptimalConnections(50000) = %d, want 2000 (capped)", conn)
	}
}

func TestCategorizeErrorMoreCases(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		expected ErrorType
	}{
		{"connection refused", errors.New("connection refused"), ErrorTypeNetwork},
		{"no such host", errors.New("no such host"), ErrorTypeNetwork},
		{"unauthorized", errors.New("unauthorized"), ErrorTypeAPI},
		{"forbidden", errors.New("forbidden"), ErrorTypeAPI},
		{"collection not found", errors.New("collection not found"), ErrorTypeAPI},
		{"marshal error", errors.New("marshal error"), ErrorTypeSDK},
		{"unmarshal error", errors.New("unmarshal error"), ErrorTypeSDK},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := categorizeError(tt.err)
			if result != tt.expected {
				t.Errorf("categorizeError() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestSearchParamWithLevelZero(t *testing.T) {
	param := &SearchParamWithLevel{Level: 0}
	params := param.Params()
	
	// Level 0 should still be included
	if params["level"] != 0 {
		t.Errorf("SearchParamWithLevel(0).Params() level = %v, want 0", params["level"])
	}
}

func TestLoadTesterExecuteQueryWithFilter(t *testing.T) {
	searchCalled := false
	filterUsed := ""
	mockClient := &mocks.MockClient{
		SearchFunc: func(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
			searchCalled = true
			filterUsed = expr
			return []client.SearchResult{}, nil
		},
	}

	lt := &LoadTester{
		clients:      []client.Client{mockClient},
		collection:   "test",
		vectorDim:    768,
		metricType:   entity.L2,
		topK:         20,
		filterExpr:   "id > 1000",
		outputFields: []string{"id", "score"},
		searchLevel:  2,
	}

	result := lt.executeQuery(context.Background(), 1)

	if !searchCalled {
		t.Error("Expected Search to be called")
	}

	if filterUsed != "id > 1000" {
		t.Errorf("Expected filter expression 'id > 1000', got '%s'", filterUsed)
	}

	if result.Error != nil {
		t.Errorf("Unexpected error: %v", result.Error)
	}
}
