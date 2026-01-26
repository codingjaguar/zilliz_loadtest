package loadtest

import (
	"context"
	"errors"
	"testing"

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

// Test utility functions
func TestCategorizeError(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		expected ErrorType
	}{
		{"timeout error", errors.New("context deadline exceeded"), ErrorTypeTimeout},
		{"network connection error", errors.New("connection refused"), ErrorTypeNetwork},
		{"API rate limit error", errors.New("rate limit exceeded"), ErrorTypeAPI},
		{"authentication error", errors.New("authentication failed"), ErrorTypeAPI},
		{"protobuf error", errors.New("protobuf marshal error"), ErrorTypeSDK},
		{"unknown error", errors.New("some random error"), ErrorTypeUnknown},
		{"nil error", nil, ErrorTypeUnknown},
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

func TestCalculatePercentile(t *testing.T) {
	tests := []struct {
		name      string
		data      []float64
		percentile int
		expected  float64
	}{
		{"empty data", []float64{}, 95, 0},
		{"single value P50", []float64{10.0}, 50, 10.0},
		{"P50 median", []float64{1.0, 2.0, 3.0, 4.0, 5.0}, 50, 2.5},
		{"P95 from sorted data", []float64{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0}, 95, 95.0},
		{"P99 from sorted data", []float64{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0}, 99, 99.0},
		{"P90 from 20 values", []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}, 90, 18.0},
		{"single element", []float64{100.0}, 95, 100.0},
		{"two elements", []float64{50.0, 100.0}, 50, 50.0}, // First element for P50 with 2 items
		{"P100 (max)", []float64{10.0, 20.0, 30.0, 40.0, 50.0}, 100, 50.0},
		{"P0 (min)", []float64{10.0, 20.0, 30.0, 40.0, 50.0}, 0, 10.0},
		{"P1", []float64{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0}, 1, 10.0},
		{"P25", []float64{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0}, 25, 25.0},
		{"P75", []float64{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0}, 75, 75.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := calculatePercentile(tt.data, tt.percentile)
			if result != tt.expected {
				t.Errorf("calculatePercentile() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestCalculateOptimalConnections(t *testing.T) {
	tests := []struct {
		name        string
		targetQPS   int
		minExpected int
		maxExpected int
	}{
		{"low QPS", 10, 5, 20},
		{"medium QPS", 100, 10, 15},
		{"high QPS", 1000, 100, 120},
		{"very high QPS", 10000, 1000, 2000},
		{"QPS = 1", 1, 5, 10},
		{"very high QPS capped", 50000, 2000, 2000},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			connections, explanation := CalculateOptimalConnections(tt.targetQPS)
			if connections < tt.minExpected || connections > tt.maxExpected {
				t.Errorf("CalculateOptimalConnections() = %d, want between %d and %d. Explanation: %s",
					connections, tt.minExpected, tt.maxExpected, explanation)
			}
			if explanation == "" {
				t.Error("CalculateOptimalConnections() should return a non-empty explanation")
			}
		})
	}
}

func TestGenerateRandomVector(t *testing.T) {
	dim := 768
	vector := generateRandomVector(dim)

	if len(vector) != dim {
		t.Errorf("generateRandomVector() length = %d, want %d", len(vector), dim)
	}

	for i, val := range vector {
		if val < 0 || val >= 1.0 {
			t.Errorf("generateRandomVector() value at index %d = %f, want in range [0, 1)", i, val)
		}
	}
}

func TestGenerateSeedingVector(t *testing.T) {
	dim := 768
	seed1 := int64(12345)
	seed2 := int64(67890)

	vector1 := generateSeedingVector(dim, seed1)
	vector2 := generateSeedingVector(dim, seed2)
	vector1Again := generateSeedingVector(dim, seed1)

	if len(vector1) != dim {
		t.Errorf("generateSeedingVector() length = %d, want %d", len(vector1), dim)
	}

	for i := range vector1 {
		if vector1[i] != vector1Again[i] {
			t.Errorf("generateSeedingVector() with same seed produced different values at index %d", i)
		}
	}

	allSame := true
	for i := range vector1 {
		if vector1[i] != vector2[i] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("generateSeedingVector() with different seeds produced same vector")
	}
}

func TestSearchParamWithLevel(t *testing.T) {
	param := &SearchParamWithLevel{Level: 3}
	params := param.Params()

	if params["level"] != 3 {
		t.Errorf("SearchParamWithLevel.Params() level = %v, want 3", params["level"])
	}

	param0 := &SearchParamWithLevel{Level: 0}
	params0 := param0.Params()
	if params0["level"] != 0 {
		t.Errorf("SearchParamWithLevel.Params() level = %v, want 0", params0["level"])
	}
}

func TestEmptySearchParam(t *testing.T) {
	param := &EmptySearchParam{}
	params := param.Params()

	if params == nil {
		t.Error("EmptySearchParam.Params() returned nil")
	}

	if len(params) != 0 {
		t.Errorf("EmptySearchParam.Params() length = %d, want 0", len(params))
	}
}

func TestEmptySearchParamMethods(t *testing.T) {
	param := &EmptySearchParam{}
	param.AddRadius(0.5)
	param.AddRangeFilter(0.5)
}

func TestSearchParamWithLevelMethods(t *testing.T) {
	param := &SearchParamWithLevel{Level: 3}
	param.AddRadius(0.5)
	param.AddRangeFilter(0.5)
}

// Test LoadTester methods
func TestLoadTesterGetClient(t *testing.T) {
	mockClient1 := &mocks.MockClient{}
	mockClient2 := &mocks.MockClient{}

	lt := &LoadTester{
		clients:    []client.Client{mockClient1, mockClient2},
		clientIdx:  0,
		collection: "test",
		vectorDim:  768,
		metricType: entity.L2,
		topK:       10,
		outputFields: []string{"id"},
		searchLevel: 1,
	}

	client1 := lt.getClient()
	client2 := lt.getClient()
	client3 := lt.getClient()

	if client1 == client2 {
		t.Error("Expected different clients from round-robin")
	}
	if client1 != client3 {
		t.Error("Expected same client after wrapping around")
	}
}

func TestLoadTesterClose(t *testing.T) {
	closeCalled := false
	mockClient := &mocks.MockClient{
		CloseFunc: func() error {
			closeCalled = true
			return nil
		},
	}

	lt := &LoadTester{
		clients: []client.Client{mockClient},
	}

	lt.Close()

	if !closeCalled {
		t.Error("Expected Close to be called on client")
	}
}

func TestLoadTesterExecuteQuery(t *testing.T) {
	searchCalled := false
	mockClient := &mocks.MockClient{
		SearchFunc: func(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
			searchCalled = true
			return []client.SearchResult{}, nil
		},
	}

	lt := &LoadTester{
		clients:      []client.Client{mockClient},
		collection:   "test",
		vectorDim:    768,
		metricType:   entity.L2,
		topK:         10,
		filterExpr:   "",
		outputFields: []string{"id"},
		searchLevel:  1,
	}

	result := lt.executeQuery(context.Background(), 1)

	if !searchCalled {
		t.Error("Expected Search to be called")
	}

	if result.Error != nil {
		t.Errorf("Unexpected error: %v", result.Error)
	}

	if result.Latency <= 0 {
		t.Error("Expected positive latency")
	}
}

func TestLoadTesterExecuteQueryWithError(t *testing.T) {
	expectedError := errors.New("context deadline exceeded")
	mockClient := &mocks.MockClient{
		SearchFunc: func(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
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

	result := lt.executeQuery(context.Background(), 1)

	if result.Error == nil {
		t.Error("Expected error from executeQuery")
		return
	}

	if result.ErrorType == ErrorTypeUnknown {
		t.Errorf("Expected error to be categorized, got ErrorTypeUnknown. Error: %v", result.Error)
		return
	}

	if result.ErrorType != ErrorTypeTimeout {
		t.Errorf("Expected error to be categorized as timeout, got %v. Error: %v", result.ErrorType, result.Error)
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

func TestExtractIDs(t *testing.T) {
	result := extractIDs(nil)
	if result != nil {
		t.Error("Expected nil for nil column")
	}
}
