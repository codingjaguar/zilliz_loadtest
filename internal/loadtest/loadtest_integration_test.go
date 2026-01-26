package loadtest

import (
	"zilliz-loadtest/internal/mocks"
	"context"
	"errors"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func TestNewLoadTester(t *testing.T) {
	// Skip this test as it requires network connection and may hang
	// We test the metric type validation in other tests
	t.Skip("Skipping test that requires network connection")
}

func TestNewLoadTesterWithConnections(t *testing.T) {
	// Skip this test - it tries to create clients which hangs
	// The metric type validation happens after client creation, so we can't test it without network
	t.Skip("Skipping test that requires network connection (creates clients before validating metric type)")
}

func TestNewLoadTesterWithOptions(t *testing.T) {
	// Skip this test - it tries to create clients which hangs
	// The metric type validation happens after client creation, so we can't test it without network
	t.Skip("Skipping test that requires network connection (creates clients before validating metric type)")
}

func TestLoadTesterGetClient(t *testing.T) {
	// Create a LoadTester with mock clients
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

	// Test round-robin selection
	client1 := lt.getClient() // Should be client 0 (mockClient1)
	client2 := lt.getClient() // Should be client 1 (mockClient2)
	client3 := lt.getClient() // Should wrap to client 0 (mockClient1)

	if client1 == client2 {
		t.Error("Expected different clients from round-robin")
	}
	if client1 != client3 {
		t.Error("Expected same client after wrapping around (client1 should equal client3)")
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
	// Use an error message that will be recognized as a timeout
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

	// Error should be categorized (not ErrorTypeUnknown when there's an actual error)
	if result.ErrorType == ErrorTypeUnknown {
		t.Errorf("Expected error to be categorized, got ErrorTypeUnknown. Error: %v", result.Error)
		return
	}
	
	// Should be categorized as timeout since error message contains "deadline exceeded"
	if result.ErrorType != ErrorTypeTimeout {
		t.Errorf("Expected error to be categorized as timeout, got %v. Error: %v", result.ErrorType, result.Error)
	}
}

func TestExtractIDs(t *testing.T) {
	// Test with nil column
	result := extractIDs(nil)
	if result != nil {
		t.Error("Expected nil for nil column")
	}

	// Note: We can't easily test the actual column types without importing
	// the full entity package internals, but we test the nil case
}

func TestEmptySearchParamMethods(t *testing.T) {
	param := &EmptySearchParam{}
	
	// Test AddRadius (should not panic)
	param.AddRadius(0.5)
	
	// Test AddRangeFilter (should not panic)
	param.AddRangeFilter(0.5)
}

func TestSearchParamWithLevelMethods(t *testing.T) {
	param := &SearchParamWithLevel{Level: 3}
	
	// Test AddRadius (should not panic)
	param.AddRadius(0.5)
	
	// Test AddRangeFilter (should not panic)
	param.AddRangeFilter(0.5)
}
