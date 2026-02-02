package main

import (
	"testing"

	"zilliz-loadtest/internal/loadtest"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func TestParseMetricType(t *testing.T) {
	tests := []struct {
		name        string
		input       string
		expected    entity.MetricType
		expectError bool
	}{
		{"L2 uppercase", "L2", entity.L2, false},
		{"L2 lowercase", "l2", entity.L2, false},
		{"IP uppercase", "IP", entity.IP, false},
		{"IP lowercase", "ip", entity.IP, false},
		{"COSINE uppercase", "COSINE", entity.COSINE, false},
		{"COSINE lowercase", "cosine", entity.COSINE, false},
		{"COSINE mixed case", "Cosine", entity.COSINE, false},
		{"invalid returns error", "INVALID", entity.L2, true},
		{"empty returns error", "", entity.L2, true},
		{"with spaces", " L2 ", entity.L2, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := loadtest.ParseMetricType(tt.input)
			if tt.expectError {
				if err == nil {
					t.Errorf("ParseMetricType(%q) expected error", tt.input)
				}
				if result != tt.expected {
					t.Errorf("ParseMetricType(%q) on error returned %v, want %v", tt.input, result, tt.expected)
				}
				return
			}
			if err != nil {
				t.Errorf("ParseMetricType(%q) error: %v", tt.input, err)
				return
			}
			if result != tt.expected {
				t.Errorf("ParseMetricType(%q) = %v, want %v", tt.input, result, tt.expected)
			}
		})
	}
}

func TestParseQPSLevels(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		expected  []int
		expectErr bool
	}{
		{"single value", "100", []int{100}, false},
		{"multiple values", "100,500,1000", []int{100, 500, 1000}, false},
		{"with spaces", "100, 500, 1000", []int{100, 500, 1000}, false},
		{"empty string returns nil", "", nil, false}, // ParseQPSLevels("") returns (nil, nil)
		{"invalid values", "100,abc,500", nil, true},
		{"negative values", "100,-50,500", nil, true},
		{"zero value", "100,0,500", nil, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := ParseQPSLevels(tt.input)
			if tt.expectErr {
				if err == nil {
					t.Errorf("ParseQPSLevels(%q) expected error", tt.input)
				}
				return
			}
			if err != nil {
				t.Errorf("ParseQPSLevels(%q) error: %v", tt.input, err)
				return
			}
			if tt.expected == nil {
				if result != nil && len(result) != 0 {
					t.Errorf("ParseQPSLevels(%q) = %v, want nil or empty", tt.input, result)
				}
				return
			}
			if len(result) != len(tt.expected) {
				t.Errorf("ParseQPSLevels(%q) length = %d, want %d", tt.input, len(result), len(tt.expected))
				return
			}
			for i, v := range result {
				if v != tt.expected[i] {
					t.Errorf("ParseQPSLevels(%q)[%d] = %d, want %d", tt.input, i, v, tt.expected[i])
				}
			}
		})
	}
}

func TestDisplayResults(t *testing.T) {
	results := []loadtest.TestResult{
		{
			QPS:          100,
			P50Latency:   40.0,
			P90Latency:   55.0,
			P95Latency:   60.0,
			P99Latency:   80.0,
			AvgLatency:   45.0,
			MinLatency:   10.0,
			MaxLatency:   200.0,
			TotalQueries: 1000,
			Errors:       5,
			ErrorBreakdown: map[loadtest.ErrorType]int{
				loadtest.ErrorTypeTimeout: 3,
				loadtest.ErrorTypeNetwork: 2,
			},
			SuccessRate: 99.5,
		},
		{
			QPS:          500,
			P50Latency:   48.0,
			P90Latency:   72.0,
			P95Latency:   85.0,
			P99Latency:   125.0,
			AvgLatency:   52.0,
			MinLatency:   15.0,
			MaxLatency:   456.0,
			TotalQueries: 5000,
			Errors:       0,
			SuccessRate:  100.0,
		},
	}

	// Just verify it doesn't panic (displayResults is in compare.go)
	displayResults(results)
}

func TestDisplayResultsEmpty(t *testing.T) {
	// Test with empty results
	displayResults([]loadtest.TestResult{})
}

func TestDisplayResultsWithErrors(t *testing.T) {
	results := []loadtest.TestResult{
		{
			QPS:          100,
			P50Latency:   40.0,
			P90Latency:   55.0,
			P95Latency:   60.0,
			P99Latency:   80.0,
			AvgLatency:   45.0,
			MinLatency:   10.0,
			MaxLatency:   200.0,
			TotalQueries: 1000,
			Errors:       10,
			ErrorBreakdown: map[loadtest.ErrorType]int{
				loadtest.ErrorTypeTimeout: 5,
				loadtest.ErrorTypeNetwork: 3,
				loadtest.ErrorTypeAPI:     2,
			},
			SuccessRate: 99.0,
		},
	}

	// Just verify it doesn't panic and shows error breakdown
	displayResults(results)
}
