package loadtest

import (
	"errors"
	"testing"
)

func TestCategorizeError(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		expected ErrorType
	}{
		{
			name:     "timeout error",
			err:      errors.New("context deadline exceeded"),
			expected: ErrorTypeTimeout,
		},
		{
			name:     "network connection error",
			err:      errors.New("connection refused"),
			expected: ErrorTypeNetwork,
		},
		{
			name:     "API rate limit error",
			err:      errors.New("rate limit exceeded"),
			expected: ErrorTypeAPI,
		},
		{
			name:     "authentication error",
			err:      errors.New("authentication failed"),
			expected: ErrorTypeAPI,
		},
		{
			name:     "protobuf error",
			err:      errors.New("protobuf marshal error"),
			expected: ErrorTypeSDK,
		},
		{
			name:     "unknown error",
			err:      errors.New("some random error"),
			expected: ErrorTypeUnknown,
		},
		{
			name:     "nil error",
			err:      nil,
			expected: ErrorTypeUnknown,
		},
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
		{
			name:      "empty data",
			data:      []float64{},
			percentile: 95,
			expected:  0,
		},
		{
			name:      "single value P50",
			data:      []float64{10.0},
			percentile: 50,
			expected:  10.0,
		},
		{
			name:      "P50 median",
			data:      []float64{1.0, 2.0, 3.0, 4.0, 5.0},
			percentile: 50,
			expected:  2.5, // Interpolated between index 1 and 2
		},
		{
			name:      "P95 from sorted data",
			data:      []float64{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0},
			percentile: 95,
			expected:  95.0,
		},
		{
			name:      "P99 from sorted data",
			data:      []float64{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0},
			percentile: 99,
			expected:  99.0,
		},
		{
			name:      "P90 from 20 values",
			data:      []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
			percentile: 90,
			expected:  18.0,
		},
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
		{
			name:        "low QPS",
			targetQPS:   10,
			minExpected: 5, // Minimum enforced
			maxExpected: 20,
		},
		{
			name:        "medium QPS",
			targetQPS:   100,
			minExpected: 10,
			maxExpected: 15,
		},
		{
			name:        "high QPS",
			targetQPS:   1000,
			minExpected: 100,
			maxExpected: 120,
		},
		{
			name:        "very high QPS",
			targetQPS:   10000,
			minExpected: 1000,
			maxExpected: 2000, // Max enforced
		},
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

	// Check that values are in valid range [0, 1)
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

	// Same seed should produce same vector
	for i := range vector1 {
		if vector1[i] != vector1Again[i] {
			t.Errorf("generateSeedingVector() with same seed produced different values at index %d", i)
		}
	}

	// Different seeds should produce different vectors
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

	// Test with level 0 (should still include it)
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
