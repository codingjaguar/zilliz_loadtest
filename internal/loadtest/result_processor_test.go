package loadtest

import (
	"testing"
	"time"
)

func TestProcessQueryResults(t *testing.T) {
	tests := []struct {
		name           string
		results        []QueryResult
		wantErrors     int
		wantLatencies  int
		wantErrorTypes map[ErrorType]int
	}{
		{
			name: "all successful",
			results: []QueryResult{
				{Latency: 100 * time.Millisecond, Error: nil},
				{Latency: 200 * time.Millisecond, Error: nil},
				{Latency: 150 * time.Millisecond, Error: nil},
			},
			wantErrors:    0,
			wantLatencies: 3,
			wantErrorTypes: map[ErrorType]int{},
		},
		{
			name: "mixed results",
			results: []QueryResult{
				{Latency: 100 * time.Millisecond, Error: nil},
				{Latency: 0, Error: &testError{msg: "timeout"}, ErrorType: ErrorTypeTimeout},
				{Latency: 200 * time.Millisecond, Error: nil},
				{Latency: 0, Error: &testError{msg: "network"}, ErrorType: ErrorTypeNetwork},
			},
			wantErrors:    2,
			wantLatencies: 2,
			wantErrorTypes: map[ErrorType]int{
				ErrorTypeTimeout: 1,
				ErrorTypeNetwork: 1,
			},
		},
		{
			name: "all errors",
			results: []QueryResult{
				{Latency: 0, Error: &testError{msg: "timeout"}, ErrorType: ErrorTypeTimeout},
				{Latency: 0, Error: &testError{msg: "timeout"}, ErrorType: ErrorTypeTimeout},
				{Latency: 0, Error: &testError{msg: "network"}, ErrorType: ErrorTypeNetwork},
			},
			wantErrors:    3,
			wantLatencies: 0,
			wantErrorTypes: map[ErrorType]int{
				ErrorTypeTimeout: 2,
				ErrorTypeNetwork: 1,
			},
		},
		{
			name:          "empty results",
			results:       []QueryResult{},
			wantErrors:    0,
			wantLatencies: 0,
			wantErrorTypes: map[ErrorType]int{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stats := processQueryResults(tt.results)

			if stats.errors != tt.wantErrors {
				t.Errorf("processQueryResults() errors = %d, want %d", stats.errors, tt.wantErrors)
			}
			if len(stats.latencies) != tt.wantLatencies {
				t.Errorf("processQueryResults() latencies = %d, want %d", len(stats.latencies), tt.wantLatencies)
			}
			if len(stats.errorBreakdown) != len(tt.wantErrorTypes) {
				t.Errorf("processQueryResults() errorBreakdown length = %d, want %d", len(stats.errorBreakdown), len(tt.wantErrorTypes))
			}
			for errType, wantCount := range tt.wantErrorTypes {
				if stats.errorBreakdown[errType] != wantCount {
					t.Errorf("processQueryResults() errorBreakdown[%v] = %d, want %d", errType, stats.errorBreakdown[errType], wantCount)
				}
			}
		})
	}
}

func TestCalculateLatencyStats(t *testing.T) {
	tests := []struct {
		name     string
		latencies []float64
		wantMin  float64
		wantMax  float64
		wantAvg  float64
		wantP50  float64
		wantP90  float64
		wantP95  float64
		wantP99  float64
	}{
		{
			name:     "empty latencies",
			latencies: []float64{},
			wantMin:   0,
			wantMax:   0,
			wantAvg:   0,
			wantP50:   0,
			wantP90:   0,
			wantP95:   0,
			wantP99:   0,
		},
		{
			name:     "single latency",
			latencies: []float64{100.0},
			wantMin:  100.0,
			wantMax:  100.0,
			wantAvg:  100.0,
			wantP50:  100.0,
			wantP90:  100.0,
			wantP95:  100.0,
			wantP99:  100.0,
		},
		{
			name:     "sorted latencies",
			latencies: []float64{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0},
			wantMin:  10.0,
			wantMax:  100.0,
			wantAvg:  55.0,
			wantP50:  50.0,
			wantP90:  90.0,
			wantP95:  95.0,
			wantP99:  99.0,
		},
		{
			name:     "unsorted latencies",
			latencies: []float64{50.0, 10.0, 90.0, 30.0, 70.0},
			wantMin:  10.0,
			wantMax:  90.0,
			wantAvg:  50.0,
			wantP50:  40.0, // After sorting: [10, 30, 50, 70, 90], P50 interpolates between 30 and 50 = 40
			wantP90:  80.0, // P90 interpolates between 70 and 90 = 80
			wantP95:  85.0, // P95 interpolates between 70 and 90 = 85
			wantP99:  89.0, // P99 interpolates between 70 and 90 = 89
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			min, max, avg, p50, p90, p95, p99 := calculateLatencyStats(tt.latencies)

			if min != tt.wantMin {
				t.Errorf("calculateLatencyStats() min = %v, want %v", min, tt.wantMin)
			}
			if max != tt.wantMax {
				t.Errorf("calculateLatencyStats() max = %v, want %v", max, tt.wantMax)
			}
			if avg != tt.wantAvg {
				t.Errorf("calculateLatencyStats() avg = %v, want %v", avg, tt.wantAvg)
			}
			if p50 != tt.wantP50 {
				t.Errorf("calculateLatencyStats() p50 = %v, want %v", p50, tt.wantP50)
			}
			if p90 != tt.wantP90 {
				t.Errorf("calculateLatencyStats() p90 = %v, want %v", p90, tt.wantP90)
			}
			if p95 != tt.wantP95 {
				t.Errorf("calculateLatencyStats() p95 = %v, want %v", p95, tt.wantP95)
			}
			if p99 != tt.wantP99 {
				t.Errorf("calculateLatencyStats() p99 = %v, want %v", p99, tt.wantP99)
			}
		})
	}
}

func TestFilterHighLatencies(t *testing.T) {
	tests := []struct {
		name          string
		latencies     []float64
		threshold     float64
		wantFiltered  int
		wantQueued    int
	}{
		{
			name:         "all below threshold",
			latencies:    []float64{100.0, 200.0, 300.0},
			threshold:    1000.0,
			wantFiltered: 3,
			wantQueued:   0,
		},
		{
			name:         "all above threshold",
			latencies:    []float64{1500.0, 2000.0, 3000.0},
			threshold:    1000.0,
			wantFiltered: 0,
			wantQueued:   3,
		},
		{
			name:         "mixed",
			latencies:    []float64{100.0, 1500.0, 200.0, 2000.0, 300.0},
			threshold:    1000.0,
			wantFiltered: 3,
			wantQueued:   2,
		},
		{
			name:         "empty",
			latencies:    []float64{},
			threshold:    1000.0,
			wantFiltered: 0,
			wantQueued:   0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filtered, queuedCount := filterHighLatencies(tt.latencies, tt.threshold)

			if len(filtered) != tt.wantFiltered {
				t.Errorf("filterHighLatencies() filtered = %d, want %d", len(filtered), tt.wantFiltered)
			}
			if queuedCount != tt.wantQueued {
				t.Errorf("filterHighLatencies() queuedCount = %d, want %d", queuedCount, tt.wantQueued)
			}

			// Verify all filtered values are below threshold
			for _, l := range filtered {
				if l >= tt.threshold {
					t.Errorf("filterHighLatencies() filtered value %v >= threshold %v", l, tt.threshold)
				}
			}
		})
	}
}

func TestReportHighLatencyWarning(t *testing.T) {
	tests := []struct {
		name         string
		min          float64
		max          float64
		avg          float64
		p95          float64
		p99          float64
		latencies    []float64
		wantFilter   bool
		wantAdjusted bool
	}{
		{
			name:         "no high latencies",
			min:          10.0,
			max:          100.0,
			avg:          50.0,
			p95:          95.0,
			p99:          99.0,
			latencies:    []float64{10.0, 50.0, 100.0},
			wantFilter:   false,
			wantAdjusted: false,
		},
		{
			name:         "high latencies present",
			min:          10.0,
			max:          2000.0,
			avg:          500.0,
			p95:          1500.0,
			p99:          1900.0,
			latencies:    []float64{10.0, 50.0, 100.0, 1500.0, 2000.0},
			wantFilter:   true,
			wantAdjusted: true,
		},
		{
			name:         "all high latencies",
			min:          1500.0,
			max:          2000.0,
			avg:          1750.0,
			p95:          1950.0,
			p99:          1990.0,
			latencies:    []float64{1500.0, 1800.0, 2000.0},
			wantFilter:   false,
			wantAdjusted: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adjustedP95, adjustedP99, shouldFilter := reportHighLatencyWarning(tt.min, tt.max, tt.avg, tt.p95, tt.p99, tt.latencies)

			if shouldFilter != tt.wantFilter {
				t.Errorf("reportHighLatencyWarning() shouldFilter = %v, want %v", shouldFilter, tt.wantFilter)
			}

			if tt.wantAdjusted {
				if adjustedP95 >= tt.p95 || adjustedP99 >= tt.p99 {
					t.Errorf("reportHighLatencyWarning() should have adjusted percentiles downward")
				}
			}
		})
	}
}

func TestReportTestSummary(t *testing.T) {
	// This function mainly logs, so we just verify it doesn't panic
	tests := []struct {
		name           string
		targetQPS      int
		totalFired     int
		completed      int
		errors         int
		errorBreakdown map[ErrorType]int
		elapsed        time.Duration
		numConnections int
	}{
		{
			name:           "successful test",
			targetQPS:      100,
			totalFired:     1000,
			completed:     1000,
			errors:         0,
			errorBreakdown: map[ErrorType]int{},
			elapsed:        10 * time.Second,
			numConnections: 10,
		},
		{
			name:      "test with errors",
			targetQPS:  100,
			totalFired: 1000,
			completed: 950,
			errors:    50,
			errorBreakdown: map[ErrorType]int{
				ErrorTypeTimeout: 30,
				ErrorTypeNetwork: 20,
			},
			elapsed:        10 * time.Second,
			numConnections: 10,
		},
		{
			name:           "low QPS",
			targetQPS:      1000,
			totalFired:     1000,
			completed:     500,
			errors:         0,
			errorBreakdown: map[ErrorType]int{},
			elapsed:        10 * time.Second,
			numConnections: 10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Should not panic
			reportTestSummary(tt.targetQPS, tt.totalFired, tt.completed, tt.errors, tt.errorBreakdown, tt.elapsed, tt.numConnections)
		})
	}
}

// testError is a simple error type for testing
type testError struct {
	msg string
}

func (e *testError) Error() string {
	return e.msg
}
