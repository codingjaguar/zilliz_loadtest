package export

import (
	"zilliz-loadtest/internal/loadtest"
	"os"
	"testing"
)

func TestCalculateSummary(t *testing.T) {
	tests := []struct {
		name     string
		results  []loadtest.TestResult
		want     ExportSummary
	}{
		{
			name: "empty results",
			results: []loadtest.TestResult{},
			want: ExportSummary{
				TotalTests: 0,
			},
		},
		{
			name: "single result",
			results: []loadtest.TestResult{
				{
					QPS:          100,
					P95Latency:   50.0,
					TotalQueries: 1000,
					Errors:       5,
				},
			},
			want: ExportSummary{
				TotalTests:      1,
				TotalQueries:    1000,
				TotalErrors:      5,
				OverallSuccessRate: 99.5,
				BestQPS:         100,
				WorstQPS:        100,
			},
		},
		{
			name: "multiple results",
			results: []loadtest.TestResult{
				{
					QPS:          100,
					P95Latency:   50.0,
					TotalQueries: 1000,
					Errors:       5,
				},
				{
					QPS:          500,
					P95Latency:   75.0,
					TotalQueries: 5000,
					Errors:       10,
				},
				{
					QPS:          1000,
					P95Latency:   100.0,
					TotalQueries: 10000,
					Errors:       20,
				},
			},
			want: ExportSummary{
				TotalTests:      3,
				TotalQueries:    16000,
				TotalErrors:      35,
				OverallSuccessRate: 99.78125,
				BestQPS:         100,  // Lowest P95 latency
				WorstQPS:        1000, // Highest P95 latency
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := calculateSummary(tt.results)

			if got.TotalTests != tt.want.TotalTests {
				t.Errorf("calculateSummary() TotalTests = %v, want %v", got.TotalTests, tt.want.TotalTests)
			}

			if got.TotalQueries != tt.want.TotalQueries {
				t.Errorf("calculateSummary() TotalQueries = %v, want %v", got.TotalQueries, tt.want.TotalQueries)
			}

			if got.TotalErrors != tt.want.TotalErrors {
				t.Errorf("calculateSummary() TotalErrors = %v, want %v", got.TotalErrors, tt.want.TotalErrors)
			}

			// Allow small floating point differences
			if abs(got.OverallSuccessRate-tt.want.OverallSuccessRate) > 0.01 {
				t.Errorf("calculateSummary() OverallSuccessRate = %v, want %v", got.OverallSuccessRate, tt.want.OverallSuccessRate)
			}

			if got.BestQPS != tt.want.BestQPS {
				t.Errorf("calculateSummary() BestQPS = %v, want %v", got.BestQPS, tt.want.BestQPS)
			}

			if got.WorstQPS != tt.want.WorstQPS {
				t.Errorf("calculateSummary() WorstQPS = %v, want %v", got.WorstQPS, tt.want.WorstQPS)
			}
		})
	}
}

func TestExportResultsJSON(t *testing.T) {
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
			SuccessRate:  99.5,
		},
	}

	// Create a temporary file
	tmpFile := "test_export.json"
	defer os.Remove(tmpFile)

	err := ExportResults(results, "json", tmpFile)
	if err != nil {
		t.Fatalf("ExportResults() error = %v", err)
	}

	// Check that file was created
	if _, err := os.Stat(tmpFile); os.IsNotExist(err) {
		t.Errorf("ExportResults() did not create file %s", tmpFile)
	}
}

func TestExportResultsCSV(t *testing.T) {
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
			SuccessRate:  99.5,
		},
	}

	// Create a temporary file
	tmpFile := "test_export.csv"
	defer os.Remove(tmpFile)

	err := ExportResults(results, "csv", tmpFile)
	if err != nil {
		t.Fatalf("ExportResults() error = %v", err)
	}

	// Check that file was created
	if _, err := os.Stat(tmpFile); os.IsNotExist(err) {
		t.Errorf("ExportResults() did not create file %s", tmpFile)
	}
}

func TestExportResultsBoth(t *testing.T) {
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
			SuccessRate:  99.5,
		},
	}

	// Create temporary files
	tmpFile := "test_export_both"
	defer os.Remove(tmpFile + ".json")
	defer os.Remove(tmpFile + ".csv")

	err := ExportResults(results, "both", tmpFile)
	if err != nil {
		t.Fatalf("ExportResults() error = %v", err)
	}

	// Check that both files were created
	if _, err := os.Stat(tmpFile + ".json"); os.IsNotExist(err) {
		t.Errorf("ExportResults() did not create JSON file")
	}
	if _, err := os.Stat(tmpFile + ".csv"); os.IsNotExist(err) {
		t.Errorf("ExportResults() did not create CSV file")
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
