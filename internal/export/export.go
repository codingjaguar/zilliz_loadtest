package export

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"zilliz-loadtest/internal/loadtest"
)

// ExportResults exports test results to JSON and/or CSV
func ExportResults(results []loadtest.TestResult, format string, outputPath string) error {
	if format == "json" || format == "both" {
		if err := exportJSON(results, outputPath); err != nil {
			return fmt.Errorf("failed to export JSON: %w", err)
		}
	}

	if format == "csv" || format == "both" {
		if err := exportCSV(results, outputPath); err != nil {
			return fmt.Errorf("failed to export CSV: %w", err)
		}
	}

	return nil
}

// exportJSON exports results to JSON format
func exportJSON(results []loadtest.TestResult, outputPath string) error {
	// Create output data structure with metadata
	exportData := struct {
		Timestamp   string       `json:"timestamp"`
		TestResults []loadtest.TestResult  `json:"test_results"`
		Summary     ExportSummary `json:"summary"`
	}{
		Timestamp:   time.Now().Format(time.RFC3339),
		TestResults: results,
		Summary:     calculateSummary(results),
	}

	// Determine output file path
	jsonPath := outputPath
	if jsonPath == "" {
		jsonPath = fmt.Sprintf("loadtest_results_%s.json", time.Now().Format("20060102_150405"))
	} else if filepath.Ext(jsonPath) != ".json" {
		jsonPath = jsonPath + ".json"
	}

	// Write JSON file
	data, err := json.MarshalIndent(exportData, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal JSON: %w", err)
	}

	if err := os.WriteFile(jsonPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write JSON file: %w", err)
	}

	fmt.Printf("Results exported to JSON: %s\n", jsonPath)
	return nil
}

// exportCSV exports results to CSV format
func exportCSV(results []loadtest.TestResult, outputPath string) error {
	// Determine output file path
	csvPath := outputPath
	if csvPath == "" {
		csvPath = fmt.Sprintf("loadtest_results_%s.csv", time.Now().Format("20060102_150405"))
	} else if filepath.Ext(csvPath) != ".csv" {
		csvPath = outputPath + ".csv"
	}

	// Create CSV file
	file, err := os.Create(csvPath)
	if err != nil {
		return fmt.Errorf("failed to create CSV file: %w", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	header := []string{
		"QPS", "P50 (ms)", "P90 (ms)", "P95 (ms)", "P99 (ms)",
		"Avg (ms)", "Min (ms)", "Max (ms)", "Total Queries",
		"Errors", "Success Rate (%)",
	}
	if err := writer.Write(header); err != nil {
		return fmt.Errorf("failed to write CSV header: %w", err)
	}

	// Write data rows
	for _, result := range results {
		row := []string{
			fmt.Sprintf("%d", result.QPS),
			fmt.Sprintf("%.2f", result.P50Latency),
			fmt.Sprintf("%.2f", result.P90Latency),
			fmt.Sprintf("%.2f", result.P95Latency),
			fmt.Sprintf("%.2f", result.P99Latency),
			fmt.Sprintf("%.2f", result.AvgLatency),
			fmt.Sprintf("%.2f", result.MinLatency),
			fmt.Sprintf("%.2f", result.MaxLatency),
			fmt.Sprintf("%d", result.TotalQueries),
			fmt.Sprintf("%d", result.Errors),
			fmt.Sprintf("%.2f", result.SuccessRate),
		}
		if err := writer.Write(row); err != nil {
			return fmt.Errorf("failed to write CSV row: %w", err)
		}
	}

	fmt.Printf("Results exported to CSV: %s\n", csvPath)
	return nil
}

// ExportSummary contains summary statistics across all test results
type ExportSummary struct {
	TotalTests      int     `json:"total_tests"`
	TotalQueries    int     `json:"total_queries"`
	TotalErrors     int     `json:"total_errors"`
	OverallSuccessRate float64 `json:"overall_success_rate"`
	BestQPS         int     `json:"best_qps"`  // QPS with lowest P95 latency
	WorstQPS        int     `json:"worst_qps"` // QPS with highest P95 latency
}

// calculateSummary calculates summary statistics from test results
func calculateSummary(results []loadtest.TestResult) ExportSummary {
	if len(results) == 0 {
		return ExportSummary{}
	}

	summary := ExportSummary{
		TotalTests: len(results),
	}

	var totalQueries, totalErrors int
	var bestLatency, worstLatency float64 = 1e9, 0

	for _, result := range results {
		totalQueries += result.TotalQueries
		totalErrors += result.Errors

		if result.P95Latency < bestLatency {
			bestLatency = result.P95Latency
			summary.BestQPS = result.QPS
		}
		if result.P95Latency > worstLatency {
			worstLatency = result.P95Latency
			summary.WorstQPS = result.QPS
		}
	}

	summary.TotalQueries = totalQueries
	summary.TotalErrors = totalErrors
	if totalQueries > 0 {
		summary.OverallSuccessRate = float64(totalQueries-totalErrors) / float64(totalQueries) * 100.0
	}

	return summary
}
