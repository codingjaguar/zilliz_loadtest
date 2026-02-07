package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"zilliz-loadtest/internal/loadtest"
)

// loadTestTableWidth is the character width of the load test results table (header/data rows).
const loadTestTableWidth = 105

// displayResults prints load test results as a formatted table.
func displayResults(results []loadtest.TestResult) {
	if len(results) == 0 {
		fmt.Println("No results to display.")
		return
	}

	// Check if any results have recall metrics
	hasRecall := false
	for _, r := range results {
		if r.RecallTested {
			hasRecall = true
			break
		}
	}

	var sep string
	if hasRecall {
		sep = strings.Repeat("=", 165) // Wider table for recall/precision columns
	} else {
		sep = strings.Repeat("=", loadTestTableWidth)
	}

	fmt.Println("\n" + sep)
	fmt.Println("Load Test Results")
	fmt.Println(sep)

	// Header
	if hasRecall {
		fmt.Printf("%-6s | %8s | %8s | %8s | %8s | %8s | %8s | %8s | %9s | %7s | %12s | %10s | %10s\n",
			"QPS", "P50(ms)", "P90(ms)", "P95(ms)", "P99(ms)", "Avg(ms)", "Min(ms)", "Max(ms)", "Success%", "Errors", "KNN Recall", "Relevance", "Precision")
		fmt.Println(strings.Repeat("-", 165))
	} else {
		fmt.Printf("%-6s | %8s | %8s | %8s | %8s | %8s | %8s | %8s | %9s | %7s\n",
			"QPS", "P50(ms)", "P90(ms)", "P95(ms)", "P99(ms)", "Avg(ms)", "Min(ms)", "Max(ms)", "Success%", "Errors")
		fmt.Println(strings.Repeat("-", loadTestTableWidth))
	}

	// Data rows
	for _, r := range results {
		if hasRecall && r.RecallTested {
			fmt.Printf("%-6d | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %8.1f%% | %7d | %11.2f%% | %9.2f%% | %9.2f%%\n",
				r.QPS, r.P50Latency, r.P90Latency, r.P95Latency, r.P99Latency, r.AvgLatency, r.MinLatency, r.MaxLatency, r.SuccessRate, r.Errors, r.MathematicalRecall, r.BusinessRecall, r.BusinessPrecision)
		} else if hasRecall {
			fmt.Printf("%-6d | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %8.1f%% | %7d | %12s | %10s | %10s\n",
				r.QPS, r.P50Latency, r.P90Latency, r.P95Latency, r.P99Latency, r.AvgLatency, r.MinLatency, r.MaxLatency, r.SuccessRate, r.Errors, "N/A", "N/A", "N/A")
		} else {
			fmt.Printf("%-6d | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %8.1f%% | %7d\n",
				r.QPS, r.P50Latency, r.P90Latency, r.P95Latency, r.P99Latency, r.AvgLatency, r.MinLatency, r.MaxLatency, r.SuccessRate, r.Errors)
		}
	}

	fmt.Println(sep)

	// Print recall explanation if available
	if hasRecall {
		fmt.Println("\nSearch Quality Metrics:")
		fmt.Println("  KNN Recall:  Index accuracy - % of exact KNN neighbors found")
		fmt.Println("  Relevance:   % of human-relevant docs found (vs qrels)")
		fmt.Println("  Precision:   % of results that are human-relevant")
	}
}

// CompareResults compares two test result sets and shows differences
func CompareResults(results1, results2 []loadtest.TestResult) {
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("Test Results Comparison")
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println()

	// Create maps by QPS for easier comparison
	result1Map := make(map[int]loadtest.TestResult)
	result2Map := make(map[int]loadtest.TestResult)

	for _, r := range results1 {
		result1Map[r.QPS] = r
	}
	for _, r := range results2 {
		result2Map[r.QPS] = r
	}

	// Find all QPS levels
	allQPS := make(map[int]bool)
	for qps := range result1Map {
		allQPS[qps] = true
	}
	for qps := range result2Map {
		allQPS[qps] = true
	}

	// Print comparison table
	fmt.Printf("%-10s | %-12s | %-12s | %-12s | %-12s | %-12s\n",
		"QPS", "P95 Change", "P99 Change", "QPS Change", "Error Change", "Status")
	fmt.Println(strings.Repeat("-", 80))

	for qps := range allQPS {
		r1, has1 := result1Map[qps]
		r2, has2 := result2Map[qps]

		if !has1 || !has2 {
			status := "Missing"
			if !has1 {
				status = "Missing in baseline"
			} else {
				status = "Missing in comparison"
			}
			fmt.Printf("%-10d | %-12s | %-12s | %-12s | %-12s | %-12s\n",
				qps, "N/A", "N/A", "N/A", "N/A", status)
			continue
		}

		p95Change := r2.P95Latency - r1.P95Latency
		p99Change := r2.P99Latency - r1.P99Latency
		qpsChange := float64(r2.TotalQueries) - float64(r1.TotalQueries)
		errorChange := r2.Errors - r1.Errors

		status := "Improved"
		if p95Change > 0 || p99Change > 0 || errorChange > 0 {
			status = "Degraded"
		}
		if p95Change == 0 && p99Change == 0 && errorChange == 0 {
			status = "Same"
		}

		fmt.Printf("%-10d | %-12.2f | %-12.2f | %-12.0f | %-12d | %-12s\n",
			qps, p95Change, p99Change, qpsChange, errorChange, status)
	}

	fmt.Println()
}

// CompareResultsFromFiles compares results from two JSON files
func CompareResultsFromFiles(file1, file2 string) error {
	data1, err := os.ReadFile(file1)
	if err != nil {
		return fmt.Errorf("failed to read file1: %w", err)
	}

	data2, err := os.ReadFile(file2)
	if err != nil {
		return fmt.Errorf("failed to read file2: %w", err)
	}

	var export1, export2 struct {
		TestResults []loadtest.TestResult `json:"test_results"`
	}

	if err := json.Unmarshal(data1, &export1); err != nil {
		return fmt.Errorf("failed to parse file1: %w", err)
	}

	if err := json.Unmarshal(data2, &export2); err != nil {
		return fmt.Errorf("failed to parse file2: %w", err)
	}

	CompareResults(export1.TestResults, export2.TestResults)
	return nil
}
