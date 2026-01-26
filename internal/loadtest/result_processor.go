package loadtest

import (
	"fmt"
	"sort"
	"time"
)

// processResults processes query results and calculates statistics
type resultStats struct {
	latencies      []float64
	errors         int
	firstError     error
	errorBreakdown map[ErrorType]int
}

// processQueryResults extracts latencies and errors from query results
func processQueryResults(results []QueryResult) resultStats {
	stats := resultStats{
		errorBreakdown: make(map[ErrorType]int),
	}

	for _, result := range results {
		if result.Error != nil {
			stats.errors++
			if stats.firstError == nil {
				stats.firstError = result.Error
			}
			errorType := categorizeError(result.Error)
			stats.errorBreakdown[errorType]++
			continue
		}
		stats.latencies = append(stats.latencies, float64(result.Latency.Milliseconds()))
	}

	return stats
}

// calculateLatencyStats calculates latency statistics from sorted latencies
func calculateLatencyStats(latencies []float64) (min, max, avg float64, p50, p90, p95, p99 float64) {
	if len(latencies) == 0 {
		return 0, 0, 0, 0, 0, 0, 0
	}

	sort.Float64s(latencies)

	min = latencies[0]
	max = latencies[len(latencies)-1]

	var sum float64
	for _, l := range latencies {
		sum += l
	}
	avg = sum / float64(len(latencies))

	p50 = calculatePercentile(latencies, 50)
	p90 = calculatePercentile(latencies, 90)
	p95 = calculatePercentile(latencies, 95)
	p99 = calculatePercentile(latencies, 99)

	return min, max, avg, p50, p90, p95, p99
}

// filterHighLatencies filters out latencies above the threshold and returns filtered stats
func filterHighLatencies(latencies []float64, threshold float64) (filtered []float64, queuedCount int) {
	filtered = make([]float64, 0, len(latencies))
	for _, l := range latencies {
		if l < threshold {
			filtered = append(filtered, l)
		}
	}
	queuedCount = len(latencies) - len(filtered)
	return filtered, queuedCount
}

// reportHighLatencyWarning reports warnings about high latencies
func reportHighLatencyWarning(min, max, avg, p95, p99 float64, latencies []float64) (adjustedP95, adjustedP99 float64, shouldFilter bool) {
	adjustedP95 = p95
	adjustedP99 = p99
	shouldFilter = false

	if max <= HighLatencyThresholdMs {
		return adjustedP95, adjustedP99, shouldFilter
	}

	fmt.Printf("\nWarning: Detected very high latencies (max: %.2f ms). This may indicate:\n", max)
	fmt.Printf("  - Server-side queuing (queries waiting for server capacity)\n")
	fmt.Printf("  - Network congestion\n")
	fmt.Printf("  - Note: These latencies include queue time, not just API response time\n")
	fmt.Printf("  Latency stats: min=%.2f ms, avg=%.2f ms, p95=%.2f ms, p99=%.2f ms, max=%.2f ms\n",
		min, avg, p95, p99, max)

	filteredLatencies, queuedCount := filterHighLatencies(latencies, HighLatencyThresholdMs)

	if len(filteredLatencies) > 0 && len(filteredLatencies) < len(latencies) {
		sort.Float64s(filteredLatencies)
		adjustedP95 = calculatePercentile(filteredLatencies, 95)
		adjustedP99 = calculatePercentile(filteredLatencies, 99)
		shouldFilter = true

		fmt.Printf("  Filtered stats (excluding queued queries >1s): p95=%.2f ms, p99=%.2f ms\n",
			adjustedP95, adjustedP99)
		fmt.Printf("  %d queries (%.1f%%) had latencies >1s, likely due to server-side queuing\n",
			queuedCount, float64(queuedCount)/float64(len(latencies))*100)
	} else if len(filteredLatencies) == 0 {
		fmt.Printf("  ⚠️  All queries had latencies >1s - severe server-side bottleneck detected\n")
	}

	return adjustedP95, adjustedP99, shouldFilter
}

// reportTestSummary prints the test summary and warnings
func reportTestSummary(targetQPS, totalFired, completed int, errors int, errorBreakdown map[ErrorType]int, elapsed time.Duration, numConnections int) {
	actualQPS := float64(completed) / elapsed.Seconds()
	completionRate := (actualQPS / float64(targetQPS)) * 100.0
	successRate := float64(completed) / float64(totalFired) * 100.0

	fmt.Printf("Fired: %d queries | Completed: %d queries in %v\n", totalFired, completed, elapsed)
	fmt.Printf("Fired at: %d QPS | Completed at: %.2f QPS (%.1f%% of target) | Errors: %d (%.2f%% success rate)\n",
		targetQPS, actualQPS, completionRate, errors, successRate)

	if errors > 0 {
		fmt.Printf("\nError Breakdown:\n")
		for errType, count := range errorBreakdown {
			percentage := float64(count) / float64(errors) * 100.0
			fmt.Printf("  - %s errors: %d (%.1f%%)\n", errType, count, percentage)
		}
	}

	if actualQPS < float64(targetQPS)*QPSWarningThreshold {
		fmt.Printf("\n⚠️  Warning: Only achieved %.1f%% of target QPS. This may indicate:\n", completionRate)
		fmt.Printf("  - Server-side rate limiting or capacity limits\n")
		fmt.Printf("  - Network bandwidth constraints\n")
		fmt.Printf("  - Server may not be able to handle %d QPS (currently using %d connections)\n", targetQPS, numConnections)
		fmt.Printf("  Note: Increasing connections from %d didn't improve throughput, suggesting server-side bottleneck\n", numConnections)
	}

	if totalFired > completed {
		pending := totalFired - completed
		fmt.Printf("Note: %d queries were still in flight when test ended (%.1f%% completion rate)\n",
			pending, float64(completed)/float64(totalFired)*100)
	}
}
