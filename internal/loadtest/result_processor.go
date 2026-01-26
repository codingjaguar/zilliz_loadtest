package loadtest

import (
	"sort"
	"time"

	"zilliz-loadtest/internal/logger"
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

	logger.Warn("Detected very high latencies",
		"max_latency_ms", max,
		"min_latency_ms", min,
		"avg_latency_ms", avg,
		"p95_latency_ms", p95,
		"p99_latency_ms", p99,
		"message", "This may indicate server-side queuing or network congestion")

	filteredLatencies, queuedCount := filterHighLatencies(latencies, HighLatencyThresholdMs)

	if len(filteredLatencies) > 0 && len(filteredLatencies) < len(latencies) {
		sort.Float64s(filteredLatencies)
		adjustedP95 = calculatePercentile(filteredLatencies, 95)
		adjustedP99 = calculatePercentile(filteredLatencies, 99)
		shouldFilter = true

		logger.Info("Filtered latency stats",
			"p95_latency_ms", adjustedP95,
			"p99_latency_ms", adjustedP99,
			"queued_queries", queuedCount,
			"queued_percent", float64(queuedCount)/float64(len(latencies))*100,
			"message", "Excluding queued queries >1s")
	} else if len(filteredLatencies) == 0 {
		logger.Warn("All queries had latencies >1s",
			"message", "Severe server-side bottleneck detected")
	}

	return adjustedP95, adjustedP99, shouldFilter
}

// reportTestSummary prints the test summary and warnings
func reportTestSummary(targetQPS, totalFired, completed int, errors int, errorBreakdown map[ErrorType]int, elapsed time.Duration, numConnections int) {
	actualQPS := float64(completed) / elapsed.Seconds()
	completionRate := (actualQPS / float64(targetQPS)) * 100.0
	successRate := float64(completed) / float64(totalFired) * 100.0

	logger.Info("Test summary",
		"target_qps", targetQPS,
		"actual_qps", actualQPS,
		"completion_rate_percent", completionRate,
		"total_fired", totalFired,
		"total_completed", completed,
		"errors", errors,
		"success_rate_percent", successRate,
		"elapsed_seconds", elapsed.Seconds())

	if errors > 0 {
		logger.Info("Error breakdown",
			"total_errors", errors)
		for errType, count := range errorBreakdown {
			percentage := float64(count) / float64(errors) * 100.0
			logger.Info("Error type",
				"error_type", string(errType),
				"count", count,
				"percentage", percentage)
		}
	}

	if actualQPS < float64(targetQPS)*QPSWarningThreshold {
		logger.Warn("Low QPS achievement",
			"completion_rate_percent", completionRate,
			"target_qps", targetQPS,
			"actual_qps", actualQPS,
			"connections", numConnections,
			"message", "This may indicate server-side rate limiting, capacity limits, or network constraints")
	}

	if totalFired > completed {
		pending := totalFired - completed
		completionRate := float64(completed) / float64(totalFired) * 100
		logger.Info("Queries still in flight",
			"pending", pending,
			"completion_rate_percent", completionRate)
	}
}
