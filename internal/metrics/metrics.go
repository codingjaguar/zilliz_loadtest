package metrics

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"zilliz-loadtest/internal/loadtest"
)

// MetricsCollector collects and exports metrics
type MetricsCollector struct {
	mu              sync.RWMutex
	queryLatencies  []timeSeriesPoint
	queryRates      []timeSeriesPoint
	errorRates      map[loadtest.ErrorType][]timeSeriesPoint
	startTime       time.Time
	enabled         bool
	exportDirectory string
}

type timeSeriesPoint struct {
	Timestamp time.Time
	Value     float64
	Labels    map[string]string
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector(enabled bool, exportDirectory string) *MetricsCollector {
	return &MetricsCollector{
		queryLatencies:  make([]timeSeriesPoint, 0),
		queryRates:      make([]timeSeriesPoint, 0),
		errorRates:      make(map[loadtest.ErrorType][]timeSeriesPoint),
		startTime:       time.Now(),
		enabled:         enabled,
		exportDirectory: exportDirectory,
	}
}

// RecordLatency records a query latency
func (mc *MetricsCollector) RecordLatency(latency time.Duration, labels map[string]string) {
	if !mc.enabled {
		return
	}

	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.queryLatencies = append(mc.queryLatencies, timeSeriesPoint{
		Timestamp: time.Now(),
		Value:     latency.Seconds() * 1000, // Convert to milliseconds
		Labels:    labels,
	})
}

// RecordQueryRate records the current query rate
func (mc *MetricsCollector) RecordQueryRate(qps float64, labels map[string]string) {
	if !mc.enabled {
		return
	}

	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.queryRates = append(mc.queryRates, timeSeriesPoint{
		Timestamp: time.Now(),
		Value:     qps,
		Labels:    labels,
	})
}

// RecordError records an error occurrence
func (mc *MetricsCollector) RecordError(errorType loadtest.ErrorType, labels map[string]string) {
	if !mc.enabled {
		return
	}

	mc.mu.Lock()
	defer mc.mu.Unlock()

	if mc.errorRates[errorType] == nil {
		mc.errorRates[errorType] = make([]timeSeriesPoint, 0)
	}

	mc.errorRates[errorType] = append(mc.errorRates[errorType], timeSeriesPoint{
		Timestamp: time.Now(),
		Value:     1, // Count
		Labels:    labels,
	})
}

// ExportPrometheus exports metrics in Prometheus format
func (mc *MetricsCollector) ExportPrometheus(outputPath string) error {
	if !mc.enabled {
		return nil
	}

	mc.mu.RLock()
	defer mc.mu.RUnlock()

	if outputPath == "" {
		outputPath = filepath.Join(mc.exportDirectory, fmt.Sprintf("metrics_%s.prom", time.Now().Format("20060102_150405")))
	}

	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create metrics file: %w", err)
	}
	defer file.Close()

	// Write latency histogram
	file.WriteString("# HELP query_latency_ms Query latency in milliseconds\n")
	file.WriteString("# TYPE query_latency_ms histogram\n")
	for _, point := range mc.queryLatencies {
		labels := formatPrometheusLabels(point.Labels)
		file.WriteString(fmt.Sprintf("query_latency_ms%s %.2f\n", labels, point.Value))
	}

	// Write query rate
	file.WriteString("\n# HELP query_rate_queries_per_second Current query rate\n")
	file.WriteString("# TYPE query_rate_queries_per_second gauge\n")
	for _, point := range mc.queryRates {
		labels := formatPrometheusLabels(point.Labels)
		file.WriteString(fmt.Sprintf("query_rate_queries_per_second%s %.2f\n", labels, point.Value))
	}

	// Write error rates
	file.WriteString("\n# HELP error_count_total Total error count by type\n")
	file.WriteString("# TYPE error_count_total counter\n")
	for errorType, points := range mc.errorRates {
		for _, point := range points {
			labels := formatPrometheusLabels(point.Labels)
			labels = fmt.Sprintf("%s,error_type=\"%s\"", labels, string(errorType))
			file.WriteString(fmt.Sprintf("error_count_total%s %.0f\n", labels, point.Value))
		}
	}

	return nil
}

// formatPrometheusLabels formats labels for Prometheus
func formatPrometheusLabels(labels map[string]string) string {
	if len(labels) == 0 {
		return ""
	}

	result := "{"
	first := true
	for k, v := range labels {
		if !first {
			result += ","
		}
		result += fmt.Sprintf("%s=\"%s\"", k, v)
		first = false
	}
	result += "}"
	return result
}

// ExportTimeSeries exports time-series data to CSV
func (mc *MetricsCollector) ExportTimeSeries(outputPath string) error {
	if !mc.enabled {
		return nil
	}

	mc.mu.RLock()
	defer mc.mu.RUnlock()

	if outputPath == "" {
		outputPath = filepath.Join(mc.exportDirectory, fmt.Sprintf("timeseries_%s.csv", time.Now().Format("20060102_150405")))
	}

	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create time-series file: %w", err)
	}
	defer file.Close()

	// Write CSV header
	file.WriteString("timestamp,metric,value,labels\n")

	// Write latency data
	for _, point := range mc.queryLatencies {
		labels := formatCSVLabels(point.Labels)
		file.WriteString(fmt.Sprintf("%s,latency_ms,%.2f,%s\n",
			point.Timestamp.Format(time.RFC3339), point.Value, labels))
	}

	// Write query rate data
	for _, point := range mc.queryRates {
		labels := formatCSVLabels(point.Labels)
		file.WriteString(fmt.Sprintf("%s,query_rate,%.2f,%s\n",
			point.Timestamp.Format(time.RFC3339), point.Value, labels))
	}

	// Write error data
	for errorType, points := range mc.errorRates {
		for _, point := range points {
			labels := formatCSVLabels(point.Labels)
			file.WriteString(fmt.Sprintf("%s,error_count,%.0f,%s,error_type=%s\n",
				point.Timestamp.Format(time.RFC3339), point.Value, labels, string(errorType)))
		}
	}

	return nil
}

// formatCSVLabels formats labels for CSV
func formatCSVLabels(labels map[string]string) string {
	if len(labels) == 0 {
		return ""
	}

	result := ""
	first := true
	for k, v := range labels {
		if !first {
			result += ";"
		}
		result += fmt.Sprintf("%s=%s", k, v)
		first = false
	}
	return result
}

// GetSummary returns a summary of collected metrics
func (mc *MetricsCollector) GetSummary() map[string]interface{} {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	return map[string]interface{}{
		"total_latency_points": len(mc.queryLatencies),
		"total_rate_points":    len(mc.queryRates),
		"error_types":          len(mc.errorRates),
		"collection_duration":  time.Since(mc.startTime).Seconds(),
	}
}
