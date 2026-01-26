package metrics

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"zilliz-loadtest/internal/loadtest"
)

func TestNewMetricsCollector(t *testing.T) {
	mc := NewMetricsCollector(true, "/tmp/test")

	if mc == nil {
		t.Fatal("NewMetricsCollector() returned nil")
	}
	if !mc.enabled {
		t.Error("NewMetricsCollector() enabled = false, want true")
	}
	if mc.exportDirectory != "/tmp/test" {
		t.Errorf("NewMetricsCollector() exportDirectory = %v, want /tmp/test", mc.exportDirectory)
	}
	if mc.queryLatencies == nil {
		t.Error("NewMetricsCollector() queryLatencies should be initialized")
	}
	if mc.queryRates == nil {
		t.Error("NewMetricsCollector() queryRates should be initialized")
	}
	if mc.errorRates == nil {
		t.Error("NewMetricsCollector() errorRates should be initialized")
	}
}

func TestRecordLatency(t *testing.T) {
	mc := NewMetricsCollector(true, "")

	labels := map[string]string{"qps": "100", "collection": "test"}
	mc.RecordLatency(100*time.Millisecond, labels)

	mc.mu.RLock()
	defer mc.mu.RUnlock()

	if len(mc.queryLatencies) != 1 {
		t.Errorf("RecordLatency() recorded %d latencies, want 1", len(mc.queryLatencies))
	}

	point := mc.queryLatencies[0]
	if point.Value != 100.0 {
		t.Errorf("RecordLatency() value = %v, want 100.0", point.Value)
	}
	if point.Labels["qps"] != "100" {
		t.Errorf("RecordLatency() labels[qps] = %v, want 100", point.Labels["qps"])
	}
}

func TestRecordLatency_Disabled(t *testing.T) {
	mc := NewMetricsCollector(false, "")

	mc.RecordLatency(100*time.Millisecond, nil)

	mc.mu.RLock()
	defer mc.mu.RUnlock()

	if len(mc.queryLatencies) != 0 {
		t.Errorf("RecordLatency() should not record when disabled, got %d", len(mc.queryLatencies))
	}
}

func TestRecordQueryRate(t *testing.T) {
	mc := NewMetricsCollector(true, "")

	labels := map[string]string{"qps": "500"}
	mc.RecordQueryRate(500.0, labels)

	mc.mu.RLock()
	defer mc.mu.RUnlock()

	if len(mc.queryRates) != 1 {
		t.Errorf("RecordQueryRate() recorded %d rates, want 1", len(mc.queryRates))
	}

	point := mc.queryRates[0]
	if point.Value != 500.0 {
		t.Errorf("RecordQueryRate() value = %v, want 500.0", point.Value)
	}
}

func TestRecordQueryRate_Disabled(t *testing.T) {
	mc := NewMetricsCollector(false, "")

	mc.RecordQueryRate(500.0, nil)

	mc.mu.RLock()
	defer mc.mu.RUnlock()

	if len(mc.queryRates) != 0 {
		t.Errorf("RecordQueryRate() should not record when disabled, got %d", len(mc.queryRates))
	}
}

func TestRecordError(t *testing.T) {
	mc := NewMetricsCollector(true, "")

	labels := map[string]string{"collection": "test"}
	mc.RecordError(loadtest.ErrorTypeNetwork, labels)

	mc.mu.RLock()
	defer mc.mu.RUnlock()

	if len(mc.errorRates[loadtest.ErrorTypeNetwork]) != 1 {
		t.Errorf("RecordError() recorded %d errors, want 1", len(mc.errorRates[loadtest.ErrorTypeNetwork]))
	}

	point := mc.errorRates[loadtest.ErrorTypeNetwork][0]
	if point.Value != 1.0 {
		t.Errorf("RecordError() value = %v, want 1.0", point.Value)
	}
}

func TestRecordError_Disabled(t *testing.T) {
	mc := NewMetricsCollector(false, "")

	mc.RecordError(loadtest.ErrorTypeTimeout, nil)

	mc.mu.RLock()
	defer mc.mu.RUnlock()

	if len(mc.errorRates) != 0 {
		t.Errorf("RecordError() should not record when disabled, got %d error types", len(mc.errorRates))
	}
}

func TestExportPrometheus(t *testing.T) {
	tmpDir := t.TempDir()
	mc := NewMetricsCollector(true, tmpDir)

	// Record some metrics
	mc.RecordLatency(100*time.Millisecond, map[string]string{"qps": "100"})
	mc.RecordQueryRate(100.0, map[string]string{"qps": "100"})
	mc.RecordError(loadtest.ErrorTypeNetwork, map[string]string{"qps": "100"})

	outputPath := filepath.Join(tmpDir, "test.prom")
	err := mc.ExportPrometheus(outputPath)
	if err != nil {
		t.Fatalf("ExportPrometheus() error = %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(outputPath); os.IsNotExist(err) {
		t.Errorf("ExportPrometheus() file does not exist: %s", outputPath)
	}

	// Read and verify content
	content, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("Failed to read exported file: %v", err)
	}

	contentStr := string(content)
	if !contains(contentStr, "query_latency_ms") {
		t.Error("ExportPrometheus() should contain query_latency_ms")
	}
	if !contains(contentStr, "query_rate_queries_per_second") {
		t.Error("ExportPrometheus() should contain query_rate_queries_per_second")
	}
	if !contains(contentStr, "error_count_total") {
		t.Error("ExportPrometheus() should contain error_count_total")
	}
}

func TestExportPrometheus_Disabled(t *testing.T) {
	mc := NewMetricsCollector(false, "")

	err := mc.ExportPrometheus("")
	if err != nil {
		t.Errorf("ExportPrometheus() when disabled should return nil, got %v", err)
	}
}

func TestExportPrometheus_DefaultPath(t *testing.T) {
	tmpDir := t.TempDir()
	mc := NewMetricsCollector(true, tmpDir)

	mc.RecordLatency(100*time.Millisecond, nil)

	err := mc.ExportPrometheus("")
	if err != nil {
		t.Fatalf("ExportPrometheus() error = %v", err)
	}

	// Check that a file was created with timestamp pattern
	files, err := os.ReadDir(tmpDir)
	if err != nil {
		t.Fatalf("Failed to read directory: %v", err)
	}

	found := false
	for _, file := range files {
		if contains(file.Name(), "metrics_") && contains(file.Name(), ".prom") {
			found = true
			break
		}
	}
	if !found {
		t.Error("ExportPrometheus() should create file with timestamp pattern")
	}
}

func TestExportTimeSeries(t *testing.T) {
	tmpDir := t.TempDir()
	mc := NewMetricsCollector(true, tmpDir)

	// Record some metrics
	mc.RecordLatency(100*time.Millisecond, map[string]string{"qps": "100"})
	mc.RecordQueryRate(100.0, map[string]string{"qps": "100"})
	mc.RecordError(loadtest.ErrorTypeNetwork, map[string]string{"qps": "100"})

	outputPath := filepath.Join(tmpDir, "test.csv")
	err := mc.ExportTimeSeries(outputPath)
	if err != nil {
		t.Fatalf("ExportTimeSeries() error = %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(outputPath); os.IsNotExist(err) {
		t.Errorf("ExportTimeSeries() file does not exist: %s", outputPath)
	}

	// Read and verify content
	content, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("Failed to read exported file: %v", err)
	}

	contentStr := string(content)
	if !contains(contentStr, "timestamp,metric,value,labels") {
		t.Error("ExportTimeSeries() should contain CSV header")
	}
	if !contains(contentStr, "latency_ms") {
		t.Error("ExportTimeSeries() should contain latency_ms")
	}
	if !contains(contentStr, "query_rate") {
		t.Error("ExportTimeSeries() should contain query_rate")
	}
	if !contains(contentStr, "error_count") {
		t.Error("ExportTimeSeries() should contain error_count")
	}
}

func TestExportTimeSeries_Disabled(t *testing.T) {
	mc := NewMetricsCollector(false, "")

	err := mc.ExportTimeSeries("")
	if err != nil {
		t.Errorf("ExportTimeSeries() when disabled should return nil, got %v", err)
	}
}

func TestGetSummary(t *testing.T) {
	mc := NewMetricsCollector(true, "")

	// Record some metrics
	mc.RecordLatency(100*time.Millisecond, nil)
	mc.RecordLatency(200*time.Millisecond, nil)
	mc.RecordQueryRate(100.0, nil)
	mc.RecordError(loadtest.ErrorTypeNetwork, nil)
	mc.RecordError(loadtest.ErrorTypeTimeout, nil)

	summary := mc.GetSummary()

	if summary["total_latency_points"] != 2 {
		t.Errorf("GetSummary() total_latency_points = %v, want 2", summary["total_latency_points"])
	}
	if summary["total_rate_points"] != 1 {
		t.Errorf("GetSummary() total_rate_points = %v, want 1", summary["total_rate_points"])
	}
	if summary["error_types"] != 2 {
		t.Errorf("GetSummary() error_types = %v, want 2", summary["error_types"])
	}
	if summary["collection_duration"] == nil {
		t.Error("GetSummary() collection_duration should be set")
	}
}

func TestFormatPrometheusLabels(t *testing.T) {
	tests := []struct {
		name   string
		labels map[string]string
		want   string
	}{
		{"empty labels", map[string]string{}, ""},
		{"single label", map[string]string{"qps": "100"}, `{qps="100"}`},
		{"multiple labels", map[string]string{"qps": "100", "collection": "test"}, `{qps="100",collection="test"}`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := formatPrometheusLabels(tt.labels)
			// Order might vary, so just check it contains the expected parts
			if tt.name == "empty labels" && got != "" {
				t.Errorf("formatPrometheusLabels() = %q, want empty string", got)
			} else if tt.name != "empty labels" {
				if !contains(got, "{") || !contains(got, "}") {
					t.Errorf("formatPrometheusLabels() = %q, should contain braces", got)
				}
				for k, v := range tt.labels {
					if !contains(got, k) || !contains(got, v) {
						t.Errorf("formatPrometheusLabels() = %q, should contain %q=%q", got, k, v)
					}
				}
			}
		})
	}
}

func TestFormatCSVLabels(t *testing.T) {
	tests := []struct {
		name   string
		labels map[string]string
		want   string
	}{
		{"empty labels", map[string]string{}, ""},
		{"single label", map[string]string{"qps": "100"}, `qps=100`},
		{"multiple labels", map[string]string{"qps": "100", "collection": "test"}, `qps=100;collection=test`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := formatCSVLabels(tt.labels)
			if tt.name == "empty labels" && got != "" {
				t.Errorf("formatCSVLabels() = %q, want empty string", got)
			} else if tt.name != "empty labels" {
				for k, v := range tt.labels {
					if !contains(got, k+"="+v) {
						t.Errorf("formatCSVLabels() = %q, should contain %q=%q", got, k, v)
					}
				}
			}
		})
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && 
		(s == substr || 
			(len(s) > len(substr) && 
				(s[:len(substr)] == substr || 
					s[len(s)-len(substr):] == substr ||
					containsSubstring(s, substr))))
}

func containsSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
