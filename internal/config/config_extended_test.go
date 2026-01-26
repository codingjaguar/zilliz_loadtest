package config

import (
	"os"
	"testing"
)

func TestSaveConfig(t *testing.T) {
	config := &Config{
		APIKey:            "test-key",
		DatabaseURL:       "https://test.zillizcloud.com",
		DefaultCollection: "test_collection",
		DefaultVectorDim:  768,
		DefaultMetricType: "L2",
		DefaultDuration:   "30s",
		DefaultQPSLevels:  []int{100, 500, 1000},
	}

	tmpFile := "test_config_save.yaml"
	defer os.Remove(tmpFile)

	err := SaveConfig(config, tmpFile)
	if err != nil {
		t.Fatalf("SaveConfig() error = %v", err)
	}

	// Verify file was created
	if _, err := os.Stat(tmpFile); os.IsNotExist(err) {
		t.Errorf("SaveConfig() did not create file %s", tmpFile)
	}

	// Try to load it back
	loadedConfig, err := LoadConfig(tmpFile)
	if err != nil {
		t.Fatalf("LoadConfig() after SaveConfig error = %v", err)
	}

	if loadedConfig.APIKey != config.APIKey {
		t.Errorf("Loaded config APIKey = %v, want %v", loadedConfig.APIKey, config.APIKey)
	}

	if loadedConfig.DefaultCollection != config.DefaultCollection {
		t.Errorf("Loaded config DefaultCollection = %v, want %v", loadedConfig.DefaultCollection, config.DefaultCollection)
	}
}

func TestLoadConfigWithFile(t *testing.T) {
	// Create a test config file
	testConfig := `api_key: "test-api-key-from-file"
database_url: "https://test-file.zillizcloud.com"
default_collection: "test_collection_file"
default_vector_dim: 512
default_metric_type: "COSINE"
default_duration: "1m"
default_qps_levels: [200, 1000]
connection_multiplier: 2.0
expected_latency_ms: 100.0
warmup_queries: 200
top_k: 20
search_level: 2
filter_expression: "id > 1000"
output_fields: ["id", "score"]
`

	tmpFile := "test_config_load.yaml"
	err := os.WriteFile(tmpFile, []byte(testConfig), 0644)
	if err != nil {
		t.Fatalf("Failed to create test config file: %v", err)
	}
	defer os.Remove(tmpFile)

	config, err := LoadConfig(tmpFile)
	if err != nil {
		t.Fatalf("LoadConfig() error = %v", err)
	}

	if config.APIKey != "test-api-key-from-file" {
		t.Errorf("LoadConfig() APIKey = %v, want test-api-key-from-file", config.APIKey)
	}

	if config.DatabaseURL != "https://test-file.zillizcloud.com" {
		t.Errorf("LoadConfig() DatabaseURL = %v, want https://test-file.zillizcloud.com", config.DatabaseURL)
	}

	if config.DefaultCollection != "test_collection_file" {
		t.Errorf("LoadConfig() DefaultCollection = %v, want test_collection_file", config.DefaultCollection)
	}

	if config.DefaultVectorDim != 512 {
		t.Errorf("LoadConfig() DefaultVectorDim = %v, want 512", config.DefaultVectorDim)
	}

	if config.DefaultMetricType != "COSINE" {
		t.Errorf("LoadConfig() DefaultMetricType = %v, want COSINE", config.DefaultMetricType)
	}

	if len(config.DefaultQPSLevels) != 2 || config.DefaultQPSLevels[0] != 200 || config.DefaultQPSLevels[1] != 1000 {
		t.Errorf("LoadConfig() DefaultQPSLevels = %v, want [200, 1000]", config.DefaultQPSLevels)
	}

	if config.ConnectionMultiplier != 2.0 {
		t.Errorf("LoadConfig() ConnectionMultiplier = %v, want 2.0", config.ConnectionMultiplier)
	}

	if config.TopK != 20 {
		t.Errorf("LoadConfig() TopK = %v, want 20", config.TopK)
	}

	if config.SearchLevel != 2 {
		t.Errorf("LoadConfig() SearchLevel = %v, want 2", config.SearchLevel)
	}

	if config.FilterExpression != "id > 1000" {
		t.Errorf("LoadConfig() FilterExpression = %v, want 'id > 1000'", config.FilterExpression)
	}

	if len(config.OutputFields) != 2 || config.OutputFields[0] != "id" || config.OutputFields[1] != "score" {
		t.Errorf("LoadConfig() OutputFields = %v, want [id, score]", config.OutputFields)
	}
}

func TestLoadConfigDefaults(t *testing.T) {
	// Load config with non-existent file (should use defaults)
	config, err := LoadConfig("non_existent_file_12345.yaml")
	if err != nil {
		t.Fatalf("LoadConfig() with non-existent file should not error, got: %v", err)
	}

	// Check defaults
	if config.DefaultVectorDim != 768 {
		t.Errorf("Default DefaultVectorDim = %v, want 768", config.DefaultVectorDim)
	}

	if config.DefaultMetricType != "L2" {
		t.Errorf("Default DefaultMetricType = %v, want L2", config.DefaultMetricType)
	}

	if config.DefaultDuration != "30s" {
		t.Errorf("Default DefaultDuration = %v, want 30s", config.DefaultDuration)
	}

	if len(config.DefaultQPSLevels) != 3 {
		t.Errorf("Default DefaultQPSLevels length = %v, want 3", len(config.DefaultQPSLevels))
	}

	if config.ConnectionMultiplier != 1.5 {
		t.Errorf("Default ConnectionMultiplier = %v, want 1.5", config.ConnectionMultiplier)
	}

	if config.WarmupQueries != 100 {
		t.Errorf("Default WarmupQueries = %v, want 100", config.WarmupQueries)
	}

	if config.TopK != 10 {
		t.Errorf("Default TopK = %v, want 10", config.TopK)
	}

	if config.SearchLevel != 1 {
		t.Errorf("Default SearchLevel = %v, want 1", config.SearchLevel)
	}
}
