package config

import (
	"fmt"
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

// Config holds the configuration for the load test tool
type Config struct {
	APIKey               string   `yaml:"api_key"`
	DatabaseURL          string   `yaml:"database_url"`
	DefaultCollection    string   `yaml:"default_collection"`
	DefaultVectorDim     int      `yaml:"default_vector_dim"`
	DefaultMetricType    string   `yaml:"default_metric_type"`
	DefaultDuration      string   `yaml:"default_duration"`
	DefaultQPSLevels     []int    `yaml:"default_qps_levels"`
	ConnectionMultiplier float64  `yaml:"connection_multiplier"`
	ExpectedLatencyMs    float64  `yaml:"expected_latency_ms"`
	WarmupQueries        int      `yaml:"warmup_queries"`
	TopK                 int      `yaml:"top_k"`
	SearchLevel          int      `yaml:"search_level"`
	FilterExpression     string   `yaml:"filter_expression"`
	OutputFields         []string `yaml:"output_fields"`

	// Seed parameters
	SeedVectorCount int `yaml:"seed_vector_count"`
	SeedVectorDim   int `yaml:"seed_vector_dim"`
	SeedBatchSize   int `yaml:"seed_batch_size"`

	// Logging and observability
	LogLevel        string `yaml:"log_level"`
	LogFormat       string `yaml:"log_format"`
	MetricsEnabled  bool   `yaml:"metrics_enabled"`
	MetricsPort     int    `yaml:"metrics_port"`
	OutputDirectory string `yaml:"output_directory"`
	MaxRetries      int    `yaml:"max_retries"`
	Timeout         string `yaml:"timeout"`
}

// LoadConfig loads configuration from file, environment variables, or returns defaults
func LoadConfig(configPath string) (*Config, error) {
	config := &Config{
		DefaultVectorDim:     768,
		DefaultMetricType:    "L2",
		DefaultDuration:      "30s",
		DefaultQPSLevels:     []int{100, 500, 1000},
		ConnectionMultiplier: 1.5,
		ExpectedLatencyMs:    75.0,
		WarmupQueries:        100,
		TopK:                 10,
		SearchLevel:          1,
		OutputFields:         []string{"id"},
	}

	// Set seed defaults
	config.SeedVectorCount = 2000000
	config.SeedVectorDim = 768
	config.SeedBatchSize = 15000

	// Set logging defaults
	config.LogLevel = "INFO"
	config.LogFormat = "text"
	config.MetricsEnabled = false
	config.MetricsPort = 9090
	config.OutputDirectory = ""
	config.MaxRetries = 3
	config.Timeout = "30s"

	// Try to load from config file
	if configPath == "" {
		// Only look in configs/config.yaml
		defaultPath := "./configs/config.yaml"
		if _, err := os.Stat(defaultPath); err == nil {
			configPath = defaultPath
		}
	}

	if configPath != "" {
		data, err := os.ReadFile(configPath)
		if err == nil {
			if err := yaml.Unmarshal(data, config); err != nil {
				return nil, fmt.Errorf("failed to parse config file: %w", err)
			}
		}
	}

	// Override with environment variables if set
	if apiKey := os.Getenv("ZILLIZ_API_KEY"); apiKey != "" {
		config.APIKey = apiKey
	}
	if dbURL := os.Getenv("ZILLIZ_DB_URL"); dbURL != "" {
		config.DatabaseURL = dbURL
	}

	return config, nil
}

// GetDuration parses the duration string from config
func (c *Config) GetDuration() (time.Duration, error) {
	if c.DefaultDuration == "" {
		return 30 * time.Second, nil
	}
	return time.ParseDuration(c.DefaultDuration)
}

// SaveConfig saves the configuration to a file
func SaveConfig(config *Config, path string) error {
	data, err := yaml.Marshal(config)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	return os.WriteFile(path, data, 0600)
}

// GetConfigValue returns a config value with fallback to default
func GetConfigValue(value string, defaultValue string) string {
	if value != "" {
		return value
	}
	return defaultValue
}
