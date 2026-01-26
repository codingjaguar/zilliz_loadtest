package config

import (
	"os"
	"testing"
	"time"
)

func TestConfigDefaults(t *testing.T) {
	config := &Config{}

	// Test default duration parsing
	duration, err := config.GetDuration()
	if err == nil && duration != 30*time.Second {
		t.Errorf("GetDuration() default = %v, want 30s", duration)
	}

	// Test with empty duration string
	config.DefaultDuration = ""
	duration, err = config.GetDuration()
	if err != nil {
		t.Errorf("GetDuration() with empty string returned error: %v", err)
	}
	if duration != 30*time.Second {
		t.Errorf("GetDuration() with empty string = %v, want 30s", duration)
	}
}

func TestConfigDurationParsing(t *testing.T) {
	tests := []struct {
		name     string
		duration string
		want     time.Duration
	}{
		{
			name:     "30 seconds",
			duration: "30s",
			want:     30 * time.Second,
		},
		{
			name:     "1 minute",
			duration: "1m",
			want:     1 * time.Minute,
		},
		{
			name:     "5 minutes",
			duration: "5m",
			want:     5 * time.Minute,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &Config{DefaultDuration: tt.duration}
			got, err := config.GetDuration()
			if err != nil {
				t.Errorf("GetDuration() error = %v", err)
				return
			}
			if got != tt.want {
				t.Errorf("GetDuration() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetConfigValue(t *testing.T) {
	tests := []struct {
		name         string
		value        string
		defaultValue string
		want         string
	}{
		{
			name:         "value provided",
			value:        "test-value",
			defaultValue: "default",
			want:         "test-value",
		},
		{
			name:         "empty value uses default",
			value:        "",
			defaultValue: "default",
			want:         "default",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetConfigValue(tt.value, tt.defaultValue)
			if got != tt.want {
				t.Errorf("GetConfigValue() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestLoadConfigWithEnvVars(t *testing.T) {
	// Save original env vars
	originalAPIKey := os.Getenv("ZILLIZ_API_KEY")
	originalDBURL := os.Getenv("ZILLIZ_DB_URL")

	// Set test env vars
	os.Setenv("ZILLIZ_API_KEY", "test-api-key")
	os.Setenv("ZILLIZ_DB_URL", "https://test.zillizcloud.com")
	defer func() {
		// Restore original env vars
		if originalAPIKey != "" {
			os.Setenv("ZILLIZ_API_KEY", originalAPIKey)
		} else {
			os.Unsetenv("ZILLIZ_API_KEY")
		}
		if originalDBURL != "" {
			os.Setenv("ZILLIZ_DB_URL", originalDBURL)
		} else {
			os.Unsetenv("ZILLIZ_DB_URL")
		}
	}()

	config, err := LoadConfig("")
	if err != nil {
		t.Fatalf("LoadConfig() error = %v", err)
	}

	if config.APIKey != "test-api-key" {
		t.Errorf("LoadConfig() APIKey = %v, want test-api-key", config.APIKey)
	}

	if config.DatabaseURL != "https://test.zillizcloud.com" {
		t.Errorf("LoadConfig() DatabaseURL = %v, want https://test.zillizcloud.com", config.DatabaseURL)
	}
}
