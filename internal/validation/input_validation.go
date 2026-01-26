package validation

import (
	"fmt"
	"regexp"
	"strings"
	"time"
)

// ValidateQPSLevels validates QPS levels
func ValidateQPSLevels(qpsLevels []int) error {
	if len(qpsLevels) == 0 {
		return fmt.Errorf("at least one QPS level is required")
	}

	for _, qps := range qpsLevels {
		if qps <= 0 {
			return fmt.Errorf("QPS level must be positive, got %d", qps)
		}
		if qps > 100000 {
			return fmt.Errorf("QPS level %d exceeds maximum recommended value of 100000", qps)
		}
	}

	return nil
}

// ValidateDuration validates a duration
func ValidateDuration(duration time.Duration) error {
	if duration <= 0 {
		return fmt.Errorf("duration must be positive, got %v", duration)
	}
	if duration < 1*time.Second {
		return fmt.Errorf("duration too short: %v (minimum: 1s). Tests shorter than 1 second may not provide accurate results", duration)
	}
	if duration > 24*time.Hour {
		return fmt.Errorf("duration too long: %v (maximum: 24h). Very long tests may cause resource issues", duration)
	}
	return nil
}

// ValidateVectorDimension validates vector dimension
func ValidateVectorDimension(dim int) error {
	if dim <= 0 {
		return fmt.Errorf("vector dimension must be positive, got %d", dim)
	}
	if dim > 32768 {
		return fmt.Errorf("vector dimension %d exceeds maximum recommended value of 32768", dim)
	}
	if dim < 8 {
		return fmt.Errorf("vector dimension %d is unusually small (minimum recommended: 8)", dim)
	}
	return nil
}

// ValidateCollectionName validates collection name format
func ValidateCollectionName(name string) error {
	if name == "" {
		return fmt.Errorf("collection name cannot be empty")
	}

	// Collection names should be alphanumeric with underscores and hyphens
	matched, err := regexp.MatchString(`^[a-zA-Z0-9_-]+$`, name)
	if err != nil {
		return fmt.Errorf("error validating collection name: %w", err)
	}

	if !matched {
		return fmt.Errorf("collection name '%s' contains invalid characters. Use only alphanumeric characters, underscores, and hyphens", name)
	}

	if len(name) > 255 {
		return fmt.Errorf("collection name '%s' exceeds maximum length of 255 characters", name)
	}

	return nil
}

// ValidateAPIKey validates API key format (basic check)
func ValidateAPIKey(apiKey string) error {
	if apiKey == "" {
		return fmt.Errorf("API key cannot be empty")
	}

	if len(apiKey) < 10 {
		return fmt.Errorf("API key appears to be too short (minimum expected length: 10 characters)")
	}

	if len(apiKey) > 1000 {
		return fmt.Errorf("API key appears to be too long (maximum expected length: 1000 characters)")
	}

	return nil
}

// ValidateDatabaseURL validates database URL format
func ValidateDatabaseURL(url string) error {
	if url == "" {
		return fmt.Errorf("database URL cannot be empty")
	}

	urlLower := strings.ToLower(url)
	if !strings.HasPrefix(urlLower, "http://") && !strings.HasPrefix(urlLower, "https://") {
		return fmt.Errorf("database URL must start with http:// or https://, got: %s", url)
	}

	if len(url) > 2048 {
		return fmt.Errorf("database URL exceeds maximum length of 2048 characters")
	}

	return nil
}

// ValidateMetricType validates metric type
func ValidateMetricType(metricType string) error {
	metricTypeUpper := strings.ToUpper(strings.TrimSpace(metricType))
	validTypes := []string{"L2", "IP", "COSINE"}
	for _, validType := range validTypes {
		if metricTypeUpper == validType {
			return nil
		}
	}
	return fmt.Errorf("invalid metric type: %s. Valid types are: %s", metricType, strings.Join(validTypes, ", "))
}

// ValidateConnections validates connection count
func ValidateConnections(connections int) error {
	if connections < 1 {
		return fmt.Errorf("number of connections must be at least 1, got %d", connections)
	}
	if connections > 2000 {
		return fmt.Errorf("number of connections %d exceeds maximum recommended value of 2000", connections)
	}
	return nil
}

// ValidateWarmupQueries validates warmup query count
func ValidateWarmupQueries(warmup int) error {
	if warmup < 0 {
		return fmt.Errorf("warmup queries cannot be negative, got %d", warmup)
	}
	if warmup > 10000 {
		return fmt.Errorf("warmup queries %d exceeds maximum recommended value of 10000", warmup)
	}
	return nil
}

// ValidateSeedParameters validates seed operation parameters
func ValidateSeedParameters(vectorDim, totalVectors, batchSize int) error {
	if err := ValidateVectorDimension(vectorDim); err != nil {
		return fmt.Errorf("invalid vector dimension: %w", err)
	}

	if totalVectors <= 0 {
		return fmt.Errorf("total vectors must be positive, got %d", totalVectors)
	}
	if totalVectors > 100000000 {
		return fmt.Errorf("total vectors %d exceeds maximum recommended value of 100,000,000", totalVectors)
	}

	if batchSize <= 0 {
		return fmt.Errorf("batch size must be positive, got %d", batchSize)
	}
	if batchSize > 50000 {
		return fmt.Errorf("batch size %d exceeds maximum recommended value of 50000 to avoid gRPC message size limits", batchSize)
	}
	if batchSize < 100 {
		return fmt.Errorf("batch size %d is too small (minimum recommended: 100 for efficiency)", batchSize)
	}

	return nil
}
