package loadtest

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
)

// RetryConfig holds retry configuration
type RetryConfig struct {
	MaxRetries        int
	InitialDelay      time.Duration
	MaxDelay          time.Duration
	BackoffMultiplier float64
}

// DefaultRetryConfig returns default retry configuration
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:        3,
		InitialDelay:      100 * time.Millisecond,
		MaxDelay:          5 * time.Second,
		BackoffMultiplier: 2.0,
	}
}

// RetryableOperation represents an operation that can be retried
type RetryableOperation func() error

// RetryWithBackoff executes an operation with exponential backoff retry
func RetryWithBackoff(ctx context.Context, operation RetryableOperation, config RetryConfig) error {
	var lastErr error
	delay := config.InitialDelay

	for attempt := 0; attempt <= config.MaxRetries; attempt++ {
		// Check context cancellation
		select {
		case <-ctx.Done():
			return fmt.Errorf("operation cancelled: %w", ctx.Err())
		default:
		}

		// Execute operation
		err := operation()
		if err == nil {
			return nil // Success
		}

		lastErr = err

		// Don't retry on last attempt
		if attempt == config.MaxRetries {
			break
		}

		// Check if error is retryable
		if !isRetryableError(err) {
			return fmt.Errorf("non-retryable error: %w", err)
		}

		// Wait before retry
		select {
		case <-ctx.Done():
			return fmt.Errorf("operation cancelled during retry: %w", ctx.Err())
		case <-time.After(delay):
		}

		// Exponential backoff
		delay = time.Duration(float64(delay) * config.BackoffMultiplier)
		if delay > config.MaxDelay {
			delay = config.MaxDelay
		}
	}

	return fmt.Errorf("operation failed after %d attempts: %w", config.MaxRetries+1, lastErr)
}

// isRetryableError checks if an error is retryable
func isRetryableError(err error) bool {
	if err == nil {
		return false
	}

	errStr := err.Error()
	errStrLower := errStr

	// Check for retryable error patterns
	retryablePatterns := []string{
		"timeout",
		"deadline exceeded",
		"connection",
		"network",
		"temporary",
		"retry",
		"unavailable",
		"busy",
		"rate limit", // Some rate limits are temporary
	}

	for _, pattern := range retryablePatterns {
		if contains(errStrLower, pattern) {
			return true
		}
	}

	return false
}

// contains checks if a string contains a substring (case-insensitive)
func contains(s, substr string) bool {
	return strings.Contains(strings.ToLower(s), strings.ToLower(substr))
}

// RetryClientCreation retries client creation with backoff
func RetryClientCreation(ctx context.Context, apiKey, databaseURL string, retryConfig RetryConfig) (client.Client, error) {
	var lastErr error

	delay := retryConfig.InitialDelay
	for attempt := 0; attempt <= retryConfig.MaxRetries; attempt++ {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("client creation cancelled: %w", ctx.Err())
		default:
		}

		c, err := CreateZillizClient(apiKey, databaseURL)
		if err == nil {
			return c, nil
		}

		lastErr = err

		if attempt == retryConfig.MaxRetries {
			break
		}

		if !isRetryableError(err) {
			return nil, fmt.Errorf("non-retryable error creating client: %w", err)
		}

		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("client creation cancelled during retry: %w", ctx.Err())
		case <-time.After(delay):
		}

		delay = time.Duration(float64(delay) * retryConfig.BackoffMultiplier)
		if delay > retryConfig.MaxDelay {
			delay = retryConfig.MaxDelay
		}
	}

	return nil, fmt.Errorf("failed to create client after %d attempts: %w", retryConfig.MaxRetries+1, lastErr)
}
