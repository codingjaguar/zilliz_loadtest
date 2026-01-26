package loadtest

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestIsRetryableError(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"timeout error", errors.New("timeout exceeded"), true},
		{"network error", errors.New("connection refused"), true},
		{"temporary error", errors.New("temporary failure"), true},
		{"non-retryable", errors.New("invalid request"), false},
		{"nil error", nil, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isRetryableError(tt.err)
			if got != tt.want {
				t.Errorf("isRetryableError() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRetryWithBackoff(t *testing.T) {
	ctx := context.Background()
	config := DefaultRetryConfig()
	config.MaxRetries = 2
	config.InitialDelay = 10 * time.Millisecond

	t.Run("success on first try", func(t *testing.T) {
		attempts := 0
		err := RetryWithBackoff(ctx, func() error {
			attempts++
			return nil
		}, config)

		if err != nil {
			t.Errorf("RetryWithBackoff() error = %v, want nil", err)
		}
		if attempts != 1 {
			t.Errorf("Expected 1 attempt, got %d", attempts)
		}
	})

	t.Run("success after retry", func(t *testing.T) {
		attempts := 0
		err := RetryWithBackoff(ctx, func() error {
			attempts++
			if attempts < 2 {
				return errors.New("timeout")
			}
			return nil
		}, config)

		if err != nil {
			t.Errorf("RetryWithBackoff() error = %v, want nil", err)
		}
		if attempts != 2 {
			t.Errorf("Expected 2 attempts, got %d", attempts)
		}
	})

	t.Run("non-retryable error", func(t *testing.T) {
		attempts := 0
		err := RetryWithBackoff(ctx, func() error {
			attempts++
			return errors.New("invalid request")
		}, config)

		if err == nil {
			t.Error("RetryWithBackoff() expected error, got nil")
		}
		if attempts != 1 {
			t.Errorf("Expected 1 attempt for non-retryable error, got %d", attempts)
		}
	})

	t.Run("context cancellation", func(t *testing.T) {
		cancelCtx, cancel := context.WithCancel(ctx)
		cancel()

		err := RetryWithBackoff(cancelCtx, func() error {
			return errors.New("timeout")
		}, config)

		if err == nil {
			t.Error("RetryWithBackoff() expected error on cancellation, got nil")
		}
	})
}
