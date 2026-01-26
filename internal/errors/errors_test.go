package errors

import (
	"errors"
	"testing"
)

func TestAppError_Error(t *testing.T) {
	tests := []struct {
		name    string
		err     *AppError
		want    string
		contains []string
	}{
		{
			name: "error with underlying error",
			err: &AppError{
				Type:    ErrorTypeValidation,
				Message: "invalid input",
				Err:     errors.New("underlying error"),
			},
			contains: []string{"validation", "invalid input", "underlying error"},
		},
		{
			name: "error without underlying error",
			err: &AppError{
				Type:    ErrorTypeNetwork,
				Message: "connection failed",
			},
			contains: []string{"network", "connection failed"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.err.Error()
			for _, substr := range tt.contains {
				if !contains(got, substr) {
					t.Errorf("AppError.Error() = %q, should contain %q", got, substr)
				}
			}
		})
	}
}

func TestAppError_Unwrap(t *testing.T) {
	underlyingErr := errors.New("underlying error")
	err := &AppError{
		Type:    ErrorTypeValidation,
		Message: "test",
		Err:     underlyingErr,
	}

	if err.Unwrap() != underlyingErr {
		t.Errorf("AppError.Unwrap() = %v, want %v", err.Unwrap(), underlyingErr)
	}

	errNoUnderlying := &AppError{
		Type:    ErrorTypeValidation,
		Message: "test",
	}
	if errNoUnderlying.Unwrap() != nil {
		t.Errorf("AppError.Unwrap() = %v, want nil", errNoUnderlying.Unwrap())
	}
}

func TestAppError_WithContext(t *testing.T) {
	err := &AppError{
		Type:    ErrorTypeValidation,
		Message: "test",
	}

	// Add first context
	err = err.WithContext("key1", "value1")
	if err.Context == nil {
		t.Fatal("Context should be initialized")
	}
	if err.Context["key1"] != "value1" {
		t.Errorf("Context[key1] = %v, want value1", err.Context["key1"])
	}

	// Add second context
	err = err.WithContext("key2", 42)
	if err.Context["key2"] != 42 {
		t.Errorf("Context[key2] = %v, want 42", err.Context["key2"])
	}

	// Verify first context still exists
	if err.Context["key1"] != "value1" {
		t.Errorf("Context[key1] = %v, want value1", err.Context["key1"])
	}
}

func TestNewValidationError(t *testing.T) {
	underlyingErr := errors.New("validation failed")
	err := NewValidationError("invalid input", underlyingErr)

	if err.Type != ErrorTypeValidation {
		t.Errorf("NewValidationError().Type = %v, want %v", err.Type, ErrorTypeValidation)
	}
	if err.Message != "invalid input" {
		t.Errorf("NewValidationError().Message = %v, want invalid input", err.Message)
	}
	if err.Err != underlyingErr {
		t.Errorf("NewValidationError().Err = %v, want %v", err.Err, underlyingErr)
	}
	if err.Context == nil {
		t.Error("NewValidationError().Context should be initialized")
	}
}

func TestNewConfigError(t *testing.T) {
	underlyingErr := errors.New("config failed")
	err := NewConfigError("config invalid", underlyingErr)

	if err.Type != ErrorTypeConfig {
		t.Errorf("NewConfigError().Type = %v, want %v", err.Type, ErrorTypeConfig)
	}
	if err.Message != "config invalid" {
		t.Errorf("NewConfigError().Message = %v, want config invalid", err.Message)
	}
}

func TestNewNetworkError(t *testing.T) {
	underlyingErr := errors.New("network failed")
	err := NewNetworkError("connection refused", underlyingErr)

	if err.Type != ErrorTypeNetwork {
		t.Errorf("NewNetworkError().Type = %v, want %v", err.Type, ErrorTypeNetwork)
	}
}

func TestNewClientError(t *testing.T) {
	underlyingErr := errors.New("client failed")
	err := NewClientError("client error", underlyingErr)

	if err.Type != ErrorTypeClient {
		t.Errorf("NewClientError().Type = %v, want %v", err.Type, ErrorTypeClient)
	}
}

func TestNewOperationError(t *testing.T) {
	underlyingErr := errors.New("operation failed")
	err := NewOperationError("operation error", underlyingErr)

	if err.Type != ErrorTypeOperation {
		t.Errorf("NewOperationError().Type = %v, want %v", err.Type, ErrorTypeOperation)
	}
}

func TestNewTimeoutError(t *testing.T) {
	underlyingErr := errors.New("timeout")
	err := NewTimeoutError("operation timed out", underlyingErr)

	if err.Type != ErrorTypeTimeout {
		t.Errorf("NewTimeoutError().Type = %v, want %v", err.Type, ErrorTypeTimeout)
	}
}

func TestWrapError(t *testing.T) {
	t.Run("wrap regular error", func(t *testing.T) {
		underlyingErr := errors.New("underlying error")
		wrapped := WrapError(underlyingErr, ErrorTypeNetwork, "wrapped message")

		if wrapped.Type != ErrorTypeNetwork {
			t.Errorf("WrapError().Type = %v, want %v", wrapped.Type, ErrorTypeNetwork)
		}
		if wrapped.Message != "wrapped message" {
			t.Errorf("WrapError().Message = %v, want wrapped message", wrapped.Message)
		}
		if wrapped.Err != underlyingErr {
			t.Errorf("WrapError().Err = %v, want %v", wrapped.Err, underlyingErr)
		}
	})

	t.Run("wrap AppError with message", func(t *testing.T) {
		appErr := NewValidationError("original message", errors.New("underlying"))
		wrapped := WrapError(appErr, ErrorTypeNetwork, "new message")

		if wrapped != appErr {
			t.Error("WrapError() should return the same AppError when wrapping an AppError")
		}
		if wrapped.Message != "new message" {
			t.Errorf("WrapError().Message = %v, want new message", wrapped.Message)
		}
	})

	t.Run("wrap AppError without message", func(t *testing.T) {
		appErr := NewValidationError("original message", errors.New("underlying"))
		wrapped := WrapError(appErr, ErrorTypeNetwork, "")

		if wrapped != appErr {
			t.Error("WrapError() should return the same AppError when wrapping an AppError")
		}
		if wrapped.Message != "original message" {
			t.Errorf("WrapError().Message = %v, want original message", wrapped.Message)
		}
	})
}

func TestIsRetryable(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"network error", NewNetworkError("network error", errors.New("connection")), true},
		{"timeout error", NewTimeoutError("timeout", errors.New("deadline")), true},
		{"validation error", NewValidationError("validation", errors.New("invalid")), false},
		{"config error", NewConfigError("config", errors.New("invalid")), false},
		{"client error", NewClientError("client", errors.New("error")), false},
		{"operation error", NewOperationError("operation", errors.New("error")), false},
		{"regular timeout error", errors.New("timeout exceeded"), true},
		{"regular connection error", errors.New("connection refused"), true},
		{"regular network error", errors.New("network failure"), true},
		{"regular temporary error", errors.New("temporary failure"), true},
		{"regular retry error", errors.New("retry later"), true},
		{"non-retryable error", errors.New("invalid request"), false},
		{"nil error", nil, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := IsRetryable(tt.err)
			if got != tt.want {
				t.Errorf("IsRetryable() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestContainsAny(t *testing.T) {
	tests := []struct {
		name       string
		s          string
		substrings []string
		want       bool
	}{
		{"contains one", "hello world", []string{"world", "foo"}, true},
		{"contains none", "hello world", []string{"foo", "bar"}, false},
		{"empty string", "", []string{"foo"}, false},
		{"empty substrings", "hello", []string{}, false},
		{"exact match", "test", []string{"test"}, true},
		{"substring at start", "test123", []string{"test"}, true},
		{"substring at end", "123test", []string{"test"}, true},
		{"substring longer than string", "hi", []string{"hello"}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := containsAny(tt.s, tt.substrings)
			if got != tt.want {
				t.Errorf("containsAny() = %v, want %v", got, tt.want)
			}
		})
	}
}

// Helper function for testing
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 || 
		(len(s) > 0 && len(substr) > 0 && 
			(s[:len(substr)] == substr || s[len(s)-len(substr):] == substr ||
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
