package errors

import (
	"fmt"
)

// ErrorType represents the category of an error
type ErrorType string

const (
	ErrorTypeValidation ErrorType = "validation"
	ErrorTypeConfig     ErrorType = "config"
	ErrorTypeNetwork    ErrorType = "network"
	ErrorTypeClient     ErrorType = "client"
	ErrorTypeOperation  ErrorType = "operation"
	ErrorTypeTimeout    ErrorType = "timeout"
	ErrorTypeUnknown    ErrorType = "unknown"
)

// AppError represents an application error with context
type AppError struct {
	Type    ErrorType
	Message string
	Err     error
	Context map[string]interface{}
}

// Error implements the error interface
func (e *AppError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("%s: %s: %v", e.Type, e.Message, e.Err)
	}
	return fmt.Sprintf("%s: %s", e.Type, e.Message)
}

// Unwrap returns the underlying error
func (e *AppError) Unwrap() error {
	return e.Err
}

// WithContext adds context to the error
func (e *AppError) WithContext(key string, value interface{}) *AppError {
	if e.Context == nil {
		e.Context = make(map[string]interface{})
	}
	e.Context[key] = value
	return e
}

// NewValidationError creates a new validation error
func NewValidationError(message string, err error) *AppError {
	return &AppError{
		Type:    ErrorTypeValidation,
		Message: message,
		Err:     err,
		Context: make(map[string]interface{}),
	}
}

// NewConfigError creates a new configuration error
func NewConfigError(message string, err error) *AppError {
	return &AppError{
		Type:    ErrorTypeConfig,
		Message: message,
		Err:     err,
		Context: make(map[string]interface{}),
	}
}

// NewNetworkError creates a new network error
func NewNetworkError(message string, err error) *AppError {
	return &AppError{
		Type:    ErrorTypeNetwork,
		Message: message,
		Err:     err,
		Context: make(map[string]interface{}),
	}
}

// NewClientError creates a new client error
func NewClientError(message string, err error) *AppError {
	return &AppError{
		Type:    ErrorTypeClient,
		Message: message,
		Err:     err,
		Context: make(map[string]interface{}),
	}
}

// NewOperationError creates a new operation error
func NewOperationError(message string, err error) *AppError {
	return &AppError{
		Type:    ErrorTypeOperation,
		Message: message,
		Err:     err,
		Context: make(map[string]interface{}),
	}
}

// NewTimeoutError creates a new timeout error
func NewTimeoutError(message string, err error) *AppError {
	return &AppError{
		Type:    ErrorTypeTimeout,
		Message: message,
		Err:     err,
		Context: make(map[string]interface{}),
	}
}

// WrapError wraps an error with additional context
func WrapError(err error, errorType ErrorType, message string) *AppError {
	if appErr, ok := err.(*AppError); ok {
		// Already an AppError, just update message if provided
		if message != "" {
			appErr.Message = message
		}
		return appErr
	}

	return &AppError{
		Type:    errorType,
		Message: message,
		Err:     err,
		Context: make(map[string]interface{}),
	}
}

// IsRetryable checks if an error is retryable
func IsRetryable(err error) bool {
	if err == nil {
		return false
	}
	if appErr, ok := err.(*AppError); ok {
		switch appErr.Type {
		case ErrorTypeNetwork, ErrorTypeTimeout:
			return true
		default:
			return false
		}
	}
	// Check error string for retryable patterns
	errStr := err.Error()
	return containsAny(errStr, []string{"timeout", "connection", "network", "temporary", "retry"})
}

// containsAny checks if a string contains any of the substrings
func containsAny(s string, substrings []string) bool {
	for _, substr := range substrings {
		if len(s) >= len(substr) {
			for i := 0; i <= len(s)-len(substr); i++ {
				if s[i:i+len(substr)] == substr {
					return true
				}
			}
		}
	}
	return false
}
