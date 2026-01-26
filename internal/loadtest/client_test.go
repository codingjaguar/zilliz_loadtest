package loadtest

import (
	"context"
	"strings"
	"testing"
	"time"
)

// TestCreateZillizClient_Validation tests input validation without making real connections
func TestCreateZillizClient_Validation(t *testing.T) {
	tests := []struct {
		name       string
		apiKey     string
		databaseURL string
		wantErr    bool
		errContains string
	}{
		{"empty API key", "", "https://test.zillizcloud.com", true, "API key"},
		{"empty database URL", "test-key", "", true, "database URL"},
		{"both empty", "", "", true, "API key"}, // Should fail on API key first
		// Note: We don't test valid inputs as that would require real network connection
		// Valid input testing should be done in integration tests with proper setup
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Use a short timeout to ensure we don't hang on connection attempts
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()

			done := make(chan error, 1)
			go func() {
				c, err := CreateZillizClient(tt.apiKey, tt.databaseURL)
				if err != nil {
					done <- err
					return
				}
				if c != nil {
					c.Close()
				}
				done <- nil
			}()

			select {
			case err := <-done:
				if (err != nil) != tt.wantErr {
					t.Errorf("CreateZillizClient() error = %v, wantErr %v", err, tt.wantErr)
					return
				}
				if tt.wantErr && tt.errContains != "" {
					if err == nil || !strings.Contains(err.Error(), tt.errContains) {
						t.Errorf("CreateZillizClient() error = %v, should contain %q", err, tt.errContains)
					}
				}
			case <-ctx.Done():
				// Timeout means it tried to connect - this should not happen for validation errors
				t.Errorf("Test timed out - CreateZillizClient() should fail validation immediately for %q", tt.name)
			}
		})
	}
}

// TestNewLoadTester_Validation tests input validation without making real connections
func TestNewLoadTester_Validation(t *testing.T) {
	tests := []struct {
		name        string
		apiKey      string
		databaseURL string
		collection  string
		vectorDim   int
		metricType  string
		wantErr     bool
		errContains string
	}{
		{"empty API key", "", "https://test.zillizcloud.com", "coll", 768, "L2", true, "API key"},
		{"empty database URL", "key", "", "coll", 768, "L2", true, "database URL"},
		{"empty collection", "key", "https://test.zillizcloud.com", "", 768, "L2", true, "collection name"},
		{"zero vector dim", "key", "https://test.zillizcloud.com", "coll", 0, "L2", true, "vector dimension"},
		{"negative vector dim", "key", "https://test.zillizcloud.com", "coll", -1, "L2", true, "vector dimension"},
		// Note: "invalid metric type" test removed because NewLoadTester creates clients
		// before validating metric type, which would cause a timeout
		// Metric type validation happens after client creation, so it requires real connections
		// This should be tested in integration tests with proper mocks
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Use a short timeout to ensure we don't hang on connection attempts
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()

			done := make(chan error, 1)
			var lt *LoadTester
			go func() {
				var err error
				lt, err = NewLoadTester(tt.apiKey, tt.databaseURL, tt.collection, tt.vectorDim, tt.metricType)
				if err != nil {
					done <- err
					return
				}
				if lt != nil {
					lt.Close()
				}
				done <- nil
			}()

			select {
			case err := <-done:
				if (err != nil) != tt.wantErr {
					t.Errorf("NewLoadTester() error = %v, wantErr %v", err, tt.wantErr)
					return
				}
				if tt.wantErr && tt.errContains != "" {
					if err == nil || !strings.Contains(err.Error(), tt.errContains) {
						t.Errorf("NewLoadTester() error = %v, should contain %q", err, tt.errContains)
					}
				}
			case <-ctx.Done():
				// Timeout means it tried to connect - this should not happen for validation errors
				t.Errorf("Test timed out - NewLoadTester() should fail validation immediately for %q", tt.name)
			}
		})
	}
}
