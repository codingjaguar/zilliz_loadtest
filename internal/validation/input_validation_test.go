package validation

import (
	"testing"
	"time"
)

func TestValidateQPSLevels(t *testing.T) {
	tests := []struct {
		name    string
		qps     []int
		wantErr bool
	}{
		{"valid single QPS", []int{100}, false},
		{"valid multiple QPS", []int{100, 500, 1000}, false},
		{"empty QPS", []int{}, true},
		{"zero QPS", []int{0}, true},
		{"negative QPS", []int{-1}, true},
		{"too high QPS", []int{200000}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateQPSLevels(tt.qps)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateQPSLevels() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateDuration(t *testing.T) {
	tests := []struct {
		name     string
		duration time.Duration
		wantErr  bool
	}{
		{"valid duration", 30 * time.Second, false},
		{"too short", 500 * time.Millisecond, true},
		{"too long", 25 * time.Hour, true},
		{"zero duration", 0, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateDuration(tt.duration)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateDuration() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateVectorDimension(t *testing.T) {
	tests := []struct {
		name    string
		dim     int
		wantErr bool
	}{
		{"valid dimension", 768, false},
		{"too small", 4, true},
		{"too large", 50000, true},
		{"zero", 0, true},
		{"negative", -1, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateVectorDimension(tt.dim)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateVectorDimension() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateCollectionName(t *testing.T) {
	tests := []struct {
		name    string
		colName string
		wantErr bool
	}{
		{"valid name", "my_collection", false},
		{"valid with hyphen", "my-collection", false},
		{"valid alphanumeric", "collection123", false},
		{"empty", "", true},
		{"invalid chars", "collection name", true},
		{"too long", string(make([]byte, 300)), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateCollectionName(tt.colName)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateCollectionName() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateAPIKey(t *testing.T) {
	tests := []struct {
		name    string
		apiKey  string
		wantErr bool
	}{
		{"valid key", "valid-api-key-12345", false},
		{"empty", "", true},
		{"too short", "short", true},
		{"too long", string(make([]byte, 2000)), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateAPIKey(tt.apiKey)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateAPIKey() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateDatabaseURL(t *testing.T) {
	tests := []struct {
		name    string
		url     string
		wantErr bool
	}{
		{"valid https", "https://cluster.zillizcloud.com", false},
		{"valid http", "http://cluster.zillizcloud.com", false},
		{"empty", "", true},
		{"no protocol", "cluster.zillizcloud.com", true},
		{"too long", "https://" + string(make([]byte, 3000)), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateDatabaseURL(tt.url)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateDatabaseURL() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateMetricType(t *testing.T) {
	tests := []struct {
		name    string
		metric  string
		wantErr bool
	}{
		{"valid L2", "L2", false},
		{"valid IP", "IP", false},
		{"valid COSINE", "COSINE", false},
		{"lowercase l2", "l2", false},
		{"invalid", "INVALID", true},
		{"empty", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateMetricType(tt.metric)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateMetricType() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateConnections(t *testing.T) {
	tests := []struct {
		name    string
		conns   int
		wantErr bool
	}{
		{"valid", 10, false},
		{"minimum", 1, false},
		{"maximum", 2000, false},
		{"zero", 0, true},
		{"too many", 3000, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateConnections(tt.conns)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateConnections() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateSeedParameters(t *testing.T) {
	tests := []struct {
		name      string
		vectorDim int
		totalVecs int
		batchSize int
		wantErr   bool
	}{
		{"valid", 768, 2000000, 15000, false},
		{"invalid dim", 0, 2000000, 15000, true},
		{"invalid total", 768, 0, 15000, true},
		{"invalid batch", 768, 2000000, 0, true},
		{"batch too large", 768, 2000000, 60000, true},
		{"batch too small", 768, 2000000, 50, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateSeedParameters(tt.vectorDim, tt.totalVecs, tt.batchSize)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateSeedParameters() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
