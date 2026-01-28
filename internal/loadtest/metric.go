package loadtest

import (
	"fmt"
	"strings"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// ParseMetricType parses a metric type string (L2, IP, COSINE) into entity.MetricType.
// Returns error for empty or unsupported values.
func ParseMetricType(s string) (entity.MetricType, error) {
	s = strings.ToUpper(strings.TrimSpace(s))
	switch s {
	case "L2":
		return entity.L2, nil
	case "IP":
		return entity.IP, nil
	case "COSINE":
		return entity.COSINE, nil
	case "":
		return entity.L2, fmt.Errorf("metric type is empty")
	default:
		return entity.L2, fmt.Errorf("unsupported metric type: %s. Supported types are: L2, IP, COSINE", s)
	}
}

// DefaultMetricType returns the default metric type (L2).
func DefaultMetricType() entity.MetricType {
	return entity.L2
}
