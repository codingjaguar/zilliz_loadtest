package loadtest

import "time"

// Constants for load testing configuration
const (
	// Latency thresholds
	HighLatencyThresholdMs = 1000.0 // Latencies above 1 second are considered high (likely queuing)

	// Timeouts
	WarmupTimeout        = 60 * time.Second
	StatusUpdateInterval = 5 * time.Second
	ProgressUpdateInterval = 2 * time.Second
	MinWaitTimeout       = 30 * time.Second
	MaxWaitTimeout       = 5 * time.Minute
	ResultChannelTimeout = 1 * time.Second
	GracePeriodAfterClose = 2 * time.Second
	ResultCollectorDelay = 100 * time.Millisecond

	// Connection calculation defaults
	ExpectedLatencyMs = 75.0
	ConnectionMultiplier = 1.5
	MinConnections = 5
	MaxConnections = 2000

	// SeedDatabase defaults
	DefaultBatchSize = 15000
	ProgressUpdateIntervalVectors = 5000

	// QPS warning threshold (80% of target)
	QPSWarningThreshold = 0.8
)
