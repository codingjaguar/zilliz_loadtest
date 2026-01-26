package loadtest

import (
	"context"

	"zilliz-loadtest/internal/logger"
)

// runWarmup executes warmup queries to prepare connections and caches
func (lt *LoadTester) runWarmup(ctx context.Context, warmupQueries int) {
	if warmupQueries <= 0 {
		return
	}

	logger.Info("Starting warmup", "queries", warmupQueries)
	warmupCtx, warmupCancel := context.WithTimeout(ctx, WarmupTimeout)
	defer warmupCancel()

	for i := 0; i < warmupQueries; i++ {
		result := lt.executeQuery(warmupCtx, i)
		if result.Error != nil && i < 5 {
			logger.Warn("Warmup query error",
				"query_num", i+1,
				"error", result.Error.Error())
		}
		if (i+1)%10 == 0 {
			logger.Debug("Warmup progress",
				"completed", i+1,
				"total", warmupQueries)
		}
	}
	logger.Info("Warmup complete")
}
