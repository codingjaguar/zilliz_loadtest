package loadtest

import (
	"context"
	"fmt"
)

// runWarmup executes warmup queries to prepare connections and caches
func (lt *LoadTester) runWarmup(ctx context.Context, warmupQueries int) {
	if warmupQueries <= 0 {
		return
	}

	fmt.Printf("Warming up with %d queries...\n", warmupQueries)
	warmupCtx, warmupCancel := context.WithTimeout(ctx, WarmupTimeout)
	defer warmupCancel()

	for i := 0; i < warmupQueries; i++ {
		result := lt.executeQuery(warmupCtx, i)
		if result.Error != nil && i < 5 {
			fmt.Printf("  Warmup query %d error: %v\n", i+1, result.Error)
		}
		if (i+1)%10 == 0 {
			fmt.Printf("  Warmup progress: %d/%d queries\n", i+1, warmupQueries)
		}
	}
	fmt.Printf("Warmup complete\n\n")
}
