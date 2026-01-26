package loadtest

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// testState tracks the state of a running test
type testState struct {
	resultsChan      chan QueryResult
	allResults       []QueryResult
	resultsMu        sync.Mutex
	queriesCompleted int64
	queriesFired     int64
	queryCount       int64
	startTime        time.Time
}

// startResultCollector starts a goroutine to collect results from the channel
func (ts *testState) startResultCollector() {
	go func() {
		for result := range ts.resultsChan {
			ts.resultsMu.Lock()
			ts.allResults = append(ts.allResults, result)
			atomic.AddInt64(&ts.queriesCompleted, 1)
			ts.resultsMu.Unlock()
		}
	}()
}

// startStatusReporter starts a goroutine to report status updates
func (ts *testState) startStatusReporter(testCtx context.Context) {
	statusTicker := time.NewTicker(StatusUpdateInterval)
	defer statusTicker.Stop()

	go func() {
		for {
			select {
			case <-testCtx.Done():
				return
			case <-statusTicker.C:
				fired := atomic.LoadInt64(&ts.queriesFired)
				completed := atomic.LoadInt64(&ts.queriesCompleted)
				elapsed := time.Since(ts.startTime)
				if elapsed.Seconds() > 0 {
					currentQPS := float64(completed) / elapsed.Seconds()
					fmt.Printf("[Status] Elapsed: %v | Fired: %d | Completed: %d | Current QPS: %.2f\n",
						elapsed.Round(time.Second), fired, completed, currentQPS)
				}
			}
		}
	}()
}

// startQueryFirer starts a goroutine that fires queries at the target QPS
func (lt *LoadTester) startQueryFirer(testCtx, queryCtx context.Context, interval time.Duration, ts *testState, wg *sync.WaitGroup) {
	ticker := time.NewTicker(interval)
	
	go func() {
		defer ticker.Stop()
		for {
			select {
			case <-testCtx.Done():
				return
			case <-ticker.C:
				currentQuery := atomic.AddInt64(&ts.queryCount, 1)
				atomic.AddInt64(&ts.queriesFired, 1)

				wg.Add(1)
				go func(queryNum int64) {
					defer wg.Done()
					defer func() {
						if r := recover(); r != nil {
							// Channel was closed, that's okay
						}
					}()

					result := lt.executeQuery(queryCtx, int(queryNum))
					select {
					case ts.resultsChan <- result:
					case <-time.After(ResultChannelTimeout):
						// Timeout - channel might be closed or full
					}
				}(currentQuery)
			}
		}
	}()
}

// waitForCompletion waits for all queries to complete with progress reporting
func (ts *testState) waitForCompletion(wg *sync.WaitGroup, waitTimeout time.Duration) {
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	progressTicker := time.NewTicker(ProgressUpdateInterval)
	defer progressTicker.Stop()

	progressDone := make(chan struct{})
	go func() {
		for {
			select {
			case <-progressDone:
				return
			case <-progressTicker.C:
				completed := atomic.LoadInt64(&ts.queriesCompleted)
				fmt.Printf("  Waiting... %d queries completed so far\n", completed)
			}
		}
	}()

	select {
	case <-done:
		close(progressDone)
		fmt.Printf("All queries completed\n")
	case <-time.After(waitTimeout):
		close(progressDone)
		fmt.Printf("Warning: Timed out waiting for queries after %v. Some queries may still be running\n", waitTimeout)
		time.Sleep(GracePeriodAfterClose)
	}
}

// calculateWaitTimeout calculates the appropriate timeout for waiting for query completion
func calculateWaitTimeout(duration time.Duration) time.Duration {
	waitTimeout := duration / 2
	if waitTimeout < MinWaitTimeout {
		waitTimeout = MinWaitTimeout
	}
	if waitTimeout > MaxWaitTimeout {
		waitTimeout = MaxWaitTimeout
	}
	return waitTimeout
}
