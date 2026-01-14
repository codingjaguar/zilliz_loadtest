package main

import (
	"context"
	"sync"
	"testing"
	"time"
)

// Simulate query execution with configurable latency
func simulateQuery(latency time.Duration) {
	time.Sleep(latency)
}

// Test to understand why we're stuck at 33 QPS
func TestWhyStuckAt33QPS(t *testing.T) {
	targetQPS := 100
	duration := 10 * time.Second

	// Test different latencies to see what gives us 33 QPS
	// If we're getting 33 QPS, that suggests queries are taking ~30ms
	// and we're only running them sequentially or with limited concurrency

	latencies := []time.Duration{
		30 * time.Millisecond,  // This would give max 33 QPS sequentially
		100 * time.Millisecond, // This would give max 10 QPS sequentially
	}

	for _, latency := range latencies {
		t.Run(latency.String(), func(t *testing.T) {
			testCtx, cancel := context.WithTimeout(context.Background(), duration)
			defer cancel()

			var wg sync.WaitGroup
			resultsChan := make(chan time.Time, targetQPS*10)

			// Strategy: Maintain In-Flight (same as our implementation)
			targetInFlight := int(float64(targetQPS) * latency.Seconds())
			if targetInFlight < 1 {
				targetInFlight = 1
			}
			targetInFlight = int(float64(targetInFlight) * 1.5)
			if targetInFlight > targetQPS*5 {
				targetInFlight = targetQPS * 5
			}

			t.Logf("Target QPS: %d, Query Latency: %v, Target In-Flight: %d",
				targetQPS, latency, targetInFlight)

			var inFlight int64
			var mu sync.Mutex
			queryCount := 0
			startTime := time.Now()

			ticker := time.NewTicker(1 * time.Millisecond)
			defer ticker.Stop()

			go func() {
				for {
					select {
					case <-testCtx.Done():
						return
					case <-ticker.C:
						mu.Lock()
						currentInFlight := int(inFlight)
						mu.Unlock()

						for currentInFlight < targetInFlight {
							mu.Lock()
							queryCount++
							currentQuery := queryCount
							inFlight++
							mu.Unlock()

							wg.Add(1)
							go func(qn int) {
								defer wg.Done()
								defer func() {
									mu.Lock()
									inFlight--
									mu.Unlock()
								}()

								start := time.Now()
								simulateQuery(latency)
								select {
								case resultsChan <- start:
								case <-testCtx.Done():
								}
							}(currentQuery)

							mu.Lock()
							currentInFlight = int(inFlight)
							mu.Unlock()
						}
					}
				}
			}()

			<-testCtx.Done()
			wg.Wait()
			close(resultsChan)

			elapsed := time.Since(startTime)
			var count int
			for range resultsChan {
				count++
			}

			actualQPS := float64(count) / elapsed.Seconds()
			theoreticalMax := 1000.0 / float64(latency.Milliseconds())

			t.Logf("Results: Actual QPS: %.2f, Theoretical Sequential Max: %.2f",
				actualQPS, theoreticalMax)
			t.Logf("Efficiency: %.1f%% (actual/theoretical), In-Flight Target: %d",
				(actualQPS/theoreticalMax)*100, targetInFlight)

			if actualQPS < float64(targetQPS)*0.8 {
				t.Errorf("Failed to achieve target QPS: got %.2f, wanted %d (%.1f%%)",
					actualQPS, targetQPS, (actualQPS/float64(targetQPS))*100)
			}
		})
	}
}

// Test to see what happens if queries take variable time
func TestVariableLatency(t *testing.T) {
	targetQPS := 100
	duration := 5 * time.Second

	testCtx, cancel := context.WithTimeout(context.Background(), duration)
	defer cancel()

	var wg sync.WaitGroup
	resultsChan := make(chan time.Time, targetQPS*10)

	// Simulate variable latency: 20-40ms (average 30ms)
	var inFlight int64
	var mu sync.Mutex
	queryCount := 0
	startTime := time.Now()

	// Use average latency for target calculation
	avgLatency := 30 * time.Millisecond
	targetInFlight := int(float64(targetQPS) * avgLatency.Seconds() * 1.5)
	if targetInFlight > targetQPS*5 {
		targetInFlight = targetQPS * 5
	}

	t.Logf("Target QPS: %d, Avg Latency: %v, Target In-Flight: %d",
		targetQPS, avgLatency, targetInFlight)

	ticker := time.NewTicker(1 * time.Millisecond)
	defer ticker.Stop()

	go func() {
		for {
			select {
			case <-testCtx.Done():
				return
			case <-ticker.C:
				mu.Lock()
				currentInFlight := int(inFlight)
				mu.Unlock()

				for currentInFlight < targetInFlight {
					mu.Lock()
					queryCount++
					currentQuery := queryCount
					inFlight++
					mu.Unlock()

					wg.Add(1)
					go func(qn int) {
						defer wg.Done()
						defer func() {
							mu.Lock()
							inFlight--
							mu.Unlock()
						}()

						// Variable latency: 20-40ms
						latency := 20*time.Millisecond + time.Duration(qn%20)*time.Millisecond
						start := time.Now()
						simulateQuery(latency)
						select {
						case resultsChan <- start:
						case <-testCtx.Done():
						}
					}(currentQuery)

					mu.Lock()
					currentInFlight = int(inFlight)
					mu.Unlock()
				}
			}
		}
	}()

	<-testCtx.Done()
	wg.Wait()
	close(resultsChan)

	elapsed := time.Since(startTime)
	var count int
	for range resultsChan {
		count++
	}

	actualQPS := float64(count) / elapsed.Seconds()
	t.Logf("Variable Latency - Actual QPS: %.2f, Target: %d, Efficiency: %.1f%%",
		actualQPS, targetQPS, (actualQPS/float64(targetQPS))*100)
}

// Test to see if there's a bottleneck in the implementation itself
func TestImplementationBottleneck(t *testing.T) {
	targetQPS := 100
	duration := 5 * time.Second
	queryLatency := 30 * time.Millisecond

	testCtx, cancel := context.WithTimeout(context.Background(), duration)
	defer cancel()

	var wg sync.WaitGroup
	resultsChan := make(chan QueryResult, targetQPS*10)

	startTime := time.Now()
	targetInFlight := int(float64(targetQPS) * queryLatency.Seconds() * 1.5)
	if targetInFlight > targetQPS*5 {
		targetInFlight = targetQPS * 5
	}

	var inFlight int64
	var mu sync.Mutex
	queryCount := 0

	ticker := time.NewTicker(1 * time.Millisecond)
	defer ticker.Stop()

	// Track how often we're checking and firing
	var checkCount int64
	var fireCount int64

	go func() {
		for {
			select {
			case <-testCtx.Done():
				return
			case <-ticker.C:
				checkCount++
				mu.Lock()
				currentInFlight := int(inFlight)
				mu.Unlock()

				for currentInFlight < targetInFlight {
					fireCount++
					mu.Lock()
					queryCount++
					currentQuery := queryCount
					inFlight++
					mu.Unlock()

					wg.Add(1)
					go func(qn int) {
						defer wg.Done()
						defer func() {
							mu.Lock()
							inFlight--
							mu.Unlock()
						}()

						start := time.Now()
						simulateQuery(queryLatency)
						result := QueryResult{Latency: time.Since(start)}
						select {
						case resultsChan <- result:
						case <-testCtx.Done():
						}
					}(currentQuery)

					mu.Lock()
					currentInFlight = int(inFlight)
					mu.Unlock()
				}
			}
		}
	}()

	<-testCtx.Done()
	wg.Wait()
	close(resultsChan)

	elapsed := time.Since(startTime)
	var count int
	for range resultsChan {
		count++
	}

	actualQPS := float64(count) / elapsed.Seconds()
	t.Logf("Implementation Test - Actual QPS: %.2f, Target: %d", actualQPS, targetQPS)
	t.Logf("Ticker checks: %d, Fires: %d, Ratio: %.2f%%",
		checkCount, fireCount, float64(fireCount)/float64(checkCount)*100)
	t.Logf("Target In-Flight: %d, Final Query Count: %d", targetInFlight, queryCount)
}

// Test with very fast ticker to see if that's the issue
func TestVeryFastTicker(t *testing.T) {
	targetQPS := 100
	duration := 5 * time.Second
	queryLatency := 30 * time.Millisecond

	testCtx, cancel := context.WithTimeout(context.Background(), duration)
	defer cancel()

	var wg sync.WaitGroup
	resultsChan := make(chan QueryResult, targetQPS*10)

	startTime := time.Now()
	targetInFlight := int(float64(targetQPS) * queryLatency.Seconds() * 2)

	var inFlight int64
	var mu sync.Mutex
	queryCount := 0

	// Use a very fast ticker (every 100 microseconds)
	ticker := time.NewTicker(100 * time.Microsecond)
	defer ticker.Stop()

	go func() {
		for {
			select {
			case <-testCtx.Done():
				return
			case <-ticker.C:
				mu.Lock()
				currentInFlight := int(inFlight)
				mu.Unlock()

				// Fire multiple queries if we're below target
				for currentInFlight < targetInFlight {
					mu.Lock()
					queryCount++
					currentQuery := queryCount
					inFlight++
					mu.Unlock()

					wg.Add(1)
					go func(qn int) {
						defer wg.Done()
						defer func() {
							mu.Lock()
							inFlight--
							mu.Unlock()
						}()

						start := time.Now()
						simulateQuery(queryLatency)
						result := QueryResult{Latency: time.Since(start)}
						select {
						case resultsChan <- result:
						case <-testCtx.Done():
						}
					}(currentQuery)

					mu.Lock()
					currentInFlight = int(inFlight)
					mu.Unlock()
				}
			}
		}
	}()

	<-testCtx.Done()
	wg.Wait()
	close(resultsChan)

	elapsed := time.Since(startTime)
	var count int
	for range resultsChan {
		count++
	}

	actualQPS := float64(count) / elapsed.Seconds()
	t.Logf("Very Fast Ticker - Actual QPS: %.2f, Target: %d, Efficiency: %.1f%%",
		actualQPS, targetQPS, (actualQPS/float64(targetQPS))*100)
}
