package loadtest

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"zilliz-loadtest/internal/mocks"
)

func TestTestState_StartResultCollector(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	ts := &testState{
		resultsChan: make(chan QueryResult, 10),
		allResults:  make([]QueryResult, 0),
	}

	ts.startResultCollector()

	// Send some results
	results := []QueryResult{
		{Latency: 100 * time.Millisecond, Error: nil},
		{Latency: 200 * time.Millisecond, Error: nil},
		{Latency: 150 * time.Millisecond, Error: nil},
	}

	for _, result := range results {
		select {
		case ts.resultsChan <- result:
		case <-ctx.Done():
			t.Fatal("Timeout sending results")
		}
	}

	// Give collector time to process
	select {
	case <-time.After(500 * time.Millisecond):
	case <-ctx.Done():
		t.Fatal("Timeout waiting for collector")
	}

	close(ts.resultsChan)

	// Give collector time to finish
	select {
	case <-time.After(500 * time.Millisecond):
	case <-ctx.Done():
		t.Fatal("Timeout waiting for collector to finish")
	}

	ts.resultsMu.Lock()
	defer ts.resultsMu.Unlock()

	if len(ts.allResults) != len(results) {
		t.Errorf("startResultCollector() collected %d results, want %d", len(ts.allResults), len(results))
	}
}

func TestTestState_StartStatusReporter(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	ts := &testState{
		startTime: time.Now(),
	}

	testCtx, testCancel := context.WithTimeout(ctx, 200*time.Millisecond)
	defer testCancel()

	ts.startStatusReporter(testCtx)

	// Wait for at least one status update
	select {
	case <-time.After(150 * time.Millisecond):
		// Status reporter should have printed at least one update
		// (we can't easily test the logger output, but we can verify it doesn't panic)
	case <-ctx.Done():
		t.Fatal("Timeout waiting for status reporter")
	}
}

func TestStartQueryFirer(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	mockClient := &mocks.MockClient{
		SearchFunc: func(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
			return []client.SearchResult{}, nil
		},
	}

	lt := &LoadTester{
		clients:      []client.Client{mockClient},
		collection:   "test",
		vectorDim:    768,
		metricType:   entity.L2,
		topK:         10,
		outputFields: []string{"id"},
		searchLevel:  1,
	}

	testCtx, testCancel := context.WithTimeout(ctx, 200*time.Millisecond)
	defer testCancel()

	queryCtx := ctx
	interval := 50 * time.Millisecond // 20 QPS
	ts := &testState{
		resultsChan: make(chan QueryResult, 100),
		startTime:   time.Now(),
	}
	var wg sync.WaitGroup

	lt.startQueryFirer(testCtx, queryCtx, interval, ts, &wg)

	// Wait for test context to expire
	select {
	case <-testCtx.Done():
	case <-ctx.Done():
		t.Fatal("Test timeout")
	}

	// Wait a bit for queries to complete with timeout
	done := make(chan struct{})
	go func() {
		time.Sleep(100 * time.Millisecond)
		close(done)
	}()

	select {
	case <-done:
	case <-ctx.Done():
		t.Fatal("Timeout waiting for queries")
	}

	fired := atomic.LoadInt64(&ts.queriesFired)
	if fired == 0 {
		t.Error("startQueryFirer() should have fired at least one query")
	}
}

func TestWaitForCompletion(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	ts := &testState{
		resultsChan: make(chan QueryResult, 10),
		startTime:   time.Now(),
	}

	var wg sync.WaitGroup

	// Add some work
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			time.Sleep(10 * time.Millisecond)
		}()
	}

	// Should complete quickly
	ts.waitForCompletion(&wg, 1*time.Second)

	// Verify all work completed
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		// Success
	case <-time.After(500 * time.Millisecond):
		t.Error("waitForCompletion() should have waited for all work")
	case <-ctx.Done():
		t.Fatal("Test timeout")
	}
}

func TestWaitForCompletion_Timeout(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	ts := &testState{
		resultsChan: make(chan QueryResult, 10),
		startTime:   time.Now(),
	}

	var wg sync.WaitGroup

	// Add work that takes longer than timeout
	wg.Add(1)
	go func() {
		defer wg.Done()
		select {
		case <-time.After(2 * time.Second):
		case <-ctx.Done():
		}
	}()

	// Should timeout - waitForCompletion waits for waitTimeout, then sleeps for GracePeriodAfterClose
	// So total time should be around waitTimeout + GracePeriodAfterClose
	start := time.Now()
	ts.waitForCompletion(&wg, 100*time.Millisecond)
	elapsed := time.Since(start)

	// waitForCompletion waits for timeout (100ms) + GracePeriodAfterClose (2s)
	// So total should be around 2.1 seconds
	expectedMin := 2*time.Second + 50*time.Millisecond
	expectedMax := 2*time.Second + 200*time.Millisecond

	if elapsed < expectedMin || elapsed > expectedMax {
		t.Errorf("waitForCompletion() elapsed = %v, want between %v and %v", elapsed, expectedMin, expectedMax)
	}
}

func TestCalculateWaitTimeout(t *testing.T) {
	tests := []struct {
		name     string
		duration time.Duration
		min      time.Duration
		max      time.Duration
	}{
		{"short duration", 10 * time.Second, MinWaitTimeout, MaxWaitTimeout},
		{"medium duration", 60 * time.Second, 30 * time.Second, MaxWaitTimeout},
		{"long duration", 10 * time.Minute, MinWaitTimeout, MaxWaitTimeout},
		{"very short", 1 * time.Second, MinWaitTimeout, MaxWaitTimeout},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			timeout := calculateWaitTimeout(tt.duration)
			if timeout < tt.min || timeout > tt.max {
				t.Errorf("calculateWaitTimeout() = %v, want between %v and %v", timeout, tt.min, tt.max)
			}
		})
	}
}
