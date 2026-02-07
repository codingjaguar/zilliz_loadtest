// CLI for Zilliz Cloud: seed and load test. Use --seed or --load-test with your cluster and collection.
package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"zilliz-loadtest/internal/config"
	"zilliz-loadtest/internal/datasource"
	"zilliz-loadtest/internal/loadtest"
	"zilliz-loadtest/internal/logger"
	"zilliz-loadtest/internal/validation"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
)

// mergedConn holds connection params after merging flags and config (flags override).
func mergedConn(cfg *config.Config, flags *Flags) (apiKey, databaseURL, collection string) {
	apiKey = or(flags.APIKey, cfg.APIKey)
	databaseURL = or(flags.DatabaseURL, cfg.DatabaseURL)
	collection = or(flags.Collection, cfg.DefaultCollection)
	return apiKey, databaseURL, collection
}

func or(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// requireConn ensures apiKey is set and exits on failure.
func requireAPIKey(apiKey string) {
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "API key is required. Set api_key in configs/config.yaml or use --api-key.")
		os.Exit(1)
	}
}

// requireDatabaseAndCollection exits with a message if database URL or collection is missing.
func requireDatabaseAndCollection(databaseURL, collection, operation string) {
	if databaseURL == "" {
		fmt.Fprintf(os.Stderr, "database_url is required for %s. Set in configs/config.yaml or use --database-url.\n", operation)
		os.Exit(1)
	}
	if collection == "" {
		fmt.Fprintf(os.Stderr, "collection is required for %s. Set default_collection in config or use --collection.\n", operation)
		os.Exit(1)
	}
}

// resolveInt returns the first positive value among flagVal, cfgVal, defaultVal.
func resolveInt(flagVal, cfgVal, defaultVal int) int {
	if flagVal > 0 {
		return flagVal
	}
	if cfgVal > 0 {
		return cfgVal
	}
	return defaultVal
}

func main() {
	flags := ParseFlags()

	if flags.Help {
		PrintHelp()
		return
	}
	if flags.Version {
		PrintVersion()
		return
	}

	cfg, err := config.LoadConfig(flags.ConfigPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Load config: %v\n", err)
		os.Exit(1)
	}

	// Initialize logger with config settings
	logger.Init(cfg.LogLevel, cfg.LogFormat)

	apiKey, databaseURL, collection := mergedConn(cfg, flags)
	requireAPIKey(apiKey)

	if flags.Seed {
		runSeed(cfg, flags, apiKey, databaseURL, collection)
		return
	}
	if flags.LoadTest {
		runLoadTest(cfg, flags, apiKey, databaseURL, collection)
		return
	}

	// No mode specified: interactive menu
	fmt.Println("Zilliz Load Test CLI")
	fmt.Println()
	fmt.Println("What do you want to do?")
	fmt.Println("  1) Seed the database")
	fmt.Println("  2) Run a read query load test")
	fmt.Print("Enter choice (1 or 2): ")
	rd := bufio.NewReader(os.Stdin)
	line, err := rd.ReadString('\n')
	if err != nil {
		fmt.Fprintf(os.Stderr, "Read input: %v\n", err)
		os.Exit(1)
	}
	choice := strings.TrimSpace(line)
	switch choice {
	case "1":
		runSeed(cfg, flags, apiKey, databaseURL, collection)
	case "2":
		runLoadTest(cfg, flags, apiKey, databaseURL, collection)
	default:
		fmt.Fprintf(os.Stderr, "Invalid choice %q. Use 1 or 2.\n", choice)
		os.Exit(1)
	}
}

func runSeed(cfg *config.Config, flags *Flags, apiKey, databaseURL, collection string) {
	requireDatabaseAndCollection(databaseURL, collection, "seed")

	vectorDim := resolveInt(flags.SeedVectorDim, cfg.SeedVectorDim, resolveInt(0, cfg.DefaultVectorDim, 768))
	totalVectors := resolveInt(flags.SeedVectorCount, cfg.SeedVectorCount, 2000000)
	batchSize := resolveInt(flags.SeedBatchSize, cfg.SeedBatchSize, loadtest.DefaultBatchSize)
	seedSource := or(flags.SeedSource, cfg.SeedSource)
	if seedSource == "" {
		seedSource = "synthetic"
	}
	metricTypeStr := or(flags.MetricType, cfg.DefaultMetricType)
	metricType, err := loadtest.ParseMetricType(metricTypeStr)
	if err != nil {
		metricType = loadtest.DefaultMetricType()
	}

	// Drop collection by default unless --keep-collection is specified
	dropCollection := !flags.KeepCollection
	if err := loadtest.SeedDatabaseWithSource(apiKey, databaseURL, collection, vectorDim, totalVectors, batchSize, metricType, seedSource, flags.SkipCollectionCreation, dropCollection); err != nil {
		fmt.Fprintf(os.Stderr, "Seed failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Seed completed successfully.")
}

func runLoadTest(cfg *config.Config, flags *Flags, apiKey, databaseURL, collection string) {
	requireDatabaseAndCollection(databaseURL, collection, "load test")

	qpsLevels, err := ParseQPSLevels(flags.QPS)
	if err != nil || len(qpsLevels) == 0 {
		if len(cfg.DefaultQPSLevels) > 0 {
			qpsLevels = cfg.DefaultQPSLevels
		} else {
			fmt.Fprintln(os.Stderr, "QPS levels required. Use --qps 100,500,1000 or set default_qps_levels in config.")
			os.Exit(1)
		}
	}

	duration, err := ParseDuration(flags.Duration)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Invalid duration: %v\n", err)
		os.Exit(1)
	}
	if duration <= 0 {
		duration, _ = cfg.GetDuration()
	}
	if duration <= 0 {
		duration = 30 * time.Second
	}

	warmup := resolveInt(flags.Warmup, cfg.WarmupQueries, 100)
	if warmup < 0 {
		warmup = 100
	}

	vectorDim := resolveInt(flags.VectorDim, cfg.DefaultVectorDim, 768)
	metricTypeStr := or(flags.MetricType, cfg.DefaultMetricType)

	ctx := context.Background()
	expectedMetricType, err := loadtest.ParseMetricType(metricTypeStr)
	if err != nil {
		expectedMetricType = loadtest.DefaultMetricType()
	}
	result, err := validation.ValidateCollection(apiKey, databaseURL, collection, vectorDim, expectedMetricType)
	if err != nil || len(result.Errors) > 0 {
		fmt.Fprintln(os.Stderr, "Pre-flight validation failed:")
		validation.PrintValidationResults(result, collection)
		os.Exit(1)
	}
	validation.PrintValidationResults(result, collection)

	// Detect if collection has real BEIR data and load queries/qrels if so
	queries, qrels := loadQueriesAndQrelsIfNeeded(ctx, apiKey, databaseURL, collection)

	connMap, _ := ParseConnections(flags.Connections)
	var results []loadtest.TestResult
	if len(queries) > 0 {
		fmt.Printf("\nRunning load test with real BEIR queries: %v QPS level(s), %v per level (warmup: %d queries).\n", qpsLevels, duration, warmup)
	} else {
		fmt.Printf("\nRunning load test: %v QPS level(s), %v per level (warmup: %d queries).\n", qpsLevels, duration, warmup)
	}

	for _, targetQPS := range qpsLevels {
		fmt.Printf("\n--- %d QPS for %v ---\n", targetQPS, duration)
		numConnections := 10
		if connMap != nil {
			if c, ok := connMap[targetQPS]; ok {
				numConnections = c
			}
		}
		if connMap == nil {
			numConnections, _ = loadtest.CalculateOptimalConnectionsWithParams(cfg.ExpectedLatencyMs, cfg.ConnectionMultiplier, targetQPS)
		}
		lt, err := loadtest.NewLoadTesterWithConnections(apiKey, databaseURL, collection, vectorDim, metricTypeStr, numConnections)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Create load tester: %v\n", err)
			os.Exit(1)
		}

		var res loadtest.TestResult
		if len(queries) > 0 {
			// Use real queries and calculate recall
			res, err = lt.RunTestWithRecall(ctx, targetQPS, duration, warmup, queries, qrels)
		} else {
			// Standard test with random vectors
			res, err = lt.RunTest(ctx, targetQPS, duration, warmup)
		}

		lt.Close()
		if err != nil {
			fmt.Fprintf(os.Stderr, "RunTest at %d QPS: %v\n", targetQPS, err)
			os.Exit(1)
		}
		results = append(results, res)
	}
	displayResults(results)
}

// loadQueriesAndQrelsIfNeeded detects if collection has BEIR or VDBBench data and loads queries/qrels
// Business recall (qrels) is only loaded for full corpus (8M+ documents for BEIR, or VDBBench datasets)
func loadQueriesAndQrelsIfNeeded(ctx context.Context, apiKey, databaseURL, collection string) ([]datasource.CohereQuery, datasource.Qrels) {
	// Connect to check collection schema
	c, err := client.NewClient(ctx, client.Config{
		Address: databaseURL,
		APIKey:  apiKey,
	})
	if err != nil {
		logger.Warn("Failed to connect for collection detection", "error", err)
		return nil, nil
	}
	defer c.Close()

	// Detect collection info including row count
	collInfo, err := loadtest.DetectCollectionInfo(ctx, c, collection)
	if err != nil {
		logger.Warn("Failed to detect collection type", "error", err)
		return nil, nil
	}

	// Check for VDBBench schema first
	if collInfo.HasVDBBenchSchema {
		datasetName := loadtest.DetectVDBBenchDataset(collInfo)
		if datasetName != "" {
			logger.Info("Detected VDBBench collection",
				"dataset", datasetName,
				"row_count", collInfo.RowCount,
				"vector_dim", collInfo.VectorDim)
			return loadVDBBenchQueriesAndQrels(datasetName)
		}
		logger.Info("VDBBench-like schema but unknown dataset, using random queries",
			"vector_dim", collInfo.VectorDim,
			"row_count", collInfo.RowCount)
		return nil, nil
	}

	// Check for BEIR schema
	if !collInfo.HasBEIRSchema {
		logger.Info("Collection does not have BEIR schema, using random queries")
		return nil, nil
	}

	logger.Info("Detected BEIR collection", "row_count", collInfo.RowCount, "is_full_corpus", collInfo.IsFullCorpus)

	// Initialize downloader and readers
	downloader := datasource.NewCohereDownloader("")
	queryReader := datasource.NewCohereQueryReader(downloader)

	// Load queries (use dev split for faster loading)
	queries, err := queryReader.ReadQueries("dev", 10000) // Load up to 10K queries
	if err != nil {
		logger.Warn("Failed to load queries", "error", err)
		return nil, nil
	}

	// Only load qrels for full corpus - business recall doesn't make sense otherwise
	// because the relevant documents won't be in a partial corpus
	var qrels datasource.Qrels
	if collInfo.IsFullCorpus {
		qrelsReader := datasource.NewCohereQrelsReader(downloader)
		qrels, err = qrelsReader.ReadQrels("dev")
		if err != nil {
			logger.Warn("Failed to load qrels", "error", err)
		} else {
			logger.Info("Loaded qrels for full corpus", "qrels_queries", len(qrels))
		}
	} else {
		logger.Info("Skipping qrels loading - business recall requires full corpus (8M+ docs)",
			"current_rows", collInfo.RowCount,
			"required_rows", loadtest.FullCorpusThreshold)
	}

	logger.Info("Loaded BEIR queries", "queries", len(queries))

	return queries, qrels
}

// loadVDBBenchQueriesAndQrels loads test queries and ground truth for a VDBBench dataset
func loadVDBBenchQueriesAndQrels(datasetName string) ([]datasource.CohereQuery, datasource.Qrels) {
	// Load VDBBench queries
	vdbQueries, err := datasource.LoadVDBBenchQueries(datasetName)
	if err != nil {
		logger.Warn("Failed to load VDBBench queries", "dataset", datasetName, "error", err)
		return nil, nil
	}

	// Convert to CohereQuery format for compatibility
	queries := make([]datasource.CohereQuery, len(vdbQueries))
	for i, q := range vdbQueries {
		queries[i] = datasource.CohereQuery{
			ID:        fmt.Sprintf("%d", q.ID),
			Embedding: q.Embedding,
		}
	}
	logger.Info("Loaded VDBBench queries", "count", len(queries))

	// Load ground truth neighbors
	neighbors, err := datasource.LoadVDBBenchNeighbors(datasetName)
	if err != nil {
		logger.Warn("Failed to load VDBBench ground truth", "dataset", datasetName, "error", err)
		return queries, nil
	}

	// Convert to Qrels format
	qrels := datasource.NewVDBBenchQrels(neighbors).ToQrels()
	logger.Info("Loaded VDBBench ground truth for recall calculation", "queries_with_neighbors", len(qrels))

	return queries, qrels
}
