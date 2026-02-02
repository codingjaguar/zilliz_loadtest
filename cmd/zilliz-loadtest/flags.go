package main

import (
	"flag"
	"fmt"
	"strings"
	"time"

	"zilliz-loadtest/internal/validation"
)

// Flags holds all command-line flags
type Flags struct {
	// Operation mode
	Seed     bool
	LoadTest bool

	// Connection parameters
	APIKey      string
	DatabaseURL string
	Collection  string

	// Seed parameters
	SeedVectorCount        int
	SeedVectorDim          int
	SeedBatchSize          int
	SeedSource             string
	SkipCollectionCreation bool
	KeepCollection         bool // If true, keep existing collection (don't drop); default is to drop

	// Load test parameters
	QPS          string
	Duration     string
	Warmup       int
	VectorDim    int
	MetricType   string
	Connections  string
	ConfigPath   string
	OutputFormat string
	OutputPath   string

	// General flags
	Quiet   bool
	Version bool
	Help    bool

	// Profiling flags
	CPUProfile string
	MemProfile string
}

// ParseFlags parses command-line flags and returns Flags struct
func ParseFlags() *Flags {
	flags := &Flags{}

	// Operation mode
	flag.BoolVar(&flags.Seed, "seed", false, "Run seed operation")
	flag.BoolVar(&flags.LoadTest, "load-test", false, "Run load test")

	// Connection parameters
	flag.StringVar(&flags.APIKey, "api-key", "", "Zilliz Cloud API key")
	flag.StringVar(&flags.DatabaseURL, "database-url", "", "Zilliz Cloud database URL")
	flag.StringVar(&flags.Collection, "collection", "", "Collection name")

	// Seed parameters
	flag.IntVar(&flags.SeedVectorCount, "seed-vector-count", 0, "Number of vectors to seed (default: 2000000)")
	flag.IntVar(&flags.SeedVectorDim, "seed-vector-dim", 0, "Vector dimension for seeding (default: 768)")
	flag.IntVar(&flags.SeedBatchSize, "seed-batch-size", 0, "Batch size for seeding (default: 15000)")
	flag.StringVar(&flags.SeedSource, "seed-source", "", "Seed data source: synthetic or cohere (default: synthetic)")
	flag.BoolVar(&flags.SkipCollectionCreation, "skip-collection-creation", false, "Skip automatic collection creation (assume collection exists)")
	flag.BoolVar(&flags.KeepCollection, "keep-collection", false, "Keep existing collection instead of dropping it before seeding")

	// Load test parameters
	flag.StringVar(&flags.QPS, "qps", "", "QPS levels (comma-separated, e.g., 100,500,1000)")
	flag.StringVar(&flags.Duration, "duration", "", "Test duration (e.g., 30s, 1m)")
	flag.IntVar(&flags.Warmup, "warmup", 0, "Number of warmup queries")
	flag.IntVar(&flags.VectorDim, "vector-dim", 0, "Vector dimension")
	flag.StringVar(&flags.MetricType, "metric-type", "", "Metric type (L2, IP, COSINE)")
	flag.StringVar(&flags.Connections, "connections", "", "Connection counts per QPS (format: qps1:count1,qps2:count2)")

	// General flags
	flag.StringVar(&flags.ConfigPath, "config", "", "Path to config file")
	flag.StringVar(&flags.OutputFormat, "output", "", "Output format (json, csv, both)")
	flag.StringVar(&flags.OutputPath, "output-path", "", "Output file path (without extension)")
	flag.BoolVar(&flags.Quiet, "quiet", false, "Suppress interactive prompts")
	flag.StringVar(&flags.CPUProfile, "cpu-profile", "", "Enable CPU profiling and write to file")
	flag.StringVar(&flags.MemProfile, "mem-profile", "", "Enable memory profiling and write to file")
	flag.BoolVar(&flags.Version, "version", false, "Print version and exit")
	flag.BoolVar(&flags.Help, "help", false, "Print help and exit")

	flag.Parse()

	return flags
}

// PrintVersion prints the version information
func PrintVersion() {
	fmt.Println("zilliz-loadtest version 1.0.0")
}

// PrintHelp prints the help message
func PrintHelp() {
	fmt.Println("Zilliz Cloud Load Test Tool")
	fmt.Println("===========================")
	fmt.Println()
	fmt.Println("Usage:")
	fmt.Println("  zilliz-loadtest [flags]")
	fmt.Println()
	fmt.Println("Operation Modes:")
	fmt.Println("  --seed              Run seed operation")
	fmt.Println("  --load-test         Run load test")
	fmt.Println("  (no flags)          Interactive mode")
	fmt.Println()
	fmt.Println("Connection Parameters:")
	fmt.Println("  --api-key string    Zilliz Cloud API key")
	fmt.Println("  --database-url string  Zilliz Cloud database URL")
	fmt.Println("  --collection string  Collection name")
	fmt.Println()
	fmt.Println("Seed Parameters:")
	fmt.Println("  --seed-vector-count int  Number of vectors to seed (default: 2000000)")
	fmt.Println("  --seed-vector-dim int    Vector dimension for seeding (default: 768)")
	fmt.Println("  --seed-batch-size int    Batch size for seeding (default: 15000)")
	fmt.Println("  --seed-source string     Seed data source: synthetic or cohere (default: synthetic)")
	fmt.Println("  --skip-collection-creation  Skip automatic collection creation (assume collection exists)")
	fmt.Println("  --keep-collection        Keep existing collection instead of dropping (default: drop collection)")
	fmt.Println()
	fmt.Println("Load Test Parameters:")
	fmt.Println("  --qps string        QPS levels (comma-separated, e.g., 100,500,1000)")
	fmt.Println("  --duration string   Test duration (e.g., 30s, 1m)")
	fmt.Println("  --warmup int        Number of warmup queries")
	fmt.Println("  --vector-dim int    Vector dimension")
	fmt.Println("  --metric-type string  Metric type (L2, IP, COSINE)")
	fmt.Println("  --connections string   Connection counts per QPS (format: qps1:count1,qps2:count2)")
	fmt.Println()
	fmt.Println("General Flags:")
	fmt.Println("  --config string     Path to config file")
	fmt.Println("  --output string     Output format (json, csv, both)")
	fmt.Println("  --output-path string  Output file path (without extension)")
	fmt.Println("  --quiet             Suppress interactive prompts")
	fmt.Println("  --version           Print version and exit")
	fmt.Println("  --help              Print this help message")
	fmt.Println()
	fmt.Println("Examples:")
	fmt.Println("  # Interactive mode")
	fmt.Println("  zilliz-loadtest")
	fmt.Println()
	fmt.Println("  # Seed database")
	fmt.Println("  zilliz-loadtest --seed --api-key KEY --database-url URL --collection COLL")
	fmt.Println()
	fmt.Println("  # Run load test")
	fmt.Println("  zilliz-loadtest --load-test --api-key KEY --database-url URL --collection COLL --qps 100,500 --duration 30s")
	fmt.Println()
	fmt.Println("  # Run with profiling")
	fmt.Println("  zilliz-loadtest --load-test ... --cpu-profile cpu.pprof --mem-profile mem.pprof")
	fmt.Println()
	fmt.Println("Exit Codes:")
	fmt.Println("  0  Success")
	fmt.Println("  1  Error")
	fmt.Println("  2  Validation failure")
}

// ParseConnections parses the connections string into a map
func ParseConnections(connStr string) (map[int]int, error) {
	if connStr == "" {
		return nil, nil
	}

	result := make(map[int]int)
	parts := strings.Split(connStr, ",")

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		kv := strings.Split(part, ":")
		if len(kv) != 2 {
			return nil, fmt.Errorf("invalid connection format: %s (expected qps:count)", part)
		}

		qps, err := parseInt(kv[0])
		if err != nil {
			return nil, fmt.Errorf("invalid QPS value: %s", kv[0])
		}

		count, err := parseInt(kv[1])
		if err != nil {
			return nil, fmt.Errorf("invalid connection count: %s", kv[1])
		}

		if qps <= 0 || count <= 0 {
			return nil, fmt.Errorf("QPS and connection count must be positive")
		}

		result[qps] = count
	}

	return result, nil
}

// parseInt is a helper to parse integers with better error messages
func parseInt(s string) (int, error) {
	s = strings.TrimSpace(s)
	var result int
	_, err := fmt.Sscanf(s, "%d", &result)
	if err != nil {
		return 0, fmt.Errorf("invalid integer: %s", s)
	}
	return result, nil
}

// IsNonInteractive returns true if flags indicate non-interactive mode
func (f *Flags) IsNonInteractive() bool {
	return f.Seed || f.LoadTest
}

// validateConnectionFlags validates API key, database URL, and collection name.
// Call this when Seed or LoadTest is set (after merging with config).
func validateConnectionFlags(f *Flags) error {
	if err := validation.ValidateAPIKey(f.APIKey); err != nil {
		return fmt.Errorf("invalid API key: %w", err)
	}
	if err := validation.ValidateDatabaseURL(f.DatabaseURL); err != nil {
		return fmt.Errorf("invalid database URL: %w", err)
	}
	if err := validation.ValidateCollectionName(f.Collection); err != nil {
		return fmt.Errorf("invalid collection name: %w", err)
	}
	return nil
}

// Validate validates flags and returns an error if invalid
func (f *Flags) Validate() error {
	if f.Version || f.Help {
		return nil
	}

	if f.Seed && f.LoadTest {
		return fmt.Errorf("cannot specify both --seed and --load-test")
	}

	if f.Seed {
		if err := validateConnectionFlags(f); err != nil {
			return err
		}
		if f.SeedVectorDim > 0 {
			if err := validation.ValidateVectorDimension(f.SeedVectorDim); err != nil {
				return fmt.Errorf("invalid seed vector dimension: %w", err)
			}
		}
		if f.SeedBatchSize > 0 {
			if err := validation.ValidateSeedParameters(f.SeedVectorDim, f.SeedVectorCount, f.SeedBatchSize); err != nil {
				return fmt.Errorf("invalid seed parameters: %w", err)
			}
		}
	}

	if f.LoadTest {
		if err := validateConnectionFlags(f); err != nil {
			return err
		}
		if f.QPS == "" {
			return fmt.Errorf("--qps is required for load test")
		}
		qpsLevels, err := ParseQPSLevels(f.QPS)
		if err != nil {
			return fmt.Errorf("invalid QPS levels: %w", err)
		}
		if err := validation.ValidateQPSLevels(qpsLevels); err != nil {
			return fmt.Errorf("invalid QPS levels: %w", err)
		}
		if f.Duration != "" {
			duration, err := ParseDuration(f.Duration)
			if err != nil {
				return fmt.Errorf("invalid duration: %w", err)
			}
			if err := validation.ValidateDuration(duration); err != nil {
				return fmt.Errorf("invalid duration: %w", err)
			}
		}
		if f.VectorDim > 0 {
			if err := validation.ValidateVectorDimension(f.VectorDim); err != nil {
				return fmt.Errorf("invalid vector dimension: %w", err)
			}
		}
		if f.MetricType != "" {
			if err := validation.ValidateMetricType(f.MetricType); err != nil {
				return fmt.Errorf("invalid metric type: %w", err)
			}
		}
		if f.Warmup < 0 {
			if err := validation.ValidateWarmupQueries(f.Warmup); err != nil {
				return fmt.Errorf("invalid warmup queries: %w", err)
			}
		}
	}

	if f.Connections != "" {
		connMap, err := ParseConnections(f.Connections)
		if err != nil {
			return fmt.Errorf("invalid --connections format: %w", err)
		}
		for qps, count := range connMap {
			if err := validation.ValidateConnections(count); err != nil {
				return fmt.Errorf("invalid connection count for QPS %d: %w", qps, err)
			}
		}
	}

	return nil
}

// ParseQPSLevels parses QPS string into slice of integers
func ParseQPSLevels(qpsStr string) ([]int, error) {
	if qpsStr == "" {
		return nil, nil
	}

	var result []int
	parts := strings.Split(qpsStr, ",")

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		qps, err := parseInt(part)
		if err != nil {
			return nil, fmt.Errorf("invalid QPS value: %s", part)
		}

		if qps <= 0 {
			return nil, fmt.Errorf("QPS must be positive: %d", qps)
		}

		result = append(result, qps)
	}

	if len(result) == 0 {
		return nil, fmt.Errorf("at least one QPS level is required")
	}

	return result, nil
}

// ParseDuration parses duration string
func ParseDuration(durationStr string) (time.Duration, error) {
	if durationStr == "" {
		return 0, nil
	}
	return time.ParseDuration(durationStr)
}
