package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

func main() {
	fmt.Println("Zilliz Cloud Load Test Configuration")
	fmt.Println("====================================")
	fmt.Println()

	// Ask for API key
	apiKey := promptInput("Enter API Key: ", true)
	
	// Ask for database URL
	databaseURL := promptInput("Enter Database URL: ", true)
	
	// Ask for collection name
	collection := promptInput("Enter Collection Name: ", true)
	
	// Ask for QPS levels (comma-separated or one at a time)
	fmt.Println("\nEnter QPS levels to test (comma-separated, e.g., 100,500,1000):")
	qpsInput := promptInput("QPS Levels: ", true)
	qpsLevels := parseQPSLevels(qpsInput)
	if len(qpsLevels) == 0 {
		fmt.Fprintf(os.Stderr, "Error: At least one QPS level is required\n")
		os.Exit(1)
	}
	
	// Ask for level parameter (1-10)
	levelInput := promptInput("Enter Level (1-10, where 10 optimizes for recall): ", false)
	level := 5 // default
	if levelInput != "" {
		parsedLevel, err := strconv.Atoi(levelInput)
		if err != nil || parsedLevel < 1 || parsedLevel > 10 {
			fmt.Fprintf(os.Stderr, "Error: Level must be an integer between 1 and 10. Using default: 5\n")
		} else {
			level = parsedLevel
		}
	}
	
	// Ask for duration (optional)
	durationInput := promptInput("Enter Duration for each QPS test (e.g., 30s, 1m) [default: 30s]: ", false)
	duration := 30 * time.Second
	if durationInput != "" {
		parsedDuration, err := time.ParseDuration(durationInput)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: Invalid duration format. Using default: 30s\n")
		} else {
			duration = parsedDuration
		}
	}
	
	// Ask for vector dimension (optional)
	vectorDimInput := promptInput("Enter Vector Dimension [default: 128]: ", false)
	vectorDim := 128
	if vectorDimInput != "" {
		parsedDim, err := strconv.Atoi(vectorDimInput)
		if err != nil || parsedDim <= 0 {
			fmt.Fprintf(os.Stderr, "Warning: Invalid vector dimension. Using default: 128\n")
		} else {
			vectorDim = parsedDim
		}
	}

	// Run the load test
	if err := runLoadTest(apiKey, databaseURL, collection, qpsLevels, level, duration, vectorDim); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func promptInput(prompt string, required bool) string {
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print(prompt)
		input, err := reader.ReadString('\n')
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
			os.Exit(1)
		}
		input = strings.TrimSpace(input)
		if input != "" || !required {
			return input
		}
		fmt.Println("This field is required. Please enter a value.")
	}
}

func parseQPSLevels(input string) []int {
	var qpsLevels []int
	parts := strings.Split(input, ",")
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		qps, err := strconv.Atoi(part)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: Invalid QPS value '%s', skipping\n", part)
			continue
		}
		if qps <= 0 {
			fmt.Fprintf(os.Stderr, "Warning: QPS must be positive, skipping '%d'\n", qps)
			continue
		}
		qpsLevels = append(qpsLevels, qps)
	}
	return qpsLevels
}

func runLoadTest(apiKey, databaseURL, collection string, qpsLevels []int, level int, duration time.Duration, vectorDim int) error {
	fmt.Printf("\n\nStarting Zilliz Cloud Load Test\n")
	fmt.Printf("==============================\n")
	fmt.Printf("Database URL: %s\n", databaseURL)
	fmt.Printf("Collection: %s\n", collection)
	fmt.Printf("Level: %d\n", level)
	fmt.Printf("Duration per QPS: %v\n", duration)
	fmt.Printf("QPS Levels: %v\n\n", qpsLevels)

	ctx := context.Background()

	// Initialize load tester
	tester, err := NewLoadTester(apiKey, databaseURL, collection, level, vectorDim)
	if err != nil {
		return fmt.Errorf("failed to initialize load tester: %w", err)
	}
	defer tester.Close()

	// Run tests for each QPS level
	var allResults []TestResult
	for _, qps := range qpsLevels {
		fmt.Printf("\n--- Running test at %d QPS for %v ---\n", qps, duration)
		
		result, err := tester.RunTest(ctx, qps, duration)
		if err != nil {
			fmt.Printf("Error running test at %d QPS: %v\n", qps, err)
			continue
		}

		allResults = append(allResults, result)
	}

	// Display results
	fmt.Printf("\n\n==============================\n")
	fmt.Printf("Load Test Results Summary\n")
	fmt.Printf("==============================\n\n")
	
	displayResults(allResults)

	return nil
}

func displayResults(results []TestResult) {
	fmt.Printf("%-10s | %-12s | %-12s | %-15s | %-15s\n", "QPS", "P95 (ms)", "P99 (ms)", "Avg Recall", "Total Queries")
	fmt.Printf("%-10s-+-%-12s-+-%-12s-+-%-15s-+-%-15s\n", 
		"----------", "------------", "------------", "---------------", "---------------")
	
	for _, result := range results {
		fmt.Printf("%-10d | %-12.2f | %-12.2f | %-15.4f | %-15d\n",
			result.QPS,
			result.P95Latency,
			result.P99Latency,
			result.AvgRecall,
			result.TotalQueries)
	}
}
