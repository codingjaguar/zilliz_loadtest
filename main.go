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
	fmt.Println("Zilliz Cloud Load Test Tool")
	fmt.Println("===========================")
	fmt.Println()
	fmt.Println("What would you like to do?")
	fmt.Println("1. Seed the database")
	fmt.Println("2. Run a read query load test")
	fmt.Println()

	choice := promptInput("Enter your choice (1 or 2): ", true)
	choice = strings.TrimSpace(choice)

	switch choice {
	case "1":
		runSeedDatabase()
	case "2":
		runLoadTestFlow()
	default:
		fmt.Fprintf(os.Stderr, "Error: Invalid choice. Please enter 1 or 2.\n")
		os.Exit(1)
	}
}

func runSeedDatabase() {
	fmt.Println("\nDatabase Seed Configuration")
	fmt.Println("===========================")
	fmt.Println()

	// Ask for API key
	apiKey := promptInput("Enter API Key: ", true)

	// Ask for database URL
	databaseURL := promptInput("Enter Database URL: ", true)

	// Ask for collection name
	collection := promptInput("Enter Collection Name: ", true)

	// Vector dimension is fixed at 768 for seeding
	vectorDim := 768
	fmt.Printf("Vector Dimension: %d (fixed for seed operation)\n", vectorDim)

	// Total vectors is fixed at 2 million
	totalVectors := 2000000
	fmt.Printf("Total Vectors: %d (fixed for seed operation)\n", totalVectors)

	// Confirm before proceeding
	fmt.Println("\nThis will upsert 2,000,000 vectors of 768 dimensions into the collection.")
	confirm := promptInput("Do you want to continue? (yes/no): ", true)
	if strings.ToLower(strings.TrimSpace(confirm)) != "yes" {
		fmt.Println("Seed operation cancelled.")
		return
	}

	// Run the seed operation
	if err := SeedDatabase(apiKey, databaseURL, collection, vectorDim, totalVectors); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func runLoadTestFlow() {
	fmt.Println("\nLoad Test Configuration")
	fmt.Println("=======================")
	fmt.Println()

	// Ask for API key
	apiKey := promptInput("Enter API Key: ", true)

	// Ask for database URL
	databaseURL := promptInput("Enter Database URL: ", true)

	// Ask for collection name
	collection := promptInput("Enter Collection Name: ", true)

	// Ask for vector dimension (required)
	vectorDimInput := promptInput("Enter Vector Dimension: ", true)
	vectorDim, err := strconv.Atoi(vectorDimInput)
	if err != nil || vectorDim <= 0 {
		fmt.Fprintf(os.Stderr, "Error: Vector dimension must be a positive integer\n")
		os.Exit(1)
	}

	// Ask for metric type (required)
	fmt.Println("\nMetric Type options: L2, IP (Inner Product), COSINE")
	metricTypeInput := promptInput("Enter Metric Type: ", true)
	metricType := parseMetricType(metricTypeInput)
	if metricType == "" {
		fmt.Fprintf(os.Stderr, "Error: Invalid metric type. Must be one of: L2, IP, COSINE\n")
		os.Exit(1)
	}

	// Ask for QPS levels (comma-separated or one at a time)
	fmt.Println("\nEnter QPS levels to test (comma-separated, e.g., 100,500,1000):")
	qpsInput := promptInput("QPS Levels: ", true)
	qpsLevels := parseQPSLevels(qpsInput)
	if len(qpsLevels) == 0 {
		fmt.Fprintf(os.Stderr, "Error: At least one QPS level is required\n")
		os.Exit(1)
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

	// Ask if user wants to customize connection counts
	fmt.Println("\nConnection Configuration")
	fmt.Println("========================")
	fmt.Println("By default, connections are estimated using: (QPS × 75ms) / 1000 × 1.5")
	fmt.Println("This is an educated guess based on Zilliz Cloud behavior - actual latency may vary.")
	customize := promptInput("Customize connection counts for each QPS level? (yes/no) [default: no]: ", false)
	customConnections := make(map[int]int)
	
	if strings.ToLower(strings.TrimSpace(customize)) == "yes" {
		for _, qps := range qpsLevels {
			defaultConnections, _ := calculateOptimalConnections(qps)
			connInput := promptInput(fmt.Sprintf("Enter number of connections for %d QPS [default: %d]: ", qps, defaultConnections), false)
			if connInput != "" {
				connCount, err := strconv.Atoi(strings.TrimSpace(connInput))
				if err != nil || connCount < 1 {
					fmt.Printf("Warning: Invalid connection count '%s', using default %d\n", connInput, defaultConnections)
					customConnections[qps] = defaultConnections
				} else {
					customConnections[qps] = connCount
				}
			} else {
				customConnections[qps] = defaultConnections
			}
		}
	}

	// Run load tests in a loop, reusing credentials
	for {
		// Run the load test
		if err := runLoadTest(apiKey, databaseURL, collection, qpsLevels, duration, vectorDim, metricType, customConnections); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		// Ask if user wants to run another test
		fmt.Println("\n" + strings.Repeat("=", 60))
		fmt.Println("Load test completed!")
		fmt.Println(strings.Repeat("=", 60))
		fmt.Println("\nWhat would you like to do?")
		fmt.Println("1. Run another test with the same configuration")
		fmt.Println("2. Run another test with new QPS levels and/or duration")
		fmt.Println("3. Exit")
		choice := promptInput("\nEnter your choice (1, 2, or 3): ", false)
		choice = strings.TrimSpace(choice)

		switch choice {
		case "1":
			// Continue with same configuration
			fmt.Println() // Add spacing before next test
			continue
		case "2":
			// Get new QPS levels
			fmt.Println("\nEnter new QPS levels to test (comma-separated, e.g., 100,500,1000):")
			qpsInput := promptInput("QPS Levels: ", true)
			newQpsLevels := parseQPSLevels(qpsInput)
			if len(newQpsLevels) == 0 {
				fmt.Fprintf(os.Stderr, "Error: At least one QPS level is required\n")
				continue
			}
			qpsLevels = newQpsLevels

			// Get new duration
			durationInput := promptInput("Enter Duration for each QPS test (e.g., 30s, 1m) [default: 30s]: ", false)
			if durationInput != "" {
				parsedDuration, err := time.ParseDuration(durationInput)
				if err != nil {
					fmt.Fprintf(os.Stderr, "Warning: Invalid duration format. Using previous duration: %v\n", duration)
				} else {
					duration = parsedDuration
				}
			}

			// Ask if user wants to customize connection counts for new QPS levels
			fmt.Println("\nConnection Configuration")
			fmt.Println("========================")
			fmt.Println("By default, connections are estimated using: (QPS × 75ms) / 1000 × 1.5")
			fmt.Println("This is an educated guess based on Zilliz Cloud behavior - actual latency may vary.")
			customize := promptInput("Customize connection counts for each QPS level? (yes/no) [default: no]: ", false)
			customConnections = make(map[int]int)
			
			if strings.ToLower(strings.TrimSpace(customize)) == "yes" {
				for _, qps := range qpsLevels {
					defaultConnections, _ := calculateOptimalConnections(qps)
					connInput := promptInput(fmt.Sprintf("Enter number of connections for %d QPS [default: %d]: ", qps, defaultConnections), false)
					if connInput != "" {
						connCount, err := strconv.Atoi(strings.TrimSpace(connInput))
						if err != nil || connCount < 1 {
							fmt.Printf("Warning: Invalid connection count '%s', using default %d\n", connInput, defaultConnections)
							customConnections[qps] = defaultConnections
						} else {
							customConnections[qps] = connCount
						}
					} else {
						customConnections[qps] = defaultConnections
					}
				}
			}

			fmt.Println() // Add spacing before next test
			continue
		case "3", "":
			// Exit
			fmt.Println("Exiting. Goodbye!")
			return
		default:
			fmt.Printf("Invalid choice '%s'. Exiting.\n", choice)
			return
		}
	}
}

func parseMetricType(input string) string {
	input = strings.ToUpper(strings.TrimSpace(input))
	switch input {
	case "L2":
		return "L2"
	case "IP":
		return "IP"
	case "COSINE":
		return "COSINE"
	default:
		return ""
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

func runLoadTest(apiKey, databaseURL, collection string, qpsLevels []int, duration time.Duration, vectorDim int, metricType string, customConnections map[int]int) error {
	fmt.Printf("\n\nStarting Zilliz Cloud Load Test\n")
	fmt.Printf("==============================\n")
	fmt.Printf("Database URL: %s\n", databaseURL)
	fmt.Printf("Collection: %s\n", collection)
	fmt.Printf("Vector Dimension: %d\n", vectorDim)
	fmt.Printf("Metric Type: %s\n", metricType)
	fmt.Printf("Duration per QPS: %v\n", duration)
	fmt.Printf("QPS Levels: %v\n\n", qpsLevels)

	ctx := context.Background()

	// Run tests for each QPS level
	var allResults []TestResult
	for _, qps := range qpsLevels {
		fmt.Printf("\n--- Running test at %d QPS for %v ---\n", qps, duration)
		
		// Get connection count (custom or calculated)
		var connections int
		var explanation string
		if customConn, ok := customConnections[qps]; ok {
			connections = customConn
			explanation = fmt.Sprintf("Using %d connections (user-specified)", connections)
		} else {
			connections, explanation = calculateOptimalConnections(qps)
		}
		
		fmt.Printf("Connection configuration: %s\n", explanation)
		fmt.Printf("Initializing %d client connections...\n", connections)
		
		// Initialize load tester with connections for this QPS
		tester, err := NewLoadTesterWithConnections(apiKey, databaseURL, collection, vectorDim, metricType, connections)
		if err != nil {
			return fmt.Errorf("failed to initialize load tester: %w", err)
		}
		result, err := tester.RunTest(ctx, qps, duration)
		if err != nil {
			fmt.Printf("Error running test at %d QPS: %v\n", qps, err)
			tester.Close()
			continue
		}

		allResults = append(allResults, result)
		tester.Close() // Close after each test to free connections
	}

	// Display results
	fmt.Printf("\n\n==============================\n")
	fmt.Printf("Load Test Results Summary\n")
	fmt.Printf("==============================\n\n")

	displayResults(allResults)

	return nil
}

func displayResults(results []TestResult) {
	fmt.Printf("%-10s | %-12s | %-12s | %-15s\n", "QPS", "P95 (ms)", "P99 (ms)", "Total Queries")
	fmt.Printf("%-10s-+-%-12s-+-%-12s-+-%-15s\n",
		"----------", "------------", "------------", "---------------")

	for _, result := range results {
		fmt.Printf("%-10d | %-12.2f | %-12.2f | %-15d\n",
			result.QPS,
			result.P95Latency,
			result.P99Latency,
			result.TotalQueries)
	}
}
