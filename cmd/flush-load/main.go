package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"zilliz-loadtest/internal/loadtest"
	"zilliz-loadtest/internal/logger"
)

func main() {
	logger.Init("INFO", "text")

	apiKey := os.Getenv("ZILLIZ_API_KEY")
	dbURL := os.Getenv("ZILLIZ_DB_URL")
	if apiKey == "" || dbURL == "" {
		// Read from config
		apiKey = "5085a4c690c0047c7a5da79e2543a10b3ef90c52c8649e174994f99b1d1503ba3b338226969287ae414a11d923072420f89abf41"
		dbURL = "https://in05-65465d731639faa.serverless.gcp-us-west1.cloud.zilliz.com"
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, err := loadtest.CreateZillizClient(apiKey, dbURL)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to connect: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	fmt.Println("Connected to Zilliz Cloud. Flushing and loading collection...")

	if err := loadtest.FlushAndLoadCollection(ctx, client, "loadtest_collection"); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to flush/load: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Collection flushed and loaded successfully!")

	// Check stats
	stats, err := client.GetCollectionStatistics(ctx, "loadtest_collection")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to get stats: %v\n", err)
	} else {
		fmt.Printf("Collection statistics: %v\n", stats)
	}
}
