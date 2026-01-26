package loadtest

import (
	"context"
	"fmt"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// SeedDatabase seeds the database with the specified number of vectors
func SeedDatabase(apiKey, databaseURL, collection string, vectorDim, totalVectors int) error {
	ctx := context.Background()

	milvusClient, err := CreateZillizClient(apiKey, databaseURL)
	if err != nil {
		return fmt.Errorf("failed to create client: %w", err)
	}
	defer milvusClient.Close()

	totalBatches := (totalVectors + DefaultBatchSize - 1) / DefaultBatchSize

	printSeedHeader(collection, vectorDim, totalVectors, DefaultBatchSize)

	startTime := time.Now()
	vectorsInserted := 0

	for batchNum := 0; batchNum < totalBatches; batchNum++ {
		batchStart := batchNum * DefaultBatchSize
		batchEnd := batchStart + DefaultBatchSize
		if batchEnd > totalVectors {
			batchEnd = totalVectors
		}
		currentBatchSize := batchEnd - batchStart

		if err := processSeedBatch(ctx, milvusClient, collection, vectorDim, batchNum+1, totalBatches, batchStart, currentBatchSize, &vectorsInserted, totalVectors, startTime); err != nil {
			return err
		}
	}

	printSeedSummary(vectorsInserted, time.Since(startTime))
	return nil
}

func printSeedHeader(collection string, vectorDim, totalVectors, batchSize int) {
	fmt.Printf("\nStarting database seed operation\n")
	fmt.Printf("================================\n")
	fmt.Printf("Collection: %s\n", collection)
	fmt.Printf("Vector Dimension: %d\n", vectorDim)
	fmt.Printf("Total Vectors: %d\n", totalVectors)
	fmt.Printf("Batch Size: %d\n\n", batchSize)
}

func processSeedBatch(ctx context.Context, milvusClient client.Client, collection string, vectorDim, batchNum, totalBatches, batchStart, currentBatchSize int, vectorsInserted *int, totalVectors int, startTime time.Time) error {
	progressPercent := float64(*vectorsInserted) / float64(totalVectors) * 100
	fmt.Printf("[Progress: %.1f%%] Generating batch %d/%d (%d vectors)...\r",
		progressPercent, batchNum, totalBatches, currentBatchSize)

	generateStart := time.Now()
	vectors := generateBatchVectors(vectorDim, batchStart, currentBatchSize, vectorsInserted, totalVectors, batchNum, totalBatches)
	generateTime := time.Since(generateStart)

	vectorColumn := entity.NewColumnFloatVector("vector", vectorDim, vectors)

	uploadProgressPercent := float64(*vectorsInserted) / float64(totalVectors) * 100
	fmt.Printf("\r[Progress: %.1f%%] Uploading batch %d/%d...",
		uploadProgressPercent, batchNum, totalBatches)

	batchStartTime := time.Now()
	_, err := milvusClient.Insert(ctx, collection, "", vectorColumn)
	if err != nil {
		return fmt.Errorf("failed to insert batch %d: %w", batchNum, err)
	}
	uploadTime := time.Since(batchStartTime)

	*vectorsInserted += currentBatchSize
	totalBatchTime := time.Since(generateStart)
	rate := float64(currentBatchSize) / totalBatchTime.Seconds()

	elapsedTotal := time.Since(startTime)
	avgRate := float64(*vectorsInserted) / elapsedTotal.Seconds()
	remainingVectors := totalVectors - *vectorsInserted
	estimatedTimeRemaining := time.Duration(float64(remainingVectors)/avgRate) * time.Second

	progressPercent = float64(*vectorsInserted) / float64(totalVectors) * 100

	fmt.Printf("\r[Progress: %.1f%%] Batch %d/%d: Inserted %d vectors (Generate: %v, Upload: %v, Total: %v, %.0f vec/s) [ETA: %v]\n",
		progressPercent, batchNum, totalBatches, currentBatchSize,
		generateTime.Round(time.Millisecond), uploadTime.Round(time.Millisecond),
		totalBatchTime.Round(time.Millisecond), rate, estimatedTimeRemaining.Round(time.Second))

	return nil
}

func generateBatchVectors(vectorDim, batchStart, currentBatchSize int, vectorsInserted *int, totalVectors, batchNum, totalBatches int) [][]float32 {
	vectors := make([][]float32, currentBatchSize)
	for i := 0; i < currentBatchSize; i++ {
		vectors[i] = generateSeedingVector(vectorDim, int64(batchStart+i))

		if (i+1)%ProgressUpdateIntervalVectors == 0 {
			progressPercent := float64(*vectorsInserted+i+1) / float64(totalVectors) * 100
			fmt.Printf("\r[Progress: %.1f%%] Generating batch %d/%d (%d/%d vectors)...",
				progressPercent, batchNum, totalBatches, i+1, currentBatchSize)
		}
	}
	return vectors
}

func printSeedSummary(vectorsInserted int, totalElapsed time.Duration) {
	avgRate := float64(vectorsInserted) / totalElapsed.Seconds()

	fmt.Printf("\n================================\n")
	fmt.Printf("Seed operation completed!\n")
	fmt.Printf("Total vectors inserted: %d\n", vectorsInserted)
	fmt.Printf("Total time: %v\n", totalElapsed)
	fmt.Printf("Average rate: %.0f vectors/sec\n", avgRate)
	fmt.Printf("================================\n")
}
