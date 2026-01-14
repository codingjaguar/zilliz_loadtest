package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"sync"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

type LoadTester struct {
	client     client.Client
	collection string
	level      int
	vectorDim  int
	metricType entity.MetricType
}

type TestResult struct {
	QPS          int
	P95Latency   float64 // in milliseconds
	P99Latency   float64 // in milliseconds
	AvgRecall    float64
	TotalQueries int
	Errors       int
}

type QueryResult struct {
	Latency time.Duration
	Recall  float64
	Error   error
}

// createZillizClient creates a Zilliz Cloud client with API key authentication
func createZillizClient(apiKey, databaseURL string) (client.Client, error) {
	ctx := context.Background()

	// Try the newer client.NewClient approach first
	milvusClient, err := client.NewClient(
		ctx,
		client.Config{
			Address:       databaseURL,
			APIKey:        apiKey,
			EnableTLSAuth: true,
		},
	)

	// If that fails, try the alternative approach with NewGrpcClient
	if err != nil {
		milvusClient, err = client.NewGrpcClient(ctx, databaseURL)
		if err != nil {
			return nil, fmt.Errorf("failed to create client: %w", err)
		}
		// Try SetToken method (common in Zilliz Cloud SDK)
		if tokenClient, ok := milvusClient.(interface{ SetToken(string) error }); ok {
			if err := tokenClient.SetToken(apiKey); err != nil {
				// If SetToken fails, try SetApiKey
				if apiKeyClient, ok := milvusClient.(interface{ SetApiKey(string) }); ok {
					apiKeyClient.SetApiKey(apiKey)
				}
			}
		} else if apiKeyClient, ok := milvusClient.(interface{ SetApiKey(string) }); ok {
			apiKeyClient.SetApiKey(apiKey)
		}
	}

	return milvusClient, nil
}

func NewLoadTester(apiKey, databaseURL, collection string, level int, vectorDim int, metricTypeStr string) (*LoadTester, error) {
	milvusClient, err := createZillizClient(apiKey, databaseURL)
	if err != nil {
		return nil, err
	}

	// Parse metric type
	var metricType entity.MetricType
	switch metricTypeStr {
	case "L2":
		metricType = entity.L2
	case "IP":
		metricType = entity.IP
	case "COSINE":
		metricType = entity.COSINE
	default:
		return nil, fmt.Errorf("unsupported metric type: %s", metricTypeStr)
	}

	return &LoadTester{
		client:     milvusClient,
		collection: collection,
		level:      level,
		vectorDim:  vectorDim,
		metricType: metricType,
	}, nil
}

func (lt *LoadTester) Close() {
	if lt.client != nil {
		lt.client.Close()
	}
}

func (lt *LoadTester) RunTest(ctx context.Context, targetQPS int, duration time.Duration) (TestResult, error) {
	// Calculate interval between requests to achieve target QPS
	interval := time.Second / time.Duration(targetQPS)

	// Create context with timeout
	testCtx, cancel := context.WithTimeout(ctx, duration)
	defer cancel()

	var wg sync.WaitGroup
	resultsChan := make(chan QueryResult, targetQPS*10) // Buffer for results

	startTime := time.Now()
	queryCount := 0
	var mu sync.Mutex

	// Start query workers
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	go func() {
		for {
			select {
			case <-testCtx.Done():
				return
			case <-ticker.C:
				mu.Lock()
				queryCount++
				currentQuery := queryCount
				mu.Unlock()

				wg.Add(1)
				go func(queryNum int) {
					defer wg.Done()
					result := lt.executeQuery(testCtx, queryNum)
					select {
					case resultsChan <- result:
					case <-testCtx.Done():
					}
				}(currentQuery)
			}
		}
	}()

	// Wait for test duration
	<-testCtx.Done()

	// Wait for all queries to complete (with timeout)
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(10 * time.Second):
		fmt.Printf("Warning: Some queries may still be running\n")
	}

	close(resultsChan)
	elapsed := time.Since(startTime)

	// Collect results
	var latencies []float64
	var recalls []float64
	errors := 0

	for result := range resultsChan {
		if result.Error != nil {
			errors++
			continue
		}
		latencies = append(latencies, float64(result.Latency.Milliseconds()))
		if result.Recall > 0 {
			recalls = append(recalls, result.Recall)
		}
	}

	// Calculate percentiles
	sort.Float64s(latencies)
	p95 := calculatePercentile(latencies, 95)
	p99 := calculatePercentile(latencies, 99)

	// Calculate average recall
	avgRecall := 0.0
	if len(recalls) > 0 {
		sum := 0.0
		for _, r := range recalls {
			sum += r
		}
		avgRecall = sum / float64(len(recalls))
	}

	actualQPS := float64(len(latencies)) / elapsed.Seconds()
	fmt.Printf("Completed: %d queries in %v (actual QPS: %.2f, errors: %d)\n",
		len(latencies), elapsed, actualQPS, errors)

	return TestResult{
		QPS:          targetQPS,
		P95Latency:   p95,
		P99Latency:   p99,
		AvgRecall:    avgRecall,
		TotalQueries: len(latencies),
		Errors:       errors,
	}, nil
}

// CustomSearchParam implements SearchParam interface with level and enable_recall_calculation
type CustomSearchParam struct {
	level                   int
	enableRecallCalculation bool
}

func (c *CustomSearchParam) Params() map[string]interface{} {
	params := make(map[string]interface{})
	params["level"] = c.level
	// Try both boolean and string "true" to see which format works
	if c.enableRecallCalculation {
		params["enable_recall_calculation"] = true
		// Also try as string in case the server expects it
		// params["enable_recall_calculation"] = "true"
	}
	return params
}

func (c *CustomSearchParam) AddRadius(radius float64) {
	// Not used for AUTOINDEX
}

func (c *CustomSearchParam) AddRangeFilter(rangeFilter float64) {
	// Not used for AUTOINDEX
}

func (lt *LoadTester) executeQuery(ctx context.Context, queryNum int) QueryResult {
	// Generate a random query vector (dummy vector for testing)
	// In a real scenario, you'd use actual query vectors from your dataset
	queryVector := generateRandomVector(lt.vectorDim)

	// Create search parameters with level and enable_recall_calculation
	// The level parameter (1-10) optimizes for recall vs latency
	// Level 10 optimizes for recall at the expense of latency, level 1 optimizes for latency
	// enable_recall_calculation tells Zilliz to calculate and return the recall rate
	searchParams := &CustomSearchParam{
		level:                   lt.level,
		enableRecallCalculation: true,
	}

	// Debug: Verify search parameters are being set correctly
	if queryNum == 1 {
		params := searchParams.Params()
		fmt.Printf("\n=== DEBUG: Search Parameters ===\n")
		fmt.Printf("Params map: %+v\n", params)
		fmt.Printf("Level: %d\n", params["level"])
		fmt.Printf("enable_recall_calculation: %v\n", params["enable_recall_calculation"])
		fmt.Printf("================================\n\n")
	}

	// Measure latency only for the search operation (not including vector generation)
	// Use high-precision timing for accurate measurements
	searchStart := time.Now()

	// Execute search
	// When enable_recall_calculation is true, Zilliz should return recall automatically
	// Try both with and without "recalls" in output fields - it may be returned automatically
	// Note: Recall calculation requires ground truth data. Random query vectors won't have
	// ground truth, so recall may always be 0 or not calculated.
	searchResults, err := lt.client.Search(
		ctx,
		lt.collection,
		[]string{}, // partition names (empty for all partitions)
		"",         // expr (empty for no filter)
		[]string{}, // output fields - try empty first, recall should be returned automatically
		[]entity.Vector{entity.FloatVector(queryVector)},
		"vector",      // vector field name
		lt.metricType, // metric type
		10,            // topK
		searchParams,
	)

	// Measure latency immediately after Search() returns
	// This includes network round-trip and server processing time
	latency := time.Since(searchStart)

	if err != nil {
		return QueryResult{
			Latency: latency,
			Error:   err,
		}
	}

	// Extract recall from search results
	// When enable_recall_calculation is true, Zilliz returns the recall in the response
	// According to docs: https://docs.zilliz.com/docs/tune-recall-rate#tune-recall-rate
	// The recall is returned as "recalls" (plural) - it may be in Fields or as metadata
	recall := 0.0

	if len(searchResults) > 0 {
		result := searchResults[0]

		// Debug: Print full search result structure
		if queryNum == 1 {
			fmt.Printf("\n=== DEBUG: Full Search Result Structure ===\n")
			fmt.Printf("Number of search results: %d\n", len(searchResults))
			fmt.Printf("ResultCount: %d\n", result.ResultCount)
			fmt.Printf("Scores: %v (length: %d)\n", result.Scores, len(result.Scores))
			fmt.Printf("IDs is nil: %v\n", result.IDs == nil)
			if result.IDs != nil {
				fmt.Printf("IDs type: %T, Len: %d\n", result.IDs, result.IDs.Len())
			}
			fmt.Printf("Fields is nil: %v\n", result.Fields == nil)
			if result.Fields != nil {
				fmt.Printf("Fields type: %T\n", result.Fields)
				// Try to see what columns are available - check common names
				testColumns := []string{"recalls", "recall", "vector", "id"}
				for _, colName := range testColumns {
					if col := result.Fields.GetColumn(colName); col != nil {
						fmt.Printf("  Found column '%s': type %T, length %d\n", colName, col, col.Len())
					}
				}
			}
			fmt.Printf("Err: %v\n", result.Err)
			fmt.Printf("SearchResult type: %T\n", result)
			fmt.Printf("==========================================\n\n")
		}

		// Try to get recall from Fields
		// First, try with "recalls" in output fields if Fields is empty or doesn't contain recall
		if result.Fields == nil {
			if queryNum == 1 {
				fmt.Printf("\n=== DEBUG: Fields is nil, trying search with 'recalls' in output fields ===\n")
			}
			// Retry with "recalls" explicitly requested as output field
			searchResultsWithRecalls, retryErr := lt.client.Search(
				ctx,
				lt.collection,
				[]string{},
				"",
				[]string{"recalls"}, // Explicitly request recalls as output field
				[]entity.Vector{entity.FloatVector(queryVector)},
				"vector",
				lt.metricType,
				10,
				searchParams,
			)
			if retryErr == nil && len(searchResultsWithRecalls) > 0 {
				result = searchResultsWithRecalls[0]
			}
		}

		if result.Fields != nil {
			// Debug: List all available columns in Fields
			if queryNum == 1 {
				fmt.Printf("\n=== DEBUG: All Columns in Fields ===\n")
				// Fields doesn't have a direct method to list all columns, but we can try common names
				// or check the structure
				fmt.Printf("Fields type: %T\n", result.Fields)
				// Try to see if there's a way to iterate columns
				// For now, just try the known column names
				fmt.Printf("Trying to get column 'recalls'...\n")
			}

			// Try "recalls" first (as shown in Python docs)
			recallColumn := result.Fields.GetColumn("recalls")
			if recallColumn == nil {
				if queryNum == 1 {
					fmt.Printf("Column 'recalls' not found, trying 'recall'...\n")
				}
				// Fallback to "recall" (singular) in case SDK version differs
				recallColumn = result.Fields.GetColumn("recall")
			}

			if recallColumn != nil {
				// Debug: Print recall column details for first query
				if queryNum == 1 {
					fmt.Printf("\n=== DEBUG: Recall Extraction (Query #1) ===\n")
					fmt.Printf("Recall column type: %T\n", recallColumn)
					fmt.Printf("Recall column name: %s\n", recallColumn.Name())
					fmt.Printf("Recall column length: %d\n", recallColumn.Len())
				}

				// The recall column is returned as ColumnDynamic (JSON field)
				// Extract the first recall value (typically one per query, but may have one per result)
				if dynamicColumn, ok := recallColumn.(*entity.ColumnDynamic); ok {
					if dynamicColumn.Len() > 0 {
						if queryNum == 1 {
							fmt.Printf("ColumnDynamic found, length: %d\n", dynamicColumn.Len())
						}

						// Try to get as double (float64) - this should work for numeric JSON values
						if val, err := dynamicColumn.GetAsDouble(0); err == nil {
							recall = val
							if queryNum == 1 {
								fmt.Printf("Got recall via GetAsDouble(0): %f\n", recall)
							}
						} else {
							if queryNum == 1 {
								fmt.Printf("GetAsDouble(0) error: %v\n", err)
							}

							// Try GetAsString to see the raw JSON value
							if strVal, err := dynamicColumn.GetAsString(0); err == nil {
								if queryNum == 1 {
									fmt.Printf("Got value as string: %s\n", strVal)
								}
								// Try to parse as float
								if parsedVal, err := strconv.ParseFloat(strVal, 64); err == nil {
									recall = parsedVal
									if queryNum == 1 {
										fmt.Printf("Parsed string to float64: %f\n", recall)
									}
								}
							} else if queryNum == 1 {
								fmt.Printf("GetAsString(0) error: %v\n", err)
							}

							// Fallback: try to get as interface{} and convert
							if val, err := dynamicColumn.Get(0); err == nil {
								if queryNum == 1 {
									fmt.Printf("Got value via Get(0): %v (type: %T)\n", val, val)
								}
								switch v := val.(type) {
								case float64:
									recall = v
									if queryNum == 1 {
										fmt.Printf("Extracted as float64: %f\n", recall)
									}
								case float32:
									recall = float64(v)
									if queryNum == 1 {
										fmt.Printf("Extracted as float32->float64: %f\n", recall)
									}
								case int64:
									recall = float64(v) / 100.0 // Convert from percentage
									if queryNum == 1 {
										fmt.Printf("Extracted as int64 (percentage): %f\n", recall)
									}
								case int:
									recall = float64(v) / 100.0 // Convert from percentage
									if queryNum == 1 {
										fmt.Printf("Extracted as int (percentage): %f\n", recall)
									}
								case string:
									// Try to parse string as float
									if parsedVal, err := strconv.ParseFloat(v, 64); err == nil {
										recall = parsedVal
										if queryNum == 1 {
											fmt.Printf("Parsed string to float64: %f\n", recall)
										}
									}
								default:
									if queryNum == 1 {
										fmt.Printf("Unknown type for recall value: %T, value: %v\n", v, v)
									}
								}
							} else {
								if queryNum == 1 {
									fmt.Printf("Get(0) error: %v\n", err)
									// Try accessing underlying ColumnJSONBytes directly
									fmt.Printf("Trying to access underlying ColumnJSONBytes...\n")

									// ColumnDynamic wraps ColumnJSONBytes - try to access the raw JSON
									// Check if we can get the underlying JSON bytes
									if jsonBytesCol, ok := recallColumn.(interface{ Data() [][]byte }); ok {
										jsonData := jsonBytesCol.Data()
										if len(jsonData) > 0 {
											fmt.Printf("Found JSON bytes, length: %d\n", len(jsonData))
											// Check all JSON values, not just the first
											for i := 0; i < len(jsonData) && i < 10; i++ {
												if len(jsonData[i]) > 0 {
													maxLen := 500
													if len(jsonData[i]) < maxLen {
														maxLen = len(jsonData[i])
													}
													fmt.Printf("JSON[%d] (full): %s\n", i, string(jsonData[i]))
												} else {
													fmt.Printf("JSON[%d]: empty\n", i)
												}
											}
										}
									}

									// Try ValueByIdx which might work for JSONBytes - check all indices
									if jsonBytesCol, ok := recallColumn.(interface{ ValueByIdx(int) ([]byte, error) }); ok {
										// Try all indices to find where the recall data is
										for idx := 0; idx < dynamicColumn.Len() && idx < 10; idx++ {
											if bytes, err := jsonBytesCol.ValueByIdx(idx); err == nil {
												fmt.Printf("JSON bytes[%d]: %s\n", idx, string(bytes))
												// Try to parse as JSON to extract recall
												var jsonVal interface{}
												if err := json.Unmarshal(bytes, &jsonVal); err == nil {
													fmt.Printf("Parsed JSON[%d]: %v (type: %T)\n", idx, jsonVal, jsonVal)

													// Try to extract recall from JSON - check various structures
													if recallMap, ok := jsonVal.(map[string]interface{}); ok {
														// Check for various possible field names
														for _, fieldName := range []string{"recall", "recalls", "recall_rate", "recallRate"} {
															if r, exists := recallMap[fieldName]; exists {
																fmt.Printf("Found field '%s' in JSON[%d]: %v (type: %T)\n", fieldName, idx, r, r)
																if rFloat, ok := r.(float64); ok {
																	recall = rFloat
																	fmt.Printf("Extracted recall from JSON map field '%s': %f\n", fieldName, recall)
																	break
																} else if rArray, ok := r.([]interface{}); ok && len(rArray) > 0 {
																	if rFloat, ok := rArray[0].(float64); ok {
																		recall = rFloat
																		fmt.Printf("Extracted recall from JSON array in field '%s': %f\n", fieldName, recall)
																		break
																	}
																}
															}
														}
														// If map is not empty, print all keys
														if len(recallMap) > 0 {
															fmt.Printf("JSON[%d] map keys: ", idx)
															for k := range recallMap {
																fmt.Printf("%s ", k)
															}
															fmt.Printf("\n")
														}
													} else if rFloat, ok := jsonVal.(float64); ok {
														recall = rFloat
														fmt.Printf("Extracted recall as direct float64 from JSON[%d]: %f\n", idx, recall)
													} else if rArray, ok := jsonVal.([]interface{}); ok && len(rArray) > 0 {
														if rFloat, ok := rArray[0].(float64); ok {
															recall = rFloat
															fmt.Printf("Extracted recall from JSON array[0] in JSON[%d]: %f\n", idx, recall)
														}
													}
												} else {
													fmt.Printf("JSON unmarshal error for JSON[%d]: %v\n", idx, err)
												}
											} else {
												fmt.Printf("ValueByIdx(%d) error: %v\n", idx, err)
											}
										}
									}
								}
							}
						}

						// Try getting multiple values to see the pattern
						if queryNum == 1 && dynamicColumn.Len() > 1 {
							fmt.Printf("Trying to get all recall values:\n")
							for i := 0; i < dynamicColumn.Len() && i < 3; i++ {
								if val, err := dynamicColumn.Get(i); err == nil {
									fmt.Printf("  recalls[%d]: %v (type: %T)\n", i, val, val)
								} else {
									fmt.Printf("  recalls[%d]: error - %v\n", i, err)
								}
								if strVal, err := dynamicColumn.GetAsString(i); err == nil {
									fmt.Printf("  recalls[%d] as string: %s\n", i, strVal)
								}
							}
						}
					} else {
						if queryNum == 1 {
							fmt.Printf("ColumnDynamic length is 0\n")
						}
					}
				} else if floatColumn, ok := recallColumn.(*entity.ColumnFloat); ok {
					if floatColumn.Len() > 0 {
						if val, err := floatColumn.ValueByIdx(0); err == nil {
							recall = float64(val)
							if queryNum == 1 {
								fmt.Printf("Got recall from ColumnFloat: %f\n", recall)
							}
						}
					}
				} else if doubleColumn, ok := recallColumn.(*entity.ColumnDouble); ok {
					if doubleColumn.Len() > 0 {
						if val, err := doubleColumn.GetAsDouble(0); err == nil {
							recall = val
							if queryNum == 1 {
								fmt.Printf("Got recall from ColumnDouble: %f\n", recall)
							}
						}
					}
				} else if int64Column, ok := recallColumn.(*entity.ColumnInt64); ok {
					// Some SDK versions might return as int64 (0-100 scale)
					if int64Column.Len() > 0 {
						if val, err := int64Column.Get(0); err == nil {
							if intVal, ok := val.(int64); ok {
								recall = float64(intVal) / 100.0 // Convert from percentage
								if queryNum == 1 {
									fmt.Printf("Got recall from ColumnInt64: %f\n", recall)
								}
							}
						}
					}
				} else {
					if queryNum == 1 {
						fmt.Printf("Unknown recall column type: %T\n", recallColumn)
					}
				}

				if queryNum == 1 {
					fmt.Printf("Final recall value: %f\n", recall)
					if recall == 0.0 {
						fmt.Printf("\n⚠️  WARNING: Recall is 0.0 - Possible reasons:\n")
						fmt.Printf("  1. ⚠️  CRITICAL: Random query vectors don't have ground truth!\n")
						fmt.Printf("     Recall calculation requires comparing results against known ground truth.\n")
						fmt.Printf("     With random vectors, Zilliz cannot calculate recall because there's no\n")
						fmt.Printf("     reference to compare against. Use actual query vectors from your dataset.\n")
						fmt.Printf("  2. The enable_recall_calculation parameter may not be working\n")
						fmt.Printf("  3. The recall data may be in a different format or location\n")
						fmt.Printf("  4. The Go SDK may not fully support recall calculation yet (feature is in Public Preview)\n")
						fmt.Printf("  5. The parameter might need to be nested differently in the request\n")
					}
					fmt.Printf("==========================================\n\n")
				}
			} else {
				if queryNum == 1 {
					fmt.Printf("\n=== DEBUG: Recall Extraction (Query #1) ===\n")
					fmt.Printf("recallColumn is nil!\n")
					fmt.Printf("The 'recalls' column was not found in the search results.\n")
					fmt.Printf("\nPossible issues:\n")
					fmt.Printf("  1. ⚠️  Random query vectors don't have ground truth - recall cannot be calculated\n")
					fmt.Printf("  2. enable_recall_calculation parameter may not be working\n")
					fmt.Printf("  3. Recall might be returned in a different location (not in Fields)\n")
					fmt.Printf("  4. Go SDK may not fully support this feature (Public Preview)\n")
					fmt.Printf("\nTroubleshooting steps:\n")
					fmt.Printf("  - Verify the parameter is being sent: check server logs or use a network proxy\n")
					fmt.Printf("  - Try with actual query vectors that have ground truth data\n")
					fmt.Printf("  - Check if recall is in result metadata or a different field\n")
					fmt.Printf("  - Contact Zilliz support if the feature should work but doesn't\n")
					fmt.Printf("==========================================\n\n")
				}
			}
		} else {
			if queryNum == 1 {
				fmt.Printf("\n=== DEBUG: Recall Extraction (Query #1) ===\n")
				fmt.Printf("result.Fields is nil or empty!\n")
				fmt.Printf("This suggests recall is not being returned in Fields.\n")
				fmt.Printf("Recall might be:\n")
				fmt.Printf("  - In result metadata (check result struct fields)\n")
				fmt.Printf("  - Not calculated due to missing ground truth (random vectors)\n")
				fmt.Printf("  - Not supported by the Go SDK version\n")
				fmt.Printf("==========================================\n\n")
			}
		}

		// Also check if recall might be in the SearchResult structure itself (not in Fields)
		// Some SDKs might return it as a separate field
		if queryNum == 1 && recall == 0.0 {
			fmt.Printf("\n=== DEBUG: Checking SearchResult structure for recall ===\n")
			fmt.Printf("SearchResult type: %T\n", result)
			// Try to use reflection or check if there are any methods that might contain recall
			// The SDK might have a GetRecall() method or similar
			fmt.Printf("Note: If recall is calculated, it should appear above in the Fields extraction.\n")
			fmt.Printf("If not found, the most likely cause is missing ground truth data.\n")
			fmt.Printf("==========================================\n\n")
		}
	}

	return QueryResult{
		Latency: latency,
		Recall:  recall,
	}
}

func generateRandomVector(dim int) []float32 {
	// Generate a pseudo-random vector for testing
	// In production, use actual query vectors from your dataset
	vector := make([]float32, dim)
	for i := range vector {
		// Use a simple pattern that varies per dimension
		vector[i] = float32((i*7+13)%100) / 100.0
	}
	return vector
}

func generateSeedingVector(dim int, seed int64) []float32 {
	// Generate a vector with better distribution for seeding
	// Uses seed to ensure different vectors for each index
	vector := make([]float32, dim)
	for i := range vector {
		// Create a more varied pattern using the seed
		value := float32((int64(i)*7919 + seed*9829) % 10000)
		vector[i] = value / 10000.0
	}
	return vector
}

func calculatePercentile(sortedData []float64, percentile int) float64 {
	if len(sortedData) == 0 {
		return 0
	}
	index := float64(percentile) / 100.0 * float64(len(sortedData))
	upper := int(math.Ceil(index)) - 1
	lower := int(math.Floor(index)) - 1

	if upper < 0 {
		upper = 0
	}
	if lower < 0 {
		lower = 0
	}
	if upper >= len(sortedData) {
		upper = len(sortedData) - 1
	}
	if lower >= len(sortedData) {
		lower = len(sortedData) - 1
	}

	if upper == lower {
		return sortedData[upper]
	}

	weight := index - float64(lower+1)
	return sortedData[lower]*(1-weight) + sortedData[upper]*weight
}

// SeedDatabase seeds the database with the specified number of vectors
func SeedDatabase(apiKey, databaseURL, collection string, vectorDim, totalVectors int) error {
	ctx := context.Background()

	// Create client
	milvusClient, err := createZillizClient(apiKey, databaseURL)
	if err != nil {
		return fmt.Errorf("failed to create client: %w", err)
	}
	defer milvusClient.Close()

	// Batch size for efficient inserts (reduced to avoid gRPC message size limits and throttling)
	// With 768 dimensions, each vector is ~3KB, so 15,000 vectors = ~45MB (well under 64MB limit)
	batchSize := 15000
	totalBatches := (totalVectors + batchSize - 1) / batchSize

	fmt.Printf("\nStarting database seed operation\n")
	fmt.Printf("================================\n")
	fmt.Printf("Collection: %s\n", collection)
	fmt.Printf("Vector Dimension: %d\n", vectorDim)
	fmt.Printf("Total Vectors: %d\n", totalVectors)
	fmt.Printf("Batch Size: %d\n\n", batchSize)

	startTime := time.Now()
	vectorsInserted := 0

	for batchNum := 0; batchNum < totalBatches; batchNum++ {
		batchStart := batchNum * batchSize
		batchEnd := batchStart + batchSize
		if batchEnd > totalVectors {
			batchEnd = totalVectors
		}
		currentBatchSize := batchEnd - batchStart

		// Show progress before generating vectors
		progressPercent := float64(vectorsInserted) / float64(totalVectors) * 100
		fmt.Printf("[Progress: %.1f%%] Generating batch %d/%d (%d vectors)...\r",
			progressPercent, batchNum+1, totalBatches, currentBatchSize)

		// Generate vectors for this batch
		generateStart := time.Now()
		vectors := make([][]float32, currentBatchSize)
		for i := 0; i < currentBatchSize; i++ {
			// Use batchStart + i as seed to ensure unique vectors
			vectors[i] = generateSeedingVector(vectorDim, int64(batchStart+i))

			// Show progress every 5000 vectors during generation
			if (i+1)%5000 == 0 {
				progressPercent := float64(vectorsInserted+i+1) / float64(totalVectors) * 100
				fmt.Printf("\r[Progress: %.1f%%] Generating batch %d/%d (%d/%d vectors)...",
					progressPercent, batchNum+1, totalBatches, i+1, currentBatchSize)
			}
		}
		generateTime := time.Since(generateStart)

		// Create vector column
		vectorColumn := entity.NewColumnFloatVector("vector", vectorDim, vectors)

		// Insert the batch (using Insert instead of Upsert since autoID is enabled)
		batchStartTime := time.Now()
		uploadProgressPercent := float64(vectorsInserted) / float64(totalVectors) * 100
		fmt.Printf("\r[Progress: %.1f%%] Uploading batch %d/%d...",
			uploadProgressPercent, batchNum+1, totalBatches)

		_, err := milvusClient.Insert(ctx, collection, "", vectorColumn)
		if err != nil {
			return fmt.Errorf("failed to insert batch %d: %w", batchNum+1, err)
		}

		vectorsInserted += currentBatchSize
		uploadTime := time.Since(batchStartTime)
		totalBatchTime := time.Since(generateStart)
		rate := float64(currentBatchSize) / totalBatchTime.Seconds()

		// Calculate estimated time remaining
		elapsedTotal := time.Since(startTime)
		avgRate := float64(vectorsInserted) / elapsedTotal.Seconds()
		remainingVectors := totalVectors - vectorsInserted
		estimatedTimeRemaining := time.Duration(float64(remainingVectors)/avgRate) * time.Second

		progressPercent = float64(vectorsInserted) / float64(totalVectors) * 100

		// Print detailed progress after each batch
		fmt.Printf("\r[Progress: %.1f%%] Batch %d/%d: Inserted %d vectors (Generate: %v, Upload: %v, Total: %v, %.0f vec/s) [ETA: %v]\n",
			progressPercent, batchNum+1, totalBatches, currentBatchSize,
			generateTime.Round(time.Millisecond), uploadTime.Round(time.Millisecond),
			totalBatchTime.Round(time.Millisecond), rate, estimatedTimeRemaining.Round(time.Second))
	}

	totalElapsed := time.Since(startTime)
	avgRate := float64(vectorsInserted) / totalElapsed.Seconds()

	fmt.Printf("\n================================\n")
	fmt.Printf("Seed operation completed!\n")
	fmt.Printf("Total vectors inserted: %d\n", vectorsInserted)
	fmt.Printf("Total time: %v\n", totalElapsed)
	fmt.Printf("Average rate: %.0f vectors/sec\n", avgRate)
	fmt.Printf("================================\n")

	return nil
}
