package loadtest

import (
	"context"
	"fmt"

	"zilliz-loadtest/internal/datasource"
	"zilliz-loadtest/internal/logger"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// RecallMetrics holds both mathematical and business recall measurements
type RecallMetrics struct {
	MathematicalRecall float64 // Recall vs brute force search
	BusinessRecall     float64 // Recall vs ground truth qrels
	QueriesTested      int
}

// RecallCalculator computes recall metrics
type RecallCalculator struct {
	client      client.Client
	collection  string
	vectorField string
	idField     string // Primary key field name
	qrels       datasource.Qrels
}

// NewRecallCalculator creates a new recall calculator
func NewRecallCalculator(c client.Client, collection string, qrels datasource.Qrels) *RecallCalculator {
	return &RecallCalculator{
		client:      c,
		collection:  collection,
		vectorField: "vector", // default field name
		idField:     "id",     // default ID field name
		qrels:       qrels,
	}
}

// NewRecallCalculatorWithField creates a new recall calculator with custom vector field name
func NewRecallCalculatorWithField(c client.Client, collection, vectorField string, qrels datasource.Qrels) *RecallCalculator {
	return &RecallCalculator{
		client:      c,
		collection:  collection,
		vectorField: vectorField,
		idField:     "id", // default ID field name
		qrels:       qrels,
	}
}

// NewRecallCalculatorWithFields creates a new recall calculator with custom field names
func NewRecallCalculatorWithFields(c client.Client, collection, vectorField, idField string, qrels datasource.Qrels) *RecallCalculator {
	return &RecallCalculator{
		client:      c,
		collection:  collection,
		vectorField: vectorField,
		idField:     idField,
		qrels:       qrels,
	}
}

// CalculateRecall computes both types of recall for given queries
func (rc *RecallCalculator) CalculateRecall(
	ctx context.Context,
	queries []datasource.CohereQuery,
	topK int,
	metricType entity.MetricType,
	searchParams entity.SearchParam,
) (*RecallMetrics, error) {
	if len(queries) == 0 {
		return nil, fmt.Errorf("no queries provided")
	}

	logger.Info("Calculating recall metrics",
		"queries", len(queries),
		"topK", topK)

	var mathRecallSum float64
	var bizRecallSum float64
	queriesWithQrels := 0

	// Sample a subset of queries for recall calculation (to avoid long runtime)
	sampleSize := min(100, len(queries))
	queryIndices := sampleIndices(len(queries), sampleSize)

	for _, idx := range queryIndices {
		query := queries[idx]

		// Get ANN search results
		annResults, err := rc.search(ctx, query.Embedding, topK, metricType, searchParams)
		if err != nil {
			logger.Warn("Failed to get ANN results", "query_id", query.ID, "error", err)
			continue
		}

		// Calculate mathematical recall (ANN vs brute force)
		bruteResults, err := rc.bruteForceSearch(ctx, query.Embedding, topK, metricType)
		if err != nil {
			logger.Warn("Failed to get brute force results", "query_id", query.ID, "error", err)
		} else {
			mathRecall := calculateOverlap(annResults, bruteResults)
			mathRecallSum += mathRecall
		}

		// Calculate business recall (ANN vs ground truth qrels)
		if rc.qrels != nil && len(rc.qrels[query.ID]) > 0 {
			relevantDocs := rc.qrels.GetRelevantDocs(query.ID)
			bizRecall := calculateBusinessRecall(annResults, relevantDocs)
			bizRecallSum += bizRecall
			queriesWithQrels++

			// Debug: Check if any relevant docs are in our results
			if queriesWithQrels <= 3 {
				logger.Debug("Business recall debug",
					"query_id", query.ID,
					"relevant_docs", relevantDocs,
					"ann_results", annResults[:min(5, len(annResults))],
					"biz_recall", bizRecall)
			}
		}
	}

	metrics := &RecallMetrics{
		MathematicalRecall: mathRecallSum / float64(sampleSize),
		BusinessRecall:     0,
		QueriesTested:      sampleSize,
	}

	if queriesWithQrels > 0 {
		metrics.BusinessRecall = bizRecallSum / float64(queriesWithQrels)
	}

	logger.Info("Recall calculation completed",
		"math_recall", fmt.Sprintf("%.2f%%", metrics.MathematicalRecall*100),
		"business_recall", fmt.Sprintf("%.2f%%", metrics.BusinessRecall*100),
		"queries_tested", sampleSize)

	return metrics, nil
}

// search performs ANN search
func (rc *RecallCalculator) search(
	ctx context.Context,
	vector []float32,
	topK int,
	metricType entity.MetricType,
	searchParams entity.SearchParam,
) ([]string, error) {
	sp, err := entity.NewIndexFlatSearchParam()
	if err != nil {
		return nil, err
	}

	results, err := rc.client.Search(
		ctx,
		rc.collection,
		[]string{},
		"",
		[]string{rc.idField},
		[]entity.Vector{entity.FloatVector(vector)},
		rc.vectorField,
		metricType,
		topK,
		sp,
	)

	if err != nil {
		return nil, err
	}

	if len(results) == 0 {
		return []string{}, nil
	}

	// Extract IDs
	ids := make([]string, 0, topK)
	idCol := results[0].Fields.GetColumn(rc.idField)
	if idCol == nil {
		return []string{}, nil
	}

	// Handle both Int64 and VarChar IDs
	if intCol, ok := idCol.(*entity.ColumnInt64); ok {
		for i := 0; i < results[0].ResultCount; i++ {
			val, err := intCol.ValueByIdx(i)
			if err == nil {
				ids = append(ids, fmt.Sprintf("%d", val))
			}
		}
	} else if strCol, ok := idCol.(*entity.ColumnVarChar); ok {
		for i := 0; i < results[0].ResultCount; i++ {
			val, err := strCol.ValueByIdx(i)
			if err == nil {
				ids = append(ids, val)
			}
		}
	}

	return ids, nil
}

// bruteForceSearch performs exact search with high search parameters
func (rc *RecallCalculator) bruteForceSearch(
	ctx context.Context,
	vector []float32,
	topK int,
	metricType entity.MetricType,
) ([]string, error) {
	// Use flat search which is exact
	return rc.search(ctx, vector, topK, metricType, nil)
}

// calculateOverlap computes recall between two result sets
func calculateOverlap(annResults, groundTruth []string) float64 {
	if len(groundTruth) == 0 {
		return 0
	}

	// Create set from ground truth
	truthSet := make(map[string]bool)
	for _, id := range groundTruth {
		truthSet[id] = true
	}

	// Count matches
	matches := 0
	for _, id := range annResults {
		if truthSet[id] {
			matches++
		}
	}

	return float64(matches) / float64(len(groundTruth))
}

// calculateBusinessRecall computes recall against qrels ground truth
// For VDBBench datasets, relevantDocs contains top-K neighbors in ranked order
// We compare search results against only the top-K ground truth neighbors (same K as search)
func calculateBusinessRecall(annResults, relevantDocs []string) float64 {
	if len(relevantDocs) == 0 || len(annResults) == 0 {
		return 0
	}

	// Only consider the top-K ground truth neighbors (same K as our search)
	// VDBBench provides 1000 neighbors but we only want to compare against topK
	k := len(annResults)
	groundTruthTopK := relevantDocs
	if len(relevantDocs) > k {
		groundTruthTopK = relevantDocs[:k]
	}

	// Create set from ground truth top-K
	truthSet := make(map[string]bool)
	for _, id := range groundTruthTopK {
		truthSet[id] = true
	}

	// Count how many ground truth top-K docs are in our results
	matches := 0
	for _, id := range annResults {
		if truthSet[id] {
			matches++
		}
	}

	// Recall = ground truth docs found / K
	return float64(matches) / float64(k)
}

// sampleIndices returns a random sample of indices
func sampleIndices(total, sample int) []int {
	if sample >= total {
		indices := make([]int, total)
		for i := range indices {
			indices[i] = i
		}
		return indices
	}

	indices := make([]int, sample)
	for i := 0; i < sample; i++ {
		indices[i] = i * total / sample
	}
	return indices
}
