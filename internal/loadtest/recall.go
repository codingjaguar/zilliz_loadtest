package loadtest

import (
	"context"
	"fmt"
	"math"

	"zilliz-loadtest/internal/datasource"
	"zilliz-loadtest/internal/logger"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// RecallMetrics holds recall and NDCG measurements
type RecallMetrics struct {
	MathematicalRecall float64 // Recall vs brute force search (ANN accuracy)
	BusinessRecall     float64 // Recall: relevant docs found / total relevant
	NDCG               float64 // Normalized Discounted Cumulative Gain
	QueriesTested      int
}

// RecallCalculator computes recall metrics
type RecallCalculator struct {
	client      client.Client
	collection  string
	vectorField string
	idField     string // Primary key field name
	searchLevel int    // Search level (1-10, higher = more accurate)
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
		searchLevel: 1, // default
		qrels:       qrels,
	}
}

// NewRecallCalculatorWithLevel creates a new recall calculator with custom search level
func NewRecallCalculatorWithLevel(c client.Client, collection, vectorField, idField string, searchLevel int, qrels datasource.Qrels) *RecallCalculator {
	return &RecallCalculator{
		client:      c,
		collection:  collection,
		vectorField: vectorField,
		idField:     idField,
		searchLevel: searchLevel,
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
	var ndcgSum float64
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

		// Calculate business recall and NDCG (ANN vs ground truth qrels)
		if rc.qrels != nil && len(rc.qrels[query.ID]) > 0 {
			relevantDocs := rc.qrels.GetRelevantDocs(query.ID)
			recall, ndcg := calculateBusinessMetrics(annResults, relevantDocs)
			bizRecallSum += recall
			ndcgSum += ndcg
			queriesWithQrels++

			}
	}

	metrics := &RecallMetrics{
		MathematicalRecall: mathRecallSum / float64(sampleSize),
		BusinessRecall:     0,
		NDCG:               0,
		QueriesTested:      sampleSize,
	}

	if queriesWithQrels > 0 {
		metrics.BusinessRecall = bizRecallSum / float64(queriesWithQrels)
		metrics.NDCG = ndcgSum / float64(queriesWithQrels)
	}

	logger.Info("Recall calculation completed",
		"math_recall", fmt.Sprintf("%.2f%%", metrics.MathematicalRecall*100),
		"biz_recall", fmt.Sprintf("%.2f%%", metrics.BusinessRecall*100),
		"ndcg", fmt.Sprintf("%.4f", metrics.NDCG),
		"queries_tested", sampleSize)

	return metrics, nil
}

// search performs ANN search with configured search level
func (rc *RecallCalculator) search(
	ctx context.Context,
	vector []float32,
	topK int,
	metricType entity.MetricType,
	searchParams entity.SearchParam,
) ([]string, error) {
	// Use search level parameter for ANN search
	sp := &SearchParamWithLevel{Level: rc.searchLevel}

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

// bruteForceSearch performs exact search (always uses flat/exact search, ignoring search level)
func (rc *RecallCalculator) bruteForceSearch(
	ctx context.Context,
	vector []float32,
	topK int,
	metricType entity.MetricType,
) ([]string, error) {
	// Use flat search which is exact (search level 10 = highest accuracy)
	sp := &SearchParamWithLevel{Level: 10}

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

// calculateBusinessMetrics computes recall and NDCG against qrels ground truth
// For VDBBench: relevantDocs contains top-K neighbors, compare against same K
// For BEIR: relevantDocs contains sparse human-labeled relevant docs (typically 1-5)
// Returns: (recall, ndcg)
func calculateBusinessMetrics(annResults, relevantDocs []string) (float64, float64) {
	if len(relevantDocs) == 0 || len(annResults) == 0 {
		return 0, 0
	}

	k := len(annResults)

	// Determine if this is VDBBench (exactly 1000 neighbors) or BEIR (variable qrels)
	// VDBBench provides exactly 1000 ordered neighbors per query
	// BEIR has variable number of unordered relevant docs (could be 1-1000+)
	isVDBBench := len(relevantDocs) == 1000

	var groundTruth []string

	if isVDBBench {
		// VDBBench: compare against top-K ground truth neighbors (ordered by distance)
		if len(relevantDocs) > k {
			groundTruth = relevantDocs[:k]
		} else {
			groundTruth = relevantDocs
		}
	} else {
		// BEIR: compare against all human-labeled relevant docs (unordered)
		groundTruth = relevantDocs
	}

	// Create set from ground truth
	truthSet := make(map[string]bool)
	for _, id := range groundTruth {
		truthSet[id] = true
	}

	// Count matches and calculate DCG
	matches := 0
	dcg := 0.0
	for i, id := range annResults {
		if truthSet[id] {
			matches++
			// DCG: relevance / log2(rank + 1), using binary relevance (1 if relevant)
			dcg += 1.0 / math.Log2(float64(i+2)) // i+2 because rank starts at 1, log2(1)=0
		}
	}

	// Calculate IDCG (ideal DCG - all relevant docs at top positions)
	idcg := 0.0
	numRelevant := min(len(groundTruth), k)
	for i := 0; i < numRelevant; i++ {
		idcg += 1.0 / math.Log2(float64(i+2))
	}

	// Recall = relevant docs found / total relevant docs
	recall := float64(matches) / float64(len(groundTruth))

	// NDCG = DCG / IDCG
	ndcg := 0.0
	if idcg > 0 {
		ndcg = dcg / idcg
	}

	return recall, ndcg
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
