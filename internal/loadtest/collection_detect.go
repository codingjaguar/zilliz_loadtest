package loadtest

import (
	"context"
	"fmt"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
)

// VDBBenchDatasetInfo contains information about a VDBBench dataset
type VDBBenchDatasetInfo struct {
	Name      string
	VectorDim int
	RowCount  int64
}

// KnownVDBBenchDatasets maps known VDBBench datasets by their characteristics
var KnownVDBBenchDatasets = []VDBBenchDatasetInfo{
	{"cohere_small_100k", 768, 100000},
	{"cohere_medium_1m", 768, 1000000},
	{"openai_small_50k", 1536, 50000},
	{"openai_medium_500k", 1536, 500000},
	{"sift_small_500k", 128, 500000},
	{"sift_medium_5m", 128, 5000000},
	{"gist_small_100k", 960, 100000},
	{"gist_medium_1m", 960, 1000000},
	{"glove_small_100k", 200, 100000},
	{"glove_medium_1m", 200, 1000000},
}

// DetectVDBBenchDataset attempts to identify which VDBBench dataset a collection matches
// Returns dataset name or empty string if no match found
func DetectVDBBenchDataset(info *CollectionInfo) string {
	if !info.HasVDBBenchSchema {
		return ""
	}

	// Match by vector dimension and approximate row count (within 10% tolerance)
	for _, ds := range KnownVDBBenchDatasets {
		if info.VectorDim == ds.VectorDim {
			// Check if row count is within 10% of expected (to account for partial imports)
			minRows := int64(float64(ds.RowCount) * 0.8)
			maxRows := int64(float64(ds.RowCount) * 1.1)
			if info.RowCount >= minRows && info.RowCount <= maxRows {
				return ds.Name
			}
		}
	}

	// If no exact match, try to match by dimension only and pick the closest row count
	for _, ds := range KnownVDBBenchDatasets {
		if info.VectorDim == ds.VectorDim && info.RowCount > 0 {
			// Return the first matching dimension (user may have partial data)
			return ds.Name
		}
	}

	return ""
}

// FullCorpusThreshold is the minimum row count to consider a collection as having the full MS MARCO corpus
// The full corpus has ~8.84M documents, so we use 8M as the threshold
const FullCorpusThreshold = 8000000

// CollectionInfo holds information about a detected collection
type CollectionInfo struct {
	HasBEIRSchema     bool   // Has text and title fields (BEIR/Cohere)
	HasVDBBenchSchema bool   // Has id and emb fields only (VDBBench)
	VectorFieldName   string // Name of the vector field (emb, vector, etc.)
	VectorDim         int    // Dimension of the vector field
	RowCount          int64  // Number of rows in the collection
	IsFullCorpus      bool   // Has full MS MARCO corpus (8M+ rows)
}

// DetectRealDataCollection checks if a collection was seeded with real BEIR data
// by checking if it has the 'text' and 'title' fields
func DetectRealDataCollection(ctx context.Context, c client.Client, collectionName string) (bool, error) {
	info, err := DetectCollectionInfo(ctx, c, collectionName)
	if err != nil {
		return false, err
	}
	return info.HasBEIRSchema, nil
}

// DetectCollectionInfo returns detailed information about a collection
func DetectCollectionInfo(ctx context.Context, c client.Client, collectionName string) (*CollectionInfo, error) {
	coll, err := c.DescribeCollection(ctx, collectionName)
	if err != nil {
		return nil, err
	}

	// Check for BEIR schema (text and title fields)
	hasText := false
	hasTitle := false
	// Check for VDBBench schema (id and emb fields only)
	hasEmb := false
	hasID := false
	vectorFieldName := ""
	vectorDim := 0
	fieldCount := len(coll.Schema.Fields)

	for _, field := range coll.Schema.Fields {
		switch field.Name {
		case "text":
			hasText = true
		case "title":
			hasTitle = true
		case "emb":
			hasEmb = true
			vectorFieldName = "emb"
			if dim, ok := field.TypeParams["dim"]; ok {
				fmt.Sscanf(dim, "%d", &vectorDim)
			}
		case "vector":
			vectorFieldName = "vector"
			if dim, ok := field.TypeParams["dim"]; ok {
				fmt.Sscanf(dim, "%d", &vectorDim)
			}
		case "id":
			hasID = true
		}
	}

	hasBEIRSchema := hasText && hasTitle
	// VDBBench schema: only id and emb fields (plus maybe a few metadata fields, but specifically "emb" for vectors)
	hasVDBBenchSchema := hasID && hasEmb && !hasBEIRSchema && fieldCount <= 3

	// Get row count
	stats, err := c.GetCollectionStatistics(ctx, collectionName)
	if err != nil {
		return &CollectionInfo{
			HasBEIRSchema:     hasBEIRSchema,
			HasVDBBenchSchema: hasVDBBenchSchema,
			VectorFieldName:   vectorFieldName,
			VectorDim:         vectorDim,
			RowCount:          0,
			IsFullCorpus:      false,
		}, nil
	}

	rowCount := int64(0)
	if rowCountStr, ok := stats["row_count"]; ok {
		fmt.Sscanf(rowCountStr, "%d", &rowCount)
	}

	return &CollectionInfo{
		HasBEIRSchema:     hasBEIRSchema,
		HasVDBBenchSchema: hasVDBBenchSchema,
		VectorFieldName:   vectorFieldName,
		VectorDim:         vectorDim,
		RowCount:          rowCount,
		IsFullCorpus:      hasBEIRSchema && rowCount >= FullCorpusThreshold,
	}, nil
}
