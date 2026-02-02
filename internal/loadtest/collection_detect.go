package loadtest

import (
	"context"
	"fmt"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
)

// FullCorpusThreshold is the minimum row count to consider a collection as having the full MS MARCO corpus
// The full corpus has ~8.84M documents, so we use 8M as the threshold
const FullCorpusThreshold = 8000000

// CollectionInfo holds information about a detected collection
type CollectionInfo struct {
	HasBEIRSchema bool  // Has text and title fields
	RowCount      int64 // Number of rows in the collection
	IsFullCorpus  bool  // Has full MS MARCO corpus (8M+ rows)
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

	// Check if collection has text and title fields (BEIR schema)
	hasText := false
	hasTitle := false

	for _, field := range coll.Schema.Fields {
		if field.Name == "text" {
			hasText = true
		}
		if field.Name == "title" {
			hasTitle = true
		}
	}

	hasBEIRSchema := hasText && hasTitle

	// Get row count
	stats, err := c.GetCollectionStatistics(ctx, collectionName)
	if err != nil {
		return &CollectionInfo{
			HasBEIRSchema: hasBEIRSchema,
			RowCount:      0,
			IsFullCorpus:  false,
		}, nil
	}

	rowCount := int64(0)
	if rowCountStr, ok := stats["row_count"]; ok {
		fmt.Sscanf(rowCountStr, "%d", &rowCount)
	}

	return &CollectionInfo{
		HasBEIRSchema: hasBEIRSchema,
		RowCount:      rowCount,
		IsFullCorpus:  hasBEIRSchema && rowCount >= FullCorpusThreshold,
	}, nil
}
