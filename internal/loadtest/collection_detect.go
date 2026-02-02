package loadtest

import (
	"context"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
)

// DetectRealDataCollection checks if a collection was seeded with real BEIR data
// by checking if it has the 'text' and 'title' fields
func DetectRealDataCollection(ctx context.Context, c client.Client, collectionName string) (bool, error) {
	coll, err := c.DescribeCollection(ctx, collectionName)
	if err != nil {
		return false, err
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

	// If it has both text and title fields, it's likely a BEIR collection
	return hasText && hasTitle, nil
}
