package datasource

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"

	"zilliz-loadtest/internal/logger"
)

// CohereQuery represents a query from the BEIR dataset
type CohereQuery struct {
	ID        string
	Text      string
	Embedding []float32
}

// CohereQueryReader reads query embeddings from BEIR dataset
type CohereQueryReader struct {
	downloader *CohereDownloader
}

// NewCohereQueryReader creates a new query reader
func NewCohereQueryReader(downloader *CohereDownloader) *CohereQueryReader {
	return &CohereQueryReader{
		downloader: downloader,
	}
}

// ReadQueries reads query embeddings from the dev or train set
func (r *CohereQueryReader) ReadQueries(split string, maxQueries int) ([]CohereQuery, error) {
	if split != "dev" && split != "train" {
		return nil, fmt.Errorf("invalid split: %s (must be 'dev' or 'train')", split)
	}

	// Download query file if needed
	parquetPath := r.getQueryFilePath(split)
	if err := r.ensureQueryFile(split); err != nil {
		return nil, fmt.Errorf("failed to ensure query file: %w", err)
	}

	// Convert to JSONL if needed
	jsonlPath, err := EnsureJSONLExists(parquetPath, maxQueries)
	if err != nil {
		return nil, fmt.Errorf("failed to convert queries to JSONL: %w", err)
	}

	// Read JSONL
	embeddings, err := ReadJSONLFile(jsonlPath, maxQueries)
	if err != nil {
		return nil, fmt.Errorf("failed to read query JSONL: %w", err)
	}

	// Convert to CohereQuery format
	queries := make([]CohereQuery, len(embeddings))
	for i, emb := range embeddings {
		queries[i] = CohereQuery{
			ID:        emb.ID,
			Text:      emb.Text,
			Embedding: emb.Embedding,
		}
	}

	logger.Info("Loaded query embeddings", "split", split, "count", len(queries))
	return queries, nil
}

// GetRandomQueries returns a random subset of queries
func (r *CohereQueryReader) GetRandomQueries(queries []CohereQuery, count int) []CohereQuery {
	if count >= len(queries) {
		return queries
	}

	// Create a copy and shuffle
	shuffled := make([]CohereQuery, len(queries))
	copy(shuffled, queries)

	rand.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})

	return shuffled[:count]
}

func (r *CohereQueryReader) getQueryFilePath(split string) string {
	filename := fmt.Sprintf("msmarco-queries-%s.parquet", split)
	return filepath.Join(r.downloader.GetCacheDir(), filename)
}

func (r *CohereQueryReader) ensureQueryFile(split string) error {
	filePath := r.getQueryFilePath(split)

	// Check if already cached
	if _, err := os.Stat(filePath); err == nil {
		logger.Info("Query file already cached", "split", split, "path", filePath)
		return nil
	}

	// Download from HuggingFace
	url := fmt.Sprintf("%s/msmarco/queries/%s.parquet", COHERE_BASE_URL, split)
	logger.Info("Downloading query file", "split", split, "url", url)

	if err := r.downloader.EnsureCacheDir(); err != nil {
		return fmt.Errorf("failed to create cache directory: %w", err)
	}

	return downloadFile(url, filePath)
}
