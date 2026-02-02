package datasource

import (
	"bufio"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"zilliz-loadtest/internal/logger"
)

// QrelEntry represents a relevance judgment (query-document pair with relevance score)
type QrelEntry struct {
	QueryID    string  `json:"query-id"`
	CorpusID   string  `json:"corpus-id"`
	Score      float64 `json:"score"` // Relevance score (0 or 1 for binary, or graded)
}

// Qrels maps query IDs to relevant document IDs
type Qrels map[string]map[string]float64 // queryID -> {docID -> relevance_score}

// CohereQrelsReader reads relevance judgments from BEIR dataset
type CohereQrelsReader struct {
	downloader *CohereDownloader
}

// NewCohereQrelsReader creates a new qrels reader
func NewCohereQrelsReader(downloader *CohereDownloader) *CohereQrelsReader {
	return &CohereQrelsReader{
		downloader: downloader,
	}
}

// ReadQrels reads relevance judgments from the dev or train set
func (r *CohereQrelsReader) ReadQrels(split string) (Qrels, error) {
	if split != "dev" && split != "train" {
		return nil, fmt.Errorf("invalid split: %s (must be 'dev' or 'train')", split)
	}

	// Ensure qrels file is downloaded
	filePath := r.getQrelsFilePath(split)
	if err := r.ensureQrelsFile(split); err != nil {
		return nil, fmt.Errorf("failed to ensure qrels file: %w", err)
	}

	// Read the JSONL.GZ file
	qrels, err := r.readQrelsFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read qrels file: %w", err)
	}

	logger.Info("Loaded qrels", "split", split, "queries", len(qrels))
	return qrels, nil
}

func (r *CohereQrelsReader) getQrelsFilePath(split string) string {
	filename := fmt.Sprintf("msmarco-qrels-%s.jsonl.gz", split)
	return filepath.Join(r.downloader.GetCacheDir(), filename)
}

func (r *CohereQrelsReader) ensureQrelsFile(split string) error {
	filePath := r.getQrelsFilePath(split)

	// Check if already cached
	if _, err := os.Stat(filePath); err == nil {
		logger.Info("Qrels file already cached", "split", split, "path", filePath)
		return nil
	}

	// Download from HuggingFace
	url := fmt.Sprintf("%s/msmarco/qrels/%s.jsonl.gz", COHERE_BASE_URL, split)
	logger.Info("Downloading qrels file", "split", split, "url", url)

	if err := r.downloader.EnsureCacheDir(); err != nil {
		return fmt.Errorf("failed to create cache directory: %w", err)
	}

	return downloadFile(url, filePath)
}

func (r *CohereQrelsReader) readQrelsFile(filePath string) (Qrels, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	gzReader, err := gzip.NewReader(file)
	if err != nil {
		return nil, fmt.Errorf("failed to create gzip reader: %w", err)
	}
	defer gzReader.Close()

	qrels := make(Qrels)
	scanner := bufio.NewScanner(gzReader)
	buf := make([]byte, 0, 1024*1024)
	scanner.Buffer(buf, 10*1024*1024)

	lineCount := 0
	for scanner.Scan() {
		var entry QrelEntry
		if err := json.Unmarshal(scanner.Bytes(), &entry); err != nil {
			logger.Debug("Failed to parse qrels JSON line", "error", err)
			continue
		}

		// Initialize map for this query if not exists
		if qrels[entry.QueryID] == nil {
			qrels[entry.QueryID] = make(map[string]float64)
		}

		// Add this relevance judgment
		qrels[entry.QueryID][entry.CorpusID] = entry.Score
		lineCount++

		if lineCount%10000 == 0 {
			logger.Debug("Reading qrels progress", "lines", lineCount)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %w", err)
	}

	logger.Info("Loaded qrels entries", "total_entries", lineCount, "unique_queries", len(qrels))
	return qrels, nil
}

// GetRelevantDocs returns the list of relevant document IDs for a query
func (q Qrels) GetRelevantDocs(queryID string) []string {
	docs := make([]string, 0, len(q[queryID]))
	for docID := range q[queryID] {
		docs = append(docs, docID)
	}
	return docs
}

// IsRelevant checks if a document is relevant for a query
func (q Qrels) IsRelevant(queryID, docID string) bool {
	if q[queryID] == nil {
		return false
	}
	score, exists := q[queryID][docID]
	return exists && score > 0
}
