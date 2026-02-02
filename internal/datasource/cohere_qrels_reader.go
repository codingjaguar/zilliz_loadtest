package datasource

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

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
	filename := fmt.Sprintf("msmarco-qrels-%s.jsonl", split)
	return filepath.Join(r.downloader.GetCacheDir(), filename)
}

func (r *CohereQrelsReader) getQrelsParquetPath(split string) string {
	filename := fmt.Sprintf("msmarco-qrels-%s.parquet", split)
	return filepath.Join(r.downloader.GetCacheDir(), filename)
}

func (r *CohereQrelsReader) ensureQrelsFile(split string) error {
	jsonlPath := r.getQrelsFilePath(split)

	// Check if JSONL already cached
	if _, err := os.Stat(jsonlPath); err == nil {
		logger.Info("Qrels file already cached", "split", split, "path", jsonlPath)
		return nil
	}

	// Check if Parquet exists, if not download it
	parquetPath := r.getQrelsParquetPath(split)
	if _, err := os.Stat(parquetPath); os.IsNotExist(err) {
		// Download from HuggingFace (parquet format)
		url := fmt.Sprintf("%s/msmarco/qrels/%s.parquet", COHERE_BASE_URL, split)
		logger.Info("Downloading qrels parquet file", "split", split, "url", url)

		if err := r.downloader.EnsureCacheDir(); err != nil {
			return fmt.Errorf("failed to create cache directory: %w", err)
		}

		if err := downloadFile(url, parquetPath); err != nil {
			return fmt.Errorf("failed to download qrels parquet: %w", err)
		}
	}

	// Convert parquet to JSONL
	logger.Info("Converting qrels parquet to JSONL", "parquet", parquetPath, "jsonl", jsonlPath)
	if err := r.convertQrelsParquetToJSONL(parquetPath, jsonlPath); err != nil {
		return fmt.Errorf("failed to convert qrels parquet: %w", err)
	}

	return nil
}

func (r *CohereQrelsReader) convertQrelsParquetToJSONL(parquetPath, jsonlPath string) error {
	// Use Python script to convert
	scriptPath := "scripts/convert_qrels_parquet.py"
	cmd := exec.Command("python3", scriptPath, parquetPath, jsonlPath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("conversion failed: %w\nOutput: %s", err, string(output))
	}
	logger.Info("Qrels conversion completed", "output", strings.TrimSpace(string(output)))
	return nil
}

func (r *CohereQrelsReader) readQrelsFile(filePath string) (Qrels, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	qrels := make(Qrels)
	scanner := bufio.NewScanner(file)
	buf := make([]byte, 0, 1024*1024)
	scanner.Buffer(buf, 10*1024*1024)

	lineCount := 0
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		// Try JSON format first
		var entry QrelEntry
		if err := json.Unmarshal(scanner.Bytes(), &entry); err == nil {
			// JSON format worked
			if qrels[entry.QueryID] == nil {
				qrels[entry.QueryID] = make(map[string]float64)
			}
			qrels[entry.QueryID][entry.CorpusID] = entry.Score
			lineCount++
			continue
		}

		// Try TSV format: query-id corpus-id score
		parts := strings.Fields(line)
		if len(parts) >= 3 {
			queryID := parts[0]
			corpusID := parts[1]
			score, err := strconv.ParseFloat(parts[2], 64)
			if err != nil {
				logger.Debug("Failed to parse qrels score", "line", line, "error", err)
				continue
			}

			if qrels[queryID] == nil {
				qrels[queryID] = make(map[string]float64)
			}
			qrels[queryID][corpusID] = score
			lineCount++
		}

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
