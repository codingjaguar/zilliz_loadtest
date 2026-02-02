package datasource

import (
	"bufio"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/sbinet/npyio"
	"zilliz-loadtest/internal/logger"
)

// CohereJSONEmbedding represents a row from the MS MARCO JSON dataset
type CohereJSONEmbedding struct {
	ID        string    `json:"_id"`
	DocID     string    `json:"docid"`
	Title     string    `json:"title"`
	Text      string    `json:"segment"`
	Embedding []float32 `json:"embedding"`
}

// ReadJSONGZFile reads embeddings from a gzipped JSON Lines file
func ReadJSONGZFile(filePath string, maxRows int) ([]CohereEmbedding, error) {
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

	scanner := bufio.NewScanner(gzReader)
	// Increase buffer size for large lines
	buf := make([]byte, 0, 1024*1024) // 1MB buffer
	scanner.Buffer(buf, 10*1024*1024)  // 10MB max

	embeddings := make([]CohereEmbedding, 0, maxRows)
	rowsRead := 0

	for scanner.Scan() && rowsRead < maxRows {
		var jsonEmb CohereJSONEmbedding
		if err := json.Unmarshal(scanner.Bytes(), &jsonEmb); err != nil {
			logger.Debug("Failed to parse JSON line", "error", err)
			continue
		}

		// Use DocID if _id is not present
		id := jsonEmb.ID
		if id == "" {
			id = jsonEmb.DocID
		}

		embeddings = append(embeddings, CohereEmbedding{
			ID:        id,
			Title:     jsonEmb.Title,
			Text:      jsonEmb.Text,
			Embedding: jsonEmb.Embedding,
		})
		rowsRead++

		if rowsRead%10000 == 0 {
			logger.Debug("Reading progress", "rows_read", rowsRead)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %w", err)
	}

	logger.Info("Completed reading JSON.GZ file", "rows_read", len(embeddings))
	return embeddings, nil
}

// ReadJSONGZWithNPY reads text from JSON.GZ and embeddings from NPY file
func ReadJSONGZWithNPY(jsonPath, npyPath string, maxRows int) ([]CohereEmbedding, error) {
	// Read text data from JSON
	textData, err := readJSONGZTextOnly(jsonPath, maxRows)
	if err != nil {
		return nil, fmt.Errorf("failed to read JSON: %w", err)
	}

	// Read embeddings from NPY
	embeddings, err := readNPYEmbeddings(npyPath, maxRows)
	if err != nil {
		return nil, fmt.Errorf("failed to read NPY: %w", err)
	}

	// Combine them
	if len(textData) != len(embeddings) {
		logger.Warn("Mismatch in text and embedding counts", "text", len(textData), "embeddings", len(embeddings))
	}

	minLen := len(textData)
	if len(embeddings) < minLen {
		minLen = len(embeddings)
	}

	result := make([]CohereEmbedding, minLen)
	for i := 0; i < minLen; i++ {
		result[i] = CohereEmbedding{
			ID:        textData[i].ID,
			Title:     textData[i].Title,
			Text:      textData[i].Text,
			Embedding: embeddings[i],
		}
	}

	logger.Info("Combined text and embeddings", "count", len(result))
	return result, nil
}

func readJSONGZTextOnly(filePath string, maxRows int) ([]CohereEmbedding, error) {
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

	scanner := bufio.NewScanner(gzReader)
	buf := make([]byte, 0, 1024*1024)
	scanner.Buffer(buf, 10*1024*1024)

	result := make([]CohereEmbedding, 0, maxRows)
	rowsRead := 0

	for scanner.Scan() && rowsRead < maxRows {
		var jsonEmb CohereJSONEmbedding
		if err := json.Unmarshal(scanner.Bytes(), &jsonEmb); err != nil {
			logger.Debug("Failed to parse JSON line", "error", err)
			continue
		}

		id := jsonEmb.ID
		if id == "" {
			id = jsonEmb.DocID
		}

		result = append(result, CohereEmbedding{
			ID:    id,
			Title: jsonEmb.Title,
			Text:  jsonEmb.Text,
		})
		rowsRead++
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %w", err)
	}

	return result, nil
}

func readNPYEmbeddings(filePath string, maxRows int) ([][]float32, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open NPY file: %w", err)
	}
	defer file.Close()

	var data []float32
	err = npyio.Read(file, &data)
	if err != nil {
		return nil, fmt.Errorf("failed to read NPY data: %w", err)
	}

	// NPY file is a flat array, need to reshape based on embedding dimension
	// Assuming the file contains embeddings of shape (n, 1024) for Cohere Embed V3
	embDim := 1024 // Cohere Embed V3 dimension
	numEmbeddings := len(data) / embDim

	if numEmbeddings > maxRows {
		numEmbeddings = maxRows
	}

	result := make([][]float32, numEmbeddings)
	for i := 0; i < numEmbeddings; i++ {
		start := i * embDim
		end := start + embDim
		result[i] = data[start:end]
	}

	logger.Info("Read embeddings from NPY", "count", len(result), "dimension", embDim)
	return result, nil
}

// GetNPYPath converts a JSON path to its corresponding NPY path
func GetNPYPath(jsonPath string) string {
	// Replace passages_jsonl with passages_npy and .json.gz with .npy
	npyPath := strings.Replace(jsonPath, "passages_jsonl", "passages_npy", 1)
	npyPath = strings.Replace(npyPath, ".json.gz", ".npy", 1)
	return npyPath
}
