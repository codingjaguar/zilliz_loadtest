package datasource

import (
	"bufio"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"

	"zilliz-loadtest/internal/logger"
)

// CohereJSONLRecord represents a record from the converted JSONL format
type CohereJSONLRecord struct {
	ID     string `json:"_id"`
	Title  string `json:"title"`
	Text   string `json:"text"`
	EmbB64 string `json:"emb_b64"`
	EmbDim int    `json:"emb_dim"`
}

// ReadJSONLFile reads embeddings from a JSONL file with base64-encoded embeddings
func ReadJSONLFile(filePath string, maxRows int) ([]CohereEmbedding, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	// Increase buffer size for large lines
	buf := make([]byte, 0, 1024*1024) // 1MB buffer
	scanner.Buffer(buf, 10*1024*1024)  // 10MB max

	embeddings := make([]CohereEmbedding, 0, maxRows)
	rowsRead := 0

	for scanner.Scan() && rowsRead < maxRows {
		var record CohereJSONLRecord
		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			logger.Debug("Failed to parse JSON line", "error", err)
			continue
		}

		// Decode base64 embedding
		embBytes, err := base64.StdEncoding.DecodeString(record.EmbB64)
		if err != nil {
			logger.Debug("Failed to decode base64 embedding", "error", err)
			continue
		}

		// Convert bytes to float32 array
		embedding := make([]float32, record.EmbDim)
		for i := 0; i < record.EmbDim; i++ {
			bits := binary.LittleEndian.Uint32(embBytes[i*4 : (i+1)*4])
			embedding[i] = math.Float32frombits(bits)
		}

		embeddings = append(embeddings, CohereEmbedding{
			ID:        record.ID,
			Title:     record.Title,
			Text:      record.Text,
			Embedding: embedding,
		})
		rowsRead++

		if rowsRead%10000 == 0 {
			logger.Debug("Reading progress", "rows_read", rowsRead)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %w", err)
	}

	logger.Info("Completed reading JSONL file", "rows_read", len(embeddings))
	return embeddings, nil
}

// ConvertParquetToJSONL converts a Parquet file to JSONL using Python helper
func ConvertParquetToJSONL(parquetPath, jsonlPath string, maxRows int) error {
	// Find the Python converter script
	scriptPath := "scripts/convert_parquet.py"

	// Check if script exists
	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		// Try absolute path from project root
		if wd, err := os.Getwd(); err == nil {
			scriptPath = filepath.Join(wd, scriptPath)
		}
	}

	logger.Info("Converting Parquet to JSONL",
		"parquet", parquetPath,
		"jsonl", jsonlPath,
		"max_rows", maxRows,
		"script", scriptPath)

	// Run Python converter
	cmd := exec.Command("python3", scriptPath, parquetPath, jsonlPath, fmt.Sprintf("%d", maxRows))
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("conversion failed: %w\nOutput: %s", err, string(output))
	}

	logger.Info("Conversion completed", "output", string(output))
	return nil
}

// EnsureJSONLExists ensures a JSONL version of the Parquet file exists
func EnsureJSONLExists(parquetPath string, maxRows int) (string, error) {
	// Derive JSONL path from Parquet path
	jsonlPath := parquetPath[:len(parquetPath)-len(filepath.Ext(parquetPath))] + ".jsonl"

	// Check if JSONL already exists
	if info, err := os.Stat(jsonlPath); err == nil && info.Size() > 0 {
		logger.Info("JSONL file already exists", "path", jsonlPath)
		return jsonlPath, nil
	}

	// Convert Parquet to JSONL
	if err := ConvertParquetToJSONL(parquetPath, jsonlPath, maxRows); err != nil {
		return "", fmt.Errorf("failed to convert parquet to jsonl: %w", err)
	}

	return jsonlPath, nil
}
