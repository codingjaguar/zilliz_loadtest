package datasource

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"zilliz-loadtest/internal/logger"
)

// GetCacheDir returns the global cache directory for datasets
func GetCacheDir() string {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "/tmp/zilliz-loadtest-cache"
	}
	return filepath.Join(homeDir, ".cache", "zilliz-loadtest", "datasets")
}

// VDBBenchRecord represents a single record from VDBBench parquet file
type VDBBenchRecord struct {
	ID  int64     `json:"id"`
	Emb []float32 `json:"emb"`
}

// DownloadFile downloads a file from URL to local path if not already cached
func DownloadFile(url, localPath string) error {
	// Check if file already exists
	if info, err := os.Stat(localPath); err == nil && info.Size() > 0 {
		logger.Info("File already cached", "path", localPath, "size_mb", info.Size()/(1024*1024))
		return nil
	}

	// Ensure directory exists
	dir := filepath.Dir(localPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", dir, err)
	}

	logger.Info("Downloading file", "url", url)
	startTime := time.Now()

	// Create temp file
	tmpPath := localPath + ".tmp"
	out, err := os.Create(tmpPath)
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}
	defer out.Close()

	// Download
	resp, err := http.Get(url)
	if err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("failed to download: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		os.Remove(tmpPath)
		return fmt.Errorf("download failed with status: %s", resp.Status)
	}

	size, err := io.Copy(out, resp.Body)
	if err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("failed to write file: %w", err)
	}

	out.Close()

	// Move to final location
	if err := os.Rename(tmpPath, localPath); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("failed to move file: %w", err)
	}

	elapsed := time.Since(startTime)
	logger.Info("Download completed",
		"path", localPath,
		"size_mb", size/(1024*1024),
		"duration", elapsed.Round(time.Second))

	return nil
}

// ConvertVDBBenchParquetToJSONL converts a VDBBench parquet file to JSONL format
// This is a no-op placeholder - we now stream directly from parquet
func ConvertVDBBenchParquetToJSONL(parquetPath, jsonlPath string) (int, error) {
	// We don't actually convert - just return 0 to indicate streaming should be used
	return 0, nil
}

// StreamVDBBenchJSONL streams records from a VDBBench parquet file via Python subprocess
// The jsonlPath is ignored - we read the parquet directly
func StreamVDBBenchJSONL(jsonlPath string, callback func(id int64, emb []float32) error) error {
	// Get parquet path from jsonl path
	parquetPath := jsonlPath[:len(jsonlPath)-6] + ".parquet" // Replace .jsonl with .parquet

	return StreamVDBBenchParquet(parquetPath, callback)
}

// StreamVDBBenchParquet streams records from a VDBBench parquet file using Python
func StreamVDBBenchParquet(parquetPath string, callback func(id int64, emb []float32) error) error {
	logger.Info("Streaming directly from parquet", "path", parquetPath)

	// Python script to stream parquet as JSONL to stdout
	script := fmt.Sprintf(`
import pyarrow.parquet as pq
import json
import sys

pf = pq.ParquetFile('%s')
count = 0
for batch in pf.iter_batches(batch_size=5000):
    df = batch.to_pandas()
    for _, row in df.iterrows():
        record = {
            'id': int(row['id']),
            'emb': [float(x) for x in row['emb']]
        }
        print(json.dumps(record), flush=True)
        count += 1
        if count %% 100000 == 0:
            print(f'PROGRESS: {count}', file=sys.stderr)
print(f'DONE: {count}', file=sys.stderr)
`, parquetPath)

	cmd := exec.Command("python3", "-c", script)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to create stdout pipe: %w", err)
	}
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start python: %w", err)
	}

	scanner := bufio.NewScanner(stdout)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 10*1024*1024)

	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := scanner.Bytes()

		var record VDBBenchRecord
		if err := json.Unmarshal(line, &record); err != nil {
			return fmt.Errorf("failed to parse line %d: %w", lineNum, err)
		}

		if err := callback(record.ID, record.Emb); err != nil {
			cmd.Process.Kill()
			return err
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading from python: %w", err)
	}

	if err := cmd.Wait(); err != nil {
		return fmt.Errorf("python process failed: %w", err)
	}

	return nil
}
