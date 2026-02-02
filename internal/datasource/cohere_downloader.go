package datasource

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"zilliz-loadtest/internal/logger"
)

const (
	// HuggingFace dataset URL for Cohere embeddings
	// Using BEIR with Embed V3 English embeddings
	COHERE_DATASET_REPO = "Cohere/beir-embed-english-v3"
	COHERE_BASE_URL     = "https://huggingface.co/datasets/" + COHERE_DATASET_REPO + "/resolve/main"

	// Default cache directory
	DEFAULT_CACHE_DIR = "~/.cache/zilliz-loadtest/datasets"

	// Total number of parquet files in the MS MARCO corpus
	// The full corpus has ~8.84M documents across 500 files
	TOTAL_CORPUS_FILES = 500
)

// CohereDownloader handles downloading and caching of Cohere Wikipedia dataset
type CohereDownloader struct {
	cacheDir string
}

// NewCohereDownloader creates a new downloader with the specified cache directory
func NewCohereDownloader(cacheDir string) *CohereDownloader {
	if cacheDir == "" {
		cacheDir = DEFAULT_CACHE_DIR
	}

	// Expand ~ to home directory
	if cacheDir[:2] == "~/" {
		home, err := os.UserHomeDir()
		if err == nil {
			cacheDir = filepath.Join(home, cacheDir[2:])
		}
	}

	return &CohereDownloader{
		cacheDir: cacheDir,
	}
}

// GetCacheDir returns the cache directory path
func (d *CohereDownloader) GetCacheDir() string {
	return d.cacheDir
}

// GetParquetFilePath returns the path to a cached file
func (d *CohereDownloader) GetParquetFilePath(fileIndex int) string {
	filename := fmt.Sprintf("msmarco-corpus-%04d.parquet", fileIndex)
	return filepath.Join(d.cacheDir, filename)
}

// IsFileCached checks if a parquet file is already downloaded
func (d *CohereDownloader) IsFileCached(fileIndex int) bool {
	filePath := d.GetParquetFilePath(fileIndex)
	info, err := os.Stat(filePath)
	if err != nil {
		return false
	}
	return info.Size() > 0
}

// EnsureCacheDir creates the cache directory if it doesn't exist
func (d *CohereDownloader) EnsureCacheDir() error {
	return os.MkdirAll(d.cacheDir, 0755)
}

// DownloadFile downloads a single parquet file with progress reporting
func (d *CohereDownloader) DownloadFile(fileIndex int) error {
	if d.IsFileCached(fileIndex) {
		logger.Info("File already cached, skipping download",
			"file_index", fileIndex,
			"path", d.GetParquetFilePath(fileIndex))
		return nil
	}

	if err := d.EnsureCacheDir(); err != nil {
		return fmt.Errorf("failed to create cache directory: %w", err)
	}

	filename := fmt.Sprintf("%04d.parquet", fileIndex)
	url := fmt.Sprintf("%s/msmarco/corpus/%s", COHERE_BASE_URL, filename)
	destPath := d.GetParquetFilePath(fileIndex)

	logger.Info("Downloading Cohere dataset file",
		"file_index", fileIndex,
		"url", url,
		"dest", destPath)

	// Download to temporary file first
	tmpPath := destPath + ".tmp"
	defer os.Remove(tmpPath) // Clean up on error

	// Create HTTP request with optional HuggingFace token
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	// Add HuggingFace token if available (from environment)
	if hfToken := os.Getenv("HF_TOKEN"); hfToken != "" {
		req.Header.Set("Authorization", "Bearer "+hfToken)
		logger.Debug("Using HuggingFace token for authentication")
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to download file: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download failed with status: %s (try setting HF_TOKEN env var if dataset requires authentication)", resp.Status)
	}

	out, err := os.Create(tmpPath)
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}
	defer out.Close()

	// Copy with progress reporting
	totalBytes := resp.ContentLength
	written, err := io.Copy(out, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	// Move temp file to final location
	if err := os.Rename(tmpPath, destPath); err != nil {
		return fmt.Errorf("failed to move file to cache: %w", err)
	}

	logger.Info("Download completed",
		"file_index", fileIndex,
		"bytes_downloaded", written,
		"expected_bytes", totalBytes)

	return nil
}

// EnsureDatasetFiles downloads the required number of parquet files (lazy init)
// Only downloads files that aren't already cached
func (d *CohereDownloader) EnsureDatasetFiles(numFiles int) error {
	if numFiles <= 0 || numFiles > 500 {
		return fmt.Errorf("invalid number of files: %d (must be 1-500)", numFiles)
	}

	logger.Info("Ensuring Cohere dataset files are available",
		"num_files", numFiles,
		"cache_dir", d.cacheDir)

	// Check which files need to be downloaded
	filesToDownload := []int{}
	for i := 0; i < numFiles; i++ {
		if !d.IsFileCached(i) {
			filesToDownload = append(filesToDownload, i)
		}
	}

	if len(filesToDownload) == 0 {
		logger.Info("All required files already cached")
		return nil
	}

	logger.Info("Downloading missing files",
		"files_to_download", len(filesToDownload),
		"total_files", numFiles)

	// Download missing files
	for _, fileIndex := range filesToDownload {
		if err := d.DownloadFile(fileIndex); err != nil {
			return fmt.Errorf("failed to download file %d: %w", fileIndex, err)
		}
	}

	logger.Info("All dataset files ready")
	return nil
}

// downloadFile is a helper function to download a file from a URL
func downloadFile(url, destPath string) error {
	// Download to temporary file first
	tmpPath := destPath + ".tmp"
	defer os.Remove(tmpPath) // Clean up on error

	// Create HTTP request with optional HuggingFace token
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	// Add HuggingFace token if available (from environment)
	if hfToken := os.Getenv("HF_TOKEN"); hfToken != "" {
		req.Header.Set("Authorization", "Bearer "+hfToken)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to download file: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download failed with status: %s", resp.Status)
	}

	out, err := os.Create(tmpPath)
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}
	defer out.Close()

	// Copy with progress reporting
	written, err := io.Copy(out, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	// Move temp file to final location
	if err := os.Rename(tmpPath, destPath); err != nil {
		return fmt.Errorf("failed to move file to cache: %w", err)
	}

	logger.Info("Download completed", "bytes_downloaded", written, "dest", destPath)
	return nil
}
