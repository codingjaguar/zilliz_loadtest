package datasource

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"zilliz-loadtest/internal/logger"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// BEIRDataset represents a BEIR benchmark dataset with human-labeled qrels
type BEIRDataset struct {
	Name       string            // Dataset name (e.g., "fiqa", "trec-covid")
	VectorDim  int               // Embedding dimension (768 for Cohere embed-english-v3)
	CorpusSize int               // Approximate corpus size
	MetricType entity.MetricType // All Cohere embeddings use COSINE
}

// Available BEIR datasets with Cohere embed-english-v3 embeddings (1024 dimensions)
var BEIRDatasets = map[string]BEIRDataset{
	// --- Original datasets ---
	"fiqa": {
		Name:       "fiqa",
		VectorDim:  1024,
		CorpusSize: 57638,
		MetricType: entity.COSINE,
	},
	"nfcorpus": {
		Name:       "nfcorpus",
		VectorDim:  1024,
		CorpusSize: 3633,
		MetricType: entity.COSINE,
	},
	"scifact": {
		Name:       "scifact",
		VectorDim:  1024,
		CorpusSize: 5183,
		MetricType: entity.COSINE,
	},
	"trec-covid": {
		Name:       "trec-covid",
		VectorDim:  1024,
		CorpusSize: 171332,
		MetricType: entity.COSINE,
	},
	"arguana": {
		Name:       "arguana",
		VectorDim:  1024,
		CorpusSize: 8674,
		MetricType: entity.COSINE,
	},
	"scidocs": {
		Name:       "scidocs",
		VectorDim:  1024,
		CorpusSize: 25657,
		MetricType: entity.COSINE,
	},
	"quora": {
		Name:       "quora",
		VectorDim:  1024,
		CorpusSize: 522931,
		MetricType: entity.COSINE,
	},
	// --- Additional BEIR datasets ---
	"webis-touche2020": {
		Name:       "webis-touche2020",
		VectorDim:  1024,
		CorpusSize: 382545,
		MetricType: entity.COSINE,
	},
	"robust04": {
		Name:       "robust04",
		VectorDim:  1024,
		CorpusSize: 528155,
		MetricType: entity.COSINE,
	},
	"trec-news": {
		Name:       "trec-news",
		VectorDim:  1024,
		CorpusSize: 594977,
		MetricType: entity.COSINE,
	},
	"nq": {
		Name:       "nq",
		VectorDim:  1024,
		CorpusSize: 2681468,
		MetricType: entity.COSINE,
	},
	"hotpotqa": {
		Name:       "hotpotqa",
		VectorDim:  1024,
		CorpusSize: 5233329,
		MetricType: entity.COSINE,
	},
	"fever": {
		Name:       "fever",
		VectorDim:  1024,
		CorpusSize: 5416568,
		MetricType: entity.COSINE,
	},
	"climate-fever": {
		Name:       "climate-fever",
		VectorDim:  1024,
		CorpusSize: 5416593,
		MetricType: entity.COSINE,
	},
	// --- CQADupStack subforums ---
	"cqadupstack-android": {
		Name:       "cqadupstack-android",
		VectorDim:  1024,
		CorpusSize: 22998,
		MetricType: entity.COSINE,
	},
	"cqadupstack-english": {
		Name:       "cqadupstack-english",
		VectorDim:  1024,
		CorpusSize: 40221,
		MetricType: entity.COSINE,
	},
	"cqadupstack-gaming": {
		Name:       "cqadupstack-gaming",
		VectorDim:  1024,
		CorpusSize: 45301,
		MetricType: entity.COSINE,
	},
	"cqadupstack-gis": {
		Name:       "cqadupstack-gis",
		VectorDim:  1024,
		CorpusSize: 37637,
		MetricType: entity.COSINE,
	},
	"cqadupstack-mathematica": {
		Name:       "cqadupstack-mathematica",
		VectorDim:  1024,
		CorpusSize: 16705,
		MetricType: entity.COSINE,
	},
	"cqadupstack-physics": {
		Name:       "cqadupstack-physics",
		VectorDim:  1024,
		CorpusSize: 38316,
		MetricType: entity.COSINE,
	},
	"cqadupstack-programmers": {
		Name:       "cqadupstack-programmers",
		VectorDim:  1024,
		CorpusSize: 32176,
		MetricType: entity.COSINE,
	},
	"cqadupstack-stats": {
		Name:       "cqadupstack-stats",
		VectorDim:  1024,
		CorpusSize: 42269,
		MetricType: entity.COSINE,
	},
	"cqadupstack-text": {
		Name:       "cqadupstack-text",
		VectorDim:  1024,
		CorpusSize: 68184,
		MetricType: entity.COSINE,
	},
	"cqadupstack-unix": {
		Name:       "cqadupstack-unix",
		VectorDim:  1024,
		CorpusSize: 47382,
		MetricType: entity.COSINE,
	},
	"cqadupstack-webmasters": {
		Name:       "cqadupstack-webmasters",
		VectorDim:  1024,
		CorpusSize: 17405,
		MetricType: entity.COSINE,
	},
	"cqadupstack-wordpress": {
		Name:       "cqadupstack-wordpress",
		VectorDim:  1024,
		CorpusSize: 48605,
		MetricType: entity.COSINE,
	},
}

// GetBEIRDataset returns a BEIR dataset by name
func GetBEIRDataset(name string) (*BEIRDataset, error) {
	dataset, ok := BEIRDatasets[name]
	if !ok {
		names := make([]string, 0, len(BEIRDatasets))
		for k := range BEIRDatasets {
			names = append(names, k)
		}
		return nil, fmt.Errorf("unknown BEIR dataset: %s (available: %s)", name, strings.Join(names, ", "))
	}
	return &dataset, nil
}

// ListBEIRDatasets returns all available BEIR dataset names
func ListBEIRDatasets() []string {
	names := make([]string, 0, len(BEIRDatasets))
	for k := range BEIRDatasets {
		names = append(names, k)
	}
	return names
}

// BEIRDataLoader handles loading BEIR datasets with Cohere embeddings
type BEIRDataLoader struct {
	datasetName string
	cacheDir    string
}

// NewBEIRDataLoader creates a new BEIR data loader
func NewBEIRDataLoader(datasetName, cacheDir string) (*BEIRDataLoader, error) {
	if _, ok := BEIRDatasets[datasetName]; !ok {
		return nil, fmt.Errorf("unknown BEIR dataset: %s", datasetName)
	}

	if cacheDir == "" {
		cacheDir = DEFAULT_CACHE_DIR
	}

	// Expand ~ to home directory
	if len(cacheDir) >= 2 && cacheDir[:2] == "~/" {
		home, err := os.UserHomeDir()
		if err == nil {
			cacheDir = filepath.Join(home, cacheDir[2:])
		}
	}

	return &BEIRDataLoader{
		datasetName: datasetName,
		cacheDir:    cacheDir,
	}, nil
}

// GetCorpusParquetPath returns the path to the corpus parquet file
func (l *BEIRDataLoader) GetCorpusParquetPath() string {
	filename := fmt.Sprintf("beir-%s-corpus.parquet", l.datasetName)
	return filepath.Join(l.cacheDir, filename)
}

// GetQueriesParquetPath returns the path to the queries parquet file
func (l *BEIRDataLoader) GetQueriesParquetPath(split string) string {
	filename := fmt.Sprintf("beir-%s-queries-%s.parquet", l.datasetName, split)
	return filepath.Join(l.cacheDir, filename)
}

// GetQrelsParquetPath returns the path to the qrels parquet file
func (l *BEIRDataLoader) GetQrelsParquetPath(split string) string {
	filename := fmt.Sprintf("beir-%s-qrels-%s.parquet", l.datasetName, split)
	return filepath.Join(l.cacheDir, filename)
}

// EnsureCorpusFile downloads the corpus parquet file(s) if not cached.
// For small datasets, returns a single file. For large datasets with multiple shards,
// downloads all shards and returns the first one (use EnsureCorpusFiles for all).
func (l *BEIRDataLoader) EnsureCorpusFile() (string, error) {
	files, err := l.EnsureCorpusFiles()
	if err != nil {
		return "", err
	}
	return files[0], nil
}

// EnsureCorpusFiles downloads all corpus parquet shard files if not cached.
// Automatically detects the number of shards by trying sequential file numbers.
func (l *BEIRDataLoader) EnsureCorpusFiles() ([]string, error) {
	// Ensure cache directory exists
	if err := os.MkdirAll(l.cacheDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create cache directory: %w", err)
	}

	var paths []string
	for shard := 0; ; shard++ {
		filename := fmt.Sprintf("beir-%s-corpus-%04d.parquet", l.datasetName, shard)
		parquetPath := filepath.Join(l.cacheDir, filename)

		if info, err := os.Stat(parquetPath); err == nil {
			logger.Info("BEIR corpus shard already cached",
				"path", parquetPath,
				"size_mb", info.Size()/1024/1024,
				"shard", shard)
			paths = append(paths, parquetPath)
			continue
		}

		// Try downloading from HuggingFace
		url := fmt.Sprintf("%s/%s/corpus/%04d.parquet", COHERE_BASE_URL, l.datasetName, shard)
		logger.Info("Downloading BEIR corpus shard", "dataset", l.datasetName, "shard", shard, "url", url)

		if err := downloadFile(url, parquetPath); err != nil {
			if shard == 0 {
				return nil, fmt.Errorf("failed to download corpus: %w", err)
			}
			// No more shards available - this is expected
			logger.Info("Found all corpus shards", "dataset", l.datasetName, "total_shards", shard)
			break
		}
		paths = append(paths, parquetPath)
	}

	return paths, nil
}

// EnsureQueriesFile downloads the queries parquet file if not cached
func (l *BEIRDataLoader) EnsureQueriesFile(split string) (string, error) {
	if split != "dev" && split != "test" && split != "train" {
		return "", fmt.Errorf("invalid split: %s (must be 'dev', 'test', or 'train')", split)
	}

	parquetPath := l.GetQueriesParquetPath(split)

	if info, err := os.Stat(parquetPath); err == nil {
		logger.Info("BEIR queries file already cached",
			"path", parquetPath,
			"size_mb", info.Size()/1024/1024)
		return parquetPath, nil
	}

	// Ensure cache directory exists
	if err := os.MkdirAll(l.cacheDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create cache directory: %w", err)
	}

	// Download from HuggingFace
	url := fmt.Sprintf("%s/%s/queries/%s.parquet", COHERE_BASE_URL, l.datasetName, split)
	logger.Info("Downloading BEIR queries", "dataset", l.datasetName, "split", split, "url", url)

	if err := downloadFile(url, parquetPath); err != nil {
		return "", fmt.Errorf("failed to download queries: %w", err)
	}

	return parquetPath, nil
}

// EnsureQrelsFile downloads the qrels parquet file if not cached
func (l *BEIRDataLoader) EnsureQrelsFile(split string) (string, error) {
	if split != "dev" && split != "test" && split != "train" {
		return "", fmt.Errorf("invalid split: %s (must be 'dev', 'test', or 'train')", split)
	}

	parquetPath := l.GetQrelsParquetPath(split)

	if info, err := os.Stat(parquetPath); err == nil {
		logger.Info("BEIR qrels file already cached",
			"path", parquetPath,
			"size_mb", info.Size()/1024/1024)
		return parquetPath, nil
	}

	// Ensure cache directory exists
	if err := os.MkdirAll(l.cacheDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create cache directory: %w", err)
	}

	// Download from HuggingFace
	url := fmt.Sprintf("%s/%s/qrels/%s.parquet", COHERE_BASE_URL, l.datasetName, split)
	logger.Info("Downloading BEIR qrels", "dataset", l.datasetName, "split", split, "url", url)

	if err := downloadFile(url, parquetPath); err != nil {
		return "", fmt.Errorf("failed to download qrels: %w", err)
	}

	return parquetPath, nil
}

// LoadQueries loads query embeddings from the dataset using Python parquet reader
func (l *BEIRDataLoader) LoadQueries(split string, maxQueries int) ([]CohereQuery, error) {
	parquetPath, err := l.EnsureQueriesFile(split)
	if err != nil {
		return nil, err
	}

	var queries []CohereQuery
	err = StreamBEIRQueriesParquetPy(parquetPath, maxQueries, func(id, text string, emb []float32) error {
		queries = append(queries, CohereQuery{
			ID:        id,
			Text:      text,
			Embedding: emb,
		})
		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to read queries: %w", err)
	}

	logger.Info("Loaded BEIR query embeddings",
		"dataset", l.datasetName,
		"split", split,
		"count", len(queries))

	return queries, nil
}

// LoadQrels loads human-labeled relevance judgments from the dataset using Python parquet reader
func (l *BEIRDataLoader) LoadQrels(split string) (Qrels, error) {
	parquetPath, err := l.EnsureQrelsFile(split)
	if err != nil {
		return nil, err
	}

	// Read parquet file using Python
	qrels := make(Qrels)
	err = StreamBEIRQrelsParquetPy(parquetPath, func(queryID, corpusID string, score float64) error {
		if qrels[queryID] == nil {
			qrels[queryID] = make(map[string]float64)
		}
		qrels[queryID][corpusID] = score
		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to read qrels parquet: %w", err)
	}

	logger.Info("Loaded BEIR qrels (human-labeled relevance judgments)",
		"dataset", l.datasetName,
		"split", split,
		"queries_with_qrels", len(qrels))

	return qrels, nil
}

// StreamBEIRCorpusParquetPy streams corpus records from a BEIR parquet file using Python
// This is used as a fallback when the Go parquet library has issues with list types
func StreamBEIRCorpusParquetPy(path string, handler func(id, title, text string, emb []float32) error) error {
	logger.Info("Streaming BEIR corpus using Python", "path", path)

	// Python script to stream corpus parquet as JSONL
	script := fmt.Sprintf(`
import pyarrow.parquet as pq
import json
import sys

pf = pq.ParquetFile('%s')
count = 0
for batch in pf.iter_batches(batch_size=1000):
    df = batch.to_pandas()
    for _, row in df.iterrows():
        record = {
            'id': str(row['_id']),
            'title': str(row.get('title', ''))[:2000],
            'text': str(row.get('text', ''))[:60000],
            'emb': [float(x) for x in row['emb']]
        }
        print(json.dumps(record), flush=True)
        count += 1
        if count %% 10000 == 0:
            print(f'PROGRESS: {count}', file=sys.stderr)
print(f'DONE: {count}', file=sys.stderr)
`, path)

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
	buf := make([]byte, 0, 256*1024)
	scanner.Buffer(buf, 64*1024*1024)

	type corpusRecord struct {
		ID    string    `json:"id"`
		Title string    `json:"title"`
		Text  string    `json:"text"`
		Emb   []float32 `json:"emb"`
	}

	rowCount := 0
	for scanner.Scan() {
		var record corpusRecord
		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			logger.Warn("Failed to parse corpus record", "error", err)
			continue
		}
		if err := handler(record.ID, record.Title, record.Text, record.Emb); err != nil {
			cmd.Process.Kill()
			return err
		}
		rowCount++
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading from python: %w", err)
	}

	if err := cmd.Wait(); err != nil {
		return fmt.Errorf("python process failed: %w", err)
	}

	logger.Info("Streamed BEIR corpus", "rows", rowCount)
	return nil
}

// StreamBEIRQueriesParquetPy streams queries from a BEIR parquet file using Python
func StreamBEIRQueriesParquetPy(path string, maxQueries int, handler func(id, text string, emb []float32) error) error {
	logger.Info("Streaming BEIR queries using Python", "path", path)

	limit := ""
	if maxQueries > 0 {
		limit = fmt.Sprintf("if count >= %d: break", maxQueries)
	}

	script := fmt.Sprintf(`
import pyarrow.parquet as pq
import json
import sys

pf = pq.ParquetFile('%s')
count = 0
for batch in pf.iter_batches(batch_size=1000):
    df = batch.to_pandas()
    for _, row in df.iterrows():
        record = {
            'id': str(row['_id']),
            'text': str(row.get('text', '')),
            'emb': [float(x) for x in row['emb']]
        }
        print(json.dumps(record), flush=True)
        count += 1
        %s
print(f'DONE: {count}', file=sys.stderr)
`, path, limit)

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

	type queryRecord struct {
		ID   string    `json:"id"`
		Text string    `json:"text"`
		Emb  []float32 `json:"emb"`
	}

	rowCount := 0
	for scanner.Scan() {
		var record queryRecord
		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			continue
		}
		if err := handler(record.ID, record.Text, record.Emb); err != nil {
			cmd.Process.Kill()
			return err
		}
		rowCount++
	}

	if err := cmd.Wait(); err != nil {
		return fmt.Errorf("python process failed: %w", err)
	}

	logger.Info("Streamed BEIR queries", "rows", rowCount)
	return nil
}

// StreamBEIRQrelsParquetPy streams qrels from a BEIR parquet file using Python
func StreamBEIRQrelsParquetPy(path string, handler func(queryID, corpusID string, score float64) error) error {
	logger.Info("Streaming BEIR qrels using Python", "path", path)

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
            'query_id': str(row.get('query-id', row.get('query_id', ''))),
            'corpus_id': str(row.get('corpus-id', row.get('corpus_id', ''))),
            'score': float(row.get('score', 1))
        }
        print(json.dumps(record), flush=True)
        count += 1
print(f'DONE: {count}', file=sys.stderr)
`, path)

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

	type qrelRecord struct {
		QueryID  string  `json:"query_id"`
		CorpusID string  `json:"corpus_id"`
		Score    float64 `json:"score"`
	}

	rowCount := 0
	for scanner.Scan() {
		var record qrelRecord
		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			continue
		}
		if err := handler(record.QueryID, record.CorpusID, record.Score); err != nil {
			cmd.Process.Kill()
			return err
		}
		rowCount++
	}

	if err := cmd.Wait(); err != nil {
		return fmt.Errorf("python process failed: %w", err)
	}

	logger.Info("Streamed BEIR qrels", "rows", rowCount)
	return nil
}
