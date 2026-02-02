package datasource

import (
	"context"
	"fmt"
	"os"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/parquet/file"
	"github.com/apache/arrow-go/v18/parquet/pqarrow"

	"zilliz-loadtest/internal/logger"
)

// CohereEmbedding represents a single row from the Cohere dataset
type CohereEmbedding struct {
	ID        string
	Title     string
	Text      string
	Embedding []float32
}

// CohereReader reads embeddings from Cohere Wikipedia Parquet files
type CohereReader struct {
	downloader *CohereDownloader
}

// NewCohereReader creates a new reader
func NewCohereReader(downloader *CohereDownloader) *CohereReader {
	return &CohereReader{
		downloader: downloader,
	}
}

// ReadEmbeddings reads up to maxVectors embeddings from the dataset
// Returns the embeddings and any error encountered
func (r *CohereReader) ReadEmbeddings(maxVectors int) ([]CohereEmbedding, error) {
	embeddings := make([]CohereEmbedding, 0, maxVectors)
	vectorsRead := 0

	// Estimate how many files we need (approximately 1M vectors per file)
	// For safety, we'll read up to 5 files to get enough vectors
	numFiles := min((maxVectors/1000000)+1, 5)

	logger.Info("Reading embeddings from Cohere dataset",
		"max_vectors", maxVectors,
		"estimated_files", numFiles)

	// Ensure files are downloaded
	if err := r.downloader.EnsureDatasetFiles(numFiles); err != nil {
		return nil, fmt.Errorf("failed to ensure dataset files: %w", err)
	}

	// Read from each file until we have enough vectors
	for fileIndex := 0; fileIndex < numFiles && vectorsRead < maxVectors; fileIndex++ {
		filePath := r.downloader.GetParquetFilePath(fileIndex)

		logger.Info("Reading from parquet file",
			"file_index", fileIndex,
			"path", filePath,
			"vectors_read_so_far", vectorsRead)

		fileEmbeddings, err := r.readParquetFile(filePath, maxVectors-vectorsRead)
		if err != nil {
			return nil, fmt.Errorf("failed to read file %d: %w", fileIndex, err)
		}

		embeddings = append(embeddings, fileEmbeddings...)
		vectorsRead += len(fileEmbeddings)

		logger.Info("Read embeddings from file",
			"file_index", fileIndex,
			"vectors_from_file", len(fileEmbeddings),
			"total_vectors", vectorsRead)
	}

	logger.Info("Completed reading embeddings",
		"total_vectors", len(embeddings),
		"requested", maxVectors)

	return embeddings, nil
}

// readParquetFile reads embeddings from a file (Parquet for BEIR)
// Converts to JSONL format first for easier reading in Go
func (r *CohereReader) readParquetFile(filePath string, maxRows int) ([]CohereEmbedding, error) {
	// Ensure JSONL version exists (will convert if needed)
	jsonlPath, err := EnsureJSONLExists(filePath, maxRows)
	if err != nil {
		// Fall back to trying direct Parquet reading
		logger.Warn("Failed to convert to JSONL, attempting direct Parquet read", "error", err)
		return r.readParquetFileSimple(filePath, maxRows)
	}

	// Read from JSONL
	return ReadJSONLFile(jsonlPath, maxRows)
}

// readParquetFileSimple uses a simpler approach to read Parquet files
func (r *CohereReader) readParquetFileSimple(filePath string, maxRows int) ([]CohereEmbedding, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer f.Close()

	// Create Parquet reader
	rdr, err := file.NewParquetReader(f)
	if err != nil {
		return nil, fmt.Errorf("failed to create parquet reader: %w", err)
	}
	defer rdr.Close()

	logger.Info("Parquet file opened",
		"num_rows", rdr.NumRows(),
		"num_row_groups", rdr.NumRowGroups())

	embeddings := make([]CohereEmbedding, 0, maxRows)
	numRowGroups := rdr.NumRowGroups()

	// Read row groups one at a time using pqarrow
	for rgIdx := 0; rgIdx < numRowGroups && len(embeddings) < maxRows; rgIdx++ {
		logger.Debug("Processing row group", "index", rgIdx, "total", numRowGroups)

		// Create Arrow reader for this specific row group
		arrowReader, err := pqarrow.NewFileReader(rdr, pqarrow.ArrowReadProperties{
			Parallel: false,
			BatchSize: 10000,
		}, nil)
		if err != nil {
			return nil, fmt.Errorf("failed to create arrow reader for row group %d: %w", rgIdx, err)
		}

		// Read just this row group's data by reading the full table
		// (Arrow doesn't provide per-row-group reading easily, so we read all and limit)
		var table arrow.Table
		readErr := func() error {
			defer func() {
				if r := recover(); r != nil {
					err = fmt.Errorf("panic while reading: %v", r)
				}
			}()
			table, err = arrowReader.ReadTable(context.Background())
			return err
		}()

		if readErr != nil || err != nil {
			logger.Warn("Failed to read table for row group", "index", rgIdx, "error", err)
			continue
		}

		if table != nil {
			defer table.Release()

			// Extract what we need
			rowGroupEmbeddings, err := r.extractEmbeddingsFromTable(table, maxRows-len(embeddings))
			if err != nil {
				logger.Warn("Failed to extract from row group", "index", rgIdx, "error", err)
				continue
			}

			embeddings = append(embeddings, rowGroupEmbeddings...)
			logger.Debug("Extracted from row group", "count", len(rowGroupEmbeddings), "total", len(embeddings))

			// If we got enough, stop
			if len(embeddings) >= maxRows {
				break
			}
		}
	}

	if len(embeddings) == 0 {
		return nil, fmt.Errorf("failed to read any embeddings from file")
	}

	logger.Info("Successfully read embeddings", "count", len(embeddings))
	return embeddings, nil
}

// extractEmbeddingsFromTable extracts embeddings from an Arrow table
func (r *CohereReader) extractEmbeddingsFromTable(table arrow.Table, maxRows int) ([]CohereEmbedding, error) {
	schema := table.Schema()

	// Find column indices
	idIdx := findFieldIndex(schema, "_id")
	titleIdx := findFieldIndex(schema, "title")
	textIdx := findFieldIndex(schema, "text")
	embIdx := findFieldIndex(schema, "emb")

	logger.Debug("Schema fields detected",
		"_id_idx", idIdx,
		"title_idx", titleIdx,
		"text_idx", textIdx,
		"emb_idx", embIdx)

	if idIdx < 0 || textIdx < 0 || embIdx < 0 {
		return nil, fmt.Errorf("missing required columns (_id:%d, text:%d, emb:%d)", idIdx, textIdx, embIdx)
	}

	numRows := int(table.NumRows())
	if numRows > maxRows {
		numRows = maxRows
	}

	embeddings := make([]CohereEmbedding, 0, numRows)

	// Read records using table reader
	tr := array.NewTableReader(table, int64(min(1000, numRows)))
	defer tr.Release()

	rowsProcessed := 0
	for tr.Next() && rowsProcessed < numRows {
		record := tr.Record()
		numRecordRows := int(record.NumRows())

		for i := 0; i < numRecordRows && rowsProcessed < numRows; i++ {
			// Extract ID
			var id string
			if idCol, ok := record.Column(idIdx).(*array.String); ok {
				id = idCol.Value(i)
			} else if idCol, ok := record.Column(idIdx).(*array.LargeString); ok {
				id = idCol.Value(i)
			}

			// Extract title (optional)
			var title string
			if titleIdx >= 0 {
				if titleCol, ok := record.Column(titleIdx).(*array.String); ok {
					title = titleCol.Value(i)
				} else if titleCol, ok := record.Column(titleIdx).(*array.LargeString); ok {
					title = titleCol.Value(i)
				}
			}

			// Extract text
			var text string
			if textCol, ok := record.Column(textIdx).(*array.String); ok {
				text = textCol.Value(i)
			} else if textCol, ok := record.Column(textIdx).(*array.LargeString); ok {
				text = textCol.Value(i)
			}

			// Extract embedding
			var embedding []float32
			embCol := record.Column(embIdx)

			switch col := embCol.(type) {
			case *array.List:
				if !col.IsNull(i) {
					embStart := int(col.Offsets()[i])
					embEnd := int(col.Offsets()[i+1])
					embLength := embEnd - embStart

					if floatValues, ok := col.ListValues().(*array.Float32); ok {
						embedding = make([]float32, embLength)
						for j := 0; j < embLength; j++ {
							embedding[j] = floatValues.Value(embStart + j)
						}
					}
				}
			case *array.FixedSizeList:
				if !col.IsNull(i) {
					embStart := i * col.Len()
					if floatValues, ok := col.ListValues().(*array.Float32); ok {
						embedding = make([]float32, col.Len())
						for j := 0; j < col.Len(); j++ {
							embedding[j] = floatValues.Value(embStart + j)
						}
					}
				}
			}

			if len(embedding) > 0 {
				embeddings = append(embeddings, CohereEmbedding{
					ID:        id,
					Title:     title,
					Text:      text,
					Embedding: embedding,
				})
				rowsProcessed++
			}
		}
	}

	if err := tr.Err(); err != nil {
		return nil, fmt.Errorf("error reading table: %w", err)
	}

	return embeddings, nil
}

// readParquetFileArrow reads embeddings from a single Parquet file using Arrow (currently broken)
func (r *CohereReader) readParquetFileArrow(filePath string, maxRows int) ([]CohereEmbedding, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open parquet file: %w", err)
	}
	defer f.Close()

	// Create Parquet file reader
	rdr, err := file.NewParquetReader(f)
	if err != nil {
		return nil, fmt.Errorf("failed to create parquet file reader: %w", err)
	}
	defer rdr.Close()

	// Use pqarrow to read as Arrow table
	// Note: Pass memory allocator explicitly
	reader, err := pqarrow.NewFileReader(rdr, pqarrow.ArrowReadProperties{Parallel: false}, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create arrow reader: %w", err)
	}

	logger.Debug("Arrow reader created", "num_row_groups", rdr.NumRowGroups(), "num_rows", rdr.NumRows())

	// Read the table with panic recovery
	var table arrow.Table
	func() {
		defer func() {
			if r := recover(); r != nil {
				err = fmt.Errorf("panic while reading table: %v", r)
			}
		}()
		table, err = reader.ReadTable(context.Background())
	}()

	if err != nil {
		return nil, fmt.Errorf("failed to read table: %w", err)
	}
	if table == nil {
		return nil, fmt.Errorf("table is nil after reading")
	}
	defer table.Release()

	// Extract embeddings
	embeddings := make([]CohereEmbedding, 0, maxRows)
	numRows := int(table.NumRows())
	if numRows > maxRows {
		numRows = maxRows
	}

	// Get column indices and log schema for debugging
	schema := table.Schema()
	logger.Debug("Parquet schema detected",
		"num_fields", schema.NumFields(),
		"num_rows", numRows)

	// Log all field names
	for i := 0; i < schema.NumFields(); i++ {
		logger.Debug("Field found", "index", i, "name", schema.Field(i).Name, "type", schema.Field(i).Type)
	}

	// Note: The Cohere dataset uses "_id" not "id"
	idIdx := findFieldIndex(schema, "_id")
	titleIdx := findFieldIndex(schema, "title")
	textIdx := findFieldIndex(schema, "text")
	embIdx := findFieldIndex(schema, "emb")

	if idIdx < 0 || titleIdx < 0 || textIdx < 0 || embIdx < 0 {
		// Log which columns are missing
		logger.Error("Missing required columns",
			"_id_found", idIdx >= 0,
			"title_found", titleIdx >= 0,
			"text_found", textIdx >= 0,
			"emb_found", embIdx >= 0)
		return nil, fmt.Errorf("missing required columns in parquet file (_id:%d, title:%d, text:%d, emb:%d)", idIdx, titleIdx, textIdx, embIdx)
	}

	logger.Info("Schema validated successfully")

	// Create table reader with batch size
	batchSize := int64(min(10000, maxRows))
	tr := array.NewTableReader(table, batchSize)
	defer tr.Release()

	rowsRead := 0
	for tr.Next() && rowsRead < maxRows {
		record := tr.Record()

		// Get columns - handle different possible types
		idCol := getStringColumn(record, idIdx)
		titleCol := getStringColumn(record, titleIdx)
		textCol := getStringColumn(record, textIdx)
		embCol := record.Column(embIdx)

		for i := 0; i < int(record.NumRows()) && rowsRead < maxRows; i++ {
			// Extract embedding based on column type
			var embedding []float32
			switch col := embCol.(type) {
			case *array.List:
				embStart := int(col.Offsets()[i])
				embEnd := int(col.Offsets()[i+1])
				embLength := embEnd - embStart

				floatValues := col.ListValues().(*array.Float32)
				embedding = make([]float32, embLength)
				for j := 0; j < embLength; j++ {
					embedding[j] = floatValues.Value(embStart + j)
				}
			case *array.FixedSizeList:
				embStart := i * col.Len()
				floatValues := col.ListValues().(*array.Float32)
				embedding = make([]float32, col.Len())
				for j := 0; j < col.Len(); j++ {
					embedding[j] = floatValues.Value(embStart + j)
				}
			default:
				return nil, fmt.Errorf("unsupported embedding column type: %T", embCol)
			}

			embeddings = append(embeddings, CohereEmbedding{
				ID:        idCol[i],
				Title:     titleCol[i],
				Text:      textCol[i],
				Embedding: embedding,
			})

			rowsRead++
		}
	}

	if err := tr.Err(); err != nil {
		return nil, fmt.Errorf("error reading records: %w", err)
	}

	return embeddings, nil
}

// Helper function to find field index by name
func findFieldIndex(schema *arrow.Schema, name string) int {
	for i, field := range schema.Fields() {
		if field.Name == name {
			return i
		}
	}
	return -1
}

// Helper function to get string values from a column (handles both String and LargeString)
func getStringColumn(record arrow.Record, colIdx int) []string {
	col := record.Column(colIdx)
	numRows := int(record.NumRows())
	result := make([]string, numRows)

	switch c := col.(type) {
	case *array.String:
		for i := 0; i < numRows; i++ {
			result[i] = c.Value(i)
		}
	case *array.LargeString:
		for i := 0; i < numRows; i++ {
			result[i] = c.Value(i)
		}
	case *array.Binary:
		for i := 0; i < numRows; i++ {
			result[i] = string(c.Value(i))
		}
	default:
		// Fallback: try to get as string
		for i := 0; i < numRows; i++ {
			result[i] = fmt.Sprintf("%v", col)
		}
	}

	return result
}

// min helper function
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
