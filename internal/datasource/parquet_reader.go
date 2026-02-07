package datasource

import (
	"fmt"

	"zilliz-loadtest/internal/logger"

	"github.com/xitongsys/parquet-go-source/local"
	"github.com/xitongsys/parquet-go/reader"
)

// BEIRCorpusRecord represents a record from BEIR corpus parquet file
type BEIRCorpusRecord struct {
	ID    string    `parquet:"name=_id, type=BYTE_ARRAY, convertedtype=UTF8"`
	Title string    `parquet:"name=title, type=BYTE_ARRAY, convertedtype=UTF8"`
	Text  string    `parquet:"name=text, type=BYTE_ARRAY, convertedtype=UTF8"`
	Emb   []float32 `parquet:"name=emb, type=LIST, valuetype=FLOAT"`
}

// BEIRQrelRecord represents a record from BEIR qrels parquet file
type BEIRQrelRecord struct {
	QueryID  string `parquet:"name=query-id, type=BYTE_ARRAY, convertedtype=UTF8"`
	CorpusID string `parquet:"name=corpus-id, type=BYTE_ARRAY, convertedtype=UTF8"`
	Score    int64  `parquet:"name=score, type=INT64"`
}

// BEIRQueryRecord represents a record from BEIR queries parquet file
type BEIRQueryRecord struct {
	ID   string    `parquet:"name=_id, type=BYTE_ARRAY, convertedtype=UTF8"`
	Text string    `parquet:"name=text, type=BYTE_ARRAY, convertedtype=UTF8"`
	Emb  []float32 `parquet:"name=emb, type=LIST, valuetype=FLOAT"`
}

// VDBBenchTrainRecord represents a record from VDBBench train.parquet file
type VDBBenchTrainRecord struct {
	ID  int64     `parquet:"name=id, type=INT64"`
	Emb []float32 `parquet:"name=emb, type=LIST, valuetype=FLOAT"`
}

// VDBBenchTestRecord represents a record from VDBBench test.parquet file
type VDBBenchTestRecord struct {
	ID  int64     `parquet:"name=id, type=INT64"`
	Emb []float32 `parquet:"name=emb, type=LIST, valuetype=FLOAT"`
}

// VDBBenchNeighborRecord represents a record from VDBBench neighbors.parquet file
type VDBBenchNeighborRecord struct {
	ID          int64   `parquet:"name=id, type=INT64"`
	NeighborIDs []int64 `parquet:"name=neighbors_id, type=LIST, valuetype=INT64"`
}

// StreamBEIRCorpusParquetGo streams corpus records using pure Go parquet reader
func StreamBEIRCorpusParquetGo(path string, handler func(id, title, text string, emb []float32) error) error {
	logger.Info("Streaming BEIR corpus using Go parquet reader", "path", path)

	fr, err := local.NewLocalFileReader(path)
	if err != nil {
		return fmt.Errorf("failed to open parquet file: %w", err)
	}
	defer fr.Close()

	pr, err := reader.NewParquetReader(fr, new(BEIRCorpusRecord), 4)
	if err != nil {
		return fmt.Errorf("failed to create parquet reader: %w", err)
	}
	defer pr.ReadStop()

	numRows := int(pr.GetNumRows())
	logger.Info("BEIR corpus parquet file opened", "rows", numRows)

	batchSize := 1000
	rowsProcessed := 0

	for rowsProcessed < numRows {
		readCount := min(batchSize, numRows-rowsProcessed)
		records := make([]BEIRCorpusRecord, readCount)

		if err := pr.Read(&records); err != nil {
			return fmt.Errorf("failed to read records: %w", err)
		}

		for _, rec := range records {
			if err := handler(rec.ID, rec.Title, rec.Text, rec.Emb); err != nil {
				return err
			}
		}

		rowsProcessed += readCount
		if rowsProcessed%10000 == 0 {
			logger.Info("BEIR corpus streaming progress", "processed", rowsProcessed, "total", numRows)
		}
	}

	logger.Info("BEIR corpus streaming completed", "total", rowsProcessed)
	return nil
}

// StreamBEIRQrelsParquetGo streams qrels using pure Go parquet reader
func StreamBEIRQrelsParquetGo(path string, handler func(queryID, corpusID string, score float64) error) error {
	logger.Info("Streaming BEIR qrels using Go parquet reader", "path", path)

	fr, err := local.NewLocalFileReader(path)
	if err != nil {
		return fmt.Errorf("failed to open parquet file: %w", err)
	}
	defer fr.Close()

	pr, err := reader.NewParquetReader(fr, new(BEIRQrelRecord), 4)
	if err != nil {
		return fmt.Errorf("failed to create parquet reader: %w", err)
	}
	defer pr.ReadStop()

	numRows := int(pr.GetNumRows())
	logger.Info("BEIR qrels parquet file opened", "rows", numRows)

	batchSize := 5000
	rowsProcessed := 0

	for rowsProcessed < numRows {
		readCount := min(batchSize, numRows-rowsProcessed)
		records := make([]BEIRQrelRecord, readCount)

		if err := pr.Read(&records); err != nil {
			return fmt.Errorf("failed to read records: %w", err)
		}

		for _, rec := range records {
			if err := handler(rec.QueryID, rec.CorpusID, float64(rec.Score)); err != nil {
				return err
			}
		}

		rowsProcessed += readCount
	}

	logger.Info("BEIR qrels streaming completed", "total", rowsProcessed)
	return nil
}

// StreamBEIRQueriesParquetGo streams queries using pure Go parquet reader
func StreamBEIRQueriesParquetGo(path string, maxQueries int, handler func(id, text string, emb []float32) error) error {
	logger.Info("Streaming BEIR queries using Go parquet reader", "path", path)

	fr, err := local.NewLocalFileReader(path)
	if err != nil {
		return fmt.Errorf("failed to open parquet file: %w", err)
	}
	defer fr.Close()

	pr, err := reader.NewParquetReader(fr, new(BEIRQueryRecord), 4)
	if err != nil {
		return fmt.Errorf("failed to create parquet reader: %w", err)
	}
	defer pr.ReadStop()

	numRows := int(pr.GetNumRows())
	if maxQueries > 0 && maxQueries < numRows {
		numRows = maxQueries
	}
	logger.Info("BEIR queries parquet file opened", "rows_to_read", numRows)

	batchSize := 1000
	rowsProcessed := 0

	for rowsProcessed < numRows {
		readCount := min(batchSize, numRows-rowsProcessed)
		records := make([]BEIRQueryRecord, readCount)

		if err := pr.Read(&records); err != nil {
			return fmt.Errorf("failed to read records: %w", err)
		}

		for _, rec := range records {
			if err := handler(rec.ID, rec.Text, rec.Emb); err != nil {
				return err
			}
		}

		rowsProcessed += readCount
	}

	logger.Info("BEIR queries streaming completed", "total", rowsProcessed)
	return nil
}

// StreamVDBBenchTrainParquetGo streams VDBBench train records using pure Go parquet reader
func StreamVDBBenchTrainParquetGo(path string, handler func(id int64, emb []float32) error) error {
	logger.Info("Streaming VDBBench train data using Go parquet reader", "path", path)

	fr, err := local.NewLocalFileReader(path)
	if err != nil {
		return fmt.Errorf("failed to open parquet file: %w", err)
	}
	defer fr.Close()

	pr, err := reader.NewParquetReader(fr, new(VDBBenchTrainRecord), 4)
	if err != nil {
		return fmt.Errorf("failed to create parquet reader: %w", err)
	}
	defer pr.ReadStop()

	numRows := int(pr.GetNumRows())
	logger.Info("VDBBench train parquet file opened", "rows", numRows)

	batchSize := 5000
	rowsProcessed := 0

	for rowsProcessed < numRows {
		readCount := min(batchSize, numRows-rowsProcessed)
		records := make([]VDBBenchTrainRecord, readCount)

		if err := pr.Read(&records); err != nil {
			return fmt.Errorf("failed to read records: %w", err)
		}

		for _, rec := range records {
			if err := handler(rec.ID, rec.Emb); err != nil {
				return err
			}
		}

		rowsProcessed += readCount
		if rowsProcessed%100000 == 0 {
			logger.Info("VDBBench train streaming progress", "processed", rowsProcessed, "total", numRows)
		}
	}

	logger.Info("VDBBench train streaming completed", "total", rowsProcessed)
	return nil
}

// StreamVDBBenchTestParquetGo streams VDBBench test queries using pure Go parquet reader
func StreamVDBBenchTestParquetGo(path string, handler func(id int64, emb []float32) error) error {
	logger.Info("Streaming VDBBench test data using Go parquet reader", "path", path)

	fr, err := local.NewLocalFileReader(path)
	if err != nil {
		return fmt.Errorf("failed to open parquet file: %w", err)
	}
	defer fr.Close()

	pr, err := reader.NewParquetReader(fr, new(VDBBenchTestRecord), 4)
	if err != nil {
		return fmt.Errorf("failed to create parquet reader: %w", err)
	}
	defer pr.ReadStop()

	numRows := int(pr.GetNumRows())
	logger.Info("VDBBench test parquet file opened", "rows", numRows)

	batchSize := 1000
	rowsProcessed := 0

	for rowsProcessed < numRows {
		readCount := min(batchSize, numRows-rowsProcessed)
		records := make([]VDBBenchTestRecord, readCount)

		if err := pr.Read(&records); err != nil {
			return fmt.Errorf("failed to read records: %w", err)
		}

		for _, rec := range records {
			if err := handler(rec.ID, rec.Emb); err != nil {
				return err
			}
		}

		rowsProcessed += readCount
	}

	logger.Info("VDBBench test streaming completed", "total", rowsProcessed)
	return nil
}

// StreamVDBBenchNeighborsParquetGo streams VDBBench neighbors using pure Go parquet reader
func StreamVDBBenchNeighborsParquetGo(path string, handler func(id int64, neighborIDs []int64) error) error {
	logger.Info("Streaming VDBBench neighbors using Go parquet reader", "path", path)

	fr, err := local.NewLocalFileReader(path)
	if err != nil {
		return fmt.Errorf("failed to open parquet file: %w", err)
	}
	defer fr.Close()

	pr, err := reader.NewParquetReader(fr, new(VDBBenchNeighborRecord), 4)
	if err != nil {
		return fmt.Errorf("failed to create parquet reader: %w", err)
	}
	defer pr.ReadStop()

	numRows := int(pr.GetNumRows())
	logger.Info("VDBBench neighbors parquet file opened", "rows", numRows)

	batchSize := 1000
	rowsProcessed := 0

	for rowsProcessed < numRows {
		readCount := min(batchSize, numRows-rowsProcessed)
		records := make([]VDBBenchNeighborRecord, readCount)

		if err := pr.Read(&records); err != nil {
			return fmt.Errorf("failed to read records: %w", err)
		}

		for _, rec := range records {
			if err := handler(rec.ID, rec.NeighborIDs); err != nil {
				return err
			}
		}

		rowsProcessed += readCount
	}

	logger.Info("VDBBench neighbors streaming completed", "total", rowsProcessed)
	return nil
}
