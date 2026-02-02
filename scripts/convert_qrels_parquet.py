#!/usr/bin/env python3
"""
Convert BEIR qrels Parquet files to TSV format for Go consumption.
"""

import sys
import pyarrow.parquet as pq

def convert_qrels_parquet_to_tsv(parquet_path, output_path):
    """Convert qrels Parquet to TSV format: query-id corpus-id score"""
    print(f"Reading qrels Parquet file: {parquet_path}")
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    print(f"Columns: {list(df.columns)}")
    print(f"Converting {len(df)} rows...")

    with open(output_path, 'w') as f:
        for idx, row in df.iterrows():
            # qrels parquet has: query_id, corpus_id, score (with underscores)
            query_id = row['query_id']
            corpus_id = row['corpus_id']
            score = row['score']

            f.write(f"{query_id}\t{corpus_id}\t{score}\n")

            if (idx + 1) % 10000 == 0:
                print(f"  Processed {idx + 1} rows...")

    print(f"Done! Wrote {len(df)} rows to {output_path}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: convert_qrels_parquet.py <input.parquet> <output.tsv>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    convert_qrels_parquet_to_tsv(input_path, output_path)
