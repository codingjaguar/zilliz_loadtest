#!/usr/bin/env python3
"""
Convert BEIR Parquet files to a Go-friendly format.
Outputs newline-delimited JSON with base64-encoded embeddings.
"""

import sys
import json
import base64
import struct
import pyarrow.parquet as pq

def convert_parquet_to_jsonl(parquet_path, output_path, max_rows=None):
    """Convert Parquet to newline-delimited JSON."""
    print(f"Reading Parquet file: {parquet_path}")
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    if max_rows:
        df = df.head(max_rows)

    print(f"Converting {len(df)} rows...")

    with open(output_path, 'w') as f:
        for idx, row in df.iterrows():
            # Convert embedding to bytes and then base64
            emb_bytes = struct.pack(f'{len(row["emb"])}f', *row["emb"])
            emb_b64 = base64.b64encode(emb_bytes).decode('ascii')

            record = {
                '_id': row['_id'],
                'text': row['text'],
                'emb_b64': emb_b64,
                'emb_dim': len(row['emb'])
            }
            # Add title only if present (corpus files have it, query files don't)
            if 'title' in row and row['title'] is not None:
                record['title'] = row['title']
            f.write(json.dumps(record) + '\n')

            if (idx + 1) % 10000 == 0:
                print(f"  Processed {idx + 1} rows...")

    print(f"Done! Wrote {len(df)} rows to {output_path}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: convert_parquet.py <input.parquet> <output.jsonl> [max_rows]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    max_rows = int(sys.argv[3]) if len(sys.argv) > 3 else None

    convert_parquet_to_jsonl(input_path, output_path, max_rows)
