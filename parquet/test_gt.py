"""
Local GT correctness + performance test.

Creates a synthetic parquet dataset (configurable size), runs the new
row-group FAISS GT computation, and validates results against a brute-force
numpy reference.

Usage:
    python3 parquet/test_gt.py

Adjust CORPUS_SIZE, NUM_QUERIES, DIM, and ROWS_PER_ROW_GROUP to stress-test
different scales without needing HuggingFace or OpenSearch.
"""

import json
import os
import shutil
import tempfile
import time

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Test parameters — adjust to simulate different corpus sizes
# ---------------------------------------------------------------------------
CORPUS_SIZE       = 50_000    # total vectors in synthetic corpus
NUM_QUERIES       = 1_000     # number of query vectors
DIM               = 1_536     # embedding dimension (matches openai-large-1536)
ROWS_PER_FILE     = 10_000    # rows per parquet file  (5 files → 50k corpus)
# Simulate real HuggingFace structure: 1 row-group per file (500k rows each).
# Using ROWS_PER_RG = ROWS_PER_FILE means 1 row-group per file in the test.
ROWS_PER_RG       = ROWS_PER_FILE
GT_BATCH          = 2_000     # _GT_BATCH from workload.py (use smaller value for fast local test)
K_GT              = 100       # neighbours to retrieve
NUM_FILES         = CORPUS_SIZE // ROWS_PER_FILE
FIELD_NAME        = "emb"
ID_COL            = "id"

print(f"Corpus: {CORPUS_SIZE:,} vectors | {NUM_FILES} files × 1 row-group × {ROWS_PER_FILE} rows | GT_BATCH={GT_BATCH} | dim={DIM} | {NUM_QUERIES} queries | K={K_GT}")
print()

# ---------------------------------------------------------------------------
# 1. Generate synthetic corpus and queries
# ---------------------------------------------------------------------------
print("Generating synthetic data...")
rng = np.random.default_rng(42)
corpus_raw = rng.standard_normal((CORPUS_SIZE, DIM)).astype(np.float32)

# Normalise corpus for cosine similarity (same as workload does)
norms = np.linalg.norm(corpus_raw, axis=1, keepdims=True)
corpus_norm = corpus_raw / norms

# Sample query vectors from corpus (first NUM_QUERIES rows), already normalised
query_vectors = corpus_norm[:NUM_QUERIES].copy()
# Generate doc IDs
all_doc_ids = [f"doc_{i}" for i in range(CORPUS_SIZE)]

# ---------------------------------------------------------------------------
# 2. Write synthetic parquet files
# ---------------------------------------------------------------------------
tmpdir = tempfile.mkdtemp(prefix="test_gt_parquet_")
print(f"Writing {NUM_FILES} parquet files to {tmpdir} ...")
t0 = time.monotonic()

parquet_paths = []
for file_idx in range(NUM_FILES):
    start = file_idx * ROWS_PER_FILE
    end   = start + ROWS_PER_FILE
    ids   = all_doc_ids[start:end]
    vecs  = corpus_raw[start:end]  # raw (unnormalised) — workload normalises during GT

    # Build table with multiple row-groups.
    # Use fixed_size_list<float32> to match the HuggingFace parquet schema.
    vec_type = pa.list_(pa.float32())
    schema   = pa.schema([
        (ID_COL,     pa.string()),
        (FIELD_NAME, vec_type),
    ])
    path = os.path.join(tmpdir, f"file_{file_idx:02d}.parquet")
    writer = pq.ParquetWriter(path, schema)
    for rg_start in range(0, ROWS_PER_FILE, ROWS_PER_RG):
        rg_end  = rg_start + ROWS_PER_RG
        rg_vecs = vecs[rg_start:rg_end]
        table = pa.table({
            ID_COL:     pa.array(ids[rg_start:rg_end], type=pa.string()),
            FIELD_NAME: pa.array(
                [row.tolist() for row in rg_vecs],
                type=vec_type,
            ),
        }, schema=schema)
        writer.write_table(table, row_group_size=ROWS_PER_RG)
    writer.close()
    parquet_paths.append(path)

print(f"  wrote {NUM_FILES} files in {time.monotonic()-t0:.1f}s")
print()

# ---------------------------------------------------------------------------
# 3. Brute-force reference (numpy, full corpus at once)
# ---------------------------------------------------------------------------
print("Computing brute-force reference (numpy full corpus) ...")
t0 = time.monotonic()
scores_ref = query_vectors @ corpus_norm.T          # (NUM_QUERIES, CORPUS_SIZE)
ref_indices = np.argsort(-scores_ref, axis=1)[:, :K_GT]  # (NUM_QUERIES, K_GT)
ref_doc_ids = [[all_doc_ids[i] for i in row] for row in ref_indices]
ref_time = time.monotonic() - t0
print(f"  reference done in {ref_time:.2f}s")
print()

# ---------------------------------------------------------------------------
# 4. New iter_batches FAISS implementation (mirrors workload.py exactly)
# ---------------------------------------------------------------------------
print(f"Running iter_batches FAISS GT (batch_size={GT_BATCH:,}) ...")

k_gt    = K_GT
num_q   = NUM_QUERIES

best_dists = np.full((num_q, k_gt), -np.inf, dtype=np.float32)
best_pos   = np.full((num_q, k_gt), -1,      dtype=np.int64)
doc_ids_out: list = []

t0 = time.monotonic()
corpus_row_offset = 0
batches_done = 0

for file_idx, path in enumerate(parquet_paths, 1):
    parquet_file = pq.ParquetFile(path)

    for batch in parquet_file.iter_batches(batch_size=GT_BATCH, columns=[ID_COL, FIELD_NAME]):
        if corpus_row_offset >= CORPUS_SIZE:
            break

        batch_rows  = len(batch)
        rows_to_use = min(batch_rows, CORPUS_SIZE - corpus_row_offset)

        ids_col = batch.column(ID_COL).to_pylist()
        doc_ids_out.extend(str(x) for x in ids_col[:rows_to_use])

        emb_col       = batch.column(FIELD_NAME).to_pylist()
        batch_vectors = np.array(emb_col[:rows_to_use], dtype=np.float32)
        del batch, emb_col, ids_col

        norms = np.linalg.norm(batch_vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        batch_vectors /= norms

        idx_b = faiss.IndexFlatIP(DIM)
        idx_b.add(batch_vectors)
        del batch_vectors

        k_this = min(k_gt, rows_to_use)
        dists, local_pos = idx_b.search(query_vectors, k_this)
        del idx_b

        abs_pos = local_pos.astype(np.int64) + corpus_row_offset
        abs_pos[local_pos == -1] = -1

        merged_dists = np.concatenate([best_dists, dists],   axis=1)
        merged_pos   = np.concatenate([best_pos,   abs_pos], axis=1)
        order        = np.argsort(-merged_dists, axis=1)[:, :k_gt]
        best_dists   = np.take_along_axis(merged_dists, order, axis=1)
        best_pos     = np.take_along_axis(merged_pos,   order, axis=1)

        corpus_row_offset += rows_to_use
        batches_done += 1

    print(f"  file {file_idx}/{NUM_FILES} (1 row-group, {batches_done} batches so far) -- {corpus_row_offset:,}/{CORPUS_SIZE:,} rows done")

faiss_time = time.monotonic() - t0
print(f"\nFAISS GT done in {faiss_time:.2f}s")
print()

# Resolve positions to doc IDs
faiss_doc_ids = [
    [doc_ids_out[pos] for pos in best_pos[i] if pos != -1 and pos < len(doc_ids_out)]
    for i in range(num_q)
]

# ---------------------------------------------------------------------------
# 5. Correctness check
# ---------------------------------------------------------------------------
print("Correctness check ...")
mismatches = 0
recall_at_k = []

for i in range(NUM_QUERIES):
    ref_set   = set(ref_doc_ids[i])
    faiss_set = set(faiss_doc_ids[i])
    if ref_set != faiss_set:
        mismatches += 1

    # Recall@K: fraction of true top-K found
    recall_at_k.append(len(ref_set & faiss_set) / K_GT)

mean_recall = np.mean(recall_at_k)
min_recall  = np.min(recall_at_k)

print(f"  Queries checked:     {NUM_QUERIES:,}")
print(f"  Set mismatches:      {mismatches}")
print(f"  Mean recall@{K_GT}:    {mean_recall:.6f}")
print(f"  Min  recall@{K_GT}:    {min_recall:.6f}")

if mismatches == 0 and mean_recall == 1.0:
    print("\n  ✅ PASS — results are identical to brute-force reference")
else:
    print(f"\n  ❌ FAIL — {mismatches} queries have different top-{K_GT} sets")
    # Print first mismatch for debugging
    for i in range(NUM_QUERIES):
        ref_set   = set(ref_doc_ids[i])
        faiss_set = set(faiss_doc_ids[i])
        if ref_set != faiss_set:
            print(f"  Query {i}: missing={ref_set - faiss_set}, extra={faiss_set - ref_set}")
            break

# ---------------------------------------------------------------------------
# 6. Performance summary
# ---------------------------------------------------------------------------
print()
print("Performance summary:")
print(f"  Reference (numpy full):  {ref_time:.2f}s")
print(f"  New (row-group FAISS):   {faiss_time:.2f}s")
print(f"  Ratio:                   {faiss_time/ref_time:.2f}x")
print(f"  Throughput:              {CORPUS_SIZE / faiss_time:,.0f} corpus vectors/s")

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
shutil.rmtree(tmpdir)
print(f"\nCleaned up {tmpdir}")
