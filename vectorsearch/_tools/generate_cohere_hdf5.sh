#!/usr/bin/env bash
# generate_cohere_hdf5.sh
#
# Slices the first N vectors from the cohere-wiki source, computes exact top-K
# ground truth via FAISS flat index, writes cohere-<label>.hdf5, and optionally
# uploads it to GCS.
#
# Requirements:
#   - python3 with h5py, numpy, faiss-cpu (pip install h5py numpy faiss-cpu)
#   - wget, bzip2
#   - gsutil authenticated to the opensearch-benchmark-datasets bucket (for upload)
#
# Usage:
#   ./generate_cohere_hdf5.sh --vectors 8000000 [OPTIONS]
#
# Options:
#   --vectors N         Number of corpus vectors to slice (required)
#   --k K               Number of ground truth neighbors (default: 100)
#   --label LABEL       Output label, e.g. "8m". Defaults to auto-formatted N
#   --workdir DIR       Directory for intermediate files (default: /tmp/cohere-gen)
#   --skip-download     Skip downloading/decompressing the source (must already exist in workdir)
#   --skip-upload       Skip the gsutil upload step
#   --gcs-bucket BUCKET GCS bucket name (default: opensearch-benchmark-datasets)
#
# Examples:
#   ./generate_cohere_hdf5.sh --vectors 8000000
#   ./generate_cohere_hdf5.sh --vectors 2000000 --label 2m --workdir /mnt/data/cohere-gen
#   ./generate_cohere_hdf5.sh --vectors 8000000 --skip-download --skip-upload

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
VECTORS=""
K=100
LABEL=""
WORKDIR="/tmp/cohere-gen"
SKIP_DOWNLOAD=false
SKIP_UPLOAD=false
GCS_BUCKET="opensearch-benchmark-datasets"
BASE_URL="https://dbyiw3u3rf9yr.cloudfront.net/corpora/vectorsearch/cohere-wikipedia-22-12-en-embeddings"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --vectors)      VECTORS="$2";    shift 2 ;;
    --k)            K="$2";          shift 2 ;;
    --label)        LABEL="$2";      shift 2 ;;
    --workdir)      WORKDIR="$2";    shift 2 ;;
    --skip-download) SKIP_DOWNLOAD=true; shift ;;
    --skip-upload)  SKIP_UPLOAD=true; shift ;;
    --gcs-bucket)   GCS_BUCKET="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$VECTORS" ]]; then
  echo "ERROR: --vectors is required"
  echo "Usage: $0 --vectors N [--k K] [--label LABEL] [--workdir DIR] [--skip-download] [--skip-upload]"
  exit 1
fi

# ── Auto-label from N if not specified ────────────────────────────────────────
if [[ -z "$LABEL" ]]; then
  if (( VECTORS % 1000000000 == 0 )); then
    LABEL="$(( VECTORS / 1000000000 ))b"
  elif (( VECTORS % 1000000 == 0 )); then
    LABEL="$(( VECTORS / 1000000 ))m"
  elif (( VECTORS % 1000 == 0 )); then
    LABEL="$(( VECTORS / 1000 ))k"
  else
    LABEL="$VECTORS"
  fi
fi

# ── Select smallest source file that covers N vectors ─────────────────────────
# Available: 1k (1000), 100k (100000), 1m (1000000), 10m (10000000)
if (( VECTORS <= 1000 )); then
  SOURCE_BZ2="documents-1k.hdf5.bz2"
  SOURCE_CAP=1000
elif (( VECTORS <= 100000 )); then
  SOURCE_BZ2="documents-100k.hdf5.bz2"
  SOURCE_CAP=100000
elif (( VECTORS <= 1000000 )); then
  SOURCE_BZ2="documents-1m.hdf5.bz2"
  SOURCE_CAP=1000000
elif (( VECTORS <= 10000000 )); then
  SOURCE_BZ2="documents-10m.hdf5.bz2"
  SOURCE_CAP=10000000
else
  echo "ERROR: --vectors $VECTORS exceeds maximum available source (10,000,000)"
  exit 1
fi

SOURCE_HDF5="${SOURCE_BZ2%.bz2}"
OUTPUT_HDF5="cohere-${LABEL}.hdf5"
GCS_DEST="gs://${GCS_BUCKET}/cohere-wiki-en-768/${OUTPUT_HDF5}"

# ── Approx RAM estimate ───────────────────────────────────────────────────────
RAM_GB=$(echo "scale=1; $VECTORS * 768 * 4 / 1073741824" | bc)

mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "=================================================="
echo "  Cohere Wiki HDF5 generator"
echo "  Vectors  : $VECTORS ($LABEL)"
echo "  K        : $K"
echo "  Source   : $SOURCE_BZ2 (capacity: $SOURCE_CAP)"
echo "  Output   : $OUTPUT_HDF5"
echo "  GCS dest : $GCS_DEST"
echo "  Workdir  : $WORKDIR"
echo "  Peak RAM : ~${RAM_GB} GB"
echo "=================================================="

# ── Step 1: Download and decompress ───────────────────────────────────────────
echo ""
if [[ "$SKIP_DOWNLOAD" == false ]]; then
  if [[ ! -f "$SOURCE_HDF5" ]]; then
    echo "Step 1: Downloading $SOURCE_BZ2..."
    wget -q --show-progress -O "$SOURCE_BZ2" "${BASE_URL}/${SOURCE_BZ2}"
    echo "Step 1: Decompressing (may take several minutes)..."
    bzip2 -d "$SOURCE_BZ2"
    echo "✅ Source file ready: $SOURCE_HDF5"
  else
    echo "Step 1: $SOURCE_HDF5 already exists, skipping download."
  fi
else
  echo "Step 1: Skipped (--skip-download)."
  if [[ ! -f "$SOURCE_HDF5" ]]; then
    echo "ERROR: $SOURCE_HDF5 not found in $WORKDIR"
    exit 1
  fi
fi

# ── Step 2: Slice N + compute ground truth ────────────────────────────────────
echo ""
echo "Step 2: Slicing ${VECTORS} vectors and computing exact top-${K} ground truth via FAISS..."

export WORKDIR K VECTORS LABEL SOURCE_HDF5 OUTPUT_HDF5

python3 - <<'PYEOF'
import sys, os, time
import h5py
import numpy as np

workdir    = os.environ["WORKDIR"]
N          = int(os.environ["VECTORS"])
K          = int(os.environ["K"])
src_path   = os.path.join(workdir, os.environ["SOURCE_HDF5"])
out_path   = os.path.join(workdir, os.environ["OUTPUT_HDF5"])

print(f"  Source  : {src_path}")
print(f"  Output  : {out_path}")
print(f"  Vectors : {N:,}")
print(f"  K       : {K}")

t0 = time.time()
with h5py.File(src_path, "r") as f:
    available = f["train"].shape[0]
    DIM       = f["train"].shape[1]
    if N > available:
        print(f"ERROR: requested {N:,} vectors but source only has {available:,}")
        sys.exit(1)
    print(f"  Reading {N:,} / {available:,} train vectors (dim={DIM})...")
    train = f["train"][:N].astype(np.float32)
    test  = f["test"][:].astype(np.float32)
print(f"  Loaded in {time.time()-t0:.1f}s  |  train={train.shape}  test={test.shape}")

# L2-normalize for innerproduct space (matching cohere-wiki-en-768 space_type)
def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)

train_n = normalize(train)
test_n  = normalize(test)

try:
    import faiss
    print(f"  Building FAISS IndexFlatIP over {N:,} vectors...")
    index = faiss.IndexFlatIP(DIM)
    index.add(train_n)
    print(f"  Searching top-{K} for {test_n.shape[0]:,} queries...")
    t1 = time.time()
    _, neighbors = index.search(test_n, K)
    print(f"  Search complete in {time.time()-t1:.1f}s")
except ImportError:
    print("  faiss not available — falling back to numpy batched brute-force (slow)...")
    BATCH = 200
    neighbors = np.empty((test_n.shape[0], K), dtype=np.int32)
    t1 = time.time()
    for i in range(0, test_n.shape[0], BATCH):
        batch = test_n[i:i+BATCH]                        # (BATCH, DIM)
        scores = batch @ train_n.T                        # (BATCH, N)
        idx    = np.argpartition(scores, -K, axis=1)[:, -K:]
        for j in range(len(batch)):
            order = np.argsort(scores[j, idx[j]])[::-1]
            neighbors[i+j] = idx[j][order]
        if ((i // BATCH) % 20) == 0:
            elapsed = time.time() - t1
            done = i + len(batch)
            eta = (elapsed / done) * (test_n.shape[0] - done) if done > 0 else 0
            print(f"    {done}/{test_n.shape[0]} queries  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
    print(f"  Search complete in {time.time()-t1:.1f}s")

print(f"  Writing {out_path}...")
with h5py.File(out_path, "w") as f:
    f.create_dataset("train",     data=train,                      compression="gzip", chunks=True)
    f.create_dataset("test",      data=test,                       compression="gzip", chunks=True)
    f.create_dataset("neighbors", data=neighbors.astype(np.int32), compression="gzip", chunks=True)
    f.attrs["dimension"]   = DIM
    f.attrs["corpus_size"] = N
    f.attrs["space_type"]  = "innerproduct"

size_gb = os.path.getsize(out_path) / 1024**3
print(f"✅ Written: {out_path}  ({size_gb:.2f} GB)")
PYEOF

# ── Step 3: Upload to GCS ─────────────────────────────────────────────────────
echo ""
if [[ "$SKIP_UPLOAD" == false ]]; then
  echo "Step 3: Uploading to $GCS_DEST..."
  gsutil -m cp "$WORKDIR/$OUTPUT_HDF5" "$GCS_DEST"
  echo "✅ Upload complete."
else
  echo "Step 3: Skipped (--skip-upload). File is at $WORKDIR/$OUTPUT_HDF5"
fi

echo ""
echo "=================================================="
echo "  Done."
echo ""
echo "  Next steps for $LABEL:"
echo "  1. Add to config/datasets.yaml gcs_cache_files:"
echo "       - corpus_size: \"$LABEL\""
echo "         gcs_path: \"$GCS_DEST\""
echo "         target_path: \"/datasets/opensearch-benchmark/.osb/benchmarks/data/cohere-${LABEL}/cohere-${LABEL}.hdf5\""
echo "  2. Add \"$LABEL\" to supported_corpus_sizes for cohere-wiki-en-768"
echo "  3. Add build+search steps to your pipeline"
echo "=================================================="
