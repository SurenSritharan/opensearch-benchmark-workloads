"""
Unit tests for ParquetBulkParamReader partition/pre-cache/seek logic.

Tests run entirely offline using synthetic local parquet files — no HuggingFace
connection required.  HfFileSystem and hf_hub_download are monkey-patched to
operate against a local temp directory.

Run with:
    python3 parquet/test_partition.py
"""

import os
import shutil
import sys
import tempfile
import types
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Synthetic dataset parameters
# ---------------------------------------------------------------------------
DIM            = 8       # tiny dimension for speed
ROWS_PER_FILE  = 1_000   # rows per parquet shard
NUM_FILES      = 5       # total files → 5,000 rows total
TOTAL_ROWS     = ROWS_PER_FILE * NUM_FILES
FIELD_NAME     = "emb"
ID_COL         = "id"


def _write_synthetic_parquet(tmpdir: str) -> list[str]:
    """Write NUM_FILES parquet shards into tmpdir and return sorted paths."""
    rng = np.random.default_rng(0)
    paths = []
    for i in range(NUM_FILES):
        start = i * ROWS_PER_FILE
        ids   = [f"doc_{j}" for j in range(start, start + ROWS_PER_FILE)]
        vecs  = rng.standard_normal((ROWS_PER_FILE, DIM)).astype(np.float32)
        table = pa.table({
            ID_COL:     pa.array(ids, type=pa.string()),
            FIELD_NAME: pa.array([v.tolist() for v in vecs], type=pa.list_(pa.float32())),
        })
        path = os.path.join(tmpdir, f"shard_{i:02d}.parquet")
        pq.write_table(table, path)
        paths.append(path)
    return sorted(paths)


def _patch_workload_module(tmpdir: str, parquet_paths: list[str]):
    """
    Monkey-patch HfFileSystem and hf_hub_download in the workload module so
    that all HuggingFace I/O is redirected to the local tmpdir.
    """
    import workload as wl

    class FakeHfFileSystem:
        def glob(self, pattern: str) -> list[str]:
            # Return fake remote paths — one per real local file.
            return [f"datasets/fake_repo/data/shard_{i:02d}.parquet"
                    for i in range(NUM_FILES)]

    def fake_hf_hub_download(repo_id, filename, repo_type=None):
        # Extract shard index from filename and return the local path.
        basename = os.path.basename(filename)
        return os.path.join(tmpdir, basename)

    wl.HfFileSystem      = FakeHfFileSystem
    wl.hf_hub_download   = fake_hf_hub_download
    return wl


def _make_reader(wl, target_docs: int, num_partitions: int = 1) -> object:
    """Instantiate a ParquetBulkParamReader via its normal __init__."""
    params = {
        "target_vector_count": target_docs,
        "target_index_dimension": DIM,
        "target_field_name": FIELD_NAME,
        "batch_size": 200,
        "bulk_size": 100,
        "num_queries": 10,
        "queries_file": "/tmp/fake_gt.json",
        "hf_repo_id": "fake/repo",
        "hf_dataset_dir": "datasets/fake_repo/data",
    }

    class FakeWorkload:
        pass

    reader = wl.ParquetBulkParamReader(FakeWorkload(), params)
    return reader


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestPreCache(unittest.TestCase):
    """Pre-cache downloads only the shards needed to cover target_docs."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="test_partition_")
        self.parquet_paths = _write_synthetic_parquet(self.tmpdir)
        self.wl = _patch_workload_module(self.tmpdir, self.parquet_paths)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _count_downloaded(self) -> int:
        """Count how many shard files exist in tmpdir (proxy for downloads)."""
        return len([f for f in os.listdir(self.tmpdir) if f.endswith(".parquet")])

    def _partition_with_tracking(self, target_docs, partition_index=0, total_partitions=1):
        """Create a partition and track _get_local_cached_path calls made during it."""
        import workload as wl

        accessed = []
        original_cls_method = wl.ParquetBulkParamReader._get_local_cached_path

        def tracking(self_inner, path):
            accessed.append(path)
            return original_cls_method(self_inner, path)

        wl.ParquetBulkParamReader._get_local_cached_path = tracking
        try:
            reader = _make_reader(self.wl, target_docs=target_docs)
            reader.partition(partition_index, total_partitions)
        finally:
            wl.ParquetBulkParamReader._get_local_cached_path = original_cls_method

        # Return only calls made during pre-cache (partition()), not during
        # _stream_and_index generator creation — filter to unique shards.
        return accessed

    def test_precache_only_needed_shards_partial(self):
        """Requesting 1,500 docs (2 shards × 1,000 rows) should pre-cache exactly 2 shards."""
        accessed = self._partition_with_tracking(target_docs=1_500)
        unique = list(dict.fromkeys(os.path.basename(p) for p in accessed))
        self.assertEqual(len(unique), 2,
            f"Expected 2 shards pre-cached for 1,500 docs, got {len(unique)}: {unique}")
        self.assertIn("shard_00.parquet", unique)
        self.assertIn("shard_01.parquet", unique)

    def test_precache_all_shards_when_target_equals_total(self):
        """Requesting all 5,000 docs should pre-cache all 5 shards."""
        accessed = self._partition_with_tracking(target_docs=TOTAL_ROWS)
        unique = list(dict.fromkeys(os.path.basename(p) for p in accessed))
        self.assertEqual(len(unique), NUM_FILES,
            f"Expected {NUM_FILES} shards pre-cached, got {len(unique)}: {unique}")

    def test_precache_single_shard_for_small_target(self):
        """Requesting 500 docs (< 1 full shard) should pre-cache exactly 1 shard."""
        accessed = self._partition_with_tracking(target_docs=500)
        unique = list(dict.fromkeys(os.path.basename(p) for p in accessed))
        self.assertEqual(len(unique), 1,
            f"Expected 1 shard pre-cached for 500 docs, got {len(unique)}: {unique}")
        self.assertIn("shard_00.parquet", unique)


class TestFileSeeking(unittest.TestCase):
    """Non-zero partitions seek directly to their starting file."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="test_partition_")
        self.parquet_paths = _write_synthetic_parquet(self.tmpdir)
        self.wl = _patch_workload_module(self.tmpdir, self.parquet_paths)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_partition_slices_are_correct(self):
        """Each partition covers the right doc range with no overlap or gap."""
        target_docs   = 3_000  # 3 shards × 1,000 rows
        num_partitions = 3
        reader = _make_reader(self.wl, target_docs=target_docs)

        partitions = [reader.partition(i, num_partitions) for i in range(num_partitions)]
        expected = [(0, 1000), (1000, 2000), (2000, 3000)]
        for i, p in enumerate(partitions):
            self.assertEqual((p._start_doc, p._end_doc), expected[i],
                f"Partition {i}: expected {expected[i]}, got ({p._start_doc}, {p._end_doc})")

    def test_ingest_docs_no_overlap_no_gap(self):
        """Docs yielded across all partitions cover exactly target_docs with no duplicates."""
        target_docs    = 3_000
        num_partitions = 3
        reader = _make_reader(self.wl, target_docs=target_docs)

        all_ids = []
        for i in range(num_partitions):
            p = reader.partition(i, num_partitions)
            for bulk in p._generator:
                body = bulk["body"]
                # body is [action, doc, action, doc, ...]
                for j in range(0, len(body), 2):
                    all_ids.append(body[j]["index"]["_id"])

        self.assertEqual(len(all_ids), target_docs,
            f"Expected {target_docs} docs total, got {len(all_ids)}")
        self.assertEqual(len(set(all_ids)), target_docs,
            f"Duplicate doc IDs found across partitions")

    def test_single_partition_yields_all_docs(self):
        """A single partition with target_docs=2,500 yields exactly 2,500 docs."""
        target_docs = 2_500
        reader = _make_reader(self.wl, target_docs=target_docs)
        p = reader.partition(0, 1)

        count = 0
        for bulk in p._generator:
            count += bulk["bulk-size"]

        self.assertEqual(count, target_docs,
            f"Expected {target_docs} docs, got {count}")

    def test_cumulative_map_does_not_touch_uncached_shards(self):
        """_stream_and_index cumulative map stops at target_docs boundary."""
        target_docs = 2_000  # needs 2 shards exactly
        reader = _make_reader(self.wl, target_docs=target_docs)
        p = reader.partition(0, 1)

        # Track which remote paths _get_local_cached_path is called with
        accessed_in_stream = []
        original = p._get_local_cached_path

        def tracking(path):
            accessed_in_stream.append(path)
            return original(path)

        p._get_local_cached_path = tracking
        # Recreate generator with patched method
        p._generator = p._stream_and_index()

        # Drain the generator
        for _ in p._generator:
            pass

        # Shards accessed: shard_00 (schema detect) + shard_00, shard_01
        # (cumulative map) + shard_00, shard_01 (ingest loop) = at most 2 unique shards
        unique_shards = set(os.path.basename(p) for p in accessed_in_stream)
        self.assertNotIn("shard_02.parquet", unique_shards,
            f"shard_02 should not be accessed for target_docs={target_docs}, accessed: {unique_shards}")
        self.assertNotIn("shard_03.parquet", unique_shards,
            f"shard_03 should not be accessed for target_docs={target_docs}")
        self.assertNotIn("shard_04.parquet", unique_shards,
            f"shard_04 should not be accessed for target_docs={target_docs}")


if __name__ == "__main__":
    # Add parquet directory to path so workload module is importable
    sys.path.insert(0, os.path.dirname(__file__))
    unittest.main(verbosity=2)
