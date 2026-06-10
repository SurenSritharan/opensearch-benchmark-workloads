import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

# Assuming your runner code is saved in a file named msmarco_runner.py
# If it's in the same file, you can just paste this below your runner class
from workload import MsMarcoVectorSearchRunner

class TestMsMarcoVectorSearchRunner(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.runner = MsMarcoVectorSearchRunner()
        
    def create_mock_os_response(self, returned_ids):
        """Helper to simulate an OpenSearch search response"""
        mock_response = {
            "hits": {
                "hits": [{"_id": str(doc_id)} for doc_id in returned_ids]
            }
        }
        mock_client = AsyncMock()
        mock_client.search.return_value = mock_response
        return mock_client

    async def test_partial_recall_calculation(self):
        """Test a normal scenario where some hits match and some miss"""
        # Ground truth top 5 docs
        mock_gt = [100, 101, 102, 103, 104] 
        # OpenSearch returned docs (3 matches: 100, 102, 104. 2 misses: 999, 888)
        mock_returned = [100, 999, 102, 888, 104]
        
        mock_client = self.create_mock_os_response(mock_returned)
        
        params = {
            "index": "test-index",
            "body": {},
            "k": 5,
            "ground_truth_indices": mock_gt
        }
        
        result = await self.runner(mock_client, params)
        
        # Verify the structure
        self.assertTrue(result["success"])
        self.assertIn("stats", result)
        
        # 3 matches out of 5 = 0.60 recall
        self.assertAlmostEqual(result["stats"]["recall@k"], 0.60)
        # OpenSearch top-1 was 100, ground truth top-1 was 100 -> Perfect top match
        self.assertEqual(result["stats"]["recall@1"], 1.0)

    async def test_zero_recall_calculation(self):
        """Test scenario where completely wrong documents are returned"""
        mock_gt = [10, 20, 30]
        mock_returned = [77, 88, 99]
        
        mock_client = self.create_mock_os_response(mock_returned)
        params = {
            "index": "test-index",
            "body": {},
            "k": 3,
            "ground_truth_indices": mock_gt
        }
        
        result = await self.runner(mock_client, params)
        
        self.assertEqual(result["stats"]["recall@k"], 0.0)
        self.assertEqual(result["stats"]["recall@1"], 0.0)

    async def test_empty_opensearch_response(self):
        """Ensure code handles an empty response gracefully without crashing"""
        mock_gt = [1, 2, 3]
        
        mock_client = AsyncMock()
        mock_client.search.return_value = {"hits": {"hits": []}} # No documents returned
        
        params = {
            "index": "test-index",
            "body": {},
            "k": 3,
            "ground_truth_indices": mock_gt
        }
        
        result = await self.runner(mock_client, params)
        
        self.assertEqual(result["stats"]["recall@k"], 0.0)
        self.assertEqual(result["stats"]["recall@1"], 0.0)

if __name__ == "__main__":
    unittest.main()