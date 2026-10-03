"""
FAISS + Ollama semantic-search integration tests.

These exercise the real local stack, so every case is skipped when Ollama is
not running or the index has not been built yet.
"""

import unittest

from backend.app.config import MIN_SEMANTIC_SIMILARITY
from backend.app.services.vector_store import get_index_stats, search_articles
from backend.app.ui.components import check_ollama

INDEX_STATS = get_index_stats()
OLLAMA = check_ollama()

REQUIRES_STACK = unittest.skipUnless(
    OLLAMA["online"] and OLLAMA["has_embedding"] and INDEX_STATS["healthy"],
    "needs a running Ollama with an embedding model and a healthy FAISS index",
)

AI_QUERIES = [
    "latest AI models and new model releases",
    "AI agents and cybersecurity",
]

OFF_TOPIC_QUERIES = [
    "best chicken biryani recipe",
    "IPL cricket final scorecard",
]


@REQUIRES_STACK
class TestFaissSemanticSearch(unittest.TestCase):

    def test_index_is_aligned_with_metadata(self):
        self.assertEqual(INDEX_STATS["indexed"], INDEX_STATS["metadata"])
        self.assertGreater(INDEX_STATS["dimension"], 0)

    def test_ai_queries_return_grounded_hits(self):
        for query in AI_QUERIES:
            with self.subTest(query=query):
                hits = search_articles(query, top_k=5)
                self.assertTrue(hits, f"no hits above threshold for {query!r}")
                for hit in hits:
                    self.assertGreaterEqual(hit["similarity"], MIN_SEMANTIC_SIMILARITY)
                    self.assertAlmostEqual(
                        hit["distance"], 1.0 - hit["similarity"], places=5
                    )

    def test_hits_are_ordered_by_similarity(self):
        hits = search_articles("AI agents and cybersecurity", top_k=5)
        scores = [hit["similarity"] for hit in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_off_topic_queries_are_filtered_out(self):
        for query in OFF_TOPIC_QUERIES:
            with self.subTest(query=query):
                self.assertEqual(
                    search_articles(query, top_k=5),
                    [],
                    "an unrelated query cleared the similarity threshold",
                )

    def test_threshold_can_be_disabled_for_inspection(self):
        raw = search_articles(
            "best chicken biryani recipe", top_k=3, min_similarity=None
        )
        self.assertTrue(raw, "raw neighbour lookup should still return rows")
        self.assertLess(raw[0]["similarity"], MIN_SEMANTIC_SIMILARITY)

    def test_blank_query_returns_nothing(self):
        self.assertEqual(search_articles("   "), [])


if __name__ == "__main__":
    unittest.main()



