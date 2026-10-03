import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import numpy as np

from backend.app.services.story_clustering import cluster_articles


def article(article_id, title, source, score=7.0, published=1000.0, category="AI Companies"):
    return {
        "id": article_id,
        "title": title,
        "source": source,
        "url": f"https://example.com/{article_id}",
        "summary": title,
        "ai_summary": title,
        "why_it_matters": "Impact",
        "category": category,
        "importance": 8,
        "career_relevance": 7,
        "ranking_score": score,
        "published_timestamp": published,
        "is_ai_news": 1,
        "should_show": 1,
    }


class TestStoryClustering(unittest.TestCase):

    def cluster(self, articles, vectors):
        with patch("backend.app.services.story_clustering.load_index", return_value=(None, [])):
            return cluster_articles(articles, vectors=vectors)

    def test_similar_event_clusters_and_preserves_sources(self):
        articles = [
            article(1, "OpenAI launches GPT X", "OpenAI", score=8, published=2000),
            article(2, "OpenAI launches GPT X for developers", "TechCrunch", score=7, published=2100),
        ]
        clusters = self.cluster(articles, {1: np.array([1.0, 0.0]), 2: np.array([0.99, 0.01])})
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["member_count"], 2)
        self.assertEqual(set(clusters[0]["sources"]), {"OpenAI", "TechCrunch"})
        self.assertEqual(clusters[0]["source_count"], 2)
        self.assertEqual(clusters[0]["representative_article"]["id"], 1)
        self.assertEqual(clusters[0]["newest_published_timestamp"], 2100)

    def test_different_events_remain_separate(self):
        articles = [
            article(1, "OpenAI releases GPT X", "OpenAI"),
            article(2, "OpenAI hires a new executive", "Reuters"),
        ]
        vectors = {1: np.array([1.0, 0.0]), 2: np.array([0.99, 0.01])}
        self.assertEqual(len(self.cluster(articles, vectors)), 2)

    def test_same_category_unrelated_articles_remain_separate(self):
        articles = [
            article(1, "New AI benchmark for medical imaging", "Lab A", category="AI Research"),
            article(2, "New AI benchmark for language models", "Lab B", category="AI Research"),
        ]
        vectors = {1: np.array([1.0, 0.0]), 2: np.array([0.99, 0.01])}
        self.assertEqual(len(self.cluster(articles, vectors)), 2)

    def test_related_arxiv_papers_remain_separate(self):
        articles = [
            article(
                1,
                "Automating Quadratic Unconstrained Binary Optimization (QUBO) Formulation Generation from Natural Language",
                "arXiv Artificial Intelligence",
                category="AI Research",
            ),
            article(
                2,
                "Improving Natural-Language Combinatorial-Optimization Accuracy in Resource-Constrained Language Models via Formal Abstractions",
                "arXiv Artificial Intelligence",
                category="AI Research",
            ),
        ]
        vectors = {1: np.array([1.0, 0.0]), 2: np.array([0.99, 0.01])}
        self.assertEqual(len(self.cluster(articles, vectors)), 2)

    def test_duplicate_url_is_safe_and_source_count_is_unique(self):
        articles = [
            article(1, "OpenAI launches GPT X", "OpenAI"),
            article(2, "OpenAI launches GPT X", "OpenAI"),
        ]
        articles[1]["url"] = articles[0]["url"]
        clusters = self.cluster(articles, {1: np.array([1.0, 0.0]), 2: np.array([1.0, 0.0])})
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["sources"], ["OpenAI"])

    def test_exact_duplicate_url_forms_one_cluster(self):
        articles = [
            article(1, "OpenAI launches GPT X", "OpenAI"),
            article(2, "Different headline for the same page", "TechCrunch"),
        ]
        articles[1]["url"] = articles[0]["url"]
        clusters = self.cluster(articles, {})
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["member_count"], 2)

    def test_near_identical_title_forms_one_cluster(self):
        articles = [
            article(1, "OpenAI launches GPT X", "OpenAI"),
            article(2, "OpenAI launches the new GPT X", "TechCrunch"),
        ]
        clusters = self.cluster(articles, {})
        self.assertEqual(len(clusters), 1)

    def test_cross_source_concrete_event_forms_one_cluster(self):
        articles = [
            article(1, "OpenAI launches GPT X model", "OpenAI", score=8),
            article(2, "OpenAI announces GPT X release for developers", "Reuters"),
            article(3, "GPT X launched by OpenAI with developer access", "TechCrunch"),
        ]
        vectors = {
            1: np.array([1.0, 0.0]),
            2: np.array([0.995, 0.01]),
            3: np.array([0.99, 0.02]),
        }
        clusters = self.cluster(articles, vectors)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["member_count"], 3)
        self.assertEqual(clusters[0]["source_count"], 3)

    def test_different_company_announcements_remain_separate(self):
        articles = [
            article(1, "OpenAI launches GPT X model", "OpenAI"),
            article(2, "Anthropic launches Claude Y model", "Anthropic"),
        ]
        vectors = {1: np.array([1.0, 0.0]), 2: np.array([0.995, 0.01])}
        self.assertEqual(len(self.cluster(articles, vectors)), 2)

    def test_qubo_and_sddl_research_papers_remain_separate(self):
        articles = [
            article(
                1,
                "Automating Quadratic Unconstrained Binary Optimization (QUBO) Formulation Generation from Natural Language",
                "arXiv Artificial Intelligence",
                category="AI Research",
            ),
            article(
                2,
                "Improving Natural-Language Combinatorial-Optimization Accuracy in Resource-Constrained Language Models via Formal Abstractions",
                "arXiv Artificial Intelligence",
                category="AI Research",
            ),
        ]
        vectors = {1: np.array([1.0, 0.0]), 2: np.array([0.999, 0.01])}
        self.assertEqual(len(self.cluster(articles, vectors)), 2)

    def test_different_research_papers_with_shared_terms_remain_separate(self):
        articles = [
            article(1, "Benchmarking language models for medical reasoning", "Research Lab A", category="AI Research"),
            article(2, "Training language models for code generation", "Research Lab B", category="AI Research"),
        ]
        vectors = {1: np.array([1.0, 0.0]), 2: np.array([0.999, 0.01])}
        self.assertEqual(len(self.cluster(articles, vectors)), 2)

    def test_representative_prefers_authority_then_score(self):
        articles = [
            article(1, "OpenAI launches GPT X", "OpenAI", score=6, published=3000),
            article(2, "OpenAI launches GPT X", "Unknown", score=10, published=4000),
        ]
        clusters = self.cluster(articles, {1: np.array([1.0, 0.0]), 2: np.array([0.99, 0.01])})
        self.assertEqual(clusters[0]["id"], 1)

    def test_cluster_metadata_and_diagnostics(self):
        articles = [
            article(1, "OpenAI launches GPT X model", "OpenAI", score=8, published=1000),
            article(2, "OpenAI announces GPT X release for developers", "Reuters", published=2000),
        ]
        vectors = {1: np.array([1.0, 0.0]), 2: np.array([0.99, 0.01])}
        output = StringIO()
        with redirect_stdout(output):
            clusters = self.cluster(articles, vectors)
        cluster = clusters[0]
        self.assertEqual(cluster["member_ids"], [2, 1])
        self.assertEqual(cluster["member_count"], 2)
        self.assertEqual(cluster["source_count"], 2)
        self.assertEqual(len(cluster["source_urls"]), 2)
        self.assertEqual(cluster["newest_published_timestamp"], 2000)
        self.assertEqual(cluster["representative_article"]["id"], 1)
        diagnostics = output.getvalue()
        for label, value in [
            ("Articles considered", "2"),
            ("Story clusters", "1"),
            ("Articles collapsed", "1"),
            ("Cross-source clusters", "1"),
            ("Same-source duplicate clusters", "0"),
            ("Largest cluster", "2"),
        ]:
            self.assertIn(f"{label}: {value}", diagnostics)

    def test_ineligible_articles_are_not_clustered(self):
        articles = [article(1, "OpenAI launches GPT X", "OpenAI")]
        articles[0]["is_ai_news"] = 0
        self.assertEqual(self.cluster(articles, {1: np.array([1.0, 0.0])}), [])


if __name__ == "__main__":
    unittest.main()


