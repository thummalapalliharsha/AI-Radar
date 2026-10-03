"""
Search-gate and RAG-grounding tests.

The keyword gate and the answer prompt are asserted without Ollama so the suite
stays deterministic; the cases that genuinely need the local models are skipped
when Ollama is not running.
"""

import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from backend.app.config import MIN_SEMANTIC_SIMILARITY
from backend.app.services.vector_store import (
    get_index_stats,
    has_temporal_intent,
    recency_score,
    search_vector_store,
)
from backend.app.ui.components import (
    NO_GROUNDED_FACTS_MESSAGE,
    build_context_block,
    check_ollama,
    classify_search_intent,
    keyword_search_intent,
    markdown_to_html,
    story_summary,
    validate_retrieved_articles,
)

OLLAMA = check_ollama()
INDEX_STATS = get_index_stats()

REQUIRES_STACK = unittest.skipUnless(
    OLLAMA["online"] and OLLAMA["has_embedding"] and INDEX_STATS["healthy"],
    "needs a running Ollama with an embedding model and a healthy FAISS index",
)


class TestQueryGate(unittest.TestCase):
    """Requirements 14 and 15: only AI questions reach retrieval."""

    def test_ai_vocabulary_is_accepted_without_the_llm(self):
        for query in [
            "What is the latest LLM from OpenAI?",
            "AI agents for cybersecurity",
            "nvidia gpu inference roadmap",
            "open source model weights release",
        ]:
            with self.subTest(query=query):
                self.assertTrue(keyword_search_intent(query))

    def test_off_topic_vocabulary_is_rejected_without_the_llm(self):
        for query in [
            "What is the cricket score in Mumbai?",
            "best biryani recipe in Chennai",
            "weather forecast for Bengaluru tomorrow",
            "bitcoin price today",
        ]:
            with self.subTest(query=query):
                self.assertFalse(keyword_search_intent(query))

    def test_mumbai_is_not_mistaken_for_ai(self):
        # "Mumbai" contains the substring "ai": the gate must match on word
        # boundaries or every cricket question looks AI-related.
        self.assertFalse(keyword_search_intent("Who won in Mumbai yesterday?"))

    def test_ambiguous_query_defers_to_the_classifier(self):
        self.assertIsNone(keyword_search_intent("what happened yesterday"))

    def test_blank_query_has_no_verdict(self):
        self.assertIsNone(classify_search_intent("   "))

    def test_classifier_agrees_on_clear_cases(self):
        self.assertTrue(classify_search_intent("What is the latest LLM from OpenAI?"))
        self.assertFalse(classify_search_intent("What is the cricket score in Mumbai?"))

    def test_temporal_terms_are_detected(self):
        for query in [
            "latest AI models",
            "recent robotics news",
            "what was announced today",
            "current AI policy",
        ]:
            with self.subTest(query=query):
                self.assertTrue(has_temporal_intent(query))

    def test_normal_explanation_query_is_not_temporal(self):
        self.assertFalse(has_temporal_intent("How do retrieval augmented systems work?"))

    def test_recent_article_gets_higher_temporal_score(self):
        now = 1_757_680_000.0
        reference = datetime.fromtimestamp(now, timezone.utc)
        recent = recency_score(now, reference)
        old = recency_score(now - 20 * 86400, reference)
        self.assertGreater(recent, old)


class TestGroundingHelpers(unittest.TestCase):
    """Requirement 16: the answer prompt can only draw on retrieved articles."""

    def test_context_block_uses_the_analysed_summary(self):
        articles = [
            {
                "title": "A new reasoning model",
                "source": "OpenAI",
                "category": "AI Models",
                "published": "2026-09-01T10:00:00+00:00",
                "summary": "Analysed summary text.",
                "why_it_matters": "Impact text.",
                "url": "https://example.com/a",
            }
        ]
        block = build_context_block(articles)
        self.assertIn("[1] TITLE: A new reasoning model", block)
        self.assertIn("SUMMARY: Analysed summary text.", block)
        self.assertIn("WHY IT MATTERS: Impact text.", block)
        self.assertNotIn("SUMMARY: None", block)

    def test_summary_prefers_the_llm_text_over_raw_rss(self):
        row = {"summary": "raw rss blurb", "ai_summary": "analysed summary"}
        self.assertEqual(story_summary(row), "analysed summary")
        self.assertEqual(story_summary({"summary": "raw only"}), "raw only")
        self.assertEqual(story_summary({}), "")

    def test_refusal_sentence_is_defined(self):
        self.assertIn("not documented", NO_GROUNDED_FACTS_MESSAGE)

    def test_answer_markup_is_escaped(self):
        html = markdown_to_html("- **Bold** [1]\n- <script>alert(1)</script>")
        self.assertIn("<strong>Bold</strong>", html)
        self.assertIn('<span class="citation">[1]</span>', html)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script>", html)

    def test_relevance_validator_keeps_only_model_selected_candidates(self):
        articles = [
            {"title": "OpenAI releases a reasoning model", "category": "AI Models", "summary": "model release"},
            {"title": "NVIDIA opens a new office", "category": "AI Companies", "summary": "office expansion"},
        ]
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"response": '{"relevant_numbers": [1]}'}
        with patch("backend.app.ui.components.requests.post", return_value=response):
            validated = validate_retrieved_articles("latest AI models", articles)
        self.assertEqual([article["title"] for article in validated], [articles[0]["title"]])

    def test_agent_query_keeps_direct_agent_article(self):
        article = {
            "title": "New AI agents improve long-horizon planning",
            "category": "AI Agents",
            "summary": "The agent system plans and executes multi-step tasks.",
        }
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"response": '{"relevant_numbers": [1]}'}
        with patch("backend.app.ui.components.requests.post", return_value=response):
            self.assertEqual(
                validate_retrieved_articles(
                    "What are the latest developments in AI agents?", [article]
                ),
                [article],
            )

    def test_specific_underwater_agriculture_rejects_generic_fluid_control(self):
        article = {
            "title": "Self-Evolving Scientific Agent Designs Physically Reasoned White-Box Fluid Control",
            "category": "AI Research",
            "summary": "An agent designs fluid-control policies for scientific simulations.",
        }
        response = Mock()
        response.raise_for_status.return_value = None
        # Even an over-permissive model response must not bypass the topic anchor.
        response.json.return_value = {"response": '{"relevant_numbers": [1]}'}
        with patch("backend.app.ui.components.requests.post", return_value=response):
            self.assertEqual(
                validate_retrieved_articles(
                    "What are the latest AI developments in underwater agriculture?",
                    [article],
                ),
                [],
            )


class TestVectorStoreContract(unittest.TestCase):

    def test_index_stats_expose_the_legacy_keys(self):
        self.assertIn("indexed_documents", INDEX_STATS)
        self.assertIn("embedding_model", INDEX_STATS)
        self.assertGreaterEqual(INDEX_STATS["indexed_documents"], 0)

    @REQUIRES_STACK
    def test_hits_carry_similarity_and_distance(self):
        hits = search_vector_store("AI models and reasoning", top_k=3)
        self.assertIsInstance(hits, list)
        self.assertTrue(hits)
        for hit in hits:
            self.assertIn("title", hit)
            self.assertIn("source", hit)
            self.assertIn("similarity", hit)
            self.assertIn("distance", hit)
            self.assertGreaterEqual(hit["similarity"], MIN_SEMANTIC_SIMILARITY)

    @REQUIRES_STACK
    def test_threshold_is_enforced_before_results_are_returned(self):
        # Requirement 17: nothing below the threshold may reach the dashboard.
        self.assertEqual(search_vector_store("how to fix a leaking tap", top_k=5), [])


if __name__ == "__main__":
    unittest.main()



