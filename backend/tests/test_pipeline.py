"""Unit tests for the collection, ranking and briefing logic."""

import unittest
from datetime import datetime, timedelta, timezone

from backend.app.config import (
    ALLOWED_CATEGORIES,
    BROWSABLE_CATEGORIES,
    MIN_SEMANTIC_SIMILARITY,
    MAX_SEMANTIC_DISTANCE,
    category_match_values,
    normalize_category,
)
from backend.app.services.briefing_generator import (
    build_sections,
    is_significant,
    parse_timestamp,
)
from backend.app.services.ai_analyzer import build_analysis_prompt
from backend.app.services.news_collector import RSS_SOURCES, clean_text, normalize_url
from backend.app.services.ranking import (
    calculate_ai_depth,
    calculate_recency,
    calculate_score,
    detect_special_category,
    rank_articles,
    remove_duplicates,
    select_top_stories,
)


class TestCollectorUnits(unittest.TestCase):

    def test_clean_text(self):
        raw = "<p>Hello &nbsp; <b>World</b>! &amp; welcome.</p>"
        self.assertEqual(clean_text(raw), "Hello World! & welcome.")

    def test_clean_text_handles_empty(self):
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(""), "")

    def test_normalize_url_strips_tracking_and_fragments(self):
        url = (
            "https://techcrunch.com/2026/08/29/ai-tool/"
            "?utm_source=twitter&utm_medium=feed#heading1"
        )
        self.assertEqual(
            normalize_url(url), "https://techcrunch.com/2026/08/29/ai-tool"
        )

    def test_every_feed_uses_a_canonical_category(self):
        for source in RSS_SOURCES:
            with self.subTest(source=source["name"]):
                self.assertIn(source["category"], ALLOWED_CATEGORIES)


class TestCategoryVocabulary(unittest.TestCase):
    """Requirement 13: one canonical spelling per category."""

    def test_legacy_developer_tools_name_maps_to_ai_tools(self):
        for spelling in [
            "Developers & Tools",
            "developers and tools",
            "Developer Tools",
            "dev tools",
        ]:
            with self.subTest(spelling=spelling):
                self.assertEqual(normalize_category(spelling), "AI Tools")

    def test_canonical_names_round_trip(self):
        for category in ALLOWED_CATEGORIES:
            self.assertEqual(normalize_category(category), category)

    def test_unknown_and_missing_values_fall_back(self):
        self.assertEqual(normalize_category(None), "Other")
        self.assertEqual(normalize_category(""), "Other")
        self.assertEqual(normalize_category("AI News"), "Other")
        self.assertEqual(normalize_category("something else entirely"), "Other")

    def test_sql_match_values_include_the_legacy_spelling(self):
        values = category_match_values("AI Tools")
        self.assertIn("ai tools", values)
        self.assertIn("developers & tools", values)
        self.assertTrue(all(value == value.lower() for value in values))

    def test_browsable_categories_exclude_the_catch_all(self):
        self.assertNotIn("Other", BROWSABLE_CATEGORIES)
        self.assertIn("AI Tools", BROWSABLE_CATEGORIES)


class TestRanking(unittest.TestCase):

    def test_incidental_india_mention_does_not_override_category(self):
        article = {
            "title": "New multimodal model expands medical reasoning",
            "summary": "The model was evaluated by a research team in Mumbai.",
            "category": "AI Models",
        }
        self.assertEqual(detect_special_category(article), "AI Models")

    def test_ai_robotics_gets_the_robotics_category(self):
        article = {
            "title": "Embodied AI model improves humanoid robot navigation",
            "summary": "A learned control policy helps the robot plan around obstacles.",
            "category": "AI Models",
        }
        self.assertEqual(detect_special_category(article), "Robotics")

    def test_generic_research_word_does_not_override_category(self):
        article = {
            "title": "Company research explores new cloud infrastructure",
            "summary": "The team published a general technology research update.",
            "category": "AI Companies",
        }
        self.assertEqual(detect_special_category(article), "AI Companies")

    def test_recency_of_a_fresh_article(self):
        now = datetime.now(timezone.utc).timestamp()
        self.assertEqual(calculate_recency(now), 10.0)

    def test_recency_decays_with_age(self):
        now = datetime.now(timezone.utc)
        two_days = (now - timedelta(days=2, hours=1)).timestamp()
        two_weeks = (now - timedelta(days=14)).timestamp()
        self.assertLess(calculate_recency(two_days), 10.0)
        self.assertLess(calculate_recency(two_weeks), calculate_recency(two_days))

    def test_recency_of_garbage_input(self):
        self.assertEqual(calculate_recency(None), 0.0)
        self.assertEqual(calculate_recency("not a date"), 0.0)

    def test_india_story_detection_and_score(self):
        article = {
            "title": "IndiaAI mission accelerates GPU clusters in Hyderabad and Bengaluru",
            "summary": "MeitY announced deep learning infrastructure grants for Indian researchers.",
            "ai_summary": "Major AI computing initiative in India focused on LLM training.",
            "why_it_matters": "Expands open-source AI capability for Indian developers.",
            "importance": 8,
            "career_relevance": 8,
            "published_timestamp": datetime.now(timezone.utc).timestamp(),
            "source": "Google India Blog",
        }
        self.assertEqual(detect_special_category(article), "India AI")
        score = calculate_score(article)
        self.assertGreaterEqual(score, 7.0)
        self.assertLessEqual(score, 10.0)

    def test_developer_story_uses_the_canonical_tools_name(self):
        article = {
            "title": "New SDK and API for code generation",
            "summary": "A developer toolkit for programming with LLMs.",
            "ai_summary": "",
            "why_it_matters": "",
            "category": "AI Models",
        }
        self.assertEqual(detect_special_category(article), "AI Tools")

    def test_special_category_is_always_canonical(self):
        article = {"title": "Nothing special here", "category": "Developers & Tools"}
        self.assertIn(detect_special_category(article), ALLOWED_CATEGORIES)

    def test_ai_depth_prefers_dense_ai_text(self):
        dense = {
            "title": "New LLM foundation model improves reasoning",
            "summary": "The neural network was trained for deep learning inference.",
        }
        thin = {"title": "Company reports quarter results", "summary": "Revenue and stock."}
        self.assertGreater(calculate_ai_depth(dense), calculate_ai_depth(thin))

    def test_duplicates_are_removed_by_url_and_title(self):
        articles = [
            {"url": "https://a.example/1", "title": "Same Story"},
            {"url": "https://a.example/1", "title": "Same Story"},
            {"url": "https://a.example/2", "title": "same story"},
            {"url": "https://a.example/3", "title": "Different"},
        ]
        self.assertEqual(len(remove_duplicates(articles)), 2)

    def test_ranking_tolerates_a_missing_timestamp(self):
        articles = [
            {"title": "A", "source": "OpenAI", "importance": 8, "published_timestamp": None},
            {
                "title": "B",
                "source": "OpenAI",
                "importance": 6,
                "published_timestamp": datetime.now(timezone.utc).timestamp(),
            },
        ]
        ranked = rank_articles(articles)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0]["title"], "B")

    def test_top_story_selection_caps_one_source(self):
        articles = [
            {
                "title": f"Story {n}",
                "source": "OpenAI",
                "special_category": "AI Models",
                "ranking_score": 9.0 - n,
            }
            for n in range(6)
        ]
        selected = select_top_stories(articles, limit=6, max_per_source=3)
        self.assertLessEqual(len(selected), 3)


class TestBriefingLogic(unittest.TestCase):

    def test_significance_thresholds(self):
        self.assertTrue(
            is_significant({"importance": 8, "career_relevance": 6, "ranking_score": 6.5})
        )
        self.assertFalse(
            is_significant({"importance": 4, "career_relevance": 4, "ranking_score": 4.5})
        )

    def test_significance_of_an_empty_article(self):
        self.assertFalse(is_significant({}))

    def test_timestamp_parsing_accepts_iso_and_epoch(self):
        parsed = parse_timestamp("2026-08-30T18:17:39.002400+00:00")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.year, 2026)

        epoch = parse_timestamp(1788284137.0)
        self.assertIsNotNone(epoch)
        self.assertEqual(epoch.tzinfo, timezone.utc)

        self.assertIsNone(parse_timestamp(None))
        self.assertIsNone(parse_timestamp(""))

    def test_sections_use_the_canonical_names(self):
        stories = [
            {"id": 1, "category": "AI Research"},
            {"id": 2, "category": "Developers & Tools"},
            {"id": 3, "category": "AI Tools"},
            {"id": 4, "category": "AI Jobs & Careers"},
            {"id": 5, "category": "India AI"},
            {"id": 6, "category": "AI Models"},
        ]
        sections = build_sections(stories)
        self.assertEqual(len(sections["research"]), 1)
        # Both the legacy and the canonical spelling land in one section.
        self.assertEqual(len(sections["developers_tools"]), 2)
        self.assertEqual(len(sections["career"]), 1)
        self.assertEqual(len(sections["india_ai"]), 1)
        self.assertEqual(len(sections["other"]), 1)
        self.assertEqual(len(sections["top_stories"]), 3)


class TestSearchThresholdConfig(unittest.TestCase):

    def test_analyzer_prompt_has_specific_category_guardrails(self):
        prompt = build_analysis_prompt("title", "source", "summary")
        for phrase in [
            "individual job postings",
            "India is central",
            "AI-powered robotics",
            "primary subject is AI/ML",
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, prompt)

    def test_similarity_and_distance_are_complementary(self):
        self.assertAlmostEqual(
            MIN_SEMANTIC_SIMILARITY + MAX_SEMANTIC_DISTANCE, 1.0, places=6
        )

    def test_threshold_is_in_a_sane_range(self):
        self.assertGreater(MIN_SEMANTIC_SIMILARITY, 0.0)
        self.assertLess(MIN_SEMANTIC_SIMILARITY, 1.0)


if __name__ == "__main__":
    unittest.main()



