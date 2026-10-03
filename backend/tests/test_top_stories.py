import unittest
from unittest.mock import patch

from backend.app.services.top_stories import get_top_stories


def story(article_id, title, importance=5, source="Test Source", **extra):
    value = {
        "id": article_id,
        "title": title,
        "url": f"https://example.com/{article_id}",
        "source": source,
        "category": "AI Models",
        "summary": title,
        "ai_summary": title,
        "why_it_matters": "Impact",
        "importance": importance,
        "career_relevance": 5,
        "published_timestamp": float(article_id),
        "is_ai_news": 1,
        "should_show": 1,
        **extra,
    }
    return value


class TestTopStories(unittest.TestCase):

    def test_only_verified_displayable_articles_are_selected(self):
        articles = [
            story(1, "Verified high-impact story", importance=10),
            story(2, "Unverified story", importance=10, is_ai_news=0),
            story(3, "Hidden story", importance=10, should_show=0),
        ]
        with patch("backend.app.services.top_stories.cluster_articles", side_effect=lambda rows: rows):
            selected = get_top_stories(10, articles=articles)
        self.assertEqual([article["id"] for article in selected], [1])

    def test_highest_ranked_story_is_selected_first(self):
        articles = [story(1, "Low impact", importance=4), story(2, "High impact", importance=10)]
        with patch("backend.app.services.top_stories.cluster_articles", side_effect=lambda rows: rows):
            selected = get_top_stories(10, articles=articles)
        self.assertEqual(selected[0]["id"], 2)

    def test_fewer_than_ten_stories_is_supported(self):
        articles = [story(1, "Only story", importance=8)]
        with patch("backend.app.services.top_stories.cluster_articles", side_effect=lambda rows: rows):
            self.assertEqual(len(get_top_stories(10, articles=articles)), 1)

    def test_cluster_is_represented_once_and_metadata_survives(self):
        articles = [story(1, "Event representative", importance=9)]
        clustered = dict(articles[0])
        clustered.update({
            "cluster_id": "story-1",
            "member_ids": [1, 2, 3],
            "member_count": 3,
            "source_count": 3,
            "source_urls": [
                {"source": "Source A", "url": "https://a.example"},
                {"source": "Source B", "url": "https://b.example"},
            ],
        })
        with patch("backend.app.services.top_stories.cluster_articles", return_value=[clustered]):
            selected = get_top_stories(10, articles=articles)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["member_count"], 3)
        self.assertEqual(selected[0]["source_count"], 3)
        self.assertEqual(len(selected[0]["source_urls"]), 2)

    def test_duplicate_urls_are_collapsed(self):
        articles = [story(1, "Same event", importance=8), story(2, "Same event follow-up", importance=7)]
        articles[1]["url"] = articles[0]["url"]
        with patch("backend.app.services.top_stories.cluster_articles", side_effect=lambda rows: rows):
            self.assertEqual(len(get_top_stories(10, articles=articles)), 1)

    def test_research_does_not_dominate_when_other_strong_categories_exist(self):
        articles = [
            *[
                story(index, f"AI benchmark research paper {index}", importance=10, source="arXiv Artificial Intelligence", category="AI Research")
                for index in range(1, 9)
            ],
            story(20, "OpenAI launches a major model", importance=8, source="OpenAI", category="AI Models"),
            story(21, "NVIDIA announces new AI infrastructure", importance=8, source="NVIDIA Newsroom", category="AI Companies"),
            story(22, "Anthropic releases a new agent platform", importance=8, source="Anthropic", category="AI Agents"),
            story(23, "Google ships a new AI developer tool", importance=8, source="Google AI Blog", category="AI Tools"),
        ]
        with patch("backend.app.services.top_stories.cluster_articles", side_effect=lambda rows: rows):
            selected = get_top_stories(10, articles=articles)
        categories = {item.get("category") for item in selected}
        self.assertIn("AI Models", categories)
        self.assertIn("AI Companies", categories)
        self.assertIn("AI Agents", categories)
        self.assertLess(sum(item.get("category") == "AI Research" for item in selected), 8)

    def test_weak_story_is_not_promoted_only_for_diversity(self):
        articles = [
            story(1, "Important model release", importance=10, category="AI Models"),
            story(2, "Minor weak robotics note", importance=1, category="Robotics"),
        ]
        with patch("backend.app.services.top_stories.cluster_articles", side_effect=lambda rows: rows):
            selected = get_top_stories(1, articles=articles)
        self.assertEqual([item["id"] for item in selected], [1])

    def test_missing_optional_metadata_does_not_crash(self):
        minimal = {
            "id": 1,
            "title": "Minimal verified story",
            "url": "https://example.com/minimal",
            "source": "Source",
            "importance": 8,
            "published_timestamp": 1.0,
            "is_ai_news": 1,
            "should_show": 1,
        }
        with patch("backend.app.services.top_stories.cluster_articles", side_effect=lambda rows: rows):
            selected = get_top_stories(10, articles=[minimal])
        self.assertEqual(len(selected), 1)


if __name__ == "__main__":
    unittest.main()


