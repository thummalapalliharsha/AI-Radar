"""
Analyser queue tests.

These pin the defect that lost a major announcement: the pending set was ordered
newest-first and truncated in SQL, so with a steady collection inflow anything
below the batch size was pushed further down on every run and never analysed -
and an unanalysed article is invisible to the feed, the index and search.
"""

import unittest

from backend.app.services.process_articles import (
    ai_signal,
    analysis_priority,
    count_strong_ai_terms,
    newest_first_queue,
    prioritise_queue,
    source_priority,
)


def row(article_id, source, title, summary="", published=0.0):
    return {
        "id": article_id,
        "source": source,
        "title": title,
        "summary": summary,
        "published_timestamp": published,
    }


class TestStrongTermCounting(unittest.TestCase):

    def test_terms_are_matched_on_word_boundaries(self):
        # "Mumbai" contains "ai"; a substring test would score it as AI coverage.
        self.assertEqual(count_strong_ai_terms("Rain in Mumbai delays play"), 0)

    def test_hyphenated_model_names_still_count(self):
        self.assertGreaterEqual(count_strong_ai_terms("GPT-6 Astra released"), 1)

    def test_empty_text(self):
        self.assertEqual(count_strong_ai_terms(None), 0)
        self.assertEqual(count_strong_ai_terms(""), 0)


class TestPriorityComponents(unittest.TestCase):

    def test_primary_labs_outrank_media_and_aggregators(self):
        self.assertGreater(source_priority("OpenAI"), source_priority("TechCrunch AI"))
        self.assertGreater(
            source_priority("TechCrunch AI"), source_priority("YourStory Technology")
        )

    def test_unknown_source_is_lowest(self):
        self.assertEqual(source_priority("Some Random Blog"), 0)
        self.assertEqual(source_priority(None), 0)

    def test_ai_dense_text_signals_higher_than_lifestyle_text(self):
        dense = ai_signal(
            "New LLM foundation model improves machine learning inference",
            "The neural network was trained with deep learning on GPUs.",
        )
        thin = ai_signal("8 powerful quotes on society and solitude", "A reading list.")
        self.assertGreater(dense, thin)
        self.assertEqual(thin, 0)

    def test_priority_is_bounded(self):
        for article in [
            row(1, "OpenAI", "GPT-6 Astra: a new generation of intelligence"),
            row(2, "YourStory Technology", "Sneaker brand raises Series B"),
        ]:
            self.assertGreaterEqual(analysis_priority(article), 0)
            self.assertLessEqual(analysis_priority(article), 5)


class TestQueueOrdering(unittest.TestCase):

    def test_important_older_article_beats_newer_noise(self):
        # The exact inversion that lost the announcement: a flagship post from a
        # primary lab, published earlier, sitting behind a stream of newer
        # low-value items that a newest-first queue would analyse first.
        flagship = row(
            100, "OpenAI", "GPT-6 Astra: a new generation of intelligence",
            "A new frontier model with improved reasoning.", published=1000.0,
        )
        noise = [
            row(200 + n, "YourStory Technology", f"Quotes on solitude, part {n}",
                "A reading list.", published=2000.0 + n)
            for n in range(20)
        ]

        selected = prioritise_queue(noise + [flagship], limit=5)
        self.assertIn(100, selected)
        self.assertEqual(selected[0], 100)

    def test_oldest_first_within_a_priority_band(self):
        older = row(1, "OpenAI", "Frontier model update", published=1000.0)
        newer = row(2, "OpenAI", "Frontier model update", published=5000.0)
        self.assertEqual(prioritise_queue([newer, older], limit=2), [1, 2])

    def test_newest_first_queue_is_bounded_and_deterministic(self):
        older = row(1, "OpenAI", "AI model release", published=1000.0)
        newer = row(2, "OpenAI", "AI model release", published=5000.0)
        newest_tie = row(3, "OpenAI", "AI model release", published=5000.0)
        self.assertEqual(
            newest_first_queue([older, newer, newest_tie], limit=2),
            [2, 3],
        )

    def test_repeated_batches_drain_every_article_exactly_once(self):
        # The anti-starvation property: taking batch after batch must consume the
        # whole pending set, with no article revisited and none left behind.
        pending = {
            n: row(
                n,
                "OpenAI" if n % 3 == 0 else "YourStory Technology",
                "AI model release" if n % 2 == 0 else "Local news roundup",
                published=float(n),
            )
            for n in range(1, 48)
        }
        expected = set(pending)
        seen = []

        for _ in range(20):
            if not pending:
                break
            batch = prioritise_queue(list(pending.values()), limit=10)
            self.assertTrue(batch, "a non-empty queue must yield a batch")
            seen.extend(batch)
            for article_id in batch:
                # Simulating analysis: the row leaves the pending set.
                del pending[article_id]

        self.assertEqual(pending, {}, "queue did not drain")
        self.assertEqual(len(seen), len(set(seen)), "an article was analysed twice")
        self.assertEqual(set(seen), expected, "an article was never analysed")

    def test_batch_size_is_respected(self):
        articles = [row(n, "OpenAI", "AI model release", published=float(n)) for n in range(30)]
        self.assertEqual(len(prioritise_queue(articles, limit=7)), 7)
        self.assertEqual(prioritise_queue(articles, limit=0), [])

    def test_empty_queue(self):
        self.assertEqual(prioritise_queue([], limit=10), [])


if __name__ == "__main__":
    unittest.main()


