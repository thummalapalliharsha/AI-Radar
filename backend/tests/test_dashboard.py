"""
Streamlit dashboard smoke tests.

Requirement 10: the dashboard must actually open and render. AppTest executes
backend/app/ui/dashboard.py exactly as `streamlit run` does, so an import error, a
missing table or a crash in any view fails here rather than in the browser.
"""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

DASHBOARD = Path(__file__).resolve().parents[1] / "app" / "ui" / "dashboard.py"

VIEWS = [
    "Live Latest AI",
    "Top Stories",
    "AI Semantic Search",
    "Intelligence Briefings",
    "AI Research",
    "AI Tools & Platforms",
    "Models & Agents",
    "AI Jobs & Careers",
    "India AI",
    "AI Robotics",
]


def start_app() -> AppTest:
    app = AppTest.from_file(str(DASHBOARD), default_timeout=300)
    app.run()
    return app


def problems(app: AppTest):
    return [str(item.value) for item in app.exception] + [
        str(item.value) for item in app.error
    ]


class TestDashboardRenders(unittest.TestCase):

    def test_refresh_service_is_available(self):
        from backend.app.services.refresh_radar import refresh_radar

        self.assertTrue(callable(refresh_radar))

    def test_dashboard_relevance_validator_export_is_available(self):
        from backend.app.ui.components import validate_retrieved_articles

        self.assertTrue(callable(validate_retrieved_articles))

    def test_app_starts_without_error(self):
        app = start_app()
        self.assertEqual(problems(app), [])
        self.assertFalse(app.exception)

    def test_hero_and_navigation_are_present(self):
        app = start_app()
        html = " ".join(str(block.value) for block in app.markdown)
        self.assertIn('class="radar-hero"', html)
        self.assertIn("AI RADAR", html)
        self.assertEqual(app.sidebar.radio[0].options, VIEWS)

    def test_stylesheet_makes_no_external_request(self):
        app = start_app()
        html = " ".join(str(block.value) for block in app.markdown)
        # Requirement 20: no Google Fonts, no CDN.
        self.assertNotIn("fonts.googleapis.com", html)
        self.assertNotIn("@import url(", html)

    def test_category_filter_offers_only_canonical_names(self):
        from backend.app.config import BROWSABLE_CATEGORIES

        app = start_app()
        self.assertEqual(
            app.selectbox[0].options, ["All Categories", *BROWSABLE_CATEGORIES]
        )

    def test_every_view_renders(self):
        for view in VIEWS:
            with self.subTest(view=view):
                app = start_app()
                app.sidebar.radio[0].set_value(view).run()
                self.assertEqual(problems(app), [])


if __name__ == "__main__":
    unittest.main()




