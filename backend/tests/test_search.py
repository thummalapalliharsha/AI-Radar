"""
Search-gate and RAG-grounding tests.

The keyword gate and the answer prompt are asserted without Ollama so the suite
stays deterministic; the cases that genuinely need the local models are skipped
when Ollama is not running.
"""

import json
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
    _append_explicit_release_timing,
    _complete_source_sentence,
    build_context_block,
    check_ollama,
    classify_search_intent,
    generate_grounded_answer,
    has_valid_answer_citations,
    keyword_search_intent,
    markdown_to_html,
    story_summary,
    validate_grounded_answer,
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
        evidence = json.loads(block)
        self.assertEqual(
            evidence,
            [
                {
                    "citation_number": 1,
                    "article_id": "",
                    "title": "A new reasoning model",
                    "summary": "Analysed summary text.",
                }
            ],
        )
        self.assertNotIn("Impact text.", block)
        self.assertNotIn("example.com", block)

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

    def test_b46_query_anchors_exclude_tangential_results(self):
        cases = [
            (
                "Kimi K3.1 Pro model launch and release",
                {
                    "id": 4820,
                    "title": "China's Moonshot AI launches Kimi K3.1 Pro; plans full model weights release by November",
                    "summary": "Chinese AI company Moonshot AI has unveiled Kimi K3.1 Pro, a 3-trillion-parameter open-weight AI model, alongside a high-speed variant, Kimi K3.1 Pro Highspeed.",
                },
                {
                    "id": 2776,
                    "title": "Kimi-maker Moonshot AI targets $2B in annual revenue",
                    "summary": "K3's usage figures have declined slightly in recent months, but the system generates 300 billion tokens each day by K3 models on the OpenRouter data.",
                },
            ),
            (
                "Muse Glimmer 30B on-device agentic model",
                {
                    "id": 5352,
                    "title": "Muse Glimmer 30B: An On-Device Agentic Model",
                    "summary": "Muse Glimmer is an open-source 30B parameter causal language model that integrates multi-step reasoning, reliable tool use, and failure recovery into a single model for autonomous agentic tasks on consumer hardware.",
                },
                {
                    "id": 2343,
                    "title": "Meta's Muse AI works and creeps me out",
                    "summary": "Meta has launched its new Muse AI assistant, which uses an AI agent to perform tasks such as online shopping, email management, and trip-planning.",
                },
            ),
            (
                "LLM inference over telco wide-area networks",
                {
                    "id": 5213,
                    "title": "InferSlice: A Reference Architecture for SLO-driven LLM Inference over a Telco Wide-Area Network",
                    "summary": "InferSlice is a reference architecture for Service-Level Objective (SLO)-driven LLM inference over a telco infrastructure, integrating network slicing, GPU partitioning, elastic serving and observability.",
                },
                {
                    "id": 3159,
                    "title": "You Cannot Pick a Provider From the Price List: Market-Aware Routing for Open-Weight LLM Inference",
                    "summary": "Open-weight inference markets introduce a new decision axis: after choosing a model, a client must choose which provider serves it.",
                },
            ),
            (
                "AI agents helping robotic engineers",
                {
                    "id": 5212,
                    "title": "Can AI Agents Help Robotic Engineers? Lessons from Two Focus Days in Industry",
                    "summary": "A case study on the use of AI agents by an SME programming industrial robots shows that the agent did not make the engineers more productive initially, but improved over time as they learned to use it.",
                },
                {
                    "id": 1426,
                    "title": "From theory to delivery: How Atos upskilled 400 engineers in agentic AI",
                    "summary": "Atos upskilled 400 engineers in agentic AI through hands-on learning with an AI League event on AWS.",
                },
            ),
            (
                "Building AI products for Indian users scale access cost",
                {
                    "id": 3896,
                    "title": "Building AI Products For Indian Users: The Challenge Of Scale, Access, And Cost",
                    "summary": "The challenge of taking AI products to millions of Indian users is making them work across budget smartphones, multiple languages, and other technical limitations.",
                },
                {
                    "id": 1494,
                    "title": "Introducing OpenAI models on Amazon Bedrock for in-country inferencing in India",
                    "summary": "Amazon Bedrock now supports the OpenAI GPT-5.6 models in India with India geographic cross-Region inference.",
                },
            ),
        ]
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"response": '{"relevant_numbers": [1, 2]}'}

        for query, relevant, tangential in cases:
            with self.subTest(query=query), patch(
                "backend.app.ui.components.requests.post", return_value=response
            ):
                validated = validate_retrieved_articles(
                    query, [relevant, tangential]
                )
            self.assertEqual([article["id"] for article in validated], [relevant["id"]])

    def test_grounded_answer_requires_valid_citations_and_support_review(self):
        article = {
            "id": 4820,
            "title": "China's Moonshot AI launches Kimi K3.1 Pro",
            "source": "Economic Times Technology",
            "category": "AI Models",
            "summary": "Moonshot AI unveiled Kimi K3.1 Pro, a 3-trillion-parameter model.",
            "url": "https://example.com/kimi",
        }
        draft = Mock()
        draft.raise_for_status.return_value = None
        draft.json.return_value = {
            "response": json.dumps(
                {"answer": "Moonshot unveiled Kimi K3.1 Pro [1]."}
            )
        }
        supported_review = Mock()
        supported_review.raise_for_status.return_value = None
        supported_review.json.return_value = {
            "response": '{"supported": true, "unsupported_claims": []}'
        }
        with patch(
            "backend.app.ui.components.requests.post",
            side_effect=[draft, supported_review],
        ) as post:
            answer = generate_grounded_answer(
                "Kimi K3.1 Pro model launch", [article]
            )
        self.assertEqual(answer, "Moonshot unveiled Kimi K3.1 Pro [1].")
        generation_payload = post.call_args_list[0].kwargs["json"]
        self.assertEqual(
            generation_payload["format"],
            {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        )
        self.assertEqual(generation_payload["options"]["temperature"], 0)
        self.assertIn("Answer the user's", generation_payload["prompt"])
        self.assertIn("sentence immediately at its end", generation_payload["prompt"])
        self.assertNotIn("example.com", generation_payload["prompt"])
        self.assertNotIn("Economic Times Technology", generation_payload["prompt"])
        self.assertNotIn("CATEGORY", generation_payload["prompt"])

        unsupported_review = Mock()
        unsupported_review.raise_for_status.return_value = None
        unsupported_review.json.return_value = {
            "response": '{"supported": false, "unsupported_claims": ["It is state of the art."]}'
        }
        unsupported_repair = Mock()
        unsupported_repair.raise_for_status.return_value = None
        unsupported_repair.json.return_value = {
            "response": json.dumps(
                {"answer": "It is state of the art [1]."}
            )
        }
        with patch(
            "backend.app.ui.components.requests.post",
            side_effect=[
                draft,
                unsupported_review,
                unsupported_repair,
                unsupported_review,
            ],
        ):
            self.assertIsNone(
                generate_grounded_answer(
                    "Kimi K3.1 Pro model launch", [article]
                )
            )

    def test_uncited_draft_gets_one_repair_then_grounding_validation(self):
        article = {
            "id": 4820,
            "title": "China's Moonshot AI launches Kimi K3.1 Pro",
            "summary": "Moonshot AI unveiled Kimi K3.1 Pro.",
        }
        draft = Mock()
        draft.raise_for_status.return_value = None
        draft.json.return_value = {
            "response": json.dumps(
                {"answer": "Moonshot AI unveiled Kimi K3.1 Pro."}
            )
        }
        repaired = Mock()
        repaired.raise_for_status.return_value = None
        repaired.json.return_value = {
            "response": json.dumps(
                {"answer": "Moonshot AI unveiled Kimi K3.1 Pro [1]."}
            )
        }
        supported_review = Mock()
        supported_review.raise_for_status.return_value = None
        supported_review.json.return_value = {
            "response": '{"supported": true, "unsupported_claims": []}'
        }

        with patch(
            "backend.app.ui.components.requests.post",
            side_effect=[draft, repaired, supported_review],
        ) as post:
            answer = generate_grounded_answer(
                "Kimi K3.1 Pro model launch", [article]
            )

        self.assertEqual(answer, "Moonshot AI unveiled Kimi K3.1 Pro [1].")
        self.assertEqual(post.call_count, 3)
        repair_prompt = post.call_args_list[1].kwargs["json"]["prompt"]
        self.assertIn("exact excerpt", repair_prompt)
        self.assertIn("DRAFT ANSWER JSON STRING", repair_prompt)
        self.assertIn('"citation_number": 1', repair_prompt)

    def test_exact_cited_source_text_corrects_a_grounding_false_negative(self):
        article = {
            "title": "Muse Glimmer 30B: An On-Device Agentic Model",
            "summary": (
                "Muse Glimmer is an open-source 30B parameter causal language "
                "model that integrates reliable tool use on consumer hardware."
            ),
        }
        rejected = Mock()
        rejected.raise_for_status.return_value = None
        rejected.json.return_value = {
            "response": (
                '{"supported": false, '
                '"unsupported_claims": ["30B parameter", "consumer hardware"]}'
            )
        }
        wrong_source = Mock()
        wrong_source.raise_for_status.return_value = None
        wrong_source.json.return_value = rejected.json.return_value
        with patch(
            "backend.app.ui.components.requests.post",
            side_effect=[
                rejected,
                Mock(
                    raise_for_status=Mock(return_value=None),
                    json=Mock(
                        return_value={
                            "response": (
                                '{"answers_query": true, '
                                '"unanswered_facets": []}'
                            )
                        }
                    ),
                ),
                wrong_source,
                rejected,
                rejected,
                rejected,
            ],
        ):
            self.assertTrue(
                validate_grounded_answer(
                    "Muse Glimmer is an open-source 30B parameter causal language "
                    "model that integrates reliable tool use on consumer hardware. [1]",
                    [article],
                    query="What is Muse Glimmer 30B and how does it use tools?",
                )
            )
            self.assertFalse(
                validate_grounded_answer(
                    "Muse Glimmer is an open-source 30B parameter causal language "
                    "model that integrates reliable tool use on consumer hardware. [2]",
                    [article, {"title": "Unrelated article", "summary": "Other facts."}],
                )
            )
            self.assertFalse(
                validate_grounded_answer(
                    "Muse Glimmer is an open-source 30B parameter causal language "
                    "model that integrates reliable tool use on consumer hardware. "
                    "[1][2]",
                    [article, {"title": "Unrelated article", "summary": "Other facts."}],
                        query="What is Muse Glimmer 30B and how does it use tools?",
                    )
            )
            self.assertFalse(
                    validate_grounded_answer(
                        "The model is safe [1].",
                        [{"title": "Safety", "summary": "The model is not safe."}],
                        query="Is the model safe?",
                    )
            )

    def test_semantically_rejected_draft_gets_one_extractive_repair(self):
        article = {
            "id": 5352,
            "title": "Muse Glimmer 30B: An On-Device Agentic Model",
            "summary": (
                "Muse Glimmer is an open-source 30B parameter causal language "
                "model that integrates reliable tool use on consumer hardware."
            ),
        }
        draft = Mock()
        draft.raise_for_status.return_value = None
        draft.json.return_value = {
            "response": json.dumps(
                {"answer": "Muse Glimmer runs on consumer hardware [1]."}
            )
        }
        rejected_review = Mock()
        rejected_review.raise_for_status.return_value = None
        rejected_review.json.return_value = {
            "response": (
                '{"supported": false, '
                '"unsupported_claims": ["runs on consumer hardware"]}'
            )
        }
        repaired = Mock()
        repaired.raise_for_status.return_value = None
        exact_review = Mock()
        exact_review.raise_for_status.return_value = None
        exact_review.json.return_value = rejected_review.json.return_value
        relevant_query = Mock()
        relevant_query.raise_for_status.return_value = None
        relevant_query.json.return_value = {
            "response": '{"answers_query": true, "unanswered_facets": []}'
        }
        repaired.json.return_value = {
            "response": json.dumps(
                {
                    "answer": (
                        "Muse Glimmer is an open-source 30B parameter causal "
                        "language model that integrates reliable tool use on "
                        "consumer hardware. [1]"
                    )
                }
            )
        }
        with patch(
            "backend.app.ui.components.requests.post",
            side_effect=[
                draft,
                rejected_review,
                repaired,
                exact_review,
                relevant_query,
            ],
        ) as post:
            answer = generate_grounded_answer(
                "Muse Glimmer 30B on-device agentic model", [article]
            )

        self.assertEqual(
            answer,
            "Muse Glimmer is an open-source 30B parameter causal language "
            "model that integrates reliable tool use on consumer hardware [1].",
        )
        self.assertEqual(post.call_count, 5)

    def test_incomplete_inferslice_repair_uses_complete_source_sentence(self):
        summary = (
            "InferSlice is a reference architecture for Service-Level Objective "
            "(SLO)-driven LLM inference over a telco infrastructure, integrating "
            "network slicing, GPU partitioning, elastic serving and observability."
        )
        article = {
            "id": 5213,
            "title": (
                "InferSlice: A Reference Architecture for SLO-driven LLM "
                "Inference over a Telco Wide-Area Network"
            ),
            "summary": summary,
        }
        draft = Mock()
        draft.raise_for_status.return_value = None
        draft.json.return_value = {
            "response": json.dumps(
                {"answer": "InferSlice is a reference architecture for LLM inference."}
            )
        }
        incomplete_repair = Mock()
        incomplete_repair.raise_for_status.return_value = None
        incomplete_repair.json.return_value = {
            "response": json.dumps(
                {
                    "answer": (
                        "InferSlice is a reference architecture for "
                        "Service-Level Objective (SLO)-driven LLM inference "
                        "over a telco infrastructure,"
                    )
                }
            )
        }
        rejected_review = Mock()
        rejected_review.raise_for_status.return_value = None
        rejected_review.json.return_value = {
            "response": (
                '{"supported": false, '
                '"unsupported_claims": ["LLM inference over a telco network"]}'
            )
        }
        relevant_query = Mock()
        relevant_query.raise_for_status.return_value = None
        relevant_query.json.return_value = {
            "response": '{"answers_query": true, "unanswered_facets": []}'
        }

        with patch(
            "backend.app.ui.components.requests.post",
            side_effect=[
                draft,
                incomplete_repair,
                rejected_review,
                relevant_query,
            ],
        ) as post:
            answer = generate_grounded_answer(
                "LLM inference over telco wide-area networks", [article]
            )

        self.assertEqual(answer, f"{summary[:-1]} [1].")
        self.assertEqual(post.call_count, 4)
        self.assertEqual(
            _complete_source_sentence(
                "LLM inference over telco wide-area networks", [article]
            ),
            summary,
        )
        self.assertTrue(has_valid_answer_citations(answer, 1))

    def test_kimi_release_timing_is_included_only_from_explicit_title(self):
        article = {
            "id": 4820,
            "title": (
                "China's Moonshot AI launches Kimi K3.1 Pro; plans full model "
                "weights release by November"
            ),
            "summary": (
                "Chinese AI company Moonshot AI has unveiled Kimi K3.1 Pro, a "
                "3-trillion-parameter open-weight AI model."
            ),
        }
        draft = Mock()
        draft.raise_for_status.return_value = None
        draft.json.return_value = {
            "response": json.dumps(
                {"answer": f"{article['summary']} [1]"}
            )
        }
        supported_summary = Mock()
        supported_summary.raise_for_status.return_value = None
        supported_summary.json.return_value = {
            "response": '{"supported": true, "unsupported_claims": []}'
        }
        reviewer_false_negative = Mock()
        reviewer_false_negative.raise_for_status.return_value = None
        reviewer_false_negative.json.return_value = {
            "response": (
                '{"supported": false, '
                '"unsupported_claims": ["by November"]}'
            )
        }
        relevant_query = Mock()
        relevant_query.raise_for_status.return_value = None
        relevant_query.json.return_value = {
            "response": '{"answers_query": true, "unanswered_facets": []}'
        }
        with patch(
            "backend.app.ui.components.requests.post",
            side_effect=[
                draft,
                supported_summary,
                reviewer_false_negative,
                relevant_query,
            ],
        ):
            answer = generate_grounded_answer(
                "Kimi K3.1 Pro model launch and release", [article]
            )

        self.assertIn("release by November", answer)
        self.assertIn('" [1].', answer)
        self.assertTrue(has_valid_answer_citations(answer, 1))

    def test_india_cost_query_does_not_invent_specific_cost(self):
        article = {
            "title": (
                "Building AI Products For Indian Users: The Challenge Of Scale, "
                "Access, And Cost"
            ),
            "summary": (
                "The challenge of taking AI products to millions of Indian users "
                "is making them work across budget smartphones, multiple "
                "languages, and other technical limitations."
            ),
        }
        answer = _append_explicit_release_timing(
            "Building AI products for Indian users scale access cost",
            article["summary"] + " [1]",
            [article],
        )
        self.assertEqual(answer, article["summary"] + " [1]")
        self.assertNotRegex(answer, r"\$\s*\d|\b\d+\s*(?:rupees|INR|dollars)\b")

    def test_grounded_answer_rejects_missing_or_unmapped_citations(self):
        article = {
            "title": "Kimi K3.1 Pro",
            "summary": "Moonshot unveiled Kimi K3.1 Pro.",
        }
        no_citation = Mock()
        no_citation.raise_for_status.return_value = None
        no_citation.json.return_value = {
            "response": json.dumps(
                {"answer": "Moonshot unveiled Kimi K3.1 Pro."}
            )
        }
        unsupported_repair = Mock()
        unsupported_repair.raise_for_status.return_value = None
        unsupported_repair.json.return_value = {
            "response": json.dumps({"answer": "Unrelated unsupported detail."})
        }
        with patch(
            "backend.app.ui.components.requests.post",
            side_effect=[no_citation, unsupported_repair],
        ) as post:
            self.assertIsNone(
                generate_grounded_answer("Kimi K3.1 Pro", [article])
            )
        self.assertEqual(post.call_count, 2)

        self.assertFalse(validate_grounded_answer("Claim [2].", [article]))
        self.assertFalse(validate_grounded_answer("Claim without citation.", [article]))

    def test_exact_source_fallback_requires_the_requested_query_facet(self):
        article = {
            "id": 3818,
            "title": "Introducing Claude Sonnet 5.5 on AWS",
            "summary": (
                "Claude Sonnet 5.5 is now available on Amazon Bedrock and "
                "Claude Platform on AWS. It's a smarter, more efficient Sonnet "
                "model for focused coding and knowledge work, with a lower cost "
                "per task at faster speed."
            ),
        }
        reviewer_rejects = Mock()
        reviewer_rejects.raise_for_status.return_value = None
        reviewer_rejects.json.return_value = {
            "response": (
                '{"supported": false, '
                '"unsupported_claims": ["not responsive to query"]}'
            )
        }
        query_rejects = Mock()
        query_rejects.raise_for_status.return_value = None
        query_rejects.json.return_value = {
            "response": (
                '{"answers_query": false, '
                '"unanswered_facets": ["availability"]}'
            )
        }
        unrelated_exact_answer = (
            "It's a smarter, more efficient Sonnet model for focused coding "
            "and knowledge work, with a lower cost per task at faster speed. [1]"
        )
        with patch(
            "backend.app.ui.components.requests.post",
            side_effect=[reviewer_rejects, query_rejects],
        ) as post:
            self.assertFalse(
                validate_grounded_answer(
                    unrelated_exact_answer,
                    [article],
                    query="Where is Claude Sonnet 5.5 available on AWS?",
                )
            )
        self.assertIn(
            "Where is Claude Sonnet 5.5 available on AWS?",
            post.call_args_list[1].kwargs["json"]["prompt"],
        )
        self.assertIn(
            "specific question",
            post.call_args_list[1].kwargs["json"]["prompt"],
        )

        query_accepts = Mock()
        query_accepts.raise_for_status.return_value = None
        query_accepts.json.return_value = {
            "response": '{"answers_query": true, "unanswered_facets": []}'
        }
        with patch(
            "backend.app.ui.components.requests.post",
            side_effect=[reviewer_rejects, query_accepts],
        ):
            self.assertTrue(
                validate_grounded_answer(
                    "Claude Sonnet 5.5 is now available on Amazon Bedrock and "
                    "Claude Platform on AWS. [1]",
                    [article],
                    query="Where is Claude Sonnet 5.5 available on AWS?",
                )
            )

    def test_exact_source_fallback_fails_closed_without_a_query(self):
        article = {
            "title": "Muse Glimmer 30B",
            "summary": "Muse Glimmer is an open-source 30B parameter model.",
        }
        rejected = Mock()
        rejected.raise_for_status.return_value = None
        rejected.json.return_value = {
            "response": (
                '{"supported": false, '
                '"unsupported_claims": ["not sufficiently grounded"]}'
            )
        }
        with patch(
            "backend.app.ui.components.requests.post",
            return_value=rejected,
        ) as post:
            self.assertFalse(
                validate_grounded_answer(
                    "Muse Glimmer is an open-source 30B parameter model. [1]",
                    [article],
                )
            )
        self.assertEqual(post.call_count, 1)

    def test_financial_title_summary_conflict_withholds_answer(self):
        article = {
            "id": 4910,
            "title": (
                "OpenAI working on new $30B raise at $1.25T valuation ahead of IPO"
            ),
            "summary": (
                "OpenAI is seeking a $30B valuation ahead of its planned IPO, "
                "potentially breaking records for the most highly valued private "
                "company."
            ),
        }
        with patch(
            "backend.app.ui.components.requests.post"
        ) as post:
            self.assertIsNone(
                generate_grounded_answer(
                    "What raise and valuation does the article report for OpenAI "
                    "ahead of an IPO?",
                    [article],
                )
            )
        post.assert_not_called()

    def test_grounded_answer_rejects_malformed_json_output(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"response": "not a JSON answer"}
        with patch(
            "backend.app.ui.components.requests.post", return_value=response
        ) as post:
            self.assertIsNone(
                generate_grounded_answer("Kimi K3.1 Pro", [{"title": "Kimi"}])
            )
        self.assertEqual(post.call_count, 1)

    def test_specific_abstention_is_normalized_to_standard_refusal(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "response": json.dumps(
                {
                    "answer": (
                        "Specific details regarding the Kimi K3.1 Pro launch "
                        "are not documented."
                    )
                }
            )
        }
        with patch("backend.app.ui.components.requests.post", return_value=response):
            self.assertEqual(
                generate_grounded_answer(
                    "Kimi K3.1 Pro model launch",
                    [{"title": "Kimi-maker Moonshot revenue", "summary": "K3 usage"}],
                ),
                NO_GROUNDED_FACTS_MESSAGE,
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
