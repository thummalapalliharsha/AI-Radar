import json
import sys
from typing import Dict, Any, Optional
import requests

from backend.app.config import (
    ALLOWED_CATEGORIES,
    OLLAMA_ANALYZE_TIMEOUT,
    OLLAMA_GENERATE_URL,
    OLLAMA_MODEL,
    normalize_category,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


AI_RELEVANCE_RULES = """
An article is genuinely AI-related ONLY when artificial intelligence or a machine learning technology is the primary subject of the actual story.

GENUINE AI TOPICS:
- New AI models, model architectures, LLMs, vision/multimodal models, reasoning models
- AI agents, multi-agent frameworks, autonomous agent workflows
- AI developer tools, SDKs, APIs, code-generation systems, developer platforms
- Machine learning research papers, training benchmarks, inference optimizations
- AI/ML datasets, evaluation methods, reproducible benchmarks, technical reports,
    and substantive research-lab announcements
- AI chips, GPUs, NPUs, datacenters, inference hardware architectures
- AI safety, guardrails, alignment, red-teaming, AI security vulnerabilities
- AI policy, legislation, compliance frameworks, government governance
- Major AI business milestones: AI startup funding, acquisitions, frontier deployments
- Meaningful AI workforce developments: major hiring or layoffs, research fellowships,
    workforce programs, or broad AI labor-market trends; not individual job postings
- AI-powered robotics and embodied AI: robots, humanoids, robotaxis, drones, or
    autonomous machines where learning, perception, planning, or an AI model is central
- Open-source AI weights, datasets, checkpoints, fine-tunes
- Major India-specific AI initiatives and deployments: IndiaAI/MeitY policy or
    infrastructure, Indian AI companies/startups, IITs or Indian research institutions,
    and deployments materially located in India

CATEGORY DECISION RULES:
- AI Research: use for a paper, technical report, benchmark, dataset, evaluation,
    research breakthrough, or research-lab announcement whose primary subject is AI/ML.
    Do not use it for ordinary science, generic technology research, or a company story
    that only mentions research incidentally.
- AI Jobs & Careers: use only for a substantial AI labor-market or workforce story,
    major hiring/layoff program, fellowship, or broad career trend. Reject individual
    vacancies, recruiting ads, job boards, and generic company hiring news.
- India AI: use only when India is central to the AI development, organization,
    institution, policy, infrastructure, or deployment. An Indian person, office,
    customer, or incidental mention is not enough.
- Robotics: use when a significant robotics or embodied-AI development is central and
    the system uses AI/ML for perception, control, planning, navigation, or autonomy.
    Generic factory automation, drones without an AI component, and ordinary hardware
    announcements are not enough.

NON-AI TOPICS (MUST BE REJECTED with is_ai_news = false):
- Generic lifestyle, philosophy, self-help, or resilience quotes (e.g., "7 Quotes from Meditations by Marcus Aurelius")
- General business investments or commercial real estate (e.g., "Smartworks to invest Rs 600 Cr in managed offices")
- Traditional IT, software, databases, or cloud migration without core AI/ML innovation
- Crypto, blockchain, Web3, generic fintech, or daily stock price movements
- General consumer electronics, gaming hardware without AI computing significance
- Generic corporate leadership changes or unrelated marketing announcements
"""


def build_analysis_prompt(title: str, source: str, summary: str) -> str:
    """Construct structured editorial prompt for Ollama classification."""
    categories_str = ", ".join([f'"{c}"' for c in ALLOWED_CATEGORIES if c != "Other"])

    return f"""You are the Chief AI News Editor for AI Radar, an enterprise intelligence platform.
Classify and analyze the following news story with strict accuracy.

{AI_RELEVANCE_RULES}

ARTICLE DETAILS:
- Title: {title}
- Source: {source}
- Description: {summary}

DECISION PROCESS:
1. Is artificial intelligence or machine learning the primary subject?
   - If NO: set is_ai_news=false, category="Other", importance=1, career_relevance=1, summary="", why_it_matters="", should_show=false.
   - If YES: set is_ai_news=true and assign appropriate fields.

2. Categories (choose exactly one):
   {categories_str}, "Other"

    Prefer the most specific category using these tie-breakers:
    Robotics for central AI/embodied robotics; India AI for India-centered AI stories;
    AI Jobs & Careers for substantial AI workforce stories; AI Research for primary
    AI/ML research, papers, datasets, benchmarks, or lab announcements. Otherwise use
    the product, company, model, agent, tools, policy, or open-source category that
    best describes the primary subject.

3. Importance (1 to 10 scale):
   - 9-10: Paradigm-shifting breakthrough, major foundation model, critical frontier security/policy
   - 7-8: Significant model update, major practical tool, high-impact research or industry move
   - 5-6: Incremental tool update, niche research benchmark, noteworthy minor announcement
   - 1-4: Minor release, repetitive marketing, or low-relevance content

4. Career Relevance (1 to 10 scale):
   - 8-10: Critical skills, new developer APIs/models, major technical workflows for AI engineers
   - 5-7: Useful industry knowledge or tooling for software developers
   - 1-4: Little direct practical utility for technical builders

5. Summary: 2 to 4 concise, factual sentences based STRICTLY on the provided text. Never fabricate unmentioned facts.
6. Why It Matters: 1 to 2 clear sentences explaining practical implications for developers, researchers, or industry.
7. Should Show: true ONLY if is_ai_news is true and importance >= 5.

OUTPUT FORMAT:
Return ONLY a valid JSON object with exact keys:
{{
    "is_ai_news": true,
    "category": "AI Models",
    "importance": 8,
    "career_relevance": 8,
    "summary": "Factual 2-4 sentence summary.",
    "why_it_matters": "Clear 1-2 sentence impact explanation.",
    "should_show": true
}}
"""


def validate_analysis_result(result: Any) -> bool:
    """Validate schema and types of Ollama output."""
    if not isinstance(result, dict):
        return False

    required_keys = {
        "is_ai_news",
        "category",
        "importance",
        "career_relevance",
        "summary",
        "why_it_matters",
        "should_show",
    }
    if not required_keys.issubset(result.keys()):
        return False

    if not isinstance(result["is_ai_news"], bool):
        return False
    if not isinstance(result["should_show"], bool):
        return False

    if result.get("category") not in ALLOWED_CATEGORIES:
        return False

    return True


def analyze_article(
    article: Dict[str, Any], timeout: int = OLLAMA_ANALYZE_TIMEOUT
) -> Optional[Dict[str, Any]]:
    """
    Analyze an article using local Ollama (llama3.2:3b).
    Enforces strict safety fallbacks if non-AI or invalid.
    """
    title = str(article.get("title", "")).strip()
    source = str(article.get("source", "")).strip()
    summary = str(article.get("summary", "")).strip()

    prompt = build_analysis_prompt(title, source, summary)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }

    try:
        response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        result_text = data.get("response", "").strip()

        if not result_text:
            return None

        result = json.loads(result_text)

        # Defensive normalization of types if model returned string booleans/ints
        if isinstance(result.get("is_ai_news"), str):
            result["is_ai_news"] = result["is_ai_news"].lower() in {"true", "1", "yes"}
        if isinstance(result.get("should_show"), str):
            result["should_show"] = result["should_show"].lower() in {"true", "1", "yes"}

        # A near-miss spelling ("Developer Tools") is corrected rather than
        # thrown away, so one loose word does not cost a whole analysis.
        if "category" in result:
            result["category"] = normalize_category(result["category"])

        try:
            result["importance"] = max(1, min(10, int(result.get("importance", 1))))
            result["career_relevance"] = max(1, min(10, int(result.get("career_relevance", 1))))
        except (ValueError, TypeError):
            result["importance"] = 1
            result["career_relevance"] = 1

        if not validate_analysis_result(result):
            return None

        # Safety enforcement for non-AI content
        if not result["is_ai_news"]:
            result["category"] = "Other"
            result["importance"] = 1
            result["career_relevance"] = 1
            result["summary"] = ""
            result["why_it_matters"] = ""
            result["should_show"] = False

        return result

    except requests.exceptions.RequestException as e:
        print(f"âš ï¸ Ollama request error: {e}")
        return None
    except json.JSONDecodeError:
        print("âš ï¸ Ollama returned non-JSON response.")
        return None
    except Exception as e:
        print(f"âš ï¸ Unexpected analyzer error: {e}")
        return None


if __name__ == "__main__":
    test_article = {
        "title": "OpenAI releases new reasoning model with improved math capabilities",
        "source": "OpenAI",
        "summary": "OpenAI introduced a new model family designed to reason through complex programming and mathematics benchmarks.",
    }
    print("Testing AI Analyzer with local Ollama...")
    res = analyze_article(test_article)
    print("Result:", json.dumps(res, indent=2))


