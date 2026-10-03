# AI Radar â€” Local AI Intelligence Platform

Tracks verified global AI developments â€” frontier models, agentic systems,
developer tooling, research and policy â€” and answers questions about them.
Everything runs on your machine: **SQLite + FAISS-CPU + local Ollama**. No cloud
services, no API keys, no ChromaDB, no per-request cost.

---

## What it does

| Capability | How |
|---|---|
| Live intelligence feed | 14 tiered RSS sources, deduplicated and classified by a local LLM |
| Semantic search with RAG | `nomic-embed-text` embeddings in a persistent FAISS `IndexFlatIP` |
| Hallucination guardrails | Domain gate, cosine-similarity threshold, and a strictly grounded answer prompt |
| Morning briefing | 9:00 AM IST baseline snapshot of the top-ranked stories |
| Evening update | 7:00 PM IST differential against the saved baseline, explicit when nothing new happened |
| Multi-factor ranking | Importance, recency, career relevance, AI depth and source authority, with diversity caps |
| Story clustering | Near-duplicate event coverage is collapsed into one story while preserving distinct source links |

---

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com/) running locally

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
ollama list          # confirm both are present
```

## Install

```bash
cd AI-Radar
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

Optional `.env` overrides (defaults shown):

```
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
EMBEDDING_MODEL=nomic-embed-text
```

---

## Run the dashboard

```bash
python -m streamlit run backend/app/ui/dashboard.py
```

Then open <http://localhost:8501>.

The dashboard creates and migrates the SQLite schema before its first query, so
it opens on a clean checkout as well as on an existing database.

## Run the pipeline

```bash
python -m backend.app.main --status                     # telemetry: database, FAISS, Ollama
python -m backend.app.main --collect                    # fetch RSS feeds
python -m backend.app.main --process --limit 10         # classify with Ollama
python -m backend.app.main --index                      # embed new stories into FAISS
python -m backend.app.main --morning                    # 9 AM IST baseline briefing
python -m backend.app.main --evening                    # 7 PM IST differential briefing
python -m backend.app.main --full-pipeline --limit 10   # every stage in order
```

Evening generation compares publication times only against a morning baseline
from the same IST calendar day. If there is no valid same-day baseline, it
records the unavailable attempt without replacing an existing evening snapshot.
When a valid evening replaces the latest snapshot, the previous full snapshot
is preserved under `data/evening_briefings/`; `data/briefing_history.json`
continues to retain the generation audit history.

Useful extras:

```bash
python -m backend.app.main --collect --process --index            # collect, one analysis batch, then reconcile
python -m backend.app.main --process --drain                     # finish analysing the window
python -m backend.app.main --process --backlog --limit 50        # work through pre-window archive entries
python -m backend.app.main --rebuild-index                       # re-embed everything from scratch
```

Each stage is safe to re-run. `--process` and `--index` are resumable: they pick
up where the last run stopped, so a long backlog can be drained across several
invocations.

The dashboard’s **Refresh radar feed** action collects the recent window,
analyzes one bounded batch of the newest pending articles, then reconciles FAISS
with every currently eligible article. Remaining unanalyzed articles are
reported as backlog, not drained by the dashboard action. Feed-level failures
are reported as partial collection, while the last successful refresh time is
advanced only after collection, analysis, and index reconciliation complete.

### Why `--drain` matters

An article that has not been classified is invisible everywhere: it is excluded
from the live feed, from the briefings, and from the FAISS index, so semantic
search cannot retrieve it. A single `--process --limit 10` run analyses ten
articles, while one collection run typically adds far more, so the pending set
grows and the newest arrivals are all that ever get looked at.

Two things prevent an announcement from being lost in that gap:

1. **The pending set is prioritised, not truncated.** Every unanalysed article in
   the window is ranked before the batch is taken off the top â€” by source tier
   (primary labs first) and by how strongly the article's own title and summary
   read as AI coverage. Ties break oldest-first, which is what guarantees the
   queue drains instead of starving.
2. **`--drain` repeats batches until the window is empty**, so the searchable
   corpus matches what was actually collected. `--full-pipeline` drains by
   default for the same reason. These explicit CLI drain operations may process
   a large backlog; the dashboard refresh deliberately uses one bounded batch.

`python -m backend.app.main --status` prints the next few articles the analyser will pick,
so a growing backlog is visible rather than silent.

## Run the tests

```bash
python -m unittest discover -s tests -t .
```

Cases that need Ollama or a built FAISS index skip themselves when either is
unavailable.

---

## Architecture

```
14 tiered RSS feeds
        |
        v
news_collector.py        URL normalisation, tracking-parameter stripping,
                         title/URL dedup, 7-day freshness window, per-feed timeout
        |
        v
data/ai_radar.db     SQLite, INSERT OR IGNORE on a UNIQUE url
        |
        v
ai_analyzer.py           llama3.2:3b via Ollama: is_ai_news, category,
process_articles.py      importance, career relevance, summary, why-it-matters
                         queue ordered by source tier + AI signal, oldest-first
                         within a band so nothing starves; --drain empties it
        |
        v
ranking.py               importance 30% + recency 30% + career 15%
                         + AI depth 15% + source authority 10%
        v
vector_store.py          nomic-embed-text -> FAISS IndexFlatIP (cosine),
                         incremental append, metadata.json aligned row-for-row
        |
        v
dashboard.py             Live feed, semantic search + RAG, briefings,
                         category sub-feeds
```

---

## How search avoids making things up

```
user query
    |
    v
domain gate            AI vocabulary accepted, off-topic vocabulary rejected,
(components.py)        ambiguous wording sent to llama3.2:3b
    |
    |-- not an AI question --> "limited to AI and machine learning" card, no answer
    v
query embedding        nomic-embed-text via Ollama
    |
    v
FAISS cosine search    IndexFlatIP over L2-normalised vectors
    |
    v
similarity threshold   keep only hits >= MIN_SEMANTIC_SIMILARITY (0.55)
    |
    |-- nothing left ------> "no sufficiently relevant information" card, no answer
    v
                       numbered [1] [2] citations, told to refuse when the
                       articles do not answer the question
    |
    v
answer + verified source cards
```

Measured on this corpus, on-topic queries with real coverage score 0.60â€“0.75
cosine similarity, while off-topic or uncovered queries top out around 0.50 â€” so
the 0.55 threshold in `app/config.py` separates "answerable" from "say nothing".

---

## Categories

`ALLOWED_CATEGORIES` in `app/config.py` is the single vocabulary, and
`normalize_category()` resolves every other spelling onto it:

```
AI Models Â· AI Agents Â· AI Tools Â· AI Research Â· AI Companies
AI Jobs & Careers Â· AI Policy Â· India AI Â· Robotics Â· Open Source AI Â· Other
```

Older rows keep whatever spelling they were stored with â€” the database is never
rewritten â€” so `"Developers & Tools"` maps to `AI Tools`, and the feed query
matches both. `"AI Hardware"` and `"AI Industry"` fold into `AI Companies`.

---

## Briefings

**Morning (9:00 AM IST)** â€” ranks the recent verified stories, applies the
diversity caps, writes `data/morning_briefing.json` (the baseline), and appends
to `data/briefing_history.json`. It leaves the most recent evening snapshot in
`data/latest_briefing.json` available until the next evening update replaces it.
**Evening (7:00 PM IST)** â€” loads the saved baseline and considers only articles
published after it that the baseline did not already carry. If any clear the
significance thresholds (importance â‰¥ 7, career â‰¥ 7, or score â‰¥ 7.0) they become
the update; otherwise the briefing states plainly that there are no major new AI
developments since the morning and repeats the baseline unchanged. Nothing is
invented to fill the slot.

## Story clustering

Multiple outlets may report the same AI event. AI Radar uses the existing
normalized FAISS article embeddings plus conservative shared-title-term checks
to collapse near-duplicate event coverage into one representative story.
Clusters retain each distinct source and original URL, while unrelated stories
in the same category remain separate. This is stricter than URL/title
deduplication because it can merge different reports of one event without

Schedule them with cron or Task Scheduler (times in UTC):

```
0 */2 * * *  cd /path/to/AI-Radar && python -m backend.app.main --collect --process --index
30 3   * * *  cd /path/to/AI-Radar && python -m backend.app.main --morning    # 9:00 AM IST
30 13  * * *  cd /path/to/AI-Radar && python -m backend.app.main --evening    # 7:00 PM IST
```

---

## Layout

```
app/
  config.py                 paths, Ollama endpoints, thresholds, category vocabulary
  main.py                   pipeline CLI
  services/
    news_collector.py       RSS ingestion
    ai_analyzer.py          Ollama classification
    process_articles.py     batch analysis runner
    ranking.py              multi-factor ranking
    vector_store.py         FAISS index + semantic search
    briefing_generator.py   morning baseline / evening differential
  ui/
    dashboard.py            Streamlit entry point
    components.py           cards, query gate, grounded answers
    styles.py               CSS (system fonts only, no CDN)
  utils/
    database.py             SQLite schema and queries
data/ai_radar.db             collected articles
data/vector_store/           ai_radar.index + metadata.json
data/evening_briefings/      archived evening snapshots
tests/                       unit and integration tests
```

## Data safety

No command in this project drops a table, deletes an article, or removes a
briefing snapshot. `initialize_database()` only adds missing schema objects and
fills NULL `published_timestamp` values, and `--index` appends to the existing
FAISS index rather than replacing it. The one destructive-by-design command is
`--rebuild-index`, which re-embeds every verified article and overwrites the
index file; the SQLite database is still left untouched.
