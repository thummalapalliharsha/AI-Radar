# AI Radar — Local AI Intelligence Platform

AI Radar is a local-first AI intelligence platform that collects, classifies,
ranks, clusters, and semantically indexes AI news from multiple RSS sources.

It focuses on major developments across AI/ML, frontier models, AI agents,
developer tools, research, AI companies, jobs and careers, AI policy,
robotics, open-source AI, and India AI.

The platform combines RSS ingestion, local LLM analysis, FAISS semantic
search, grounded RAG answers, story clustering, and morning/evening
intelligence briefings into one pipeline.

> **Local-first:** AI classification and embedding run through Ollama on your
> machine. SQLite stores the news corpus and FAISS provides local semantic
> retrieval.

---

## ✨ Key Features

| Capability | Description |
|---|---|
| 🌐 AI News Collection | Collects AI-related articles from tiered RSS sources |
| 🧠 Local AI Classification | Uses Ollama to classify and analyze articles |
| 🔎 Semantic AI Search | Searches the local AI news corpus using embeddings |
| 📚 Grounded RAG | Generates answers only from retrieved AI Radar articles |
| 🛡️ Search Guardrails | Domain filtering and similarity thresholds reduce irrelevant answers |
| 📰 Latest AI Feed | Displays the newest verified AI stories |
| 🌅 Morning Briefing | Creates a 9:00 AM IST baseline briefing |
| 🌙 Evening Briefing | Detects significant developments since the morning baseline |
| 📊 Multi-factor Ranking | Combines importance, recency, career relevance, AI depth and source authority |
| 🧩 Story Clustering | Groups near-duplicate coverage of the same event |
| 💼 AI Jobs & Careers | Tracks relevant AI career developments |
| 🇮🇳 India AI | Provides a dedicated India-focused AI category |
| 💻 Developer Intelligence | Tracks developer tools, APIs, frameworks and infrastructure |
| 🤖 Models & Agents | Tracks model launches and agentic AI developments |
| 🔬 AI Research | Tracks important research and technical breakthroughs |
| 🔒 Local Storage | Uses SQLite and FAISS locally |
| 💰 No Paid AI API Required | Local Ollama models handle classification and embeddings |

---

# 🏗️ Architecture

```text
                    AI RADAR
                       │
                       ▼
              ┌─────────────────┐
              │   RSS Sources   │
              │ Tiered Sources  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ News Collector  │
              │ URL / Title     │
              │ Deduplication   │
              │ Freshness       │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ SQLite Database │
              │ Article Corpus  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  AI Analyzer    │
              │ Local Ollama    │
              │ Classification  │
              │ Summarization   │
              └────────┬────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
       ┌──────────────┐  ┌───────────────┐
       │   Ranking    │  │   Clustering  │
       │ Importance   │  │ Near-Duplicate│
       │ Recency      │  │ Events        │
       │ AI Depth     │  └───────────────┘
       │ Career       │
       │ Authority    │
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │   Ollama     │
       │ Embeddings   │
       │ nomic-embed  │
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │ FAISS-CPU    │
       │ Vector Store │
       └──────┬───────┘
              │
       ┌──────┴─────────────┐
       ▼                    ▼
┌───────────────┐   ┌──────────────────┐
│ AI Semantic   │   │ Morning / Evening│
│ Search + RAG  │   │    Briefings     │
└───────┬───────┘   └──────────────────┘
        │
        ▼
┌─────────────────────────────┐
│        Streamlit UI         │
│ Latest • Search • Briefings │
│ Research • Tools • Models   │
│ Jobs • India AI             │
└─────────────────────────────┘
🧰 Technology Stack
Backend
Python 3.12+
SQLite
Ollama
llama3.2:3b
nomic-embed-text
FAISS-CPU
NumPy
Feedparser
Requests
BeautifulSoup
Python-dotenv
Frontend
Next.js
React
TypeScript
Tailwind/CSS-based UI
AI / Retrieval
Local LLM classification
Local embeddings
FAISS IndexFlatIP
Cosine-similarity retrieval
Grounded RAG
Retrieval thresholding
Query-domain classification
🖥️ Requirements
Python 3.12+
Node.js / npm for the frontend
Ollama
Windows, Linux, or macOS

Ollama must be running locally.

Install the required models:

ollama pull llama3.2:3b
ollama pull nomic-embed-text

Verify:

ollama list

You should see both models available.

🚀 Installation
1. Clone the repository
git clone https://github.com/thummalapalliharsha/AI-Radar.git
cd AI-Radar
2. Create the Python environment
Windows
python -m venv .venv
.venv\Scripts\activate
Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
3. Install Python dependencies
pip install -r requirements.txt
⚙️ Configuration

AI Radar is designed to work with local Ollama defaults.

Optional .env configuration:

OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
EMBEDDING_MODEL=nomic-embed-text

No paid AI API key is required for the core local pipeline.

🗄️ Local Data

Runtime data is intentionally kept out of GitHub.

The application uses:

data/
├── ai_radar.db
├── faiss_index.bin
├── faiss_meta.json
├── vector_store/
└── briefing files / snapshots

The database and generated vector data are local runtime artifacts and are
excluded through .gitignore.

A fresh clone can rebuild its local database and FAISS index by running the
pipeline.

▶️ Run the Dashboard

From the project root:

python -m streamlit run backend/app/ui/dashboard.py

Then open:

http://localhost:8501

The dashboard initializes the local database before querying it.

🔄 Run the AI Radar Pipeline

The pipeline is divided into independent stages.

Check system status
python -m backend.app.main --status

This reports:

Database status
Number of collected articles
Verified AI stories
Selected feed stories
Active sources
Pending analysis
FAISS health
Vector count
Embedding model
Ollama status
Collect RSS articles
python -m backend.app.main --collect

This fetches articles from the configured RSS sources.

The collector performs:

URL normalization
Tracking-parameter removal
Title/URL deduplication
Freshness filtering
Feed-level timeouts
Source prioritization
Process articles with Ollama

Process a bounded batch:

python -m backend.app.main --process --limit 10

Process the pending analysis queue:

python -m backend.app.main --process --drain

Process older backlog entries:

python -m backend.app.main --process --backlog --limit 50

The local LLM determines information such as:

Whether the article is actually AI-related
AI category
Importance
Career relevance
AI depth
Concise summary
Why the development matters
Build / reconcile the FAISS index
python -m backend.app.main --index

The vector store uses:

nomic-embed-text
       ↓
L2-normalized embeddings
       ↓
FAISS IndexFlatIP

The current retrieval system uses cosine similarity through normalized
inner-product search.

🧠 Semantic Search + RAG

AI Radar does not simply send every user question to an LLM.

The search pipeline is:

User Question
      │
      ▼
AI Domain Gate
      │
      ├── Not AI-related
      │       ↓
      │   Reject / guidance
      │
      ▼
Query Embedding
      │
      ▼
FAISS Semantic Search
      │
      ▼
Similarity Threshold
      │
      ├── No sufficiently relevant articles
      │       ↓
      │   No answer generated
      │
      ▼
Retrieved AI Articles
      │
      ▼
Grounded Local LLM
      │
      ▼
Answer + Supporting Sources

The answer-generation prompt is designed to use only the retrieved AI Radar
articles.

This reduces the chance of producing unsupported answers when the local
corpus does not contain sufficient information.

🛡️ Search Guardrails

AI Radar applies multiple checks before generating a response.

1. Domain Gate

Questions unrelated to AI/ML are rejected instead of being answered as
general-purpose questions.

2. Semantic Similarity Threshold

Only sufficiently relevant FAISS results are passed to the answer-generation
stage.

3. Grounded Answer Prompt

The local LLM is instructed to answer from the retrieved article context.

4. No Relevant Evidence

If the local corpus does not contain sufficiently relevant information, the
system returns a "no sufficiently relevant information" response instead of
inventing an answer.

📰 News Categories

AI Radar uses a controlled category vocabulary:

AI Models
AI Agents
AI Tools
AI Research
AI Companies
AI Jobs & Careers
AI Policy
India AI
Robotics
Open Source AI
Other

The classifier uses article content rather than relying only on the RSS
category.

This is important because broad technology feeds can contain articles that
are not actually about AI.

📊 Story Ranking

Stories are ranked using multiple signals:

Importance       30%
Recency          30%
Career relevance 15%
AI depth         15%
Source authority 10%

The ranking system also applies diversity controls so that the feed does not
become dominated by one source or one category.

🧩 Story Clustering

Different publications may report the same underlying AI event.

AI Radar performs an additional clustering stage after normal URL/title
deduplication.

Article A ──┐
Article B ──┼── Same underlying event
Article C ──┘
                 ↓
          Representative Story
                 +
          Distinct Source Links

The objective is to reduce repetitive coverage while preserving access to
different source reports.

🌅 Morning Briefing

The morning briefing establishes the day's baseline.

Target time:

9:00 AM IST

The process:

Recent verified AI stories
        ↓
Ranking
        ↓
Diversity filtering
        ↓
Top stories
        ↓
Morning baseline

Each briefing item can contain:

Headline
Concise summary
Why it matters
Category
Source
Source URL

The morning snapshot becomes the reference point for the evening update.

🌙 Evening Briefing

The evening briefing is designed as a differential update, rather than
another copy of the morning briefing.

Target time:

7:00 PM IST

The system:

Morning baseline
       +
Newly published AI stories
       ↓
Compare
       ↓
Significance filtering
       ↓
Major new developments
       ↓
Evening update

The evening briefing considers developments published after the morning
baseline and avoids repeating stories that were already included.

If no sufficiently significant new developments are found, the system can
explicitly report:

No major new AI developments since the morning briefing.

This prevents the system from inventing stories merely to fill an evening
briefing slot.

🔁 Full Pipeline

Run all major stages together:

python -m backend.app.main --full-pipeline --limit 10

For a larger pending queue, the pipeline can process the analysis backlog
using the available drain functionality.

Individual stages can also be run independently:

python -m backend.app.main --collect
python -m backend.app.main --process --limit 10
python -m backend.app.main --index
python -m backend.app.main --morning
python -m backend.app.main --evening
⏱️ Refresh Workflow

The dashboard's refresh operation is intentionally bounded.

It can:

Collect recent articles
Analyze a bounded number of pending articles
Reconcile the FAISS index
Refresh the feed

It does not automatically drain an unlimited backlog during a UI refresh.

This keeps the dashboard responsive while allowing the CLI to process larger
backlogs separately.

🧪 Testing

Run the test suite with:

python -m unittest discover -s backend/tests -t .

Tests cover areas including:

Database behavior
Analysis queue
Search
FAISS retrieval
Dashboard behavior
Briefings
Pipeline execution
Story clustering
Refresh behavior
API behavior
Status reporting

Tests requiring Ollama or an existing FAISS index can skip themselves when
those dependencies are unavailable.

📁 Project Structure
AI-Radar/
│
├── backend/
│   ├── __init__.py
│   │
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── main.py
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── server.py
│   │   │
│   │   ├── services/
│   │   │   ├── ai_analyzer.py
│   │   │   ├── briefing_generator.py
│   │   │   ├── news_collector.py
│   │   │   ├── process_articles.py
│   │   │   ├── ranking.py
│   │   │   ├── refresh_radar.py
│   │   │   ├── story_clustering.py
│   │   │   ├── top_stories.py
│   │   │   └── vector_store.py
│   │   │
│   │   ├── ui/
│   │   │   ├── components.py
│   │   │   ├── dashboard.py
│   │   │   ├── styles.py
│   │   │   └── top_story_components.py
│   │   │
│   │   └── utils/
│   │       ├── database.py
│   │       └── url_normalization.py
│   │
│   ├── tests/
│   │   ├── test_analysis_queue.py
│   │   ├── test_api.py
│   │   ├── test_briefing.py
│   │   ├── test_dashboard.py
│   │   ├── test_database.py
│   │   ├── test_faiss_search.py
│   │   ├── test_pipeline.py
│   │   ├── test_refresh.py
│   │   ├── test_search.py
│   │   ├── test_status.py
│   │   ├── test_story_clustering.py
│   │   └── test_top_stories.py
│   │
│   └── test_clusters.py
│
├── frontend/
│   ├── app/
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   └── lab/
│   │       └── hero-object/
│   │
│   ├── package.json
│   ├── package-lock.json
│   ├── next.config.ts
│   └── tsconfig.json
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   └── local runtime data
│
├── .gitignore
├── README.md
└── requirements.txt
🔐 Data & Security

Local runtime data is excluded from version control.

The repository does not commit:

.env
.env.local
*.db
FAISS indexes
FAISS metadata
node_modules/
.next/
frontend_old/
generated briefing files
logs

The local database and vector store remain on the developer's machine.

Do not commit API keys, passwords, tokens, private credentials, or other
secrets to the repository.

🛠️ Useful Commands
System status
python -m backend.app.main --status
Collect news
python -m backend.app.main --collect
Analyze a batch
python -m backend.app.main --process --limit 10
Drain pending analysis
python -m backend.app.main --process --drain
Process archive backlog
python -m backend.app.main --process --backlog --limit 50
Update FAISS
python -m backend.app.main --index
Rebuild FAISS
python -m backend.app.main --rebuild-index

--rebuild-index re-embeds the verified article corpus and replaces the
existing FAISS index. The SQLite database is not deleted.

Morning briefing
python -m backend.app.main --morning
Evening briefing
python -m backend.app.main --evening
Full pipeline
python -m backend.app.main --full-pipeline --limit 10
Start Streamlit
python -m streamlit run backend/app/ui/dashboard.py
🔄 Recommended Daily Workflow
                  ┌────────────────────┐
                  │     RSS Sources    │
                  └─────────┬──────────┘
                            │
                            ▼
                     Collect Articles
                            │
                            ▼
                    Local AI Analysis
                            │
                            ▼
                     Rank + Cluster
                            │
                            ▼
                      FAISS Index
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
       9:00 AM IST                  7:00 PM IST
       Morning Baseline             Evening Delta
              │                           │
              └─────────────┬─────────────┘
                            ▼
                     AI Radar Dashboard
🎯 Project Goals

AI Radar is designed around five principles:

1. Verified over viral

Prioritize reliable and meaningful AI developments rather than rumors or
unverified claims.

2. Local over dependent

Use local models and local storage wherever practical.

3. Grounded over generative

Answers should be based on retrieved AI Radar evidence rather than unsupported
model knowledge.

4. Differential over repetitive

The evening briefing should identify what changed since the morning instead
of simply repeating the day's news.

5. Useful over noisy

Ranking, classification, deduplication, and clustering are used to reduce
the amount of repetitive information presented to the user.

📌 Current Status

AI Radar currently includes:

RSS-based AI news collection
Local Ollama classification
SQLite article storage
FAISS semantic search
Local embedding generation
Grounded RAG answers
Similarity-based retrieval filtering
Multi-factor story ranking
Story clustering
Latest AI feed
Morning briefing generation
Evening differential briefing
Research/category feeds
Developer and tooling coverage
Models and agents coverage
AI jobs and careers coverage
India AI coverage
Streamlit dashboard
Automated pipeline CLI
Unit/integration tests
🚧 Future Improvements

Potential future extensions include:

Automated Windows Task Scheduler integration
Daily email delivery
AI briefing podcast generation
More multilingual Indian AI coverage
Improved event-level clustering
Additional source verification
More detailed source-quality scoring
Production deployment
Expanded analytics and historical trend views
📜 License

Add the project's chosen open-source license here before publishing a formal
release.

Author

Harsha Thummalapalli

GitHub:

https://github.com/thummalapalliharsha
