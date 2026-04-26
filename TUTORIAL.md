# NewsWeave - Build a Personal AI News Intelligence System
### End-to-End Tutorial: Agentic Systems with Ollama + Knowledge Graphs

---

## What You're Building

**NewsWeave** is a personal intelligence product that:
- Reads 500+ articles/day from 16 curated RSS feeds across finance, tech, business, politics, geopolitics, and science
- Uses a **local Ollama LLM** (gemma3:1b - fastest available) to extract entities, themes, sentiment, and causal links
- Builds a **knowledge graph** linking articles, people, organisations, places, and themes
- Exposes a **real-time dashboard** with D3 force-directed graph, article feed, and AI Q&A
- Runs entirely on your laptop - no cloud, no data leaks

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR                             │
│                    (asyncio event loop)                          │
└───┬──────────────┬────────────────┬──────────────┬──────────────┘
    │              │                │              │
    ▼              ▼                ▼              ▼
NewsAgent    AnalystAgent      GraphAgent    StorageAgent
(RSS fetch)  (Ollama LLM)     (NetworkX)    (SQLite)
    │              │                │              │
    └──────────────┴────────────────┴──────────────┘
                           │
                     FastAPI Server
                           │
                    Dashboard (HTML/D3)
```

### Agent Responsibilities

| Agent | Input | Output | Key Lib |
|-------|-------|--------|---------|
| **NewsAgent** | RSS feed URLs | `list[Article]` | feedparser, httpx |
| **AnalystAgent** | Articles | Enriched articles (entities, themes, sentiment) | httpx → Ollama |
| **GraphAgent** | Enriched articles | `nx.DiGraph` | networkx |
| **StorageAgent** | Articles + graph | SQLite rows | sqlite3 |
| **FastAPI** | HTTP requests | JSON responses | fastapi, uvicorn |

---

## Prerequisites

```bash
# 1. Python 3.12+
python --version  # 3.12+

# 2. Install Ollama
curl https://ollama.ai/install.sh | sh   # macOS/Linux
# or visit https://ollama.com on Windows

# 3. Pull the fastest local model
ollama pull gemma3:1b    # 815MB - ~50-100 tok/s on M1

# 4. Verify Ollama is running
curl http://localhost:11434/api/tags
```

---

## Step 1: Project Setup

```bash
git clone https://github.com/yourname/newsweave  # or use the provided files
cd newsweave

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create data directory
mkdir -p data
```

---

## Step 2: Understanding the NewsAgent

The NewsAgent is an **agentic fetch loop** - it doesn't just call one function; it:
1. Concurrently pulls 16 RSS feeds using `asyncio.gather`
2. Parses and normalises each entry
3. Deduplicates by URL hash
4. Optionally fetches full article text

### Key Design Pattern: Async Concurrency

```python
# All 16 feeds fetched simultaneously - not one-by-one
tasks = [self._parse_feed(f) for f in self.feeds]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

**Why this matters**: Sequential fetching would take 16 × ~2s = 32s. Concurrent fetching takes ~2-3s total.

### Adding Your Own Feeds

```python
# In news_agent.py, add to FEEDS list:
FEEDS.append({
    "url": "https://your-favourite-blog.com/rss",
    "source": "My Blog",
    "category": "technology",  # finance|technology|business|politics|geopolitics|science
})
```

### Running NewsAgent Standalone

```python
import asyncio
from agents.news_agent import NewsAgent

async def test():
    agent = NewsAgent(max_articles=10)
    articles = await agent.fetch()
    for a in articles:
        print(f"[{a.category}] {a.title[:60]} - {a.source}")

asyncio.run(test())
```

---

## Step 3: Understanding the AnalystAgent

The AnalystAgent uses **structured output prompting** to extract intelligence from each article using your local Ollama model.

### The Prompt Strategy

The key is returning **pure JSON** - no markdown, no explanation:

```
Return this exact JSON structure (no markdown, no explanation):
{
  "entities": [...],
  "themes": [...],
  "sentiment": "positive|neutral|negative",
  "impact_score": 0.0-1.0,
  "causal_edges": [...],
  "one_line_insight": "..."
}
```

### Model Selection Guide

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| `gemma3:1b` | 815MB | ⚡⚡⚡ | ✓✓ | High-volume pipeline (default) |
| `gemma3:4b` | 2.5GB | ⚡⚡ | ✓✓✓ | Better entity extraction |
| `llama3.2:3b` | 2.0GB | ⚡⚡ | ✓✓✓ | Good balance |
| `mistral:7b` | 4.1GB | ⚡ | ✓✓✓✓ | Best quality, slower |

```bash
# Switch model at runtime
export OLLAMA_MODEL=gemma3:4b
python agents/orchestrator.py
```

### Concurrency Tuning

```python
# AnalystAgent concurrency (default: 4 simultaneous Ollama calls)
analyst = AnalystAgent(
    model="gemma3:1b",
    concurrency=4,     # increase if Ollama is fast enough
    timeout=30,        # seconds per article
)
```

### Testing Without Ollama (Heuristic Fallback)

The agent automatically falls back to keyword heuristics if Ollama is offline:
```python
# Simply don't start Ollama - the agent handles it gracefully
# Sentiment detection uses keyword matching
# Entities use capitalised-word extraction (rough NER)
```

---

## Step 4: Understanding the GraphAgent

The knowledge graph is the product's core differentiator. It uses **NetworkX DiGraph** to model relationships.

### Node Types

```
Article node  → the news story itself
  art:a1f3b2  label="Fed holds rates" sentiment="neutral" impact=0.82

Entity node   → person, org, place, product, currency
  ent:nvidia  label="Nvidia" entity_type="ORG"

Theme node    → abstract topic
  theme:ai    label="AI"
```

### Edge Types

```
MENTIONS      Article → Entity     (weight = LLM relevance score)
COVERS        Article → Theme      (weight = 1.0)
CO_OCCURS     Entity  → Entity     (weight = shared article count, ≥2)
CAUSAL        Entity  → Entity     (extracted by LLM: "X causes Y")
ASSOCIATED_WITH Entity → Theme
```

### Graph Queries

```python
from agents.graph_agent import GraphAgent
import networkx as nx

g = GraphAgent()
# ... after running pipeline ...

# Top entities by degree centrality
centrality = nx.degree_centrality(g.G)
top_entities = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]

# Community detection - clusters of related entities
communities = nx.algorithms.community.greedy_modularity_communities(g.G.to_undirected())

# Shortest path between two entities (e.g. Fed → Nvidia)
path = nx.shortest_path(g.G, 'ent:fed', 'ent:nvidia')
# Shows how Fed policy → inflation → semiconductor demand → Nvidia

# What themes connect to a specific article?
article_node = 'art:a1f3b2'
theme_neighbours = [
    n for n in g.G.neighbors(article_node)
    if g.G.nodes[n].get('type') == 'theme'
]
```

### Visualising Impact Propagation

```python
# Find all articles connected to "Nvidia" within 2 hops
nvidia_node = 'ent:nvidia'
within_2 = nx.single_source_shortest_path_length(g.G, nvidia_node, cutoff=2)
# Returns: {node_id: distance, ...}
```

---

## Step 5: Running the Full Pipeline

```bash
# Start Ollama (if not already running)
ollama serve

# In a new terminal, run the pipeline once:
cd newsweave
python agents/orchestrator.py

# Output:
# 2025-04-26 08:00:01 [orchestrator] INFO: === NewsWeave pipeline starting ===
# 2025-04-26 08:00:02 [orchestrator] INFO: [1/4] Fetching articles...
# 2025-04-26 08:00:04 [orchestrator] INFO:       38 articles fetched
# 2025-04-26 08:00:04 [orchestrator] INFO: [2/4] Analysing with Ollama (gemma3:1b)...
# 2025-04-26 08:00:22 [orchestrator] INFO:       38 articles analysed
# 2025-04-26 08:00:22 [orchestrator] INFO: [3/4] Building knowledge graph...
# 2025-04-26 08:00:22 [orchestrator] INFO:       Graph: 127 nodes, 189 edges
# 2025-04-26 08:00:22 [orchestrator] INFO: [4/4] Persisting to database...
# 2025-04-26 08:00:22 [orchestrator] INFO: === Pipeline done in 21.4s ===
```

### Running Continuously (every 30 minutes)

```python
# In orchestrator.py, change main() to:
async def main():
    orc = NewsWeaveOrchestrator(run_interval_minutes=30)
    await orc.run_forever()   # runs indefinitely
```

---

## Step 6: Starting the API & Dashboard

```bash
# Terminal 1: Run the backend
uvicorn api:app --reload --port 8000

# Terminal 2: Serve the dashboard
cd ui
python -m http.server 3000
# or: npx serve . -p 3000

# Open in browser
open http://localhost:3000/dashboard.html
```

### API Endpoints

```bash
# Get articles (filterable)
curl "http://localhost:8000/api/articles?category=technology&limit=20"

# Get graph data (D3-ready JSON)
curl "http://localhost:8000/api/graph"

# Get stats
curl "http://localhost:8000/api/stats"

# Ask a question (RAG with Ollama)
curl "http://localhost:8000/api/ask?q=What+is+driving+tech+stocks%3F"

# Trigger pipeline manually
curl -X POST "http://localhost:8000/api/run"

# Top entities by graph centrality
curl "http://localhost:8000/api/entities/top?n=15"
```

---

## Step 7: Docker Deployment

```bash
# Build and start everything
docker compose up -d

# Check logs
docker compose logs -f api
docker compose logs -f scheduler

# The model puller automatically downloads gemma3:1b on first run
# Dashboard served at: http://localhost:3000
# API at: http://localhost:8000/docs
```

### Environment Variables

```bash
# .env file
OLLAMA_MODEL=gemma3:1b       # or gemma3:4b for better quality
NEWSAPI_KEY=your_key_here     # optional - adds more sources
```

---

## Step 8: Extending the System

### Add a New Intelligence Source

```python
# Create a new agent: agents/earnings_agent.py
class EarningsAgent:
    """Fetches SEC 8-K filings."""

    async def fetch(self) -> list[Article]:
        # Call SEC EDGAR API
        url = "https://efts.sec.gov/LATEST/search-index?q=%228-K%22&dateRange=custom&startdt=2025-04-25"
        ...

# Register in orchestrator.py
self.earnings_agent = EarningsAgent()

# In run_once():
earnings = await self.earnings_agent.fetch()
articles.extend(earnings)
```

### Add Custom Entity Types

```python
# In analyst_agent.py EXTRACT_PROMPT, extend the entity types:
# "type": "PERSON|ORG|PLACE|PRODUCT|CURRENCY|INDEX|ETF|CRYPTO|COMMODITY"
```

### Add Alerts

```python
# In orchestrator.py, after pipeline completes:
async def _check_alerts(self, articles):
    for art in articles:
        if art.impact_score >= 0.9 and art.sentiment == 'negative':
            await self._notify(f"⚠️ High-impact negative: {art.title}")

async def _notify(self, msg):
    # Send to Slack, Telegram, macOS notification, etc.
    import subprocess
    subprocess.run(['osascript', '-e', f'display notification "{msg}" with title "NewsWeave"'])
```

### Add Embeddings for Semantic Search

```python
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB, runs locally

def embed_articles(articles):
    texts = [f"{a.title}. {a.summary}" for a in articles]
    embeddings = model.encode(texts)
    return embeddings

# Store embeddings in SQLite blob column
# Query: find articles semantically similar to a user query
```

### Add Portfolio Monitoring

```python
# Track your watchlist against news entities
WATCHLIST = ['Apple', 'Nvidia', 'Microsoft', 'Tesla', 'TSMC']

def watchlist_alerts(articles, watchlist):
    alerts = []
    for art in articles:
        for ent in art.entities:
            if any(w.lower() in ent['name'].lower() for w in watchlist):
                alerts.append({
                    'ticker': ent['name'],
                    'headline': art.title,
                    'sentiment': art.sentiment,
                    'impact': art.impact_score,
                })
    return sorted(alerts, key=lambda x: x['impact'], reverse=True)
```

---

## Troubleshooting

### Ollama not responding
```bash
ollama serve                   # start the server
ollama list                    # verify model is downloaded
ollama run gemma3:1b "hello"   # test it manually
```

### RSS feeds returning empty
```bash
# Test a specific feed
python -c "
import feedparser
f = feedparser.parse('https://feeds.bbci.co.uk/news/world/rss.xml')
print(len(f.entries), 'entries')
print(f.entries[0].title)
"
```

### Database locked
```bash
# SQLite write conflict - restart the scheduler
docker compose restart scheduler
# or
pkill -f orchestrator.py && python agents/orchestrator.py
```

### Graph is too sparse
- Lower the CO_OCCURS threshold from 2 to 1 in `graph_agent.py`
- Increase `max_articles` in the orchestrator
- Run the pipeline more frequently (every 15 minutes)

---

## Performance Benchmarks

| Component | Time (M2 MacBook Pro) |
|-----------|----------------------|
| Fetch 40 articles (RSS) | ~3s |
| Analyse 40 articles (gemma3:1b) | ~18-25s |
| Build graph | <0.1s |
| SQLite save | <0.1s |
| **Total pipeline** | **~22-30s** |

For faster analysis, use `concurrency=8` in AnalystAgent, or switch to `gemma3:1b` if you're currently on a larger model.

---

## What Makes This a Product (Not Just a Script)

1. **Persistence** - SQLite survives restarts; graph accumulates over days/weeks
2. **Incremental** - deduplication ensures articles are never re-processed
3. **Resilient** - heuristic fallback means it works even without Ollama
4. **Observable** - pipeline logs, run table, dashboard stats
5. **Extensible** - new agents plug into the orchestrator without touching existing code
6. **Privacy-first** - everything runs locally; no data sent to cloud APIs
7. **Interactive** - D3 graph lets you explore relationships visually
8. **RAG-enabled** - ask questions grounded in today's actual news

---

## Next Steps

- [ ] Add **RSS OPML import** (import your current feed reader)
- [ ] Add **Telegram/Slack bot** for push alerts
- [ ] Add **portfolio watchlist** monitoring
- [ ] Add **weekly digest** PDF generation (ReportLab)
- [ ] Connect **financial data** (yfinance for price charts alongside news)
- [ ] Add **timeline view** showing how a story evolves over days
- [ ] Export graph to **Gephi** for advanced visual analysis
- [ ] Deploy to **Raspberry Pi** for always-on home intelligence server
