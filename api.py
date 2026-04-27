"""
NewsWeave FastAPI Backend
=========================
Serves the dashboard with:
  GET  /api/articles          → paginated article list
  GET  /api/graph             → graph JSON for D3 visualisation
  GET  /api/stats             → pipeline stats
  POST /api/run               → trigger a manual pipeline run
  GET  /api/entities/top      → top entities by degree
  GET  /api/ask               → ask a question (Ollama RAG)

Run:  uvicorn api:app --reload --port 8000
"""

import os
import asyncio
import re
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, Query, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import httpx

from paths import DB_PATH
from storage_agent import StorageAgent
from orchestrator import NewsWeaveOrchestrator
from world_impact import aggregate_world_impact
try:
    from agentic_workflow import NewsWeaveLangGraph
except Exception:  # pragma: no cover - optional dependency at runtime
    NewsWeaveLangGraph = None

app = FastAPI(title="NewsWeave API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def _ollama_base() -> str:
    return (os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")


api_log = logging.getLogger("api")


def _quick_topic_type_rules(name: str) -> Optional[str]:
    n = str(name or "").strip()
    if not n:
        return None
    k = n.lower().strip(" .")
    if k in {
        "donald trump", "trump", "joe biden", "biden", "vladimir putin",
        "xi jinping", "narendra modi", "elon musk", "jerome powell",
        "tim cook", "jensen huang",
    }:
        return "PERSON"
    if k in {
        "us", "u.s.", "u.s", "usa", "u.s.a.", "united states", "uk", "u.k.",
        "united kingdom", "china", "india", "russia", "japan", "france",
        "germany", "canada", "australia", "israel", "iran", "ukraine",
        "european union", "south korea", "north korea", "taiwan", "uae",
        "united arab emirates", "saudi arabia", "new zealand", "singapore",
        "mexico", "brazil", "pakistan", "bangladesh", "nepal", "sri lanka",
        "bhutan", "myanmar", "afghanistan", "indonesia", "malaysia",
        "philippines", "vietnam", "thailand", "nigeria", "south africa",
        "egypt", "turkey", "spain", "italy", "netherlands",
    }:
        return "COUNTRY"
    if k in {
        "washington", "new york", "london", "paris", "berlin", "tokyo", "beijing",
        "shanghai", "moscow", "kyiv", "jerusalem", "tel aviv", "delhi", "mumbai",
        "bangalore", "sydney", "melbourne", "toronto", "vancouver", "dubai",
        "abu dhabi", "riyadh", "tehran", "istanbul", "seoul", "taipei", "brussels",
        "lahore", "karachi", "islamabad", "dhaka", "kathmandu", "colombo", "yangon",
        "manila", "jakarta", "bangkok", "hanoi", "kuala lumpur", "lagos", "cairo",
        "madrid", "rome", "amsterdam", "barcelona", "milan",
        "kerala", "punjab", "sindh", "balochistan", "khyber pakhtunkhwa",
        "tamil nadu", "karnataka", "maharashtra", "gujarat", "rajasthan",
        "american", "british", "chinese", "indian", "russian", "japanese", "french",
        "german", "canadian", "australian", "israeli", "iranian", "ukrainian",
        "saudi", "emirati", "korean", "taiwanese", "singaporean", "mexican", "brazilian",
    }:
        return "PLACE"
    if re.search(r"\b(country|nation)\b", k):
        return "COUNTRY"
    if re.search(r"\b(state|province|city|island|region)\b", k):
        return "PLACE"
    if any(
        h in k
        for h in (
            "inc", "corp", "corporation", "ltd", "llc", "plc", "group", "bank",
            "ministry", "department", "committee", "agency", "administration",
            "university", "institute", "federal reserve", "nato", "opec", "who",
            "imf", "world bank",
        )
    ):
        return "ORG"
    if k in {
        "apple", "microsoft", "google", "alphabet", "amazon", "meta", "tesla",
        "nvidia", "openai", "anthropic", "deepmind", "blackrock", "goldman sachs",
        "morgan stanley", "jpmorgan", "fed", "ecb", "ofgem", "reuters",
        "bloomberg", "bbc", "techcrunch", "financial times", "wsj",
    }:
        return "ORG"
    if re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+$", n):
        return "PERSON"
    if re.match(r"^[A-Z]{2,4}$", n):
        if k in {"us", "uk"}:
            return "COUNTRY"
        if k in {"eu", "nato", "opec", "un"}:
            return "ORG"
    if re.match(r"^[A-Z][A-Za-z0-9&.-]{2,}$", n):
        return "ORG"
    return None


@app.get("/")
def index():
    return FileResponse("dashboard.html", media_type="text/html")


# Shared state (in-process; last run result for the UI)
last_pipeline_result: dict = {}
last_scheduler_tick: dict = {"last_run": None, "alerts_created": 0}

# Shared instances - DB path is absolute so API and CLI see the same SQLite file
storage = StorageAgent(db_path=DB_PATH)
orchestrator = NewsWeaveOrchestrator(
    ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b"),
    ollama_url=_ollama_base(),
    db_path=DB_PATH,
    max_articles=int(os.getenv("MAX_ARTICLES", "20")),
    run_interval_minutes=30,
)
agentic_runner = (
    NewsWeaveLangGraph(
        news_agent=orchestrator.news_agent,
        analyst_agent=orchestrator.analyst_agent,
        storage_agent=storage,
    )
    if NewsWeaveLangGraph is not None
    else None
)

# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/articles")
def get_articles(
    category: Optional[str] = None,
    sentiment: Optional[str] = None,
    limit: int = Query(500, le=5000),
    today: bool = Query(False),
    since: Optional[str] = None,
):
    articles = storage.get_articles(
        category=category,
        sentiment=sentiment,
        limit=limit,
        today_only=today,
        fetched_since=since,
    )
    return {"articles": articles, "total": len(articles)}


@app.get("/api/graph")
def get_graph():
    return storage.get_graph_json()


@app.get("/api/world/impact")
def world_impact(
    category: Optional[str] = None,
    limit: int = Query(200, le=500),
):
    """
    Country-level news impact (from titles, summaries, entities) for the world map.
    `category` matches article categories (same as /api/articles?category=); omit for all.
    """
    if category in (None, "", "all"):
        category = None
    articles = storage.get_articles(category=category, limit=limit)
    return aggregate_world_impact(articles)


@app.get("/api/stats")
def get_stats():
    graph = storage.get_graph_json()
    articles = storage.get_articles(limit=1000)
    sentiments = {"positive": 0, "neutral": 0, "negative": 0}
    categories = {}
    for a in articles:
        sentiments[a.get("sentiment", "neutral")] = \
            sentiments.get(a.get("sentiment", "neutral"), 0) + 1
        cat = a.get("category", "other")
        categories[cat] = categories.get(cat, 0) + 1

    return {
        "total_articles": len(articles),
        "total_nodes": len(graph["nodes"]),
        "total_edges": len(graph["links"]),
        "sentiment_breakdown": sentiments,
        "category_breakdown": categories,
        "last_run_at": storage.get_last_run_at(),
    }


@app.post("/api/run")
async def trigger_run(background_tasks: BackgroundTasks):
    """Trigger a pipeline run in the background."""
    global last_pipeline_result
    last_pipeline_result = {
        "status": "running",
        "started": datetime.utcnow().isoformat(),
    }
    background_tasks.add_task(_run_pipeline)
    return {"status": "Pipeline started", "timestamp": datetime.utcnow().isoformat()}


@app.post("/api/agent/run")
async def trigger_agentic_run():
    """Run one LangGraph-powered pipeline pass and return summary."""
    if agentic_runner is None:
        return {
            "status": "error",
            "message": "LangGraph not installed. Add 'langgraph' to requirements and reinstall.",
        }
    return await agentic_runner.run_once()


@app.get("/api/pipeline/last")
def pipeline_last():
    """Result of the most recent /api/run (ok or error, with summary when ok)."""
    return last_pipeline_result or {"status": "none", "message": "No run yet"}


def _emit_progress(p: Dict[str, Any]) -> None:
    """Merge live pipeline stage into last_pipeline_result for the UI."""
    global last_pipeline_result
    try:
        m = p.get("metrics") or {}
        api_log.debug(
            "pipeline_progress stage=%s message=%s current=%s total=%s metrics={articles:%s regions:%s edges:%s pos:%s neg:%s}",
            p.get("stage"),
            p.get("message"),
            p.get("current"),
            p.get("total"),
            m.get("articles_total_live", m.get("articles_fetched")),
            m.get("regions_count"),
            m.get("relations_built", m.get("graph_edges")),
            m.get("positive_signals"),
            m.get("negative_signals"),
        )
    except Exception:
        pass
    started = last_pipeline_result.get("started") or datetime.utcnow().isoformat()
    last_pipeline_result = {
        "status": "running",
        "started": started,
        "progress": p,
    }


async def _run_pipeline():
    global last_pipeline_result
    try:
        summary = await orchestrator.run_once(on_progress=_emit_progress)
        last_pipeline_result = {
            "status": "ok",
            "db_path": DB_PATH,
            "summary": summary,
        }
        print("Pipeline completed:", summary)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("Pipeline error:", e, "\n", tb)
        last_pipeline_result = {
            "status": "error",
            "message": str(e),
            "db_path": DB_PATH,
        }


async def _background_watchlist_loop():
    """Periodic evaluation loop for watchlist alerts."""
    while True:
        try:
            created = storage.evaluate_watchlists()
            last_scheduler_tick["last_run"] = datetime.utcnow().isoformat()
            last_scheduler_tick["alerts_created"] = len(created)
        except Exception:
            # Keep scheduler resilient in long-running mode.
            pass
        await asyncio.sleep(600)


@app.get("/api/entities/top")
def top_entities(
    n: int = Query(15, le=50),
    category: Optional[str] = None,
):
    """When ``category`` is set, rank entities by connections into that article slice."""
    if category in (None, "", "all"):
        category = None
    entities = storage.get_top_entities(n=n, category=category)
    return {"entities": entities}


@app.get("/api/watchlists")
def list_watchlists():
    return {"watchlists": storage.list_watchlists()}


@app.post("/api/watchlists")
def create_watchlist(
    name: str = Query(..., min_length=2),
    category: Optional[str] = None,
    query: Optional[str] = None,
    min_impact: float = Query(0.7, ge=0.0, le=1.0),
    min_articles: int = Query(2, ge=1, le=50),
):
    watchlist = storage.create_watchlist(
        name=name,
        category=category,
        query=query,
        min_impact=min_impact,
        min_articles=min_articles,
    )
    return {"watchlist": watchlist}


@app.delete("/api/watchlists/{watchlist_id}")
def delete_watchlist(watchlist_id: int):
    ok = storage.delete_watchlist(watchlist_id)
    return {"deleted": ok}


@app.post("/api/watchlists/evaluate")
def evaluate_watchlists():
    alerts = storage.evaluate_watchlists()
    return {"alerts_created": len(alerts), "alerts": alerts}


@app.get("/api/alerts")
def list_alerts(limit: int = Query(50, ge=1, le=500)):
    return {"alerts": storage.list_alerts(limit=limit)}


@app.get("/api/scheduler/status")
def scheduler_status():
    return last_scheduler_tick


@app.get("/api/brief")
def get_brief(category: Optional[str] = None, limit: int = Query(12, ge=3, le=100)):
    if category in (None, "", "all"):
        category = None
    return storage.build_brief(category=category, limit=limit)


@app.post("/api/topics/categorize")
async def categorize_topics(
    names: list[str] = Body(default=[]),
):
    """
    Quickly categorize topic labels into PERSON|ORG|COUNTRY|PLACE|PRODUCT|EVENT|UNKNOWN.
    Uses lightweight local rules first, then one short Ollama pass for unknowns.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for n in names[:80]:
        s = str(n or "").strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s)
    if not cleaned:
        return {"types": {}}

    types: dict[str, str] = {}
    unknown: list[str] = []
    for n in cleaned:
        t = _quick_topic_type_rules(n)
        if t:
            types[n] = t
        else:
            unknown.append(n)

    if unknown:
        prompt = (
            "Classify each topic label as exactly one of PERSON, ORG, COUNTRY, PLACE, PRODUCT, EVENT, UNKNOWN.\n"
            "Return minified JSON only in this schema: "
            '{"items":[{"name":"...", "type":"PERSON|ORG|COUNTRY|PLACE|PRODUCT|EVENT|UNKNOWN"}]}.\n'
            "Labels:\n- " + "\n- ".join(unknown[:40])
        )
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.post(
                    f"{_ollama_base()}/api/generate",
                    json={
                        "model": os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b"),
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.0, "num_predict": 220},
                    },
                )
                r.raise_for_status()
                raw = str(r.json().get("response", "")).strip()
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                parsed = json.loads(m.group())
                for it in parsed.get("items", []):
                    nm = str((it or {}).get("name") or "").strip()
                    tp = str((it or {}).get("type") or "").strip().upper()
                    if nm and tp in {"PERSON", "ORG", "COUNTRY", "PLACE", "PRODUCT", "EVENT", "UNKNOWN"}:
                        types[nm] = tp
        except Exception:
            # Keep endpoint fast and resilient - unknowns fall back to UNKNOWN.
            pass

    for n in cleaned:
        types[n] = types.get(n) or "UNKNOWN"
    return {"types": types}


@app.on_event("startup")
async def _start_background_scheduler():
    asyncio.create_task(_background_watchlist_loop())


@app.get("/api/ask")
async def ask_question(
    q: str = Query(..., min_length=3),
    category: Optional[str] = None,
):
    """
    Mini-RAG: retrieve top articles, answer via OpenAI (if OPENAI_API_KEY) else Ollama.
    Optional ``category`` restricts context to that topic (same labels as the sidebar).
    """
    if category in (None, "", "all"):
        category = None
    articles = storage.get_articles(category=category, limit=25)
    if not articles:
        return {
            "question": q,
            "answer": (
                "No articles in this category yet. Run the news pipeline, or choose 'All news' in the sidebar."
            ),
        }
    context_articles = articles[:8]
    context = "\n\n".join(
        f"[{a.get('source', '')}] {a.get('title', '')}: {str(a.get('summary', ''))[:200]}"
        for a in context_articles
    )
    prompt = (
        f"Based on these recent news articles:\n\n{context}\n\n"
        f"Answer this question concisely: {q}"
    )
    oa_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if oa_key:
        ob = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        mod = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    f"{ob}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {oa_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": mod,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 500,
                        "temperature": 0.2,
                    },
                )
                r.raise_for_status()
                answer = r.json()["choices"][0]["message"]["content"] or "No answer."
        except Exception:
            answer = "OpenAI request failed. Check key and model."
    else:
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
                # Prefer /api/chat first (newer Ollama), then fallback to /api/generate.
                try:
                    r = await client.post(
                        f"{_ollama_base()}/api/chat",
                        json={
                            "model": model_name,
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False,
                            "options": {"temperature": 0.2, "num_predict": 300},
                        },
                    )
                    if r.status_code == 404:
                        raise httpx.HTTPStatusError(
                            "chat endpoint missing",
                            request=r.request,
                            response=r,
                        )
                    r.raise_for_status()
                    j = r.json()
                    answer = (
                        ((j.get("message") or {}).get("content"))
                        or j.get("response")
                        or ""
                    ).strip()
                except Exception:
                    r2 = await client.post(
                        f"{_ollama_base()}/api/generate",
                        json={
                            "model": model_name,
                            "prompt": prompt,
                            "stream": False,
                            "options": {"temperature": 0.2, "num_predict": 300},
                        },
                    )
                    r2.raise_for_status()
                    j2 = r2.json()
                    answer = str(j2.get("response") or "").strip()

                if not answer:
                    answer = (
                        "No answer was returned by Ollama. Check model availability with: "
                        f"ollama list (expected: {model_name})"
                    )
        except Exception:
            answer = (
                "Ollama request failed. Make sure Ollama is running and model is pulled:\n"
                "1) ollama serve\n"
                f"2) ollama pull {os.getenv('OLLAMA_MODEL', 'qwen2.5:0.5b')}"
            )

    citations = [
        {
            "id": a.get("id"),
            "title": a.get("title"),
            "source": a.get("source"),
            "url": a.get("url"),
            "published_at": a.get("published_at"),
        }
        for a in context_articles[:5]
    ]
    confidence = (
        "high" if len({c.get("source") for c in citations if c.get("source")}) >= 4
        else "medium" if len(citations) >= 3
        else "low"
    )
    return {
        "question": q,
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
    }


# ── Dev entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
