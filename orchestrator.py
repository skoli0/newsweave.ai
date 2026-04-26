"""
NewsWeave Agent Orchestrator
============================
Coordinates the full agentic pipeline:
  1. NewsAgent   → fetches headlines from RSS + NewsAPI
  2. AnalystAgent → uses Ollama to extract entities, sentiment, themes
  3. GraphAgent  → builds/updates the knowledge graph (NetworkX)
  4. StorageAgent → persists nodes/edges to SQLite

Run:  python orchestrator.py
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from news_agent import NewsAgent, Article
from analyst_agent import AnalystAgent
from graph_agent import GraphAgent
from storage_agent import StorageAgent
from paths import DB_PATH
from world_impact import detect_countries_for_article

_LOG_LEVEL = (os.getenv("LOG_LEVEL") or "DEBUG").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.DEBUG),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("orchestrator")


class NewsWeaveOrchestrator:
    """Top-level agent that coordinates the full pipeline."""

    def __init__(
        self,
        ollama_model: str = "qwen2.5:0.5b",    # sub‑1B, fast; override OLLAMA_MODEL
        ollama_url: Optional[str] = None,
        db_path: Optional[str] = None,
        max_articles: int = 20,
        run_interval_minutes: int = 30,
    ):
        self.ollama_model = ollama_model
        self.ollama_url = (ollama_url or os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        self.db_path = db_path or DB_PATH
        self.max_articles = max_articles
        self.run_interval = run_interval_minutes * 60
        self.graph_context_articles = int(os.getenv("GRAPH_CONTEXT_ARTICLES", "500"))

        # Initialise sub-agents (use self.ollama_url, not the raw arg - the arg can be None when env is set)
        self.news_agent = NewsAgent(max_articles=max_articles)
        self.analyst_agent = AnalystAgent(
            model=ollama_model or os.getenv("OLLAMA_MODEL") or "qwen2.5:0.5b",
            base_url=self.ollama_url,
        )
        self.storage_agent = StorageAgent(db_path=self.db_path)

    def _dict_to_article(self, row: Dict[str, Any]) -> Article:
        """Convert persisted article rows back into Article dataclass for graph build."""
        return Article(
            id=str(row.get("id") or ""),
            title=str(row.get("title") or "Untitled"),
            url=str(row.get("url") or ""),
            source=str(row.get("source") or ""),
            category=str(row.get("category") or "general"),
            published_at=str(row.get("published_at") or datetime.utcnow().isoformat()),
            summary=str(row.get("summary") or ""),
            full_text=str(row.get("full_text") or ""),
            entities=list(row.get("entities") or []),
            themes=list(row.get("themes") or []),
            sentiment=str(row.get("sentiment") or "neutral"),
            impact_score=float(row.get("impact_score") or 0.5),
            related_ids=list(row.get("related_ids") or []),
        )

    def _articles_for_graph(self, current_batch: list[Article]) -> list[Article]:
        """
        Build graph over current batch + rolling historical context.
        This keeps fetch fast (small batch) while preserving dense graph relations.
        """
        by_id: dict[str, Article] = {a.id: a for a in current_batch if a.id}
        try:
            rows = self.storage_agent.get_articles(limit=max(50, self.graph_context_articles))
            for r in rows:
                aid = str(r.get("id") or "")
                if not aid or aid in by_id:
                    continue
                by_id[aid] = self._dict_to_article(r)
        except Exception as e:
            log.debug("Historical graph context load failed: %s", e)
        return list(by_id.values())

    def _pg(
        self,
        on_progress: Optional[Callable[[Dict[str, Any]], None]],
        payload: Dict[str, Any],
    ) -> None:
        try:
            stage = payload.get("stage")
            msg = payload.get("message")
            metrics = payload.get("metrics") or {}
            if isinstance(metrics, dict):
                log.debug(
                    "progress stage=%s message=%s metrics={articles:%s regions:%s edges:%s pos:%s neg:%s}",
                    stage,
                    msg,
                    metrics.get("articles_total_live", metrics.get("articles_fetched")),
                    metrics.get("regions_count"),
                    metrics.get("relations_built", metrics.get("graph_edges")),
                    metrics.get("positive_signals"),
                    metrics.get("negative_signals"),
                )
        except Exception:
            pass
        if on_progress:
            try:
                on_progress(payload)
            except Exception as e:
                log.debug("on_progress error (ignored): %s", e)

    def _metrics_from_articles(
        self,
        articles: list[Article],
        *,
        graph_nodes: Optional[int] = None,
        graph_edges: Optional[int] = None,
        graph_articles_used: Optional[int] = None,
        articles_total_live: Optional[int] = None,
        saved_articles_total: Optional[int] = None,
    ) -> Dict[str, Any]:
        sentiments = {"positive": 0, "neutral": 0, "negative": 0}
        regions: set[str] = set()
        relations_live = 0
        for a in articles:
            s = str(getattr(a, "sentiment", "neutral") or "neutral").lower()
            if s not in sentiments:
                s = "neutral"
            sentiments[s] += 1
            try:
                rel = getattr(a, "related_ids", []) or []
                if isinstance(rel, list):
                    relations_live += len(rel)
            except Exception:
                pass
            try:
                ad = {
                    "title": a.title,
                    "summary": a.summary,
                    "source": a.source,
                    "entities": a.entities,
                }
                regions.update(detect_countries_for_article(ad))
            except Exception:
                pass
        out: Dict[str, Any] = {
            "articles_fetched": len(articles),
            "articles_total_live": int(articles_total_live if articles_total_live is not None else len(articles)),
            "sentiment_breakdown": sentiments,
            "regions_count": len(regions),
            "positive_signals": sentiments["positive"],
            "negative_signals": sentiments["negative"],
            # Live relation signal from analysed/saved article causal links.
            "relations_built": int(relations_live),
        }
        if saved_articles_total is not None:
            out["saved_articles_total"] = int(saved_articles_total)
        if graph_articles_used is not None:
            out["graph_articles_used"] = int(graph_articles_used)
            out["articles_total_live"] = int(graph_articles_used)
        if graph_nodes is not None:
            out["graph_nodes"] = int(graph_nodes)
        if graph_edges is not None:
            out["graph_edges"] = int(graph_edges)
            out["relations_built"] = int(graph_edges)
        return out

    def _live_saved_metrics(self) -> Dict[str, Any]:
        """Absolute metrics from currently saved (today) articles in DB."""
        rows = self.storage_agent.get_articles(limit=5000, today_only=True)
        saved_articles: list[Article] = [self._dict_to_article(r) for r in rows]
        m = self._metrics_from_articles(saved_articles, saved_articles_total=len(rows))
        try:
            g = self.storage_agent.get_graph_json()
            m["graph_nodes"] = int(len(g.get("nodes") or []))
            m["graph_edges"] = int(len(g.get("links") or []))
        except Exception:
            pass
        return m

    async def run_once(
        self, on_progress: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> dict:
        """Execute one full pipeline pass. Returns stats dict.

        on_progress: optional callback with stage updates for live UIs. Articles are
        written to the DB as soon as they are fetched, then after each Ollama analysis.
        """
        start = datetime.utcnow()
        log.info("=== NewsWeave pipeline starting ===")
        saved_total_live = self.storage_agent.count_articles(today_only=True)

        self._pg(
            on_progress,
            {
                "stage": "fetch",
                "message": "Fetching RSS feeds…",
                "current": 0,
                "total": 0,
                "metrics": self._live_saved_metrics(),
            },
        )

        # Step 1: Fetch news
        log.info("[1/4] Fetching articles...")
        articles = await self.news_agent.fetch()
        log.info(f"      {len(articles)} articles fetched")
        for idx, a in enumerate(articles, start=1):
            log.debug(
                "fetched[%d/%d] id=%s category=%s source=%s title=%s",
                idx,
                len(articles),
                a.id,
                a.category,
                a.source,
                (a.title or "")[:160],
            )

        n = len(articles)
        fetched_so_far: list[Article] = []
        self._pg(
            on_progress,
            {
                "stage": "fetched",
                "message": f"Loaded {n} article(s) from RSS - writing to database…",
                "articles_fetched": n,
                "current": 0,
                "total": n,
                "metrics": self._live_saved_metrics(),
            },
        )
        for idx, art in enumerate(articles, start=1):
            self.storage_agent.upsert_articles([art])
            saved_total_live = self.storage_agent.count_articles(today_only=True)
            fetched_so_far.append(art)
            log.debug(
                "stored fetched[%d/%d] id=%s title=%s",
                idx,
                n,
                art.id,
                (art.title or "")[:120],
            )
            self._pg(
                on_progress,
                {
                    "stage": "fetched",
                    "message": f"Fetched {idx}/{n} article(s)…",
                    "articles_fetched": n,
                    "current": idx,
                    "total": n,
                    "metrics": self._live_saved_metrics(),
                },
            )

        self._pg(
            on_progress,
            {
                "stage": "fetched",
                "message": f"Loaded {n} article(s) - {self.analyst_agent.backend} (0/{n})…",
                "articles_fetched": n,
                "current": 0,
                "total": n,
                "metrics": self._live_saved_metrics(),
            },
        )

        # Step 2: Batched LLM (1 request per chunk - OpenAI or Ollama; much faster)
        b = self.analyst_agent.backend
        m = self.analyst_agent.model
        # For real-time UI visibility, process analysis one-by-one and emit per-article progress.
        step = 1
        log.info(
            "[2/4] Analysing with %s (%s), chunk=%d...",
            b,
            m,
            step,
        )
        n_batches = (n + step - 1) // max(step, 1) if n else 0
        for bi, i in enumerate(range(0, n, step)):
            chunk = articles[i : i + step]
            await self.analyst_agent.enrich_chunk(chunk)
            self.storage_agent.upsert_articles(chunk)
            saved_total_live = self.storage_agent.count_articles(today_only=True)
            done = min(i + len(chunk), n)
            for a in chunk:
                log.debug(
                    "analysed/stored id=%s sentiment=%s entities=%d themes=%d impact=%.3f",
                    a.id,
                    a.sentiment,
                    len(a.entities or []),
                    len(a.themes or []),
                    float(a.impact_score or 0.5),
                )
            self._pg(
                on_progress,
                {
                    "stage": "analyse",
                    "message": f"Analysed {done}/{n} article(s)…",
                    "articles_fetched": n,
                    "current": done,
                    "total": n,
                    "metrics": self._live_saved_metrics(),
                },
            )
        analysed = articles
        log.info("      %d articles enriched via %s", len(analysed), b)

        # Step 3: Build knowledge graph using rolling corpus for richer relations.
        log.info("[3/4] Building knowledge graph...")
        graph_articles = self._articles_for_graph(analysed)
        self._pg(
            on_progress,
            {
                "stage": "graph",
                "message": "Building knowledge graph (0%)…",
                "articles_fetched": n,
                "current": n,
                "total": n,
                "metrics": self._metrics_from_articles(
                    graph_articles,
                    graph_articles_used=len(graph_articles),
                    saved_articles_total=saved_total_live,
                ),
            },
        )
        g_builder = GraphAgent()
        graph = g_builder.update(graph_articles)
        stats = g_builder.stats()
        log.debug(
            "graph updated articles_used=%d nodes=%d edges=%d",
            len(graph_articles),
            int(stats.get("nodes") or 0),
            int(stats.get("edges") or 0),
        )
        self._pg(
            on_progress,
            {
                "stage": "graph",
                "message": "Building knowledge graph (100%)…",
                "articles_fetched": n,
                "current": n,
                "total": n,
                "metrics": self._metrics_from_articles(
                    graph_articles,
                    graph_nodes=stats.get("nodes"),
                    graph_edges=stats.get("edges"),
                    graph_articles_used=len(graph_articles),
                    saved_articles_total=saved_total_live,
                ),
            },
        )
        log.info(
            "      Graph: %d nodes, %d edges (from %d articles)",
            stats["nodes"],
            stats["edges"],
            len(graph_articles),
        )

        # Step 4: Persist graph + edges to SQLite
        log.info("[4/4] Persisting to database…")
        self._pg(
            on_progress,
            {
                "stage": "persist",
                "message": "Saving graph and edges…",
                "articles_fetched": n,
                "current": n,
                "total": n,
                "metrics": self._metrics_from_articles(
                    graph_articles,
                    graph_nodes=stats.get("nodes"),
                    graph_edges=stats.get("edges"),
                    graph_articles_used=len(graph_articles),
                    saved_articles_total=saved_total_live,
                ),
            },
        )
        d3 = g_builder.to_json()
        self.storage_agent.save(graph_articles, graph, d3)
        log.debug(
            "persist complete saved_articles=%d graph_nodes=%d graph_edges=%d",
            len(graph_articles),
            int(stats.get("nodes") or 0),
            int(stats.get("edges") or 0),
        )

        elapsed = (datetime.utcnow() - start).total_seconds()
        summary = {
            "timestamp": start.isoformat(),
            "articles_fetched": len(articles),
            "articles_analysed": len(analysed),
            "graph_articles_used": len(graph_articles),
            "graph_nodes": stats["nodes"],
            "graph_edges": stats["edges"],
            "elapsed_seconds": round(elapsed, 2),
        }
        log.info("=== Pipeline done in %.1fs ===", elapsed)
        return summary

    async def run_forever(self):
        """Continuous loop - runs every run_interval seconds."""
        while True:
            try:
                await self.run_once()
            except Exception as e:
                log.error("Pipeline error: %s", e, exc_info=True)
            log.info("Sleeping %d minutes...", self.run_interval // 60)
            await asyncio.sleep(self.run_interval)


async def main():
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
    ollama_url = (os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
    max_articles = int(os.getenv("MAX_ARTICLES", "20"))
    interval = int(os.getenv("RUN_INTERVAL_MINUTES", "30"))
    orc = NewsWeaveOrchestrator(
        ollama_model=model,
        ollama_url=ollama_url,
        db_path=DB_PATH,
        max_articles=max_articles,
        run_interval_minutes=interval,
    )
    if os.getenv("SCHEDULER_MODE", "").lower() in ("1", "true", "yes"):
        await orc.run_forever()
    else:
        summary = await orc.run_once()
        print("\n--- Pipeline Summary ---")
        for k, v in summary.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
