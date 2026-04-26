"""
LangGraph-based orchestration for NewsWeave.

This keeps the existing agents (NewsAgent, AnalystAgent, GraphAgent, StorageAgent)
but runs them through an explicit state graph so the pipeline is easier to evolve,
test, and extend with human-in-the-loop controls.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from graph_agent import GraphAgent
from news_agent import NewsAgent, Article
from analyst_agent import AnalystAgent
from storage_agent import StorageAgent


class PipelineState(TypedDict, total=False):
    started_at: str
    completed_at: str
    articles: list[Article]
    graph_json: dict[str, Any]
    graph_nodes: int
    graph_edges: int
    error: str


class NewsWeaveLangGraph:
    """LangGraph runtime wrapping the existing NewsWeave pipeline."""

    def __init__(
        self,
        news_agent: NewsAgent,
        analyst_agent: AnalystAgent,
        storage_agent: StorageAgent,
    ) -> None:
        self.news_agent = news_agent
        self.analyst_agent = analyst_agent
        self.storage_agent = storage_agent
        self._graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(PipelineState)
        g.add_node("fetch_news", self._fetch_news)
        g.add_node("analyse_news", self._analyse_news)
        g.add_node("build_graph", self._build_graph_node)
        g.add_node("persist_graph", self._persist_graph)
        g.set_entry_point("fetch_news")
        g.add_edge("fetch_news", "analyse_news")
        g.add_edge("analyse_news", "build_graph")
        g.add_edge("build_graph", "persist_graph")
        g.add_edge("persist_graph", END)
        return g.compile()

    async def run_once(self) -> dict[str, Any]:
        state: PipelineState = {"started_at": datetime.utcnow().isoformat()}
        result = await self._graph.ainvoke(state)
        completed_at = datetime.utcnow().isoformat()
        result["completed_at"] = completed_at
        return self._to_summary(result)

    async def _fetch_news(self, state: PipelineState) -> PipelineState:
        articles = await self.news_agent.fetch()
        return {**state, "articles": articles}

    async def _analyse_news(self, state: PipelineState) -> PipelineState:
        articles = list(state.get("articles", []))
        if not articles:
            return state
        step = max(self.analyst_agent.chunk_size, 1)
        for i in range(0, len(articles), step):
            chunk = articles[i : i + step]
            await self.analyst_agent.enrich_chunk(chunk)
            self.storage_agent.upsert_articles(chunk)
        return {**state, "articles": articles}

    async def _build_graph_node(self, state: PipelineState) -> PipelineState:
        articles = list(state.get("articles", []))
        builder = GraphAgent()
        graph = builder.update(articles)
        stats = builder.stats()
        graph_json = builder.to_json()
        return {
            **state,
            "graph_json": graph_json,
            "graph_nodes": int(stats.get("nodes", 0)),
            "graph_edges": int(stats.get("edges", 0)),
            "articles": articles,
        }

    async def _persist_graph(self, state: PipelineState) -> PipelineState:
        articles = list(state.get("articles", []))
        graph_json = state.get("graph_json") or {"nodes": [], "links": []}
        builder = GraphAgent()
        graph = builder.update(articles)
        self.storage_agent.save(articles, graph, graph_json)
        return state

    def _to_summary(self, result: PipelineState) -> dict[str, Any]:
        started = result.get("started_at")
        ended = result.get("completed_at")
        elapsed = None
        if started and ended:
            try:
                elapsed = round(
                    (
                        datetime.fromisoformat(ended)
                        - datetime.fromisoformat(started)
                    ).total_seconds(),
                    2,
                )
            except Exception:
                elapsed = None
        return {
            "status": "ok",
            "started_at": started,
            "completed_at": ended,
            "elapsed_seconds": elapsed,
            "articles_fetched": len(result.get("articles", [])),
            "graph_nodes": int(result.get("graph_nodes", 0)),
            "graph_edges": int(result.get("graph_edges", 0)),
            "framework": "langgraph",
        }
