"""
StorageAgent - Persists articles and graph data to SQLite.

Tables:
  articles  - full article records + enrichment
  entities  - deduplicated entity registry
  edges     - graph edges for fast querying
  runs      - pipeline run logs
"""

import json
import logging
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import networkx as nx

from news_agent import Article
from graph_agent import nx_digraph_to_d3

log = logging.getLogger("storage")

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id            TEXT PRIMARY KEY,
    title         TEXT,
    url           TEXT,
    source        TEXT,
    category      TEXT,
    published_at  TEXT,
    summary       TEXT,
    sentiment     TEXT DEFAULT 'neutral',
    impact_score  REAL DEFAULT 0.5,
    themes        TEXT,          -- JSON list
    entities      TEXT,          -- JSON list
    fetched_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS entities (
    id            TEXT PRIMARY KEY,   -- ent:name_normalised
    label         TEXT,
    entity_type   TEXT,
    degree        INTEGER DEFAULT 0,
    last_seen     TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS edges (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source        TEXT,
    target        TEXT,
    relation      TEXT,
    weight        REAL DEFAULT 1.0,
    created_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ran_at        TEXT DEFAULT (datetime('now')),
    articles_in   INTEGER,
    nodes         INTEGER,
    edges_count   INTEGER,
    elapsed_s     REAL
);

CREATE INDEX IF NOT EXISTS idx_articles_category ON articles(category);
CREATE INDEX IF NOT EXISTS idx_articles_sentiment ON articles(sentiment);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);

-- Full D3 graph snapshot (all node types: article, entity, theme) - /api/graph reads this
CREATE TABLE IF NOT EXISTS graph_d3 (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    payload TEXT NOT NULL,
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS watchlists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT,
    query TEXT,
    min_impact REAL DEFAULT 0.7,
    min_articles INTEGER DEFAULT 2,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    watchlist_id INTEGER,
    title TEXT NOT NULL,
    payload TEXT NOT NULL, -- JSON
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(watchlist_id) REFERENCES watchlists(id)
);

CREATE INDEX IF NOT EXISTS idx_alerts_watchlist ON alerts(watchlist_id);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);
"""


class StorageAgent:
    """SQLite persistence layer."""

    def __init__(self, db_path: str = "data/newsweave.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def save(
        self,
        articles: list[Article],
        graph: nx.DiGraph,
        d3: Optional[Dict[str, Any]] = None,
    ):
        d3 = d3 if d3 is not None else nx_digraph_to_d3(graph)
        with self._conn() as conn:
            self._upsert_articles(conn, articles)
            self._upsert_entities(conn, graph)
            self._upsert_edges(conn, graph)
            self._store_graph_d3(conn, d3)
            conn.execute(
                "INSERT INTO runs (articles_in, nodes, edges_count) VALUES (?, ?, ?)",
                (len(articles), graph.number_of_nodes(), graph.number_of_edges()),
            )
        log.info("Saved %d articles + graph to %s", len(articles), self.db_path)

    def upsert_articles(self, articles: list[Article]):
        """Write or update article rows only (for live updates during a pipeline run)."""
        if not articles:
            return
        with self._conn() as conn:
            self._upsert_articles(conn, articles)

    def _upsert_articles(self, conn, articles: list[Article]):
        conn.executemany(
            """
            INSERT INTO articles (id, title, url, source, category, published_at,
                                  summary, sentiment, impact_score, themes, entities)
            VALUES (:id,:title,:url,:source,:category,:published_at,
                    :summary,:sentiment,:impact_score,:themes,:entities)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                url=excluded.url,
                source=excluded.source,
                category=excluded.category,
                published_at=excluded.published_at,
                summary=excluded.summary,
                sentiment=excluded.sentiment,
                impact_score=excluded.impact_score,
                themes=excluded.themes,
                entities=excluded.entities
            """,
            [
                {
                    "id": a.id,
                    "title": a.title,
                    "url": a.url,
                    "source": a.source,
                    "category": a.category,
                    "published_at": a.published_at,
                    "summary": a.summary[:1000],
                    "sentiment": a.sentiment,
                    "impact_score": a.impact_score,
                    "themes": json.dumps(a.themes),
                    "entities": json.dumps(a.entities),
                }
                for a in articles
            ],
        )

    def _upsert_entities(self, conn, graph: nx.DiGraph):
        rows = [
            {
                "id": nid,
                "label": data.get("label", nid),
                "entity_type": data.get("entity_type", "ORG"),
                "degree": graph.degree(nid),
            }
            for nid, data in graph.nodes(data=True)
            if data.get("type") == "entity"
        ]
        conn.executemany(
            """
            INSERT INTO entities (id, label, entity_type, degree)
            VALUES (:id,:label,:entity_type,:degree)
            ON CONFLICT(id) DO UPDATE SET degree=excluded.degree, last_seen=datetime('now')
            """,
            rows,
        )

    def _upsert_edges(self, conn, graph: nx.DiGraph):
        # Wipe and rebuild edges (simple for tutorial; production would diff)
        conn.execute("DELETE FROM edges")
        rows = [
            {
                "source": u,
                "target": v,
                "relation": data.get("relation", "RELATED"),
                "weight": data.get("weight", 1.0),
            }
            for u, v, data in graph.edges(data=True)
        ]
        conn.executemany(
            "INSERT INTO edges (source, target, relation, weight) VALUES (:source,:target,:relation,:weight)",
            rows,
        )

    def _store_graph_d3(self, conn, d3: Dict[str, Any]) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO graph_d3 (id, payload, updated_at) VALUES (1, ?, datetime('now'))",
            (json.dumps(d3),),
        )

    # ── query helpers (used by the FastAPI backend) ──────────────────────────

    def get_articles(
        self,
        category: str = None,
        sentiment: str = None,
        limit: int = 50,
        *,
        today_only: bool = False,
        fetched_since: Optional[str] = None,
    ) -> list[dict]:
        sql = "SELECT * FROM articles WHERE 1=1"
        params = []
        if category:
            # Case-insensitive so UI filters match DB regardless of how feeds stored labels
            sql += " AND lower(trim(coalesce(category, ''))) = lower(trim(?))"
            params.append(category)
        if sentiment:
            sql += " AND sentiment=?"
            params.append(sentiment)
        if today_only:
            # SQLite stores fetched_at as UTC text; this keeps dashboard scoped to today's batch.
            sql += " AND date(fetched_at) = date('now')"
        if fetched_since:
            sql += " AND fetched_at >= ?"
            params.append(fetched_since)
        # For feed display, newest first is easiest to reason about.
        sql += " ORDER BY fetched_at DESC, impact_score DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_article(dict(r)) for r in rows]

    def count_articles(self, *, today_only: bool = False) -> int:
        sql = "SELECT COUNT(*) FROM articles WHERE 1=1"
        params = []
        if today_only:
            sql += " AND date(fetched_at) = date('now')"
        with self._conn() as conn:
            row = conn.execute(sql, params).fetchone()
        try:
            return int(row[0]) if row else 0
        except (TypeError, ValueError):
            return 0

    def _row_to_article(self, row: dict) -> dict:
        """Deserialize JSON fields for API / dashboard consumers."""
        out = dict(row)
        for key in ("themes", "entities"):
            raw = out.get(key)
            if isinstance(raw, str) and raw:
                try:
                    out[key] = json.loads(raw)
                except json.JSONDecodeError:
                    out[key] = []
            elif raw is None:
                out[key] = []
        if out.get("impact_score") is not None:
            try:
                out["impact_score"] = float(out["impact_score"])
            except (TypeError, ValueError):
                out["impact_score"] = 0.5
        return out

    def get_last_run_at(self) -> Optional[str]:
        """Return the most recent pipeline run timestamp as a UTC ISO string (with Z suffix)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT ran_at FROM runs ORDER BY ran_at DESC LIMIT 1"
            ).fetchone()
            if row and row[0]:
                ts = str(row[0]).rstrip("Z") + "Z"
                return ts
            # Fallback: max fetched_at from articles (always set on insert)
            row2 = conn.execute(
                "SELECT MAX(fetched_at) FROM articles"
            ).fetchone()
            if row2 and row2[0]:
                return str(row2[0]).rstrip("Z") + "Z"
        return None

    def get_graph_json(self) -> dict:
        """Return the last stored D3 graph (all node types). Fallback: empty graph."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT payload FROM graph_d3 WHERE id=1"
            ).fetchone()
        if row and row[0]:
            try:
                data = json.loads(row[0])
                if isinstance(data, dict) and "nodes" in data and "links" in data:
                    return data
            except json.JSONDecodeError:
                pass
        return {"nodes": [], "links": []}

    def get_top_entities(self, n: int = 15, category: Optional[str] = None) -> list[dict]:
        """
        Return top entity rows. When ``category`` is set, rank entities that appear
        in edges next to an article in that category (by touch count in that slice).
        """
        with self._conn() as conn:
            if not category or str(category).strip().lower() in ("", "all", "none"):
                rows = conn.execute(
                    "SELECT label, entity_type, degree FROM entities ORDER BY degree DESC LIMIT ?",
                    (n,),
                ).fetchall()
                return [dict(r) for r in rows]
            cat = str(category).strip()
            arows = conn.execute(
                "SELECT id FROM articles WHERE lower(trim(coalesce(category, ''))) = lower(trim(?))",
                (cat,),
            ).fetchall()
        art_set = {f"art:{r[0]}" for r in arows}
        if not art_set:
            return []
        with self._conn() as conn:
            erows = conn.execute("SELECT source, target FROM edges").fetchall()
        ent_touch: dict[str, int] = defaultdict(int)
        for s, t in erows:
            if s in art_set and t.startswith("ent:"):
                ent_touch[t] += 1
            if t in art_set and s.startswith("ent:"):
                ent_touch[s] += 1
        if not ent_touch:
            return []
        ranked = sorted(ent_touch.keys(), key=lambda e: -ent_touch[e])[:n]
        out: list[dict] = []
        with self._conn() as conn:
            for eid in ranked:
                row = conn.execute(
                    "SELECT label, entity_type, degree FROM entities WHERE id = ?",
                    (eid,),
                ).fetchone()
                if row:
                    out.append(
                        {
                            "label": row[0],
                            "entity_type": row[1] or "ORG",
                            "degree": int(ent_touch[eid]),
                        }
                    )
        return out

    # ── watchlists / alerts / briefs ────────────────────────────────────────

    def create_watchlist(
        self,
        *,
        name: str,
        category: Optional[str] = None,
        query: Optional[str] = None,
        min_impact: float = 0.7,
        min_articles: int = 2,
    ) -> dict:
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO watchlists (name, category, query, min_impact, min_articles)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(name).strip(),
                    (str(category).strip() if category else None),
                    (str(query).strip() if query else None),
                    float(min_impact),
                    max(1, int(min_articles)),
                ),
            )
            wid = cur.lastrowid
        return self.get_watchlist(wid) or {}

    def get_watchlist(self, watchlist_id: int) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM watchlists WHERE id = ?",
                (int(watchlist_id),),
            ).fetchone()
        return dict(row) if row else None

    def list_watchlists(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM watchlists ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_watchlist(self, watchlist_id: int) -> bool:
        with self._conn() as conn:
            conn.execute("DELETE FROM alerts WHERE watchlist_id = ?", (int(watchlist_id),))
            cur = conn.execute("DELETE FROM watchlists WHERE id = ?", (int(watchlist_id),))
        return cur.rowcount > 0

    def list_alerts(self, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT a.*, w.name AS watchlist_name
                FROM alerts a
                LEFT JOIN watchlists w ON w.id = a.watchlist_id
                ORDER BY a.created_at DESC, a.id DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            payload = d.get("payload")
            if isinstance(payload, str):
                try:
                    d["payload"] = json.loads(payload)
                except json.JSONDecodeError:
                    d["payload"] = {}
            out.append(d)
        return out

    def evaluate_watchlists(self, lookback_limit: int = 250) -> list[dict]:
        alerts_created: list[dict] = []
        watchlists = self.list_watchlists()
        if not watchlists:
            return alerts_created
        recent = self.get_articles(limit=max(20, int(lookback_limit)))
        for w in watchlists:
            cat = (w.get("category") or "").strip() or None
            query = (w.get("query") or "").strip().lower()
            min_imp = float(w.get("min_impact") or 0.7)
            min_articles = max(1, int(w.get("min_articles") or 2))
            matched = []
            for a in recent:
                if cat and str(a.get("category", "")).strip().lower() != cat.lower():
                    continue
                if query:
                    blob = " ".join(
                        [
                            str(a.get("title") or ""),
                            str(a.get("summary") or ""),
                            " ".join(
                                str((e or {}).get("name", e))
                                for e in (a.get("entities") or [])
                            ),
                        ]
                    ).lower()
                    if query not in blob:
                        continue
                try:
                    impact = float(a.get("impact_score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    impact = 0.0
                if impact < min_imp:
                    continue
                matched.append(a)
            if len(matched) < min_articles:
                continue
            top = sorted(
                matched,
                key=lambda x: float(x.get("impact_score", 0.0) or 0.0),
                reverse=True,
            )[:5]
            payload = {
                "match_count": len(matched),
                "top_articles": [
                    {
                        "id": a.get("id"),
                        "title": a.get("title"),
                        "url": a.get("url"),
                        "source": a.get("source"),
                        "impact_score": a.get("impact_score"),
                        "sentiment": a.get("sentiment"),
                    }
                    for a in top
                ],
            }
            title = f"{w.get('name', 'Watchlist')} triggered ({len(matched)} matches)"
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO alerts (watchlist_id, title, payload) VALUES (?, ?, ?)",
                    (int(w["id"]), title, json.dumps(payload)),
                )
            alerts_created.append(
                {
                    "watchlist_id": int(w["id"]),
                    "title": title,
                    "payload": payload,
                }
            )
        return alerts_created

    def build_brief(self, *, category: Optional[str] = None, limit: int = 12) -> dict:
        articles = self.get_articles(category=category, limit=max(3, int(limit)))
        if not articles:
            return {
                "summary": "No fresh articles found for this scope.",
                "highlights": [],
                "sentiment_breakdown": {"positive": 0, "neutral": 0, "negative": 0},
                "sources": [],
            }
        sentiments = {"positive": 0, "neutral": 0, "negative": 0}
        for a in articles:
            s = str(a.get("sentiment", "neutral")).lower()
            sentiments[s if s in sentiments else "neutral"] += 1
        ranked = sorted(
            articles,
            key=lambda a: float(a.get("impact_score", 0.0) or 0.0),
            reverse=True,
        )
        highlights = [
            {
                "title": a.get("title"),
                "source": a.get("source"),
                "impact_score": a.get("impact_score"),
                "sentiment": a.get("sentiment"),
                "url": a.get("url"),
            }
            for a in ranked[:5]
        ]
        sources = sorted({str(a.get("source") or "") for a in articles if a.get("source")})
        summary = (
            f"Scanned {len(articles)} recent stories"
            + (f" in {category}" if category else "")
            + f". Top signal concentration is {sentiments['negative']} negative, "
              f"{sentiments['neutral']} neutral, and {sentiments['positive']} positive items."
        )
        return {
            "summary": summary,
            "highlights": highlights,
            "sentiment_breakdown": sentiments,
            "sources": sources[:12],
            "generated_at": datetime.utcnow().isoformat(),
        }
