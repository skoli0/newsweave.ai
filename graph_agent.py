"""
GraphAgent - Builds and maintains the knowledge graph.

Nodes:
  - Article  (id, title, sentiment, impact, category)
  - Entity   (name, type)  - PERSON, ORG, PLACE, PRODUCT, …
  - Theme    (name)

Edges:
  - article → entity      (MENTIONS,  weight = relevance)
  - article → theme       (COVERS,    weight = 1.0)
  - entity  → entity      (CO_OCCURS, weight = count)
  - article → article     (CAUSAL,    from LLM causal_edges)
  - entity  → theme       (ASSOCIATED_WITH)

Uses NetworkX for in-memory graph; exported as JSON for the dashboard.
"""

import logging
from collections import defaultdict
from typing import Any

import networkx as nx

from news_agent import Article

log = logging.getLogger("graph")


class GraphAgent:
    """Maintains a persistent multigraph across pipeline runs."""

    def __init__(self):
        self.G: nx.DiGraph = nx.DiGraph()
        self._entity_article_index: dict[str, list[str]] = defaultdict(list)

    # ── public ──────────────────────────────────────────────────────────────

    def update(self, articles: list[Article]) -> nx.DiGraph:
        for art in articles:
            self._add_article_node(art)
            self._add_entity_nodes(art)
            self._add_theme_nodes(art)
            self._add_causal_edges(art)

        self._add_co_occurrence_edges(articles)
        log.debug("Graph updated: %d nodes, %d edges", self.G.number_of_nodes(), self.G.number_of_edges())
        return self.G

    def stats(self) -> dict:
        return {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "articles": sum(1 for _, d in self.G.nodes(data=True) if d.get("type") == "article"),
            "entities": sum(1 for _, d in self.G.nodes(data=True) if d.get("type") == "entity"),
            "themes":   sum(1 for _, d in self.G.nodes(data=True) if d.get("type") == "theme"),
        }

    def to_json(self) -> dict:
        """
        Export graph as D3-friendly JSON:
          { nodes: [{id, label, type, ...}], links: [{source, target, relation, weight}] }
        """
        return nx_digraph_to_d3(self.G)

    def top_entities(self, n: int = 10) -> list[dict]:
        """Return top-N entities by degree centrality."""
        entity_nodes = [
            n for n, d in self.G.nodes(data=True) if d.get("type") == "entity"
        ]
        centrality = nx.degree_centrality(self.G)
        ranked = sorted(
            entity_nodes, key=lambda n: centrality.get(n, 0), reverse=True
        )
        result = []
        for node in ranked[:n]:
            data = self.G.nodes[node]
            result.append({
                "name": data.get("label", node),
                "entity_type": data.get("entity_type", "ORG"),
                "degree": self.G.degree(node),
                "centrality": round(centrality.get(node, 0), 4),
            })
        return result

    def find_clusters(self) -> list[list[str]]:
        """Community detection using greedy modularity."""
        try:
            undirected = self.G.to_undirected()
            communities = nx.algorithms.community.greedy_modularity_communities(undirected)
            return [list(c) for c in communities]
        except Exception:
            return []

    # ── private ─────────────────────────────────────────────────────────────

    def _add_article_node(self, art: Article):
        self.G.add_node(
            f"art:{art.id}",
            type="article",
            label=art.title[:60],
            sentiment=art.sentiment,
            impact=art.impact_score,
            category=art.category,
            source=art.source,
            url=art.url,
            published_at=art.published_at,
        )

    def _add_entity_nodes(self, art: Article):
        for ent in art.entities:
            name = ent.get("name", "").strip()
            if not name or len(name) < 2:
                continue
            ent_id = f"ent:{name.lower().replace(' ', '_')}"

            if not self.G.has_node(ent_id):
                self.G.add_node(
                    ent_id,
                    type="entity",
                    label=name,
                    entity_type=ent.get("type", "ORG"),
                )

            self.G.add_edge(
                f"art:{art.id}", ent_id,
                relation="MENTIONS",
                weight=ent.get("relevance", 0.5),
            )
            self._entity_article_index[ent_id].append(f"art:{art.id}")

    def _add_theme_nodes(self, art: Article):
        for theme in art.themes:
            theme = theme.strip().lower()
            if not theme:
                continue
            theme_id = f"theme:{theme.replace(' ', '_')}"

            if not self.G.has_node(theme_id):
                self.G.add_node(theme_id, type="theme", label=theme)

            self.G.add_edge(
                f"art:{art.id}", theme_id,
                relation="COVERS",
                weight=1.0,
            )

    def _add_causal_edges(self, art: Article):
        """Add cause→effect edges extracted by the LLM."""
        for edge in art.related_ids:       # related_ids holds causal_edges list
            if not isinstance(edge, dict):
                continue
            cause = edge.get("cause", "").strip().lower()
            effect = edge.get("effect", "").strip().lower()
            relation = edge.get("relation", "CAUSES")
            if cause and effect and cause != effect:
                cause_id  = f"ent:{cause.replace(' ', '_')}"
                effect_id = f"ent:{effect.replace(' ', '_')}"
                # Ensure nodes exist
                for nid, label in [(cause_id, cause), (effect_id, effect)]:
                    if not self.G.has_node(nid):
                        self.G.add_node(nid, type="entity", label=label, entity_type="CONCEPT")
                self.G.add_edge(
                    cause_id, effect_id,
                    relation=relation,
                    source_article=art.id,
                    weight=art.impact_score,
                )

    def _add_co_occurrence_edges(self, articles: list[Article]):
        """
        If two entities appear in 2+ articles together, add a CO_OCCURS edge.
        Weight = number of shared articles.
        """
        # Build article→entities map
        art_entities: dict[str, set[str]] = {}
        for art in articles:
            ents = {
                f"ent:{e['name'].lower().replace(' ', '_')}"
                for e in art.entities
                if e.get("name")
            }
            art_entities[f"art:{art.id}"] = ents

        # Count co-occurrences
        co: dict[tuple, int] = defaultdict(int)
        for ents in art_entities.values():
            ent_list = sorted(ents)
            for i in range(len(ent_list)):
                for j in range(i + 1, len(ent_list)):
                    co[(ent_list[i], ent_list[j])] += 1

        # Add edges with weight ≥ 2
        for (e1, e2), count in co.items():
            if count >= 2 and self.G.has_node(e1) and self.G.has_node(e2):
                if self.G.has_edge(e1, e2):
                    self.G[e1][e2]["weight"] = self.G[e1][e2].get("weight", 0) + count
                else:
                    self.G.add_edge(e1, e2, relation="CO_OCCURS", weight=count)


def nx_digraph_to_d3(g: nx.DiGraph) -> dict:
    """Serialize a NetworkX graph for the dashboard and SQLite snapshot (JSON-safe)."""
    nodes = []
    for node_id, data in g.nodes(data=True):
        row = {"id": node_id}
        for k, v in data.items():
            if k == "id":
                continue
            if isinstance(v, (str, int, float, bool)) or v is None:
                row[k] = v
            else:
                row[k] = str(v)
        nodes.append(row)

    links = []
    for u, v, data in g.edges(data=True):
        rec = {"source": u, "target": v}
        for k, v2 in (data or {}).items():
            if k in ("source", "target", "u", "v"):
                continue
            if isinstance(v2, (str, int, float, bool)) or v2 is None:
                rec[k] = v2
            else:
                rec[k] = str(v2)
        links.append(rec)

    return {"nodes": nodes, "links": links}
