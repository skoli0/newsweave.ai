"""
NewsAgent - Fetches articles from curated RSS feeds.

Covers: Finance, Technology, Business, Politics, Science, Geopolitics.
No API key required for RSS; optional NewsAPI key for richer results.
"""

import asyncio
import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import feedparser          # pip install feedparser
import httpx               # pip install httpx


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Article:
    id: str
    title: str
    url: str
    source: str
    category: str
    published_at: str
    summary: str = ""
    full_text: str = ""
    # Filled by AnalystAgent
    entities: list = field(default_factory=list)
    themes: list = field(default_factory=list)
    sentiment: str = "neutral"
    impact_score: float = 0.5
    related_ids: list = field(default_factory=list)

    @staticmethod
    def make_id(url: str) -> str:
        return hashlib.sha1(url.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Feed registry
# ---------------------------------------------------------------------------

FEEDS = [
    # India / subcontinent coverage
    {"url": "https://www.thehindu.com/news/national/feeder/default.rss", "source": "The Hindu India", "category": "politics"},
    {"url": "https://indianexpress.com/section/india/feed/",             "source": "Indian Express India", "category": "politics"},
    {"url": "https://www.firstpost.com/category/india/feed",             "source": "Firstpost India", "category": "geopolitics"},
    {"url": "https://www.livemint.com/rss/news",                         "source": "Mint India", "category": "business"},
    {"url": "https://www.livemint.com/rss/markets",                      "source": "Mint Markets", "category": "finance"},
    {"url": "https://www.business-standard.com/rss/home_page_top_stories.rss", "source": "Business Standard India", "category": "business"},
    {"url": "https://www.thehindu.com/sci-tech/science/feeder/default.rss", "source": "The Hindu Science", "category": "science"},

    # Finance
    {"url": "https://feeds.bloomberg.com/markets/news.rss",            "source": "Bloomberg",       "category": "finance"},
    {"url": "https://feeds.reuters.com/reuters/businessNews",          "source": "Reuters Business", "category": "business"},
    {"url": "https://www.ft.com/rss/home",                             "source": "Financial Times",  "category": "finance"},
    {"url": "https://feeds.marketwatch.com/marketwatch/topstories",    "source": "MarketWatch",      "category": "finance"},
    # Technology
    {"url": "https://feeds.arstechnica.com/arstechnica/index",         "source": "Ars Technica",     "category": "technology"},
    {"url": "https://www.wired.com/feed/rss",                          "source": "Wired",            "category": "technology"},
    {"url": "https://techcrunch.com/feed/",                            "source": "TechCrunch",       "category": "technology"},
    {"url": "https://feeds.feedburner.com/TheHackersNews",             "source": "The Hacker News",  "category": "technology"},
    # Business & Macro
    {"url": "https://feeds.reuters.com/reuters/topNews",               "source": "Reuters Top",      "category": "business"},
    {"url": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml","source": "NYT Business",   "category": "business"},
    # Politics
    {"url": "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml","source": "NYT Politics",   "category": "politics"},
    {"url": "https://feeds.bbci.co.uk/news/politics/rss.xml",          "source": "BBC Politics",     "category": "politics"},
    # Geopolitics / World
    {"url": "https://feeds.bbci.co.uk/news/world/rss.xml",             "source": "BBC World",        "category": "geopolitics"},
    {"url": "https://feeds.reuters.com/Reuters/worldNews",             "source": "Reuters World",    "category": "geopolitics"},
    # Science
    {"url": "https://www.nature.com/nature.rss",                       "source": "Nature",           "category": "science"},
    {"url": "https://rss.sciencemag.org/rss/current.xml",              "source": "Science Mag",      "category": "science"},
]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class NewsAgent:
    """
    Agentic loop:
      1. Pull each RSS feed concurrently (asyncio + feedparser)
      2. De-duplicate by URL hash
      3. Optionally fetch full text (first 2 000 chars)
      4. Return list[Article]
    """

    def __init__(
        self,
        feeds: list = FEEDS,
        max_articles: int = 60,
        fetch_full_text: bool = False,
        newsapi_key: Optional[str] = None,
    ):
        self.feeds = feeds
        self.max_articles = max_articles
        self.fetch_full_text = fetch_full_text
        self.newsapi_key = newsapi_key or os.getenv("NEWSAPI_KEY")

    # ── public ──────────────────────────────────────────────────────────────

    async def fetch(self) -> list[Article]:
        """Main entry point. Returns deduplicated articles."""
        tasks = [self._parse_feed(f) for f in self.feeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        seen: set[str] = set()
        articles: list[Article] = []
        batches: list[list[Article]] = [
            batch for batch in results if not isinstance(batch, Exception)
        ]
        idx = 0
        # Round-robin across feeds so one region/source does not monopolize the cap.
        while len(articles) < self.max_articles:
            added = False
            for batch in batches:
                if idx >= len(batch):
                    continue
                art = batch[idx]
                if art.id not in seen:
                    seen.add(art.id)
                    articles.append(art)
                    added = True
                    if len(articles) >= self.max_articles:
                        return articles
            if not added:
                break
            idx += 1

        return articles

    # ── private ─────────────────────────────────────────────────────────────

    _HTTP_HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; NewsWeave/1.0; +https://github.com) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
    }

    async def _parse_feed(self, feed_cfg: dict) -> list[Article]:
        """Fetch feed over HTTP, then parse in a thread (feedparser is sync)."""
        try:
            async with httpx.AsyncClient(
                timeout=25.0,
                headers=self._HTTP_HEADERS,
                follow_redirects=True,
            ) as client:
                r = await client.get(feed_cfg["url"])
                r.raise_for_status()
                content = r.text
        except Exception:
            return []

        loop = asyncio.get_running_loop()
        try:
            parsed = await loop.run_in_executor(
                None, feedparser.parse, content
            )
        except Exception:
            return []

        articles = []
        for entry in parsed.entries[:8]:          # max 8 per feed
            url = getattr(entry, "link", None) or ""
            if not url:
                continue

            summary = (
                getattr(entry, "summary", "")
                or getattr(entry, "description", "")
            )
            # Strip HTML tags naively
            import re
            summary = re.sub(r"<[^>]+>", "", summary)[:800]

            pub = getattr(entry, "published", None) or datetime.utcnow().isoformat()

            art = Article(
                id=Article.make_id(url),
                title=getattr(entry, "title", "Untitled")[:200],
                url=url,
                source=feed_cfg["source"],
                category=feed_cfg["category"],
                published_at=str(pub),
                summary=summary,
            )
            articles.append(art)

        return articles

    async def _fetch_full_text(self, url: str) -> str:
        """Best-effort fetch of article body (first 2 000 chars)."""
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(url, follow_redirects=True)
                import re
                text = re.sub(r"<[^>]+>", " ", r.text)
                text = re.sub(r"\s+", " ", text)
                return text[:2000]
        except Exception:
            return ""
