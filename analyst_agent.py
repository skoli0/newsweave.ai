"""
AnalystAgent - Enriches articles via a single request per batch (fast) or
OpenAI (OPENAI_API_KEY) with gpt-4o-mini, otherwise Ollama with a sub-1B model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Optional

import httpx

from news_agent import Article

log = logging.getLogger("analyst")

# Default local model: sub‑1B, very fast; override with OLLAMA_MODEL
DEFAULT_OLLAMA_MODEL = "qwen2.5:0.5b"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
# Compact batches: many small Ollama models have limited context; OpenAI can take more at once
DEFAULT_CHUNK_OLLAMA = int(os.getenv("ANALYSE_CHUNK_OLLAMA", "20"))
DEFAULT_CHUNK_OPENAI = int(os.getenv("ANALYSE_CHUNK_OPENAI", "20"))

BATCH_INTRO = """You are a news intelligence analyst. For EACH numbered article, extract structured data.
Output ONLY valid minified JSON with this format (no markdown):
{"items":[
  {"i":0,"id":"<article id from input>","entities":[{"name":str,"type":"PERSON|ORG|COUNTRY|PLACE|PRODUCT|EVENT|UNKNOWN","relevance":0.0-1.0}],"themes":[str],
   "sentiment":"positive|neutral|negative","impact_score":0.0-1.0,
   "causal_edges":[{"cause":str,"effect":str,"relation":str}],"one_line_insight":str}
]}
Rules: max 4 entities, max 2 themes, max 1 causal edge per item; "i" is the 0-based index from the list below; "id" must match exactly.
ARTICLES (index | id | title | category | source | summary snippet):
"""

SENTIMENT_KEYWORDS = {
    "positive": ["surge", "rally", "gain", "beat", "record", "approve", "growth"],
    "negative": ["crash", "fall", "drop", "ban", "war", "risk", "loss", "cut", "sanction"],
}

THEME_MAP = {
    "finance": ["markets", "monetary policy", "earnings"],
    "technology": ["AI", "semiconductor", "regulation"],
    "business": ["M&A", "supply chain", "labour"],
    "politics": ["elections", "legislation", "geopolitics"],
    "geopolitics": ["trade war", "sanctions", "conflict"],
    "science": ["research", "climate", "health"],
}

KNOWN_PERSONS = {
    "donald trump",
    "trump",
    "joe biden",
    "biden",
    "vladimir putin",
    "putin",
    "xi jinping",
    "narendra modi",
    "elon musk",
    "jerome powell",
    "tim cook",
    "jensen huang",
}

COUNTRY_ALIASES = {
    "us",
    "u.s.",
    "u.s",
    "usa",
    "u.s.a.",
    "united states",
    "uk",
    "u.k.",
    "united kingdom",
    "china",
    "india",
    "russia",
    "japan",
    "france",
    "germany",
    "canada",
    "australia",
    "israel",
    "iran",
    "ukraine",
    "european union",
    "south korea",
    "north korea",
    "taiwan",
    "uae",
    "united arab emirates",
    "saudi arabia",
    "new zealand",
    "singapore",
    "mexico",
    "brazil",
    "pakistan",
    "bangladesh",
    "nepal",
    "sri lanka",
    "bhutan",
    "myanmar",
    "afghanistan",
    "indonesia",
    "malaysia",
    "philippines",
    "vietnam",
    "thailand",
    "nigeria",
    "south africa",
    "egypt",
    "turkey",
    "spain",
    "italy",
    "netherlands",
}

CITY_ALIASES = {
    "washington", "new york", "london", "paris", "berlin", "tokyo", "beijing",
    "shanghai", "moscow", "kyiv", "jerusalem", "tel aviv", "delhi", "mumbai",
    "bangalore", "sydney", "melbourne", "toronto", "vancouver", "dubai",
    "abu dhabi", "riyadh", "tehran", "istanbul", "seoul", "taipei", "brussels",
    "lahore", "karachi", "islamabad", "dhaka", "kathmandu", "colombo", "yangon",
    "manila", "jakarta", "bangkok", "hanoi", "kuala lumpur", "lagos", "cairo",
    "madrid", "rome", "amsterdam", "barcelona", "milan",
    "kerala", "punjab", "sindh", "balochistan", "khyber pakhtunkhwa",
    "tamil nadu", "karnataka", "maharashtra", "gujarat", "rajasthan",
}

DEMONYMS = {
    "american", "british", "chinese", "indian", "russian", "japanese", "french",
    "german", "canadian", "australian", "israeli", "iranian", "ukrainian",
    "saudi", "emirati", "korean", "taiwanese", "singaporean", "mexican", "brazilian",
}

ORG_HINTS = (
    "inc", "corp", "corporation", "ltd", "llc", "plc", "group", "bank",
    "ministry", "department", "committee", "agency", "administration", "university",
    "institute", "federal reserve", "nato", "opec", "who", "imf", "world bank",
)

KNOWN_ORGS = {
    "apple", "microsoft", "google", "alphabet", "amazon", "meta", "tesla",
    "nvidia", "openai", "anthropic", "deepmind", "blackrock", "goldman sachs",
    "morgan stanley", "jpmorgan", "fed", "federal reserve", "ecb", "imf",
    "world bank", "nato", "opec", "un", "who", "ofgem", "reuters", "bloomberg",
    "bbc", "techcrunch", "financial times", "wsj",
}

VALID_TYPES = {"PERSON", "ORG", "COUNTRY", "PLACE", "PRODUCT", "EVENT", "UNKNOWN"}


def _use_openai() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


class AnalystAgent:
    """
    Batched path: one LLM call per chunk (tens of seconds total vs minutes with per-article).
    Set OPENAI_API_KEY for API route; else Ollama on OLLAMA_HOST with OLLAMA_MODEL.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: str = "http://localhost:11434",
        concurrency: int = 4,
        timeout: int = 30,
    ):
        self.use_openai = _use_openai()
        self.openai_model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip()
        self.ollama_model = (model or os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL).strip()
        if self.use_openai:
            self.model = self.openai_model
        else:
            self.model = self.ollama_model
        self.base_url = (base_url or "http://127.0.0.1:11434").rstrip("/")
        self.openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        self.openai_base = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip(
            "/"
        )
        # Single in-flight Ollama request; OpenAI is one HTTPS call
        self.sem = asyncio.Semaphore(1)
        self.timeout = 90 if self.use_openai else max(120, timeout)
        self._chunk_ollama = max(1, DEFAULT_CHUNK_OLLAMA)
        self._chunk_openai = max(1, DEFAULT_CHUNK_OPENAI)

    @property
    def backend(self) -> str:
        return "openai" if self.use_openai else "ollama"

    @property
    def chunk_size(self) -> int:
        return self._chunk_openai if self.use_openai else self._chunk_ollama

    async def enrich_chunk(self, articles: list[Article]) -> list[Article]:
        """One batched request for this chunk. Mutates article objects in place."""
        if not articles:
            return articles
        if self.use_openai:
            return await self._enrich_openai_chunk(articles)
        return await self._enrich_ollama_chunk(articles)

    async def _enrich_openai_chunk(self, articles: list[Article]) -> list[Article]:
        user_content = BATCH_INTRO + _build_batch_user_content(articles)
        url = f"{self.openai_base}/chat/completions"
        req = {
            "model": self.openai_model,
            "messages": [
                {
                    "role": "user",
                    "content": user_content,
                }
            ],
            "temperature": 0.1,
            "max_tokens": min(8000, 500 + 350 * len(articles)),
            "response_format": {"type": "json_object"},
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.openai_key}",
                        "Content-Type": "application/json",
                    },
                    json=req,
                )
                r.raise_for_status()
                data = r.json()
            text = data["choices"][0]["message"]["content"]
            return self._apply_batch_parsed(articles, self._parse_json_object(text))
        except Exception as e:
            log.warning("OpenAI batch failed: %s - trying local Ollama fallback", e)
            return await self._enrich_ollama_chunk(articles)

    async def _enrich_ollama_chunk(self, articles: list[Article]) -> list[Article]:
        prompt = BATCH_INTRO + _build_batch_user_content(articles)
        payload = {
            "model": self.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": min(4096, 200 + 280 * len(articles)),
            },
        }
        raw: Optional[str] = None
        try:
            async with self.sem:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    r = await client.post(
                        f"{self.base_url}/api/chat", json=payload
                    )
                    r.raise_for_status()
                    j = r.json()
                    raw = (j.get("message") or {}).get("content") or j.get("response", "")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                gpayload = {
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": min(4096, 200 + 280 * len(articles)),
                    },
                }
                try:
                    async with self.sem:
                        async with httpx.AsyncClient(
                            timeout=self.timeout
                        ) as client:
                            r2 = await client.post(
                                f"{self.base_url}/api/generate", json=gpayload
                            )
                            r2.raise_for_status()
                            raw = r2.json().get("response", "")
                except Exception as ge:
                    log.debug("Ollama /api/generate fallback: %s", ge)
            else:
                raise
        except Exception as e:
            log.warning("Ollama batch failed: %s - heuristics for chunk", e)
            for a in articles:
                self._apply_heuristics(a)
            return articles

        if not raw or not str(raw).strip():
            for a in articles:
                self._apply_heuristics(a)
            return articles

        try:
            return self._apply_batch_parsed(articles, self._parse_json_object(str(raw)))
        except Exception as e:
            log.debug("Ollama JSON parse: %s", e)
            for a in articles:
                self._apply_heuristics(a)
            return articles

    def _parse_json_object(self, raw: str) -> dict:
        raw = raw.strip()
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            raise ValueError("No JSON in model output")
        return json.loads(m.group())

    def _apply_batch_parsed(self, articles: list[Article], data: dict) -> list[Article]:
        items = data.get("items")
        if not isinstance(items, list):
            raise ValueError("missing items")
        by_id = {a.id: a for a in articles}
        by_idx = list(articles)
        applied: set[str] = set()
        for it in items:
            if not isinstance(it, dict):
                continue
            a = by_id.get(str(it.get("id") or "").strip())
            if a is None and "i" in it:
                try:
                    idx = int(it["i"])
                    if 0 <= idx < len(by_idx):
                        a = by_idx[idx]
                except (TypeError, ValueError):
                    a = None
            if a is None:
                continue
            self._apply(a, it)
            applied.add(a.id)
        for a in articles:
            if a.id not in applied:
                self._apply_heuristics(a)
        return articles

    def _apply(self, article: Article, data: dict):
        raw_entities = (data.get("entities") or [])[:6]
        norm_entities = []
        for e in raw_entities:
            ne = self._normalize_entity(e)
            if ne:
                norm_entities.append(ne)
        article.entities = norm_entities
        article.themes = (data.get("themes") or [])[:4]
        article.sentiment = (data.get("sentiment") or "neutral").lower()
        if article.sentiment not in ("positive", "neutral", "negative"):
            article.sentiment = "neutral"
        try:
            article.impact_score = float(data.get("impact_score", 0.5))
        except (TypeError, ValueError):
            article.impact_score = 0.5
        article.related_ids = data.get("causal_edges", []) or []
        ins = (data.get("one_line_insight") or "").strip()
        if ins and ins not in (article.summary or ""):
            article.summary = f"{ins} | {article.summary}"

    def _apply_heuristics(self, article: Article):
        text = (article.title + " " + (article.summary or "")).lower()
        pos = sum(1 for w in SENTIMENT_KEYWORDS["positive"] if w in text)
        neg = sum(1 for w in SENTIMENT_KEYWORDS["negative"] if w in text)
        article.sentiment = (
            "positive" if pos > neg else "negative" if neg > pos else "neutral"
        )
        article.themes = THEME_MAP.get(article.category, ["general"])[:2]
        words = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", article.title)
        raw_entities = [
            {"name": w, "type": "ORG", "relevance": 0.5}
            for w in dict.fromkeys(words)
        ][:4]
        norm_entities = []
        for e in raw_entities:
            ne = self._normalize_entity(e)
            if ne:
                norm_entities.append(ne)
        article.entities = norm_entities
        article.impact_score = 0.5
        article.related_ids = []

    def _normalize_entity(self, ent: object) -> Optional[dict]:
        """Normalize noisy LLM entity typing for better graph quality."""
        if not isinstance(ent, dict):
            return None
        name = str(ent.get("name") or "").strip()
        if not name:
            return None
        raw_type = str(ent.get("type") or "UNKNOWN").strip().upper()
        if raw_type in {"GPE", "LOCATION", "LOC"}:
            raw_type = "PLACE"
        if raw_type in {"NORP"}:
            raw_type = "COUNTRY"
        if raw_type in {"CURRENCY", "INDEX", "MONEY"}:
            raw_type = "PRODUCT"
        etype = raw_type if raw_type in VALID_TYPES else "UNKNOWN"
        key = re.sub(r"\s+", " ", name.lower()).strip(" .")

        # Rule 1: explicit well-known people
        if key in KNOWN_PERSONS:
            etype = "PERSON"
        # Rule 1b: explicit known organizations
        elif key in KNOWN_ORGS:
            etype = "ORG"
        # Rule 2: country aliases map to COUNTRY, city aliases map to PLACE
        elif key in COUNTRY_ALIASES:
            etype = "COUNTRY"
        elif key in CITY_ALIASES:
            etype = "PLACE"
        # Rule 2b: demonyms imply country context
        elif key in DEMONYMS:
            etype = "COUNTRY"
        # Rule 2c: geographic cue words
        elif re.search(r"\b(country|nation|state|province|city|island|region)\b", key):
            etype = "COUNTRY" if re.search(r"\b(country|nation)\b", key) else "PLACE"
        # Rule 2d: organization suffixes should stay ORG
        elif any(h in key for h in ORG_HINTS):
            etype = "ORG"
        # Rule 3: two-word human-like names (Title Case) -> PERSON
        elif re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+$", name):
            etype = "PERSON"
        # Rule 4: all-caps short codes like US/UK/EU are places/orgs, never person
        elif re.match(r"^[A-Z]{2,4}$", name):
            if key in {"us", "uk"}:
                etype = "COUNTRY"
            elif key in {"eu", "nato", "opec", "un"}:
                etype = "ORG"
        # Rule 5: fallback for single-token title case labels often used for org brands
        elif etype == "UNKNOWN" and re.match(r"^[A-Z][A-Za-z0-9&.-]{2,}$", name):
            etype = "ORG"

        rel = ent.get("relevance", 0.5)
        try:
            relevance = float(rel)
        except (TypeError, ValueError):
            relevance = 0.5
        return {
            "name": name,
            "type": etype,
            "relevance": max(0.0, min(1.0, relevance)),
        }

    # ── legacy / tests ─────────────────────────────────────────────────────

    async def analyse_batch(self, articles: list[Article]) -> list[Article]:
        if not articles:
            return []
        if self.use_openai:
            step = self._chunk_openai
        else:
            step = self._chunk_ollama
        for i in range(0, len(articles), step):
            ch = articles[i : i + step]
            await self.enrich_chunk(ch)
        return articles

    async def analyse_one(self, article: Article) -> Article:
        await self.enrich_chunk([article])
        return article

    # Old per-article Ollama (unused in main path) removed in favour of batch


def _build_batch_user_content(articles: list[Article]) -> str:
    lines = []
    for i, a in enumerate(articles):
        sm = re.sub(r"\s+", " ", a.summary)[:300]
        safe_title = a.title.replace("|", " ")
        lines.append(
            f"{i} | {a.id} | {safe_title} | {a.category} | {a.source} | {sm}"
        )
    return "\n".join(lines)
