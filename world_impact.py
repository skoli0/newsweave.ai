"""
Map article text and entities to countries; aggregate impact and sentiment.
Used for the dashboard world view (GET /api/world/impact).
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

# (regex pattern, ISO2) - longer / more specific patterns first
# Patterns use case-insensitive search on a lowercased blob.
_COUNTRY_PATTERNS: list[tuple[re.Pattern, str]] = []
for pat, iso2 in [
    (r"united states|u\.s\.a\.?|u\.s\.?|america(?!n express)|\bus\b(?!-)", "US"),
    (r"united kingdom|u\.k\.?|britain|british|england|scotland|wales|northern ireland|london(?!, ontario)", "GB"),
    (r"european union|\beu parliament\b|brussels(?!,)", "DE"),  # proxy EU institutions → main EU state for centroid
    (r"china|chinese|beijing|shanghai|guangzhou|shenzhen|hong kong|taiwan|taipei", "CN"),
    (r"japan|japanese|tokyo|osaka|nikkei|yen\b", "JP"),
    (r"india|indian|delhi|mumbai|bangalore|bengaluru|rupee(?!, trinidad)", "IN"),
    (r"germany|german|berlin|frankfurt|munich|dax\b", "DE"),
    (r"france|french|paris(?!, texas)|\bifop\b|macron|euronext", "FR"),
    (r"canada|canadian|toronto|vancouver|montreal|loonie|tsx\b", "CA"),
    (r"australia|australian|sydney(?!, ohio)|melbourne(?!, fla)|asx\b", "AU"),
    (r"brazil|brazilian|brasil|são paulo|sao paulo|rio de janeiro|bovespa", "BR"),
    (r"russia|moscow|kremlin|russian(?! d)", "RU"),
    (r"south korea|korean peninsula|seoul(?!, ohio)", "KR"),
    (r"north korea|pyongyang|kim jong|dprk|demilitarized", "KP"),
    (r"mexico|mexican(?!, mo)|\bmxn\b|peso (?:loss|gains?)", "MX"),
    (r"israel(?!i? palestinian conflict)|gaza(?!,)|tel aviv|jerusalem(?!, tennessee)|netanyahu|hamas|hezbollah|west bank|palestin", "IL"),
    (r"saudi|riyadh(?!,)|opec|oil (?:exporters?|producers?)", "SA"),
    (r"iran(?!,)|tehran(?!,)", "IR"),
    (r"ukraine|ukrainian(?!,)|kyiv(?!,)|kiev(?!,)|putin(?!'s dacha)|zapor|crimea(?!,)", "UA"),
    (r"turkey|türkiye|turkish|ankara(?!,)|erdogan|istanbul(?!,)", "TR"),
    (r"spain(?!,)|spanish(?!,)|madrid(?!,)|\bibex\b|ibex", "ES"),
    (r"italy|italian(?!,)|milan(?!, georgia)|rome(?!, new)|\bfiat\b", "IT"),
    (r"argentina(?!,)|argentine(?!,)|merval|buenos aires(?!,)", "AR"),
    (r"indonesia(?!,)|jakarta(?!,)", "ID"),
    (r"poland|warsaw(?!, indiana)", "PL"),
    (r"netherlands(?!,)|dutch(?!,)|\beuronext\b|amsterdam(?!,)", "NL"),
    (r"sweden(?!,)|stockholm(?!,)", "SE"),
    (r"switzerland(?!,)|swiss(?!,)|\bsnb\b|zurich(?!,)", "CH"),
    (r"norway(?!,)|norsk|oslo(?!,)", "NO"),
    (r"denmark(?!,)|copenhagen(?!,)", "DK"),
    (r"finland(?!,)|helsinki(?!,)", "FI"),
    (r"ireland(?!,)|dublin(?!, georgia)", "IE"),
    (r"belgium(?!,)|brussels(?!,)", "BE"),
    (r"austria(?!,)|vienna(?!,)", "AT"),
    (r"greece(?!,)|athens(?!, georgia|alabama|ohio)", "GR"),
    (r"portugal(?!,)|lisbon(?!, iowa|maine)", "PT"),
    (r"egypt(?!,)|cairo(?!, illinois)", "EG"),
    (r"south africa(?!,)|johannesburg(?!,)", "ZA"),
    (r"nigeria(?!,)|lagos(?!, portugal|spain|texas|ohio|california)", "NG"),
    (r"pakistan(?!,)|islamabad(?!,)|karachi(?!,)", "PK"),
    (r"\bbangladesh\b(?!,)|\bdhaka\b(?!,)", "BD"),
    (r"vietnam(?!,)|hanoi(?!,)|hanoi(?!,)|ho chi minh|hanoi(?!,)", "VN"),
    (r"thailand(?!,)|bangkok(?!,)", "TH"),
    (r"philippines(?!,)|manila(?!,)", "PH"),
    (r"malaysia(?!,)|kuala lumpur(?!,)", "MY"),
    (r"singapore(?!,)", "SG"),
    (r"new zealand(?!,)|\bnz\b(?!,)", "NZ"),
    (r"algeria(?!,)|algiers(?!,)", "DZ"),
    (r"chile(?!,)|santiago(?!,)", "CL"),
    (r"colombia(?!,)|bogotá(?!,)|bogota(?!,)", "CO"),
    (r"peru(?!,)|lima(?!,)", "PE"),
    (r"venezuela(?!,)|caracas(?!,)", "VE"),
    (r"israel(?!,)|\bibt\b|shekel", "IL"),
    (r"uae|dubai(?!,)|abu dhabi(?!,)|emirati(?!,)", "AE"),
    (r"qatar(?!,)|doha(?!, georgia)", "QA"),
    (r"kuwait(?!,)", "KW"),
    (r"iraq(?!,)|baghdad(?!, georgia)", "IQ"),
    (r"syria(?!,)|damascus(?!,)", "SY"),
    (r"lebanon(?!,)|beirut(?!,)", "LB"),
    (r"afghanistan(?!,)|kabul(?!,)", "AF"),
    (r"ethiopia(?!,)|addis(?!,)", "ET"),
    (r"kenya(?!,)|nairobi(?!,)", "KE"),
    (r"morocco(?!,)|casablanca(?!,)", "MA"),
    (r"ukraine(?!,)|zelensky|zelenskyy", "UA"),
    (r"armenia(?!,)|azerbaijan(?!,)|baku(?!,)", "AZ"),
    (r"kazakhstan(?!,)|almaty(?!,)", "KZ"),
    (r"uzbekistan(?!,)", "UZ"),
    (r"\bbangladesh\b(?!,)", "BD"),
    (r"cyprus(?!,)", "CY"),
    (r"malta(?!,)", "MT"),
    (r"luxembourg(?!,)", "LU"),
    (r"romania(?!,)|bucharest(?!,)", "RO"),
    (r"bulgaria(?!,)|sofia(?!,| west virginia|kentucky)", "BG"),
    (r"serbia(?!,)|belgrade(?!,)", "RS"),
    (r"croatia(?!,)|zagreb(?!,)", "HR"),
    (r"ukraine(?!,)|\bzapor\b", "UA"),
    (r"cuba(?!,)|havana(?!,| florida| georgia)", "CU"),
    (r"panama(?!,)|\bcanal zone\b(?!,)", "PA"),
    (r"alaska(?!,)", "US"),
    (r"california(?!,)", "US"),
    (r"texas(?!,)", "US"),
    (r"florida(?!,)", "US"),
    (r"silicon valley|wall street(?!,)|\bfed chair\b(?!,)|federal reserve|white house|pentagon|capitol hill|congress(?!,)", "US"),
    (r"downing street|scottish(?!,)|\bbbc\b(?!,)", "GB"),
    (r"euro(?!,)|ecb\b(?!,)|lagarde(?!,)", "DE"),
    (r"nato\b(?!,)", "US"),  # western-led → US centroid for org news
    (r"opec\+|g7\b|g20\b", "US"),
]:
    try:
        _COUNTRY_PATTERNS.append((re.compile(pat, re.IGNORECASE), iso2))
    except re.error:
        continue

# Centroids (lat, lon) for choropleth / circles - one per ISO2
CENTROIDS: dict[str, tuple[float, float]] = {
    "US": (39.8, -98.5), "GB": (54.0, -2.0), "DE": (51.0, 10.0), "FR": (46.2, 2.2),
    "CN": (35.0, 105.0), "JP": (36.0, 138.0), "IN": (22.0, 79.0), "CA": (56.0, -96.0),
    "BR": (-14.0, -51.0), "AU": (-25.0, 134.0), "RU": (60.0, 100.0), "MX": (23.0, -102.0),
    "KR": (36.0, 128.0), "KP": (40.0, 127.0), "IL": (31.0, 35.0), "SA": (24.0, 45.0),
    "IR": (32.0, 53.0), "UA": (49.0, 32.0), "TR": (39.0, 35.0), "ES": (40.0, -3.0),
    "IT": (42.0, 12.5), "AR": (-34.0, -64.0), "ID": (-2.0, 118.0), "PL": (52.0, 20.0),
    "NL": (52.0, 5.0), "SE": (62.0, 15.0), "CH": (47.0, 8.0), "NO": (64.0, 10.0),
    "DK": (56.0, 10.0), "FI": (64.0, 26.0), "IE": (53.0, -8.0), "BE": (50.5, 4.5),
    "AT": (47.5, 14.5), "GR": (39.0, 22.0), "PT": (39.5, -8.0), "EG": (27.0, 30.0),
    "ZA": (-30.0, 25.0), "NG": (10.0, 8.0), "PK": (30.0, 70.0), "BD": (24.0, 90.0),
    "VN": (16.0, 108.0), "TH": (15.0, 100.0), "PH": (12.0, 122.0), "MY": (3.0, 102.0),
    "SG": (1.3, 103.8), "NZ": (-42.0, 174.0), "DZ": (28.0, 3.0), "CL": (-30.0, -71.0),
    "CO": (4.0, -72.0), "PE": (-10.0, -75.0), "VE": (7.0, -66.0), "AE": (24.0, 54.0),
    "QA": (25.3, 51.2), "KW": (29.3, 47.4), "IQ": (33.0, 44.0), "SY": (35.0, 38.0),
    "LB": (33.8, 35.8), "AF": (33.0, 66.0), "ET": (9.0, 40.0), "KE": (0.0, 38.0),
    "MA": (32.0, -6.0), "AZ": (40.0, 47.0), "KZ": (48.0, 66.0), "UZ": (42.0, 64.0),
    "CY": (35.0, 33.0), "MT": (35.9, 14.4), "LU": (49.6, 6.1), "RO": (46.0, 25.0),
    "BG": (42.7, 25.5), "RS": (44.0, 20.5), "HR": (45.0, 16.0), "CU": (22.0, -79.0),
    "PA": (8.5, -80.0), "AL": (41.0, 20.0), }

_COUNTRY_NAMES: dict[str, str] = {
    "US": "United States", "GB": "United Kingdom", "DE": "Germany", "FR": "France",
    "CN": "China", "JP": "Japan", "IN": "India", "CA": "Canada", "BR": "Brazil", "AU": "Australia",
    "RU": "Russia", "MX": "Mexico", "KR": "South Korea", "KP": "North Korea", "IL": "Israel & Palestine",
    "SA": "Saudi Arabia", "IR": "Iran", "UA": "Ukraine", "TR": "Turkey", "ES": "Spain", "IT": "Italy",
    "AR": "Argentina", "ID": "Indonesia", "PL": "Poland", "NL": "Netherlands", "SE": "Sweden", "CH": "Switzerland",
    "NO": "Norway", "DK": "Denmark", "FI": "Finland", "IE": "Ireland", "BE": "Belgium", "AT": "Austria", "GR": "Greece",
    "PT": "Portugal", "EG": "Egypt", "ZA": "South Africa", "NG": "Nigeria", "PK": "Pakistan", "BD": "Bangladesh",
    "VN": "Vietnam", "TH": "Thailand", "PH": "Philippines", "MY": "Malaysia", "SG": "Singapore", "NZ": "New Zealand",
    "DZ": "Algeria", "CL": "Chile", "CO": "Colombia", "PE": "Peru", "VE": "Venezuela", "AE": "United Arab Emirates",
    "QA": "Qatar", "KW": "Kuwait", "IQ": "Iraq", "SY": "Syria", "LB": "Lebanon", "AF": "Afghanistan", "ET": "Ethiopia",
    "KE": "Kenya", "MA": "Morocco", "AZ": "Azerbaijan", "KZ": "Kazakhstan", "UZ": "Uzbekistan", "CY": "Cyprus", "MT": "Malta",
    "LU": "Luxembourg", "RO": "Romania", "BG": "Bulgaria", "RS": "Serbia", "HR": "Croatia", "CU": "Cuba", "PA": "Panama",
    "AL": "Albania",
}

def _text_blob(article: dict) -> str:
    parts = [
        article.get("title") or "",
        article.get("summary") or "",
        article.get("source") or "",
    ]
    ents = article.get("entities")
    if isinstance(ents, list):
        for e in ents:
            if isinstance(e, dict):
                parts.append(e.get("name", "") or "")
    return " ".join(parts)


def detect_countries_for_article(article: dict) -> set[str]:
    """Return ISO2 codes that appear to be related to this story."""
    blob = f" {_text_blob(article).lower()} "
    found: set[str] = set()
    for rx, iso2 in _COUNTRY_PATTERNS:
        if rx.search(blob):
            found.add(iso2)
    return found


def aggregate_world_impact(articles: list[dict]) -> dict[str, Any]:
    """Roll up impact_score and sentiment by country (ISO2)."""
    by_iso: dict[str, dict] = {}
    for a in articles:
        isos = detect_countries_for_article(a)
        if not isos:
            continue
        imp = 0.5
        try:
            imp = float(a.get("impact_score", 0.5) or 0.5)
        except (TypeError, ValueError):
            pass
        sent = (a.get("sentiment") or "neutral").lower()
        if sent not in ("positive", "negative", "neutral"):
            sent = "neutral"
        t = a.get("title", "")[:80]
        for code in isos:
            d = by_iso.setdefault(
                code,
                {
                    "iso2": code,
                    "name": _COUNTRY_NAMES.get(code, code),
                    "impact": 0.0,
                    "count": 0,
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0,
                    "headlines": [],
                },
            )
            d["impact"] += imp
            d["count"] += 1
            d[sent] = d.get(sent, 0) + 1
            if len(d["headlines"]) < 5 and t and t not in d["headlines"]:
                d["headlines"].append(t)

    # Add coordinates for mapping
    points = []
    for code, d in sorted(by_iso.items(), key=lambda x: -x[1]["impact"]):
        ll = CENTROIDS.get(code)
        if not ll:
            continue
        lat, lon = ll
        points.append(
            {
                **d,
                "lat": lat,
                "lon": lon,
            }
        )
    return {
        "countries": points,
        "max_impact": max((d["impact"] for d in points), default=0.0) or 1.0,
    }
