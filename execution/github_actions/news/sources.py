# 뉴스 브리핑 RSS 카탈로그 (data only, 동작 없음).
# 카탈로그 변경 = contract test 갱신만 필요, 로직 코드 변경 없음.
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class Category(str, Enum):
    """카테고리 — 메시지 섹션 순서·제목 매핑에 사용.

    StrEnum 이 3.11+ 지만 문자열 호환 안정성 위해 (str, Enum) 다중 상속 사용 (Py3.12 호환).
    """
    INDEX = "index"              # 📈 시장 지수 (RSS 아님, market_indices.py 가 처리)
    WEATHER = "weather"          # ⛅ 오늘 날씨 (RSS 아님, weather.py 가 처리)
    TOP = "top"                  # 🔝 주요 뉴스
    MARKETS = "markets"          # 💼 시장 뉴스
    TECH = "tech"                # 🤖 AI 뉴스
    RESTAURANT = "restaurant"    # 🍽 요식업
    LOCAL = "local"              # 🏙 사용자 거주 지역


@dataclass(frozen=True)
class RssSource:
    slug: str                    # DB source 컬럼·dedup 키 (stable)
    display_name: str            # 메시지 prefix '[Reuters]' 형태
    url: str                     # RSS endpoint
    category: Category
    quality_tier: int            # 1~5 — source_quality.TIER_SCORES 참조 (T1=최객관)
    max_items_per_fetch: int = 5 # feedparser 최근 N 항목만 스캔 (후보 선발)
    requires_points: int = 0     # HN 전용 — 이 점수 이상만 수집


# RSS 소스 — 헤드라인급 거시 뉴스 위주 (개별 업체 micro 제외).
# LOCAL 카테고리 소스는 배포자별 env(USER_LOCAL_RSS)로 런타임 주입 (코드에 지역 식별자 미포함).
CATALOG: tuple[RssSource, ...] = (
    # 🔝 TOP — 미국·국제 헤드라인 (macro · 정책 · 거시)
    RssSource("npr_news", "[NPR]",
              "https://feeds.npr.org/1001/rss.xml",
              Category.TOP, quality_tier=1),
    RssSource("bbc_us", "[BBC]",
              "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml",
              Category.TOP, quality_tier=1),
    # 💼 MARKETS — 시장 전문
    RssSource("cnbc", "[CNBC]",
              "https://www.cnbc.com/id/100003114/device/rss/rss.html",
              Category.MARKETS, quality_tier=3),
    RssSource("wsj_markets", "[WSJ]",
              "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
              Category.MARKETS, quality_tier=2),
    RssSource("marketwatch", "[MarketWatch]",
              "https://feeds.content.dowjones.io/public/rss/mw_topstories",
              Category.MARKETS, quality_tier=3),
    # 🤖 TECH — AI 전문 RSS.
    # HN 200pt+ 는 AI 논문·model release frontpage 도 포함 (cluster_select hint 가 비-AI 필터).
    RssSource("hn", "[HN]",
              "https://hnrss.org/frontpage?points=200",
              Category.TECH, quality_tier=4, requires_points=200),
    RssSource("techcrunch_ai", "[TechCrunch]",
              "https://techcrunch.com/category/artificial-intelligence/feed/",
              Category.TECH, quality_tier=3),
    RssSource("verge_ai", "[The Verge]",
              "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
              Category.TECH, quality_tier=2),
    RssSource("wired_ai", "[Wired]",
              "https://www.wired.com/feed/tag/ai/latest/rss",
              Category.TECH, quality_tier=2),
    RssSource("venturebeat_ai", "[VentureBeat]",
              "https://venturebeat.com/category/ai/feed",
              Category.TECH, quality_tier=3),
    # 🍽 RESTAURANT — 요식업 트렌드·정책·공급망
    RssSource("nrn", "[NRN]",
              "https://www.nrn.com/rss.xml",
              Category.RESTAURANT, quality_tier=3),
    RssSource("restaurant_business", "[RB]",
              "https://www.restaurantbusinessonline.com/rss.xml",
              Category.RESTAURANT, quality_tier=3),
    # 🏙 LOCAL 소스는 코드에 두지 않고 by_category 가 런타임 env(USER_LOCAL_RSS)로 주입.
)

# 카테고리별 최종 발송 건수 상한 — voucher 용으로 fetcher 는 +1 건 더 수집.
# INDEX 는 별도 경로(yfinance)라 본 dict 에 포함 안 됨.
CATEGORY_MAX_ITEMS: dict[Category, int] = {
    Category.TOP: 2,
    Category.MARKETS: 2,
    Category.TECH: 2,
    Category.RESTAURANT: 2,
    Category.LOCAL: 2,
}

# 메시지 상단부터의 출력 순서 (날씨·지수 먼저, 지역 마지막).
CATEGORY_ORDER: tuple[Category, ...] = (
    Category.INDEX,
    Category.WEATHER,
    Category.TOP,
    Category.MARKETS,
    Category.TECH,
    Category.RESTAURANT,
    Category.LOCAL,
)

CATEGORY_HEADERS: dict[Category, str] = {
    Category.INDEX: "📈 시장 지수",
    Category.WEATHER: "⛅ 오늘 날씨",
    Category.TOP: "🔝 미국·국제 헤드라인",
    Category.MARKETS: "💼 시장",
    Category.TECH: "🤖 AI 뉴스",
    Category.RESTAURANT: "🍽 요식업 트렌드",
    Category.LOCAL: "🏙 지역 주요",
}


def _env_local_sources() -> tuple[RssSource, ...]:
    """USER_LOCAL_RSS env ('label|url') → LOCAL 소스 1건.  미설정 시 () (지역 섹션 skip).

    코드에 특정 지역 식별자(도시·매체 URL)를 두지 않기 위해 런타임 env 로 주입한다.
    """
    raw = os.environ.get("USER_LOCAL_RSS", "").strip()
    if "|" not in raw:
        return ()
    label, _, url = raw.partition("|")
    label, url = label.strip(), url.strip()
    if not url:
        return ()
    return (RssSource("local_media", label or "[Local]", url, Category.LOCAL, quality_tier=2),)


def by_slug(slug: str) -> RssSource | None:
    """CATALOG(+env LOCAL) 내 slug 조회.  없으면 None (호출자가 로깅·skip 판단)."""
    for s in CATALOG:
        if s.slug == slug:
            return s
    for s in _env_local_sources():
        if s.slug == slug:
            return s
    return None


def by_category(category: Category) -> tuple[RssSource, ...]:
    """해당 카테고리 소스만 반환.  LOCAL 은 env 주입, INDEX 는 비어있음 (RSS 아님)."""
    base = tuple(s for s in CATALOG if s.category == category)
    if category == Category.LOCAL:
        return base + _env_local_sources()
    return base


__all__ = [
    "Category",
    "RssSource",
    "CATALOG",
    "CATEGORY_MAX_ITEMS",
    "CATEGORY_ORDER",
    "CATEGORY_HEADERS",
    "by_slug",
    "by_category",
]
