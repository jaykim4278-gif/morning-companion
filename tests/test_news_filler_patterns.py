# W8a 확장 (2026-04-21) — 키워드 기반 filler 필터 테스트.
from __future__ import annotations

from execution.github_actions.news.filler_patterns import is_filler
from execution.github_actions.news.sources import Category


class TestGlobalPatterns:
    def test_podcast_promo_blocked(self) -> None:
        assert is_filler("Listen to our Restaurant Daily podcast", "", Category.RESTAURANT)
        assert is_filler("Get all the headlines in today's Restaurant Daily podcast", "", Category.TOP)

    def test_top_n_list_blocked(self) -> None:
        assert is_filler("Jersey Mike's, Red Lobster, and the Top 500 Restaurant Chains", "", Category.RESTAURANT)
        assert is_filler("Top 100 QSR Ranking 2026", "", Category.MARKETS)

    def test_see_the_list_blocked(self) -> None:
        assert is_filler("Check out the full list of 2026 winners", "", Category.TECH)
        assert is_filler("Read the full episode notes", "", Category.TOP)

    def test_sponsored_blocked(self) -> None:
        assert is_filler("Sponsored: How X company transforms Y", "", Category.TECH)

    def test_substantive_news_passes(self) -> None:
        assert is_filler("Fed raises rates 25bp", "Powell signals May cut", Category.MARKETS) is None
        assert is_filler("Anthropic secures $4B investment from Amazon", "", Category.TECH) is None

    def test_radio_episode_series_blocked(self) -> None:
        # 라디오 단편 시리즈 패턴 차단 ("Episode: ..." / "The Engines of Our Ingenuity ...").
        assert is_filler("Episode: 1567 Christopher Wren, Physician", "", Category.LOCAL)
        assert is_filler("The Engines of Our Ingenuity 1567: Christopher Wren", "", Category.LOCAL)
        assert is_filler("Episode 1234 — A great architect first learns medicine", "", Category.LOCAL)
        # 'Episode' 가 단어 중간에 들어간 경우는 통과 (단순 anchored 패턴).
        assert is_filler("New downtown metro line begins service", "", Category.LOCAL) is None


class TestRestaurantSpecific:
    def test_lto_blocked(self) -> None:
        assert is_filler("McDonald's unveils new menu item", "", Category.RESTAURANT)
        assert is_filler("Taco Bell announces new flavor", "", Category.RESTAURANT)

    def test_industry_trend_passes(self) -> None:
        assert is_filler(
            "Restaurant labor costs hit 5-year high",
            "Industry-wide wage pressure from California AB-1228",
            Category.RESTAURANT,
        ) is None
        assert is_filler(
            "FDA recalls romaine lettuce after E. coli outbreak",
            "",
            Category.RESTAURANT,
        ) is None


class TestLocalSpecific:
    def test_personal_crime_blocked(self) -> None:
        assert is_filler(
            "Local singer pleads not guilty to murder",
            "", Category.LOCAL,
        )
        assert is_filler(
            "Man charged with first-degree murder",
            "", Category.LOCAL,
        )
        assert is_filler(
            "Suspect indicted for armed robbery on Main St",
            "", Category.LOCAL,
        )

    def test_accident_fire_blocked(self) -> None:
        assert is_filler("Major crash on the freeway", "", Category.LOCAL)
        assert is_filler("Apartment fire displaces 20 residents", "", Category.LOCAL)
        assert is_filler("Girl missing since Tuesday found safe", "", Category.LOCAL)

    def test_gossip_blocked(self) -> None:
        assert is_filler(
            "Local singer pleads not guilty to murder in death of 14-year-old",
            "", Category.LOCAL,
        )

    def test_macro_local_news_passes(self) -> None:
        assert is_filler(
            "City Council approves $5B Metro expansion",
            "", Category.LOCAL,
        ) is None
        assert is_filler(
            "Port container traffic hits record",
            "", Category.LOCAL,
        ) is None
        assert is_filler(
            "State Medical Board sanctions three doctors for delayed care",
            "", Category.LOCAL,
        ) is None  # 'sanctions' 는 pattern 에 없음 — macro 규제/판결 판단


class TestCategoryScope:
    def test_local_pattern_not_applied_to_tech(self) -> None:
        # 테크 카테고리의 "crash" 는 S/W 충돌 의미 — 차단 안 해야 함.
        # 현재 _LOCAL_PATTERNS 는 Category.LOCAL 에만 적용되므로 통과.
        assert is_filler("V8 engine crash on null deref", "", Category.TECH) is None

    def test_restaurant_pattern_not_applied_to_markets(self) -> None:
        # MARKETS 카테고리에서는 restaurant_lto 패턴 미적용.
        # "LTO" 약어는 다른 뜻일 수 있음.
        assert is_filler("LTO stock surges on AI news", "", Category.MARKETS) is None


class TestEmptyInput:
    def test_empty_title_and_summary_returns_none(self) -> None:
        assert is_filler("", "", Category.TOP) is None

    def test_whitespace_only_returns_none(self) -> None:
        assert is_filler("   ", "\n\t", Category.TOP) is None
