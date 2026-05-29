# W8a PR #5 — MarkdownV2 escape + 메시지 조립 + 4000자 cap 테스트.
# docs/design/w8a-news-briefing-design.md §7 · §10.1.
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from execution.github_actions.news.market_indices import IndexQuote
from execution.github_actions.news.message_builder import (
    build_message,
    escape_mdv2,
    escape_url,
    link,
)
from execution.github_actions.news.rss_fetcher import RssItem
from execution.github_actions.news.sources import Category


class TestEscape:
    def test_escapes_all_mdv2_special(self) -> None:
        # _*[]()~`>#+-=|{}.! 전부 \ 로 escape.
        raw = "a_b*c[d](e)f~g`h>i#j+k-l=m|n{o}p.q!"
        result = escape_mdv2(raw)
        # 총 18개 특수문자 (_ * [ ] ( ) ~ ` > # + - = | { } . !).
        assert result.count("\\") == 18

    def test_plain_text_unchanged(self) -> None:
        assert escape_mdv2("hello world") == "hello world"

    def test_url_escape_only_closing_paren_and_backslash(self) -> None:
        # URL 내부는 ) 와 \ 만.  점·괄호·등등은 그대로.
        assert escape_url("https://example.com/a?b=1") == "https://example.com/a?b=1"
        assert escape_url("https://example.com/a(b)") == "https://example.com/a(b\\)"

    def test_link_composes_correctly(self) -> None:
        result = link("연준, 금리 인하", "https://reuters.com/article")
        # 링크 텍스트는 mdv2, URL 은 url escape.
        assert result.startswith("[연준, 금리 인하](")
        assert result.endswith(")")


class TestBuildMessage:
    def _item(self, src: str, title_ko: str, url: str) -> RssItem:
        return RssItem(
            source=src, url=url, url_hash=url,
            title_en="en", pub_date=datetime(2026, 4, 20, tzinfo=timezone.utc),
            category=Category.TOP, title_ko=title_ko,
        )

    def test_empty_selected_still_renders_header(self) -> None:
        msg = build_message(indices=[], selected={}, now=datetime(2026, 4, 20, 11, 1, tzinfo=timezone.utc))
        assert "미국 뉴스 브리핑" in msg
        # 헤더·footer 만.  에러 없이 렌더.

    def test_indices_section_rendered(self) -> None:
        q = IndexQuote(
            symbol="^GSPC", display_name="S&P 500",
            last_close=5123.45, day_change_pct=0.85, ytd_change_pct=8.2,
            unit="", as_of=date(2026, 4, 19),
        )
        msg = build_message(indices=[q], selected={}, now=datetime(2026, 4, 20, tzinfo=timezone.utc))
        assert "S&P 500" in msg
        assert "5,123" in msg  # 종가 (escape 됨)
        assert "▲" in msg      # 상승 화살표

    def test_indices_failed_fallback_message(self) -> None:
        msg = build_message(indices=[], selected={}, indices_failed=True,
                             now=datetime(2026, 4, 20, tzinfo=timezone.utc))
        assert "지수 데이터 일시 불가" in msg

    def test_category_items_with_link(self) -> None:
        # 2026-05-13 redesign: title 은 bold (*..*), URL 은 "자세히 →" 별도 링크.
        items = [self._item("npr_news", "연준 금리 인하", "https://reuters.com/a")]
        msg = build_message(indices=[], selected={Category.TOP: items},
                             now=datetime(2026, 4, 20, tzinfo=timezone.utc))
        assert "미국·국제 헤드라인" in msg
        # 새 디자인: title 은 bold 마커 `*..*` 으로 감쌈 (link 아님 → 파란색 X).
        assert "*연준 금리 인하*" in msg
        # URL 은 "자세히 →" 단어로 별도 노출.
        assert "자세히" in msg
        assert "https://reuters.com/a" in msg

    def test_title_is_bold_not_link(self) -> None:
        # 2026-05-13 redesign: title 은 link 아님 (가독성).  URL 은 "자세히 →" 단어만 link.
        items = [self._item("npr_news", "연준 금리 인하", "https://reuters.com/a")]
        msg = build_message(indices=[], selected={Category.TOP: items},
                             now=datetime(2026, 4, 20, tzinfo=timezone.utc))
        # 옛 형식 `[title](url)` 은 없어야 함 (link 가 아니므로).
        assert "[연준 금리 인하]" not in msg
        # 신규 link 는 "자세히 →" 텍스트만.
        assert "[자세히 →](" in msg or "[자세히 →\\](" in msg or "자세히 \\→" in msg

    def test_display_name_prefix(self) -> None:
        # 매체명은 italic + 📰 이모지 메타 라인에 노출.  대괄호 떼고 표시.
        items = [self._item("npr_news", "x", "https://r.com/1")]
        msg = build_message(indices=[], selected={Category.TOP: items},
                             now=datetime(2026, 4, 20, tzinfo=timezone.utc))
        assert "NPR" in msg
        assert "📰" in msg
        # 옛 형식의 대괄호 prefix `[NPR]` 은 더 이상 없음 (italic 형태로 변경).
        assert "\\[NPR\\]" not in msg

    def test_category_order_respected(self) -> None:
        # TOP 먼저, LOCAL 나중.
        items_top = [self._item("npr_news", "top1", "https://r.com/a")]
        items_local = [self._item("local_public_media", "local1", "https://k.com/a")]
        msg = build_message(
            indices=[], selected={Category.LOCAL: items_local, Category.TOP: items_top},
            now=datetime(2026, 4, 20, tzinfo=timezone.utc),
        )
        top_pos = msg.find("미국·국제 헤드라인")
        local_pos = msg.find("지역 주요")
        assert 0 < top_pos < local_pos

    def test_empty_category_section_hidden(self) -> None:
        # 빈 리스트인 카테고리는 헤더도 안 보여야 함.
        msg = build_message(
            indices=[], selected={Category.TOP: []},
            now=datetime(2026, 4, 20, tzinfo=timezone.utc),
        )
        assert "미국·국제 헤드라인" not in msg

    def test_message_under_4096_even_with_full_catalog(self) -> None:
        # 긴 제목 많이 넣어서 길이 초과시 drop 작동.
        long_items = [
            self._item(f"npr_news", "매우매우 긴 제목을 가진 뉴스 " * 10, f"https://r.com/{i}")
            for i in range(20)
        ]
        selected = {
            Category.TOP: long_items[:4],
            Category.MARKETS: long_items[4:8],
            Category.TECH: long_items[8:12],
            Category.RESTAURANT: long_items[12:16],
            Category.LOCAL: long_items[16:20],
        }
        msg = build_message(
            indices=[], selected=selected,
            now=datetime(2026, 4, 20, tzinfo=timezone.utc),
        )
        assert len(msg) <= 4096

    def test_weekday_korean_label(self) -> None:
        # 2026-04-20 = 월요일.  괄호는 MDv2 escape 되어 \( \) 로 렌더.
        msg = build_message(indices=[], selected={},
                             now=datetime(2026, 4, 20, tzinfo=timezone.utc))
        assert "월" in msg
        assert "4월 20일" in msg

    def _item_with_summary(
        self, src: str, title_ko: str, url: str,
        summary_ko: str | None = None, summary_en: str = "",
    ) -> RssItem:
        return RssItem(
            source=src, url=url, url_hash=url,
            title_en="en", pub_date=datetime(2026, 4, 20, tzinfo=timezone.utc),
            category=Category.TOP, title_ko=title_ko,
            summary_en=summary_en, summary_ko=summary_ko,
        )

    def test_summary_renders_as_blockquote_italic(self) -> None:
        # hotfix 2026-04-21: 제목 아래 2줄 요약 blockquote italic 으로 렌더.
        items = [self._item_with_summary(
            "npr_news", "연준 금리 인하",
            "https://r.com/a",
            summary_ko="파월 의장이 5월 FOMC 에서 인하 여지를 열어뒀다.",
        )]
        msg = build_message(indices=[], selected={Category.TOP: items},
                             now=datetime(2026, 4, 20, tzinfo=timezone.utc))
        # blockquote (>) + italic (_..._) wrapper 존재.
        assert ">_" in msg
        assert "파월 의장" in msg

    def test_summary_english_blocked_when_no_korean(self) -> None:
        # 2026-05-03 정책 전환: summary_ko 없으면 빈 라인.  영문 summary_en 폴백 차단.
        # 사용자 강력 요구 — "3개 AI 연결했으니 무조건 번역, 안 되면 차라리 안 나오게."
        items = [self._item_with_summary(
            "npr_news", "Fed 금리 인하 시그널",
            "https://r.com/a",
            summary_en="Powell hints at a rate cut at the May FOMC meeting.",
        )]
        msg = build_message(indices=[], selected={Category.TOP: items},
                             now=datetime(2026, 4, 20, tzinfo=timezone.utc))
        # 영문 요약은 절대 메시지에 포함되면 안 됨.
        assert "Powell hints" not in msg
        # 한국어 제목은 정상 렌더.
        assert "Fed 금리 인하 시그널" in msg

    def test_no_summary_renders_single_line(self) -> None:
        # summary 전무 → blockquote 생략.
        items = [self._item_with_summary("npr_news", "제목만", "https://r.com/a")]
        msg = build_message(indices=[], selected={Category.TOP: items},
                             now=datetime(2026, 4, 20, tzinfo=timezone.utc))
        assert "제목만" in msg
        # blockquote marker 가 이 기사 섹션에는 없어야 함.
        # (footer 에도 > 는 없어야 — 현재 footer 에는 없음)
        lines = msg.split("\n")
        article_line_idx = next(i for i, line in enumerate(lines) if "제목만" in line)
        if article_line_idx + 1 < len(lines):
            next_line = lines[article_line_idx + 1]
            assert not next_line.startswith(">")

    def test_footer_clarifies_translation_fallback(self) -> None:
        # 2026-05-04: footer 카피 — "LLM 실패 시 사전 폴백 — 영문 미노출" 명시.
        msg = build_message(indices=[], selected={},
                             now=datetime(2026, 4, 20, 11, 1, tzinfo=timezone.utc))
        assert "AI 번역" in msg
        assert "사전 폴백" in msg
        assert "영문 미노출" in msg

    def test_url_utm_params_stripped(self) -> None:
        # 2026-05-14 hotfix — 5/14 06:08 CT cron 실패 (Telegram 400 "byte offset N Italic")
        # 의 root cause 의심: URL 안 utm_* 의 raw `_` 가 MDv2 italic 파서 트리거.
        # normalize_url 적용으로 utm_*·fbclid·gclid·_ga 등 제거 검증.
        url_with_tracking = (
            "https://example-local-media.org/articles/news/2026/05/13/"
            "551633/stolen-wage-complaints/"
            "?utm_source=rss-article&utm_medium=link&utm_campaign=local-rss-link"
            "&fbclid=ABC123&gclid=XYZ789"
        )
        items = [self._item("local_public_media", "임금 갈취 신고", url_with_tracking)]
        msg = build_message(indices=[], selected={Category.LOCAL: items},
                             now=datetime(2026, 4, 20, tzinfo=timezone.utc))
        # 트래킹 파라미터가 메시지 본문에서 사라져야 함.
        assert "utm_source" not in msg
        assert "utm_medium" not in msg
        assert "utm_campaign" not in msg
        assert "fbclid" not in msg
        assert "gclid" not in msg
        # 원본 URL 의 path 는 유지되어야 함 (링크 작동 보장).
        assert "stolen-wage-complaints" in msg
        assert "example-local-media.org" in msg

    def test_weather_advisory_with_newline_flattened(self) -> None:
        # 2026-05-14 hotfix — root cause: LLM weather_advisory 가 \\n 포함 시
        # `>_line1\\nline2_` 가 되면서 line 2 가 blockquote 밖이 되어 italic 만 살아남고
        # close 못 찾아 Telegram 400 "Can't find end of Italic entity at byte offset N".
        from execution.github_actions.news.weather import WeatherSnapshot
        weather = WeatherSnapshot(
            temp_c=21.1, temp_f=70.0, weather_label="맑음",
            today_high_c=31.1, today_low_c=21.1,
            today_high_f=88.0, today_low_f=70.0,
            precip_prob_pct=1,
        )
        # LLM 이 multi-line advisory 출력한 실제 케이스 재현.
        advisory_multiline = (
            "안녕하세요,\n오늘 날씨는 70°F로 맑겠으며, "
            "최고 88°F, 최저 70°F를 보이겠습니다. "
            "공기질 경보가 발령되었으니 참고하시기 바랍니다."
        )
        msg = build_message(
            indices=[], selected={},
            now=datetime(2026, 5, 14, 11, 1, tzinfo=timezone.utc),
            weather=weather,
            weather_advisory=advisory_multiline,
        )
        # advisory blockquote-italic 라인은 single-line 이어야 함.
        bq_lines = [ln for ln in msg.split("\n") if ln.startswith(">_")]
        assert len(bq_lines) >= 1
        for bq_line in bq_lines:
            # 평탄화 후 `\\n` 이 라인 내부에 없어야 함 (line.split("\\n") 결과 1줄).
            assert bq_line.count("\n") == 0
            # 라인이 `_` 로 끝나면서 italic close 되어 있어야 함.
            assert bq_line.endswith("_")

    def test_summary_with_newline_flattened(self) -> None:
        # 2026-05-14 hotfix — 5/14 06:08 CT cron 실패 root cause.
        # LLM 출력에 \\n 포함 시 `>_line1\\nline2_` 구조가 되어 line 2 가 blockquote
        # 밖이 되면서 italic 만 살아남아 Telegram 파서 실패 (byte offset N Italic).
        # 평탄화로 single-line 보장.
        items = [self._item_with_summary(
            "npr_news", "테스트 제목",
            "https://r.com/a",
            summary_ko="첫 문장입니다.\n두 번째 문장도 있습니다.\n세 번째 문장.",
        )]
        msg = build_message(indices=[], selected={Category.TOP: items},
                             now=datetime(2026, 4, 20, tzinfo=timezone.utc))
        # 평탄화 후 blockquote-italic 한 줄에 들어가야 함.
        assert ">_첫 문장입니다\\. 두 번째 문장도 있습니다\\. 세 번째 문장\\._" in msg
        # 원본의 raw \\n 이 본문에 그대로 들어가면 안 됨 (이게 root cause).
        # 평탄화 후 blockquote 라인에는 \\n 이 없어야 함.
        bq_line = next(ln for ln in msg.split("\n") if ln.startswith(">_"))
        assert "\n" not in bq_line  # blockquote 라인 자체에 줄바꿈 없음

    def test_url_underscores_count_balanced(self) -> None:
        # MDv2 italic `_..._` 짝맞춤 검증 — utm_* 제거 후 URL 에 `_` 가 남지 않아
        # message 전체 unescaped `_` 개수가 짝수여야 함 (italic 짝맞춤).
        items = [
            self._item("local_public_media", "테스트 제목",
                       "https://example.com/article/?utm_source=foo&utm_medium=bar"),
            self._item("npr_news", "두번째 제목",
                       "https://example.com/?utm_campaign=baz"),
        ]
        msg = build_message(indices=[], selected={Category.TOP: items},
                             now=datetime(2026, 4, 20, tzinfo=timezone.utc))
        # un-escaped (preceding char != '\\') `_` 카운트.
        unesc_count = sum(
            1 for i, c in enumerate(msg)
            if c == "_" and (i == 0 or msg[i - 1] != "\\")
        )
        assert unesc_count % 2 == 0, (
            f"Unbalanced italic `_` markers ({unesc_count}) — would break Telegram MDv2 parser.\n"
            f"Message:\n{msg}"
        )
