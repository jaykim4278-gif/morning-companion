# W8a PR #4 — LLM 클러스터링 + best 선택 + 결정론 폴백.
# docs/design/w8a-news-briefing-design.md §5 · §10.1.
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest

from execution.ai_client.fallback_chain import ChainConfig, FallbackError
from execution.github_actions.news.cluster_select import (
    Cluster,
    ClusterParseError,
    _build_category_hint,
    _parse_cluster_response,
    cluster_and_select,
    cluster_deterministic,
    cluster_llm,
    select_best_per_cluster,
)
from execution.github_actions.news.rss_fetcher import RssItem
from execution.github_actions.news.sources import Category


@dataclass
class FakeProvider:
    name: str
    responses: list[Any] = field(default_factory=list)
    call_count: int = 0

    def generate(self, payload: dict, prompt: str) -> str:
        idx = self.call_count
        self.call_count += 1
        if idx >= len(self.responses):
            return "[]"
        item = self.responses[idx]
        if isinstance(item, BaseException):
            raise item
        return item


FAST = ChainConfig(max_retries_per_provider=0, base_backoff_seconds=0.0)


def _item(src: str, title: str, url: str = None) -> RssItem:
    url = url or f"https://{src}.com/{title.replace(' ', '-')}"
    return RssItem(
        source=src, url=url, url_hash=url,
        title_en=title,
        pub_date=datetime(2026, 4, 20, tzinfo=timezone.utc),
        category=Category.TOP,
    )


# ============================================================
class TestParseClusterResponse:
    def test_valid_single_cluster(self) -> None:
        raw = json.dumps([{"topic": "Fed", "items": [0, 1], "best": 0}])
        clusters = _parse_cluster_response(raw, n_items=2)
        assert len(clusters) == 1
        assert clusters[0].topic == "Fed"
        assert clusters[0].items == (0, 1)
        assert clusters[0].best_idx == 0

    def test_markdown_fence_stripped(self) -> None:
        raw = "```json\n" + json.dumps([{"topic": "x", "items": [0], "best": 0}]) + "\n```"
        clusters = _parse_cluster_response(raw, n_items=1)
        assert clusters[0].topic == "x"

    def test_missing_idx_raises(self) -> None:
        # 2건 candidates 지만 클러스터가 idx=0 만 다룸 → 1 누락.
        raw = json.dumps([{"topic": "x", "items": [0], "best": 0}])
        with pytest.raises(ClusterParseError, match="누락"):
            _parse_cluster_response(raw, n_items=2)

    def test_duplicate_idx_raises(self) -> None:
        raw = json.dumps([
            {"topic": "a", "items": [0], "best": 0},
            {"topic": "b", "items": [0, 1], "best": 0},
        ])
        with pytest.raises(ClusterParseError, match="중복"):
            _parse_cluster_response(raw, n_items=2)

    def test_best_not_in_items_raises(self) -> None:
        raw = json.dumps([{"topic": "x", "items": [0, 1], "best": 5}])
        with pytest.raises(ClusterParseError, match="best"):
            _parse_cluster_response(raw, n_items=2)

    def test_out_of_range_idx_raises(self) -> None:
        raw = json.dumps([{"topic": "x", "items": [0, 5], "best": 0}])
        with pytest.raises(ClusterParseError, match="out-of-range"):
            _parse_cluster_response(raw, n_items=2)

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ClusterParseError, match="JSON"):
            _parse_cluster_response("not-json", n_items=1)


# ============================================================
class TestClusterDeterministic:
    def test_each_item_becomes_solo_cluster(self) -> None:
        items = [_item("reuters", "a"), _item("ap", "b"), _item("cnbc", "c")]
        clusters = cluster_deterministic(items)
        assert len(clusters) == 3
        for i, c in enumerate(clusters):
            assert c.items == (i,)
            assert c.best_idx == i


class TestSelectBestPerCluster:
    def test_returns_best_per_cluster(self) -> None:
        items = [_item("reuters", "a"), _item("ap", "b"), _item("cnbc", "c")]
        clusters = [
            Cluster(topic="x", items=(0, 1), best_idx=1),  # AP 가 best
            Cluster(topic="y", items=(2,), best_idx=2),    # CNBC 단독
        ]
        bests = select_best_per_cluster(items, clusters)
        assert [b.source for b in bests] == ["ap", "cnbc"]


# ============================================================
class TestClusterAndSelect:
    def test_llm_success_returns_bests(self) -> None:
        items = [
            _item("reuters", "Fed rate cut"),
            _item("ap", "Fed hints June cut"),
            _item("cnbc", "Tech rally"),
        ]
        # LLM: reuters+ap 는 같은 Fed 사건, reuters=best.  cnbc 는 독립.
        raw = json.dumps([
            {"topic": "Fed", "items": [0, 1], "best": 0},
            {"topic": "Tech", "items": [2], "best": 2},
        ])
        provider = FakeProvider("gemini", [raw])
        bests, topics = cluster_and_select(items, [provider], config=FAST)
        assert [b.source for b in bests] == ["reuters", "cnbc"]
        assert topics[items[0].url] == "Fed"
        assert topics[items[2].url] == "Tech"

    def test_llm_failure_falls_back_to_deterministic(self) -> None:
        items = [_item("reuters", "a"), _item("ap", "b")]
        # LLM 완전 실패 → 결정론 폴백.
        provider = FakeProvider("gemini", [FallbackError("down")])
        bests, topics = cluster_and_select(items, [provider], config=FAST)
        # deterministic = 모든 아이템 solo → 2건 모두 best.
        assert len(bests) == 2

    def test_llm_bad_json_falls_back(self) -> None:
        items = [_item("reuters", "a")]
        provider = FakeProvider("gemini", ["not valid json"])
        bests, _ = cluster_and_select(items, [provider], config=FAST)
        assert len(bests) == 1    # 폴백에서 단독 클러스터

    def test_empty_candidates_returns_empty(self) -> None:
        bests, topics = cluster_and_select([], [FakeProvider("g")], config=FAST)
        assert bests == []
        assert topics == {}


# ============================================================
class TestClusterLlmContract:
    def test_invariant_all_indices_covered(self) -> None:
        items = [_item("npr_news", "a"), _item("bbc_us", "b")]
        raw = json.dumps([
            {"topic": "x", "items": [0], "best": 0},
            {"topic": "y", "items": [1], "best": 1},
        ])
        clusters = cluster_llm(items, [FakeProvider("gemini", [raw])], config=FAST)
        covered = {i for c in clusters for i in c.items}
        assert covered == {0, 1}


# ============================================================
# 2026-04-21 hotfix — filler exclude 필터.
# ============================================================
class TestClusterExclude:
    def test_exclude_true_cluster_dropped_from_bests(self) -> None:
        # 2 items, 두 번째는 팟캐스트 홍보 → exclude=true → bests 에 1개만.
        items = [_item("nrn", "Fed raises rates"), _item("nrn", "Listen to our podcast")]
        raw = json.dumps([
            {"topic": "fed_rates", "items": [0], "best": 0, "exclude": False},
            {"topic": "filler_podcast", "items": [1], "best": 1, "exclude": True},
        ])
        bests, topic_map = cluster_and_select(
            items, [FakeProvider("gemini", [raw])], config=FAST,
        )
        assert len(bests) == 1
        assert bests[0].title_en == "Fed raises rates"
        # excluded item 은 topic_map 에도 없어야 news_sent_log 에 기록 안 됨.
        assert items[1].url not in topic_map

    def test_exclude_missing_defaults_to_false(self) -> None:
        # 기존 응답 포맷 (exclude 필드 없음) 하위 호환.
        items = [_item("npr_news", "a")]
        raw = json.dumps([{"topic": "x", "items": [0], "best": 0}])
        bests, _ = cluster_and_select(
            items, [FakeProvider("gemini", [raw])], config=FAST,
        )
        assert len(bests) == 1

    def test_all_excluded_returns_empty(self) -> None:
        # 모든 후보가 filler — 빈 결과 (카테고리 숨김).
        items = [_item("nrn", "Top 500"), _item("nrn", "podcast")]
        raw = json.dumps([
            {"topic": "rankings", "items": [0], "best": 0, "exclude": True},
            {"topic": "filler_podcast", "items": [1], "best": 1, "exclude": True},
        ])
        bests, topic_map = cluster_and_select(
            items, [FakeProvider("gemini", [raw])], config=FAST,
        )
        assert bests == []
        assert topic_map == {}

    def test_invalid_exclude_value_treated_as_false(self) -> None:
        # exclude 가 boolean 아닌 경우 (LLM 오작동) 안전 쪽으로 false 간주.
        items = [_item("npr_news", "a")]
        raw = json.dumps([{"topic": "x", "items": [0], "best": 0, "exclude": "yes"}])
        bests, _ = cluster_and_select(
            items, [FakeProvider("gemini", [raw])], config=FAST,
        )
        assert len(bests) == 1  # 문자열 "yes" 는 bool 검증 실패 → 포함


# ============================================================
# PR #89 — 사용자 관심 키워드 hint 주입 검증.
# ============================================================
class TestUserInterestsHint:
    def test_empty_interests_returns_base_hint_only(self) -> None:
        hint = _build_category_hint(Category.MARKETS, user_interests=())
        # 키워드 섹션 헤더 없음.
        assert "사용자 관심 키워드" not in hint
        # 카테고리 본문은 그대로.
        assert "FOMC" in hint or "M&A" in hint

    def test_tech_hint_is_ai_focused(self) -> None:
        # 2026-05-13 사용자 요청 — TECH 카테고리는 AI 중심.
        hint = _build_category_hint(Category.TECH, user_interests=())
        # AI 관련 핵심 키워드가 hint 에 명시되어야 함.
        assert "AI" in hint
        # frontier labs 명시 — AI 모델 발표 우선
        assert any(lab in hint for lab in ("OpenAI", "Anthropic", "Google"))
        # AI 반도체 명시
        assert "NVIDIA" in hint or "AI 반도체" in hint
        # AI 규제 명시
        assert "EU AI Act" in hint or "AI 규제" in hint

    def test_tech_hint_excludes_non_ai_strictly(self) -> None:
        # 2026-05-13 hotfix — 첫 dispatch 에서 정치(Trump drug policy) + 금융(Fed chair) 가 TECH 에 흘러옴.
        # hint 가 정치·금융·헬스케어 무조건 exclude 명시해야 LLM 이 거르도록.
        hint = _build_category_hint(Category.TECH, user_interests=())
        assert "정치" in hint
        assert "금융" in hint
        assert "헬스케어" in hint
        # AI 후보 0건일 때 카테고리 빈 출력 허용 명시.
        assert "0건" in hint or "빈 출력" in hint

    def test_interests_appended_to_hint(self) -> None:
        hint = _build_category_hint(
            Category.MARKETS,
            user_interests=("AlphaTopic", "BetaTopic", "GammaTopic"),
        )
        assert "사용자 관심 키워드" in hint
        assert "AlphaTopic" in hint
        assert "BetaTopic" in hint
        assert "GammaTopic" in hint

    def test_interests_passed_to_cluster_llm_prompt(self) -> None:
        # cluster_llm 호출 시 user_interests 가 프롬프트에 실제 포함되는지 확인.
        items = [_item("reuters", "Fed rate cut")]
        seen_prompts: list[str] = []

        class _CapturingProvider:
            name = "gemini"

            def generate(self, payload: dict, prompt: str) -> str:
                seen_prompts.append(prompt)
                return json.dumps([
                    {"topic": "x", "items": [0], "best": 0, "exclude": False},
                ])

        cluster_llm(
            items, [_CapturingProvider()],
            category=Category.MARKETS,
            config=FAST,
            user_interests=("AlphaTopic", "BetaPolicy"),
        )
        assert any("AlphaTopic" in p for p in seen_prompts)
        assert any("BetaPolicy" in p for p in seen_prompts)
        assert any("사용자 관심 키워드" in p for p in seen_prompts)
