# W8a PR #3 — 뉴스 브리핑 PII 가드 contract (LLM payload 에 사용자 데이터 부재).
# docs/design/w8a-news-briefing-design.md §10.1.
#
# 설계: 뉴스 번역 payload 는 뉴스 제목 자체가 본문이라 사용자 PII 들어갈 여지 원래 없음.
# 그래도 향후 누군가 user_id 나 약물 정보를 payload 에 실수로 넣는 회귀를 막는 contract test.
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from execution.ai_client.fallback_chain import ChainConfig
from execution.github_actions.news.rss_fetcher import RssItem
from execution.github_actions.news.sources import Category
from execution.github_actions.news.translator import translate_one


@dataclass
class CapturingProvider:
    name: str = "gemini"
    response: str = "번역"
    seen_payloads: list[dict] = field(default_factory=list)
    seen_prompts: list[str] = field(default_factory=list)

    def generate(self, payload: dict, prompt: str) -> str:
        self.seen_payloads.append(payload)
        self.seen_prompts.append(prompt)
        return self.response


_FAST = ChainConfig(max_retries_per_provider=0, base_backoff_seconds=0.0)


def _item(title: str = "Fed rate cut") -> RssItem:
    return RssItem(
        source="reuters",
        url="https://r.com/1",
        url_hash="h",
        title_en=title,
        pub_date=datetime(2026, 4, 20, tzinfo=timezone.utc),
        category=Category.TOP,
    )


def test_payload_does_not_contain_user_id_or_drug_names() -> None:
    p = CapturingProvider()
    translate_one(_item(), [p], config=_FAST)
    payload = p.seen_payloads[0]
    # 사용자 PII 키워드 부재 확인.
    forbidden_keys = {"user_id", "drug_map", "medications", "patient", "name", "phone", "address"}
    assert not (set(payload.keys()) & forbidden_keys), (
        f"뉴스 payload 에 PII 키 발견: {set(payload.keys()) & forbidden_keys}"
    )


def test_prompt_does_not_contain_user_identifiers() -> None:
    # 뉴스 제목에 사용자 이름·DOB 가 들어갈 이유가 없지만 contract 로 방어.
    p = CapturingProvider()
    translate_one(_item(title="Fed rate cut"), [p], config=_FAST)
    prompt = p.seen_prompts[0]
    # 흔한 PII 패턴
    import re
    dob_pat = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
    ssn_pat = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    phone_pat = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")
    assert not dob_pat.search(prompt), "DOB 패턴 감지"
    assert not ssn_pat.search(prompt), "SSN 패턴 감지"
    assert not phone_pat.search(prompt), "전화 패턴 감지"


def test_title_is_only_news_content_not_user_data() -> None:
    # payload.title 은 item.title_en 그대로 — 사용자 입력 경로 없음.
    p = CapturingProvider()
    item = _item(title="Fed rate cut")
    translate_one(item, [p], config=_FAST)
    assert p.seen_payloads[0]["news"]["title"] == "Fed rate cut"
