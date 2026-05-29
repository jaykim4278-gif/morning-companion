# W8a PR #4 — L2 LLM 클러스터링 + best 선택 (+ 결정론적 폴백).
# docs/design/w8a-news-briefing-design.md §5 근거.
#
# 목적: 사용자 요구 "그룹 내에서 가장 객관적·중요·많이 본 뉴스를 선택해줘."
#
# 동작:
#   1) candidates 제목들을 JSON 으로 LLM 에 전달
#   2) LLM 이 같은 사건끼리 클러스터링 + 각 클러스터에서 best 1건 index 반환
#   3) LLM 실패/JSON 파싱 실패 시 → cluster_deterministic 폴백 (독립 클러스터, tier 정렬 상위)
#
# 폴백 전략은 "보수적" — 같은 사건이 2건 들어갈 위험은 있어도 메시지 발송 자체는 보장.
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from execution.ai_client.fallback_chain import (
    AllProvidersFailed,
    ChainConfig,
    Provider,
    call_with_fallback,
)
from execution.github_actions.news.rss_fetcher import RssItem
from execution.github_actions.news.source_quality import score_item
from execution.github_actions.news.sources import Category

logger = logging.getLogger(__name__)


# 2026-04-21 재작성 — 사용자 피드백 반영.
# 기존: 객관성·1차 출처·최신성 3원칙
# 추가: "헤드라인급 거시 뉴스 우선" rubric + 카테고리별 선호 힌트
#
# 실제 피드백 예시 (제거 대상):
# - [TechCrunch] "Yelp AI 업데이트, Blue Energy 3.8억 펀딩" → 개별 업체 micro
# - [NRN] "피자헛 로열티, 맥도날드 음료" → 개별 체인 마이너 메뉴
# - [지역방송] 실종·살인 등 개인 범죄 단발 사건 → 거시성 없음
CLUSTER_PROMPT = """다음 영문 뉴스 제목들을 같은 사건/주제별로 그룹화하고,
각 그룹에서 '헤드라인급 거시 뉴스' 기준으로 best 1건의 인덱스를 선택하세요.
동시에 **정보 가치가 없는 filler 클러스터**는 `"exclude": true` 로 표시하세요 (메시지에서 제거됨).

【선택 기준 (우선순위)】
1. 거시성: 산업 전체·정책·경제 흐름·대형 M&A(>$1B) > 개별 업체 제품/펀딩·인사 뉴스
2. 객관성: 사실 보도 > 의견·해설
3. 1차 출처: 원본 보도 > 인용·재가공
4. 최신성: 가장 최근 pubDate

【카테고리별 추가 선호】
{category_hint}

【배제 대상 (같은 그룹 내라면 best 에서 제외)】
- 개별 스타트업 펀딩 라운드 (<$1B 는 제외, ≥$1B 대형은 포함)
- 개별 제품 업데이트·신메뉴·LTO (Limited Time Offer)
- 개인 범죄·교통사고·화재·실종 (집단·정책 영향 없는 로컬 사건)
- 연예인 가십·개인사 (industry·정책 영향 없는 것)
- 인사 이동 (CEO 교체 등 대형 M&A·거시 영향 없으면 제외)

【🚫 FILLER — 반드시 exclude=true 로 표시하고 제거】
이런 기사는 정보 가치 0 이므로 메시지에서 제외해야 합니다:
- 팟캐스트/뉴스레터/유튜브 홍보 ("Listen to our podcast", "Subscribe to", "See the full episode")
- 순위 나열 리스트 ("Top 500 Chains", "Top 100 Ranking", 체인 이름만 나열)
- 콘텐츠 포인터 ("Read more in our daily briefing", "See the list", "Check out...")
- 후원·광고성 ("Sponsored", "Partner content", 기사 형식 위장한 프로모션)
- 낚시성 제목만 있고 본문 없는 경우 (click-bait without substance)
- 같은 매체 다른 섹션 안내문 ("Today's top stories in [매체이름]")

예시:
- "Jersey Mike's, Red Lobster on Top 500 Restaurants list" → filler (순위 나열)
- "Listen to our Restaurant Daily podcast for all headlines" → filler (팟캐스트 홍보)
- "See the full Top 100 QSR ranking" → filler (리스트 포인터)
- "Fed raises rates 25bp, signals May cut" → **거시 뉴스 유지**

【예외 (거시 영향 있으면 포함)】
- 기업 인사라도 시가총액 $100B+ 기업의 CEO 교체, 이사회 개편
- 대형 데이터 유출 (>1M 사용자), 법적 판결 (산업 기준 바뀜)
- 규제·법안·FDA 승인·리콜 (산업 전반 영향)

입력 (0-indexed):
{items_json}

출력 형식 (반드시 유효 JSON, 배열):
[
  {{
    "topic": "사건 주제 (영문 짧게)",
    "items": [0, 3, 5],
    "best": 0,
    "exclude": false
  }},
  {{
    "topic": "filler_podcast_promo",
    "items": [2],
    "best": 2,
    "exclude": true
  }}
]

중요:
- 모든 입력 인덱스가 정확히 하나의 클러스터에 속해야 함
- 독립 사건은 items=[해당_인덱스 하나] 단독 클러스터
- best 는 거시성 최우선, 그 다음 객관성·1차 출처·최신성
- exclude 필드는 boolean (true/false), 생략하면 false 간주
- JSON 외 텍스트(설명·마크다운) 금지
- 모든 아이템이 filler 여도 각자 cluster 만들고 exclude=true — 인덱스 누락 금지"""


# 카테고리별 선호 hint — 프롬프트에 삽입.  편집 시 테스트 test_news_cluster_select 참조.
_CATEGORY_HINTS: dict[Category, str] = {
    Category.TOP: (
        "- 연방 정책·의회 입법·대법원 판결·대선/선거 정국\n"
        "- 거시 경제지표(고용·CPI·GDP·금리)·연준 정책\n"
        "- 외교·무역 협상·관세·지정학 분쟁\n"
        "- 전국 규모 자연재해·공중보건·인프라 사고"
    ),
    Category.MARKETS: (
        "- FOMC 회의·금리 결정·국채 수익률 반전\n"
        "- $10B+ 대형 M&A·IPO·파산\n"
        "- 산업 전반 실적 트렌드(빅테크·에너지·금융 등)\n"
        "- 대외 충격(유가·환율·지정학)"
    ),
    Category.TECH: (
        # 2026-05-13 사용자 요청 — 일반 테크 → AI 전용 카테고리.
        "🎯 이 카테고리는 AI 뉴스 전용입니다.  best 선정은 AI 관련 기사 중에서만.\n"
        "  - AI 모델·연구·제품 발표 (OpenAI / Anthropic / Google DeepMind / Meta / xAI / Mistral 등 frontier labs)\n"
        "  - AI 반도체·인프라 (NVIDIA / AMD / TSMC / 하이퍼스케일러 AI 데이터센터·전력)\n"
        "  - AI 규제·소송·정책 (EU AI Act, US Executive Order, 저작권 판결, 안전 표준)\n"
        "  - 기업 AI 도입·AI agent·enterprise AI·생산성 도구\n"
        "  - AI 안전·alignment·red-teaming·산업 표준\n"
        "  - AI 학계·논문 발표·model release (Llama, Gemini, Claude, GPT 신버전 등)\n"
        "🚫 무조건 exclude (best 후보 X, AI 카테고리에 부적합):\n"
        "  - 정치 뉴스 (선거·법안·당파·외교·의회·대통령 정책) — AI 규제 법안 아니라면 모두 제외\n"
        "  - 금융·경제 뉴스 (Fed·금리·시장·M&A) — AI 기업 IPO/펀딩 아니라면 모두 제외\n"
        "  - 헬스케어·약가·의료정책 — AI 의료 응용 아니라면 모두 제외\n"
        "  - 일반 소비자 하드웨어 (스마트폰·이어폰·자동차·게임) — AI feature 핵심 아니라면 제외\n"
        "  - 단발 사이버 사고·기업 보안 사건 — AI 보안 연구 아니라면 제외\n"
        "  - 개별 스타트업 펀딩 (<$500M) — AI 산업 trends 아니라면 제외\n"
        "⚠️ 후보 중 AI 직접 관련 기사가 0건이면 모든 후보를 exclude=true 로 표시 (카테고리 빈 출력 허용).\n"
        "   AI 와 무관한 정치·금융 기사로 카테고리를 채우지 마세요."
    ),
    Category.RESTAURANT: (
        "- 업계 노동 규제·최저임금·팁 제도 변화\n"
        "- 공급망 쇼크·식자재 가격·관세\n"
        "- 소비자 지출 트렌드·QSR vs 캐주얼 시장 점유 변화\n"
        "- 체인 M&A·대형 파산 (>$500M 또는 500+ 매장)\n"
        "- 식품 안전·FDA 리콜·규제 개정\n"
        "🚫 특히 제외 (이 카테고리에 filler 가 많음):\n"
        "  - 'Top 500 Chains' / 'Top 100 Ranking' 등 순위 나열 리스트\n"
        "  - 'Listen to our Restaurant Daily podcast' 등 팟캐스트 홍보\n"
        "  - 'See today's top stories' 등 매체 내부 안내문\n"
        "  - 개별 체인 신메뉴·LTO 기사 (산업 트렌드 아니면 제외)"
    ),
    Category.LOCAL: (
        "- 시정·예산·선거·자연재해 등 거시 지역 이슈\n"
        "- 지역 경제·고용·주요 산업 대형 이슈\n"
        "- 교통·인프라 프로젝트 (대중교통·공항·고속도로)\n"
        "- 공중보건·환경·학군 정책\n"
        "- 지역 관련 없으면 전국 주요 뉴스 대체 허용\n"
        "🚫 개인 범죄·교통사고·화재·실종·가정폭력·성범죄 등 개별 사건 제외\n"
        "   (집단 범죄, 공직자 비리, 산업 전반 영향 있는 법원 판결은 예외)"
    ),
}


def _build_category_hint(
    category: Category | None,
    user_interests: Sequence[str] = (),
) -> str:
    """프롬프트에 삽입할 카테고리별 선호 hint + 사용자 관심 키워드.

    user_interests 가 비어있지 않으면 cluster_and_select 의 best 선택 시 가중치로 활용.
    """
    base = (
        "(카테고리 정보 없음 — 거시성·객관성 기준만 적용)"
        if category is None
        else _CATEGORY_HINTS.get(category, "(카테고리 선호 hint 미정의)")
    )
    if not user_interests:
        return base
    interests_line = ", ".join(user_interests)
    return (
        f"{base}\n\n"
        f"【사용자 관심 키워드 — 다음 키워드 관련 기사가 클러스터 내에 있다면 best 로 우선 선택】\n"
        f"{interests_line}"
    )


@dataclass(frozen=True)
class Cluster:
    topic: str           # 'Fed rate decision', 'AI regulation' 등
    items: tuple[int, ...]  # 입력 candidates 인덱스
    best_idx: int        # candidates 의 best 인덱스 (items 중 하나)
    exclude: bool = False  # 2026-04-21 hotfix — filler (팟캐스트 홍보·순위 나열 등) → 렌더 제외


class ClusterParseError(ValueError):
    """LLM 응답이 유효 JSON 아니거나 invariant 위반."""


def _build_items_json(candidates: Sequence[RssItem]) -> str:
    """candidates 를 LLM 입력 JSON 문자열로."""
    data = [
        {
            "idx": i,
            "source": c.source,
            "title": c.title_en,
            "pub_date": c.pub_date.isoformat(),
        }
        for i, c in enumerate(candidates)
    ]
    return json.dumps(data, ensure_ascii=False, indent=2)


def _parse_cluster_response(raw: str, n_items: int) -> list[Cluster]:
    """LLM 응답 → Cluster 리스트.  검증 실패 시 ClusterParseError.

    검증:
    - JSON 배열 파싱 성공
    - 모든 0..n_items-1 인덱스가 정확히 하나의 클러스터에 속함
    - best 가 items 중 하나
    """
    try:
        # LLM 이 ```json ... ``` 마크다운 감쌌을 수 있어 앞뒤 제거.
        text = raw.strip()
        if text.startswith("```"):
            # 첫 라인과 마지막 ``` 제거.
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ClusterParseError(f"JSON 파싱 실패: {exc}") from exc

    if not isinstance(data, list):
        raise ClusterParseError(f"expected list, got {type(data).__name__}")

    clusters: list[Cluster] = []
    seen: set[int] = set()
    for entry in data:
        if not isinstance(entry, dict):
            raise ClusterParseError(f"클러스터 항목이 dict 아님: {entry}")
        topic = str(entry.get("topic") or "unknown")
        items = entry.get("items")
        best = entry.get("best")
        if not isinstance(items, list) or not items:
            raise ClusterParseError(f"invalid items: {items}")
        items_int = tuple(int(i) for i in items)
        if any(i < 0 or i >= n_items for i in items_int):
            raise ClusterParseError(f"out-of-range idx: {items_int}")
        if any(i in seen for i in items_int):
            raise ClusterParseError(f"중복 idx: {items_int}")
        seen.update(items_int)
        if best not in items_int:
            raise ClusterParseError(f"best={best} 가 items={items_int} 에 없음")
        exclude_raw = entry.get("exclude", False)
        exclude = bool(exclude_raw) if isinstance(exclude_raw, bool) else False
        clusters.append(Cluster(
            topic=topic, items=items_int, best_idx=int(best), exclude=exclude,
        ))

    missing = set(range(n_items)) - seen
    if missing:
        raise ClusterParseError(f"누락 idx: {sorted(missing)}")
    return clusters


def cluster_llm(
    candidates: list[RssItem],
    providers: Sequence[Provider],
    *,
    category: Category | None = None,
    config: ChainConfig | None = None,
    user_interests: Sequence[str] = (),
) -> list[Cluster]:
    """LLM 호출 + 파싱.  AllProvidersFailed / ClusterParseError → 호출자가 폴백 판단.

    category: 카테고리별 선호 hint 를 프롬프트에 삽입 (2026-04-21 추가).
    user_interests: 사용자 관심 키워드 (PR #89) — 프롬프트 hint 에 붙어 best 선택 가중치.
    """
    if not candidates:
        return []
    cfg = config or ChainConfig(max_retries_per_provider=1, base_backoff_seconds=1.0)
    items_json = _build_items_json(candidates)
    prompt = CLUSTER_PROMPT.format(
        items_json=items_json,
        category_hint=_build_category_hint(category, user_interests),
    )
    payload = {"news": {"items": items_json}}  # _ALLOWED_TOP_KEYS["news"] 활용
    result = call_with_fallback(
        payload=payload,
        prompt=prompt,
        providers=list(providers),
        drug_map={},
        config=cfg,
    )
    return _parse_cluster_response(result.text, len(candidates))


def cluster_deterministic(candidates: list[RssItem]) -> list[Cluster]:
    """LLM 폴백 — 모든 아이템을 독립 클러스터로 취급 (의미 클러스터링 생략).

    best = 각 클러스터의 유일 아이템.  후속 단계에서 tier 정렬 후 상위 N 건 선발하면
    일반적으로 같은 사건이 연속으로 들어가지 않음 (source 다변화 덕).
    """
    return [
        Cluster(topic=f"solo_{i}_{c.source}", items=(i,), best_idx=i)
        for i, c in enumerate(candidates)
    ]


def select_best_per_cluster(
    candidates: list[RssItem],
    clusters: list[Cluster],
) -> list[RssItem]:
    """각 클러스터의 best 를 뽑아 RssItem 리스트로.  클러스터 순서 유지.

    2026-04-21 hotfix — exclude=True 클러스터는 제외 (filler 제거).
    """
    return [candidates[c.best_idx] for c in clusters if not c.exclude]


def cluster_and_select(
    candidates: list[RssItem],
    providers: Sequence[Provider],
    *,
    category: Category | None = None,
    config: ChainConfig | None = None,
    user_interests: Sequence[str] = (),
) -> tuple[list[RssItem], dict[str, str | None]]:
    """상위 진입점 — LLM 클러스터링 시도 → 실패 시 결정론 폴백.

    category: 프롬프트에 카테고리별 macro 선호 hint 삽입 (2026-04-21).
    user_interests: 사용자 관심 키워드 (PR #89) — 프롬프트 hint 끝에 붙여 best 선택 가중치.

    Returns:
        (bests, cluster_topic_by_url) — 발송 대상 아이템 + news_sent_log.cluster_topic 매핑.
    """
    if not candidates:
        return [], {}
    try:
        clusters = cluster_llm(
            candidates, providers,
            category=category, config=config, user_interests=user_interests,
        )
        logger.info(
            "LLM 클러스터링 성공 — category=%s, interests=%d, %d → %d 클러스터",
            category, len(user_interests), len(candidates), len(clusters),
        )
    except (AllProvidersFailed, ClusterParseError) as exc:
        logger.warning("LLM 클러스터링 실패 — 결정론 폴백: %s", exc)
        clusters = cluster_deterministic(candidates)

    # exclude=True 클러스터는 filler 이므로 news_sent_log 기록·렌더 모두 제외.
    bests = select_best_per_cluster(candidates, clusters)
    topic_by_url: dict[str, str | None] = {}
    for cluster in clusters:
        if cluster.exclude:
            continue
        best_item = candidates[cluster.best_idx]
        topic_by_url[best_item.url] = cluster.topic
    return bests, topic_by_url


__all__ = [
    "CLUSTER_PROMPT",
    "Cluster",
    "ClusterParseError",
    "cluster_llm",
    "cluster_deterministic",
    "select_best_per_cluster",
    "cluster_and_select",
]
