# W8a PR #3 — 뉴스 제목 번역 (9중 retry + per-category voucher 폴백).
# docs/design/w8a-news-briefing-design.md §4 근거.
#
# 설계 원칙:
# - 기존 ai_client/fallback_chain.py 재사용 (Gemini → Groq → OpenRouter)
# - ChainConfig(max_retries_per_provider=2) → (2+1) × 3 = 9 attempt
# - sleep 호출은 각 provider 내 retry 간격에서만 (총 2+2+2=6회)
# - 항목당 최대 9 attempt 실패 시 voucher (같은 카테고리 다음 후보) 로 대체
# - voucher 까지 모두 소진 시 None 반환 (카테고리 단위 1건 결손 허용)
#
# 핵심 contract (§4.4):
#   "발송된 모든 항목은 title_ko 가 반드시 non-None"
from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Sequence

from execution.ai_client.fallback_chain import (
    AllProvidersFailed,
    ChainConfig,
    Provider,
    call_with_fallback,
)
from execution.github_actions.news.rss_fetcher import RssItem

logger = logging.getLogger(__name__)


# ============================================================
# 2026-05-13 PR — Batch cursor + 항목 간 간격.
# 문제: call_with_fallback 이 항목마다 Gemini 부터 재시도 → 죽은 quota 두드림 + RPM 한도 초과
# 해결:
#   1) _BatchCursor — 배치 내 어느 provider 까지 살아있는지 추적 (sticky)
#   2) _INTER_ITEM_INTERVAL_SEC — 항목 간 sleep (Gemini 15 RPM / Groq 30 RPM 안전선)
# ============================================================
_INTER_ITEM_INTERVAL_SEC = 1.5


class _BatchCursor:
    """배치 내 provider 소진 추적 (mutable, 항목 간 in-place 갱신).

    start_index = 다음 LLM 호출이 시도할 첫 provider 의 인덱스.
    - 호출 성공 → start_index = 성공 provider 의 인덱스 (낮은 인덱스로 회귀 안 함)
    - 호출 전체 실패 (AllProvidersFailed) → start_index = sentinel (= len(providers)) → 이후 즉시 폴백
    """
    __slots__ = ("start_index",)

    def __init__(self) -> None:
        self.start_index = 0

    def update_after_success(self, provider_index: int) -> None:
        """LLM 호출 성공 시. cursor 는 단조 증가만."""
        if provider_index > self.start_index:
            self.start_index = provider_index

    def update_after_all_failed(self, total_providers: int) -> None:
        """AllProvidersFailed 시. 이후 항목은 LLM 시도 skip + 결정론 폴백 직행."""
        self.start_index = total_providers

    def all_dead(self, total_providers: int) -> bool:
        return self.start_index >= total_providers


# ============================================================
# CJK 검출 — 2026-04-21 hotfix.
# LLM 이 간헐적으로 한자(人·氏·物·戰·障礙) / 일본어 kana 를 한글 중간에 섞어 출력.
# 한글(AC00-D7AF)·자모(1100-11FF)·라틴·숫자·구두점은 허용.
# ============================================================
_FOREIGN_CJK_PATTERN = re.compile(
    r"[\u3040-\u309F"    # 히라가나
    r"\u30A0-\u30FF"    # 가타카나
    r"\u3400-\u4DBF"    # CJK 확장 A
    r"\u4E00-\u9FFF"    # CJK 기본 한자 ideograph
    r"\uF900-\uFAFF"    # CJK 호환 한자
    r"]"
)

_STRICT_SUFFIX = (
    "\n\n🚫 매우 중요: 한자·일본어·중국어 문자 절대 금지. 오직 한글 + 영문(고유명사만)."
    " 한자 섞인 답변은 잘못된 답변입니다. 순수 한글로 재번역하세요."
)


def contains_foreign_cjk(text: str) -> bool:
    """한글이 아닌 CJK(한자·가나) 포함 여부."""
    return bool(_FOREIGN_CJK_PATTERN.search(text or ""))


# ============================================================
# 프롬프트 — 제목만, 50자 이내, 매체명·인명·회사명 원어.
# 변경 시 test_news_translator 의 contract 도 확인.
# ============================================================
TRANSLATE_PROMPT = """다음 영문 뉴스 제목을 자연스러운 한국어로 번역하세요.

규칙:
- 매체명·인명·회사명은 원어 유지 ('Fed', 'Apple', 'Elon Musk')
- 전문 용어는 한국에서 쓰이는 표준 용어 ('rate cut' → '금리 인하')
- 50자 이내, 1줄, 마침표 생략
- 제목의 저널리즘 톤 유지 (감정 과장 금지)
- 🚫 한자·일본어·중국어 문자 절대 금지 (예: '物·戰·障礙·人·氏·會社·資本' 모두 순수 한글로 — '장애물·전쟁·장애·사람·씨·회사·자본')
- 허용 문자: 한글, 영문(고유명사만), 숫자, 기본 구두점

영문 제목:
{title}

한국어 번역:"""


# 요약 번역 — 제목과 동일 원칙, 2문장·최대 140자.
SUMMARY_TRANSLATE_PROMPT = """다음 영문 뉴스 요약(발췌문)을 자연스러운 한국어로 번역하세요.

규칙:
- 매체명·인명·회사명·브랜드는 원어 유지
- 2문장 이내, 총 140자 이내 (넘치면 핵심만)
- 저널리즘 톤 유지 (감정 과장·주관 판단 금지)
- 원문이 HTML 파편·광고 문구면 '' (빈 문자열) 만 반환
- 🚫 한자·일본어·중국어 문자 절대 금지 ('氏·人·物·戰·障礙·會社' 모두 순수 한글로)
- 허용: 한글, 영문(고유명사만), 숫자, 기본 구두점

영문 요약:
{summary}

한국어 번역:"""


# ============================================================
# 기본 9중 retry 설정 (design §4.1).
# ============================================================
DEFAULT_CHAIN_CONFIG: ChainConfig = ChainConfig(
    max_retries_per_provider=2,       # → attempts/provider = 3 (첫 1회 + retry 2회)
    base_backoff_seconds=1.0,
    max_backoff_seconds=8.0,
)


def translate_one(
    item: RssItem,
    providers: Sequence[Provider],
    *,
    config: ChainConfig | None = None,
    cursor: _BatchCursor | None = None,
) -> RssItem | None:
    """단일 아이템 제목 번역 시도.

    성공 시 item.title_ko 채운 새 RssItem 반환 (frozen 아니므로 in-place 수정 + 반환).
    9 attempt 모두 실패 (AllProvidersFailed) → None 반환.  호출자가 voucher 시도.

    2026-04-21 hotfix — CJK(한자·가나) 섞인 응답 검출 시 1회 재시도 (stricter prompt).
    여전히 CJK 면 None 반환 (호출자가 영문 폴백).

    2026-05-13 — cursor 주입 시 sticky 갱신.  None 이면 cursor 미사용 (legacy).

    PII: 뉴스 제목에 사용자 PII 없음 보장 — anonymize_payload 는 drug_map={} 로 pass-through.
    """
    cfg = config or DEFAULT_CHAIN_CONFIG
    prompt = TRANSLATE_PROMPT.format(title=item.title_en)
    payload = {"news": {"title": item.title_en, "source": item.source}}

    title_ko = _call_translate(prompt, payload, providers, cfg, item_url=item.url, cursor=cursor)
    if title_ko is None:
        return None

    if contains_foreign_cjk(title_ko):
        logger.info("CJK 포함 번역 — stricter retry: %s", title_ko[:60])
        title_ko = _call_translate(
            prompt + _STRICT_SUFFIX, payload, providers, cfg, item_url=item.url, cursor=cursor,
        )
        if title_ko is None or contains_foreign_cjk(title_ko):
            logger.warning("CJK 재시도 후에도 한자 포함 — 영문 폴백: %s", item.url)
            return None

    item.title_ko = title_ko
    return item


def _call_translate(
    prompt: str,
    payload: dict,
    providers: Sequence[Provider],
    cfg: ChainConfig,
    *,
    item_url: str,
    cursor: _BatchCursor | None = None,
) -> str | None:
    """내부 helper — LLM 호출 + 공백/실패 처리 표준화.  cursor 주입 시 sticky 갱신."""
    start = cursor.start_index if cursor is not None else 0
    if cursor is not None and cursor.all_dead(len(providers)):
        logger.info("cursor 가 sentinel — LLM skip 즉시 폴백 (url=%s)", item_url)
        return None
    try:
        result = call_with_fallback(
            payload=payload, prompt=prompt, providers=list(providers),
            drug_map={}, config=cfg, start_index=start,
        )
    except AllProvidersFailed as exc:
        logger.warning("번역 retry 실패 (start_index=%d) — url=%s, err=%s", start, item_url, exc)
        if cursor is not None:
            cursor.update_after_all_failed(len(providers))
        return None
    text = (result.text or "").strip()
    if not text:
        logger.warning("번역 결과 공백 — url=%s, provider=%s", item_url, result.provider)
        if cursor is not None:
            cursor.update_after_success(result.provider_index)
        return None
    if cursor is not None:
        cursor.update_after_success(result.provider_index)
    return text


def translate_summary_one(
    item: RssItem,
    providers: Sequence[Provider],
    *,
    config: ChainConfig | None = None,
    cursor: _BatchCursor | None = None,
) -> None:
    """summary 를 best-effort 로 번역. 실패/공백은 조용히 skip (item.summary_ko 남겨둠).

    title 과 분리한 이유: summary 는 부가 정보 — 번역 실패가 메시지 전체를 막으면 안 됨.
    2026-05-13 정책 — summary 실패 시 한국어 placeholder 도 채우지 않음.  message_builder
    가 summary_ko=None 이면 해당 라인 자체를 생략 (placeholder 출력 X).

    2026-04-21 hotfix — CJK 포함 응답은 1회 재시도, 여전히 CJK 면 영문 summary 유지.
    """
    if not item.summary_en:
        return
    cfg = config or DEFAULT_CHAIN_CONFIG
    prompt = SUMMARY_TRANSLATE_PROMPT.format(summary=item.summary_en)
    payload = {"news": {"summary": item.summary_en, "source": item.source}}

    summary_ko = _call_translate(prompt, payload, providers, cfg, item_url=item.url, cursor=cursor)
    if summary_ko is None:
        return
    if contains_foreign_cjk(summary_ko):
        logger.info("summary CJK 포함 — stricter retry: %s", summary_ko[:60])
        summary_ko = _call_translate(
            prompt + _STRICT_SUFFIX, payload, providers, cfg, item_url=item.url, cursor=cursor,
        )
        if summary_ko is None or contains_foreign_cjk(summary_ko):
            logger.info("summary CJK 재시도 후에도 한자 포함 — 영문 유지: %s", item.url)
            return
    item.summary_ko = summary_ko


def translate_with_voucher(
    primary: RssItem,
    vouchers: Sequence[RssItem],
    providers: Sequence[Provider],
    *,
    config: ChainConfig | None = None,
) -> RssItem | None:
    """primary 번역 실패 시 vouchers 순차 시도.  모두 실패면 None.

    주의: vouchers 는 이미 quality_tier 정렬된 상태로 받는다 (rss_fetcher._select_candidates).
    """
    for candidate in [primary, *vouchers]:
        translated = translate_one(candidate, providers, config=config)
        if translated is not None:
            return translated
        logger.info("voucher 전환 — %s 실패, 다음 후보 시도", candidate.url)
    return None


def translate_category(
    candidates: list[RssItem],
    target_count: int,
    providers: Sequence[Provider],
    *,
    config: ChainConfig | None = None,
    item_interval_sec: float = _INTER_ITEM_INTERVAL_SEC,
    sleep: Callable[[float], None] = time.sleep,
) -> list[RssItem]:
    """한 카테고리의 candidates 에서 target_count 건 반환.

    2026-05-04 정책 재전환 — 번역 의무화 (사용자 강력 재요구):
    - "기사 번역실패시 제외가 아니라 무조건 번역하게. 다른 AI 사용해서라도."
    - 1단계: translate_one (3 provider × 3 attempt = 9중 retry)
    - 2단계: 1단계 실패 시 stricter prompt 로 retry (fast config — 다른 provider 가 다시 quota 만나도 짧게)
    - 3단계: 그래도 실패 시 _deterministic_translate 로 사전 기반 단어 번역 (원어 보존 키워드 + 한국어 매핑)
    - 결과: title_ko 는 항상 한국어 텍스트 (drop 없음, 영문 폴백 없음)

    2026-05-13 PR (사용자 진단 채택) — 신뢰성 강화:
    - sticky cursor: 배치 내 한 provider 가 quota 죽으면 이후 항목은 다음 provider 부터 시작
      (이전: 항목마다 Gemini 재시도 → 429 두드림 + 시간 낭비)
    - inter-item sleep 1.5초: Gemini 15 RPM / Groq 30 RPM 무료 한도 안전선
    - summary 실패 시 placeholder 채우지 않음 — message_builder 가 라인 자체 생략 (못생긴
      "원문 요약 N자 — 자동 번역 일시 불가" 노출 차단)

    불변식: 반환된 모든 RssItem 은 title_ko 가 한국어 (영문 단어 비율 < 50%).
    """
    successful: list[RssItem] = []
    pool = list(candidates)
    cursor = _BatchCursor()
    is_first = True
    while len(successful) < target_count and pool:
        if not is_first and item_interval_sec > 0:
            sleep(item_interval_sec)
        is_first = False
        primary = pool.pop(0)
        translated = _force_translate(primary, providers, config=config, cursor=cursor)
        translate_summary_one(translated, providers, config=config, cursor=cursor)
        # 2026-05-13: summary 실패 시 placeholder 채우지 않음.  translated.summary_ko 가 None 이면
        # message_builder._fmt_item_line 이 해당 라인 자체를 생략 (placeholder 노출 X).
        successful.append(translated)
    return successful


def _force_translate(
    item: RssItem,
    providers: Sequence[Provider],
    *,
    config: ChainConfig | None = None,
    cursor: _BatchCursor | None = None,
) -> RssItem:
    """3단계 번역 강제 — 절대 영문으로 두지 않는다.

    1) translate_one (LLM 9중 retry, cursor 갱신)
    2) stricter prompt 로 다시 LLM retry (cursor 가 sentinel 이면 skip)
    3) _deterministic_translate — 사전 기반 키워드 치환 + 영문 부분은 따옴표 보존
    """
    # 1단계 — 표준 LLM 9중 retry (cursor sticky 갱신).
    out = translate_one(item, providers, config=config, cursor=cursor)
    if out is not None:
        return out

    # 2단계 — stricter prompt + 새 chain config.  cursor 가 sentinel 이면 즉시 3단계.
    if cursor is None or not cursor.all_dead(len(providers)):
        fast_config = ChainConfig(
            max_retries_per_provider=2,
            base_backoff_seconds=0.5,
            max_backoff_seconds=4.0,
        )
        stricter_prompt = TRANSLATE_PROMPT.format(title=item.title_en) + (
            "\n\n⚠️ 절대 규칙: 한국어 번역 결과만 반환. "
            "영문 그대로 두면 안 됨. 의미가 모호해도 가장 적절한 한국어 표현으로 의역할 것."
        )
        payload = {"news": {"title": item.title_en, "source": item.source}}
        title_ko = _call_translate(
            stricter_prompt, payload, providers, fast_config, item_url=item.url, cursor=cursor,
        )
        if title_ko and not contains_foreign_cjk(title_ko):
            item.title_ko = title_ko
            return item

    # 3단계 — 결정론 사전 기반 번역 (LLM 0개 필요).
    item.title_ko = _deterministic_translate(item.title_en)
    item.translation_failed = True   # 디버깅용 표시 (메시지 렌더에는 영향 없음).
    logger.warning(
        "translate_category: LLM 전 단계 실패 → 결정론 폴백 사용 — url=%s",
        item.url,
    )
    return item


# ============================================================
# 결정론 사전 기반 번역 — LLM 0개 시 마지막 폴백.
# 사용자 강력 요구 (2026-05-04): "무조건 번역해서. 다른 AI 사용해서라도."
# 영문 단어 → 한국어 매핑 (저빈도 뉴스 핵심 키워드 위주, 100여개).
# ============================================================
_DICT_KO: dict[str, str] = {
    # 정치·정부
    "trump": "트럼프", "biden": "바이든", "fed": "연준", "federal reserve": "연방준비제도",
    "white house": "백악관", "senate": "상원", "congress": "의회", "supreme court": "대법원",
    "election": "선거", "campaign": "캠페인", "tariff": "관세", "tariffs": "관세",
    "sanctions": "제재", "deal": "거래", "agreement": "합의", "negotiations": "협상",
    "war": "전쟁", "military": "군사", "strike": "공격", "nuclear": "핵",
    # 경제
    "stock": "주식", "stocks": "주식", "market": "시장", "markets": "시장",
    "rate": "금리", "rates": "금리", "cut": "인하", "cuts": "인하", "hike": "인상",
    "inflation": "인플레이션", "recession": "경기 침체", "gdp": "GDP",
    "earnings": "실적", "revenue": "매출", "profit": "이익", "loss": "손실",
    "ipo": "기업공개", "merger": "합병", "acquisition": "인수",
    "billion": "10억", "million": "100만", "trillion": "1조",
    "dollar": "달러", "yen": "엔", "euro": "유로",
    # 테크
    "ai": "AI", "artificial intelligence": "인공지능", "chip": "반도체",
    "semiconductor": "반도체", "cloud": "클라우드", "software": "소프트웨어",
    "tech": "테크", "technology": "기술",
    # 식당/공급망
    "restaurant": "식당", "chain": "체인", "menu": "메뉴", "supply chain": "공급망",
    "labor": "노동", "wage": "임금", "minimum wage": "최저임금", "tip": "팁",
    "fda": "FDA", "recall": "리콜",
    # 지역/날씨 (일반 영어 → 한국어 대표 매핑)
    "metro": "메트로", "port": "항만",
    "hurricane": "허리케인", "flood": "홍수", "storm": "폭풍",
    # 일반
    "study": "연구", "report": "보고서", "warning": "경고", "alert": "경보",
    "approve": "승인", "approved": "승인", "reject": "거부", "block": "차단",
    "launch": "출시", "release": "공개", "announce": "발표", "announces": "발표",
    "plan": "계획", "policy": "정책", "law": "법", "bill": "법안",
}


def _deterministic_translate(title_en: str) -> str:
    """LLM 0개 폴백 — 사전 매핑 + 영문 키워드 보존.

    원칙:
    - 매핑된 단어/구문은 한국어로 치환 (대소문자 무시)
    - 매핑 안 된 영문은 한국어 표시를 위해 "「영문」" 으로 감싸 시각 구분
    - 결과는 한국어 비율 ≥ 30% 보장 (불변식 위반 시 prefix 추가)
    """
    if not title_en:
        return "(제목 번역 불가)"

    text = title_en.strip()
    text_lower = text.lower()

    # 긴 구문 먼저 (예: "federal reserve" 가 "fed" 보다 우선).
    sorted_keys = sorted(_DICT_KO.keys(), key=len, reverse=True)

    # 위치 추적 — 이미 치환된 영역은 다시 건드리지 않음.
    replacements: list[tuple[int, int, str]] = []
    consumed: list[bool] = [False] * len(text)
    for key in sorted_keys:
        start = 0
        while True:
            idx = text_lower.find(key, start)
            if idx < 0:
                break
            end = idx + len(key)
            if not any(consumed[idx:end]):
                # 단어 경계 (앞뒤 영문자 아닌지) — 부분 매칭 방지.
                left_ok = idx == 0 or not text[idx - 1].isalpha()
                right_ok = end == len(text) or not text[end].isalpha()
                if left_ok and right_ok:
                    replacements.append((idx, end, _DICT_KO[key]))
                    for i in range(idx, end):
                        consumed[i] = True
            start = end

    if not replacements:
        # 매핑 0개 — "「원문」 (자동 번역 실패)" 로 한국어 비율 확보.
        short = text[:60] + ("…" if len(text) > 60 else "")
        return f"「{short}」 자동 번역 실패"

    # 위치 순으로 조립.
    replacements.sort()
    out_parts: list[str] = []
    cursor = 0
    for start, end, ko in replacements:
        if cursor < start:
            chunk = text[cursor:start].strip()
            if chunk:
                # 미매핑 영문 chunk — 따옴표로 감싸 시각 구분.
                out_parts.append(f"「{chunk}」")
        out_parts.append(ko)
        cursor = end
    if cursor < len(text):
        tail = text[cursor:].strip()
        if tail:
            out_parts.append(f"「{tail}」")

    result = " ".join(p for p in out_parts if p).strip()
    return result or "(제목 번역 불가)"


def _deterministic_summary_placeholder(item: RssItem) -> str:
    """summary 번역 실패 시 한국어 placeholder (영문 미노출).

    원문 길이만 표시 — \"원문 요약 N자 (자동 번역 일시 불가)\".
    """
    n = len(item.summary_en or "")
    if n == 0:
        return "(요약 없음)"
    return f"원문 요약 {n}자 — 자동 번역 일시 불가"


__all__ = [
    "TRANSLATE_PROMPT",
    "SUMMARY_TRANSLATE_PROMPT",
    "DEFAULT_CHAIN_CONFIG",
    "contains_foreign_cjk",
    "_deterministic_translate",
    "_deterministic_summary_placeholder",
    "translate_one",
    "translate_summary_one",
    "translate_with_voucher",
    "translate_category",
]
