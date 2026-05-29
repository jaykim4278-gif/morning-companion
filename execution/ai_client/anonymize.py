# AI API 전송 직전 PII 제거 — 외부 공급자는 개인정보를 영영 보지 못한다.
from __future__ import annotations

import re
from typing import Any

# directives/security-principles.md §1 금지 항목에 대응하는 정규식.
# 과탐지는 허용(false positive), 누락은 불허(false negative).
_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone_us": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "email": re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "dob": re.compile(r"\b(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])\b"),
    "zipcode_us": re.compile(r"\b\d{5}(?:-\d{4})?\b"),
}

# 실제 약명 → 코드명 매핑. DB의 medications 테이블이 권위 있는 소스지만,
# 익명화기는 오프라인에서도 동작해야 하므로 로컬 캐시를 받는다.
_REAL_DRUG_NAMES = {
    # 고혈압 계열 일반명 일부 (대소문자 무시)
    "lisinopril", "losartan", "valsartan", "amlodipine", "hydrochlorothiazide",
    "metoprolol", "atenolol", "enalapril",
    # 고지혈증 계열
    "atorvastatin", "rosuvastatin", "simvastatin", "pravastatin", "ezetimibe",
}

# AI 페이로드에서 허용되는 최상위 키. 이 외의 키는 drop 한다 (deny-by-default).
_ALLOWED_TOP_KEYS = frozenset({
    "vitals", "body_composition", "sleep", "exercise", "glucose",
    "medications", "trends", "anomalies", "locale", "briefing_hint",
    # LabCorp 직전 검사 baseline — metric 이름·수치·플래그·날짜만, PII 무관.
    "lab_baseline", "next_lab_appointment",
    # W6 주간 리포트 — 집계 수치와 일자별 점수만 (PII 無).
    "weekly_stats", "weekly_change", "daily_scores", "lab_context",
    # W3 오늘 건강 코치 — 어제 비교 델타 + 체중 변화 패턴 (PR #66).
    "yesterday", "deltas_vs_yesterday", "weight_change_pattern",
    # W8a 뉴스 번역 — 뉴스 제목/출처만 (사용자 PII 無).  payload={"news": {title, source}}.
    "news",
    # W9 영어회화 — 테마 키워드만 (사용자 PII 無).  payload={"english_phrase": {theme}}.
    "english_phrase",
})

# user_id 같이 외부에 노출되면 안 되는 키. 포함 시 제거.
_DENY_KEYS = frozenset({
    "user_id", "email", "phone", "name", "first_name", "last_name",
    "full_name", "address", "dob", "birth_date", "ssn", "mrn",
    "insurance_id", "labcorp_id", "accession_id", "physician",
    "doctor", "hospital", "clinic",
})


class PIILeakError(ValueError):
    """익명화기가 탐지한 잔여 PII — 전송 중단 신호."""


def _scrub_string(s: str, drug_map: dict[str, str]) -> str:
    """정규식 및 약명 치환을 한 문자열에 적용."""
    for pat in _PII_PATTERNS.values():
        s = pat.sub("[REDACTED]", s)
    # 약명은 단어 경계 기준 대소문자 무시 치환. 매핑 없으면 코드명 미지정으로 redact.
    lowered = s.lower()
    for real in _REAL_DRUG_NAMES:
        if real in lowered:
            code = drug_map.get(real.lower(), "[DRUG]")
            s = re.compile(rf"\b{re.escape(real)}\b", re.IGNORECASE).sub(code, s)
    return s


def _walk(value: Any, drug_map: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _scrub_string(value, drug_map)
    if isinstance(value, dict):
        return {
            k: _walk(v, drug_map)
            for k, v in value.items()
            if k not in _DENY_KEYS
        }
    if isinstance(value, list):
        return [_walk(v, drug_map) for v in value]
    return value


def anonymize_payload(
    payload: dict[str, Any],
    drug_map: dict[str, str] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """AI API 호출 직전에 호출되는 유일한 익명화 진입점.

    - 최상위 키는 `_ALLOWED_TOP_KEYS` 화이트리스트로 제한
    - `_DENY_KEYS` 와 일치하는 키는 재귀 제거
    - 문자열 값은 PII 정규식 및 약명 매핑으로 치환
    - strict=True 이면 치환 후에도 PII 패턴이 남아있을 경우 `PIILeakError`

    Args:
        payload: AI 전송 후보 딕셔너리
        drug_map: real_name(소문자) -> code_name(예: "lisinopril" -> "고혈압약A")
        strict: 잔여 PII 탐지 시 예외 발생 여부 (기본 True — 프로덕션 경로)

    Returns:
        익명화된 새 딕셔너리 (입력은 변경하지 않음)
    """
    drug_map = {k.lower(): v for k, v in (drug_map or {}).items()}
    filtered = {k: v for k, v in payload.items() if k in _ALLOWED_TOP_KEYS}
    scrubbed = _walk(filtered, drug_map)

    if strict:
        leaks = _detect_leaks(scrubbed)
        if leaks:
            raise PIILeakError(f"Residual PII detected: {leaks}")

    return scrubbed


def _detect_leaks(value: Any) -> list[str]:
    """재귀적으로 순회하며 남아있는 PII 패턴을 수집."""
    leaks: list[str] = []
    if isinstance(value, str):
        for name, pat in _PII_PATTERNS.items():
            if pat.search(value):
                leaks.append(name)
        lowered = value.lower()
        for real in _REAL_DRUG_NAMES:
            if re.search(rf"\b{re.escape(real)}\b", lowered):
                leaks.append(f"drug:{real}")
    elif isinstance(value, dict):
        for v in value.values():
            leaks.extend(_detect_leaks(v))
    elif isinstance(value, list):
        for v in value:
            leaks.extend(_detect_leaks(v))
    return leaks
