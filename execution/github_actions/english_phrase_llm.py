# 영어회화 LLM (JSON 출력 + 응답 예시 + TTS 친화 구조).
#
# 출력 형식: LLM 이 JSON 반환 → renderer 가 HTML Telegram 메시지로 조립.
# 모든 영문 문장이 독립 필드로 분리되어 TTS 합성·링크 임베드가 용이.
from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from execution.ai_client.fallback_chain import (
    AllProvidersFailed,
    ChainConfig,
    Provider,
    call_with_fallback,
)
from execution.github_actions.english_phrase import (
    EnglishPhrase,
    EnglishPhraseInputs,
    Theme,
    hash_phrase,
)

logger = logging.getLogger(__name__)


# ============================================================
# 데이터 구조 — JSON 의 1:1 매핑.
# ============================================================
@dataclass(frozen=True)
class Scenario:
    """직접 써먹기 — 일상 장면 + 실제 영어 예시 + 한국어 직역.

    korean 필드는 과거 데이터 호환을 위해 빈 문자열 허용 — 빈 값이면 renderer 가 라인 생략.
    """
    setup: str        # "동네 커피숍에서 우연히 만난 이웃에게"
    english: str      # "Hello, Sarah! Long time no see. What have you been up to?"
    korean: str = ""  # "안녕하세요, 사라! 오랜만이에요. 그동안 뭘 하고 지내셨어요?"


@dataclass(frozen=True)
class ResponseExample:
    """대답 예시 — 영어 응답 + 한국어 직역."""
    english: str
    korean: str


@dataclass(frozen=True)
class EnglishPhraseData:
    """LLM 출력 JSON 의 구조화 표현 — renderer 에 전달.

    모든 영문 문장(phrase_en, scenarios[].english, responses[].english)이
    독립 필드로 분리되어 TTS 생성·링크 임베드가 단순해짐.
    """
    theme: Theme
    phrase_en: str                  # 메인 영어 표현 (예: "What have you been up to?")
    phrase_ko: str                  # 직역 (예: "그동안 뭘 하고 지내셨어요?")
    context: str                    # 💡 미국 현장에서 — 3~4문장 설명
    scenarios: tuple[Scenario, ...]
    responses: tuple[ResponseExample, ...]
    pronunciation: str              # 🗣️ 발음 가이드 (한국어 표기)
    etymology: str                  # 📌 어원·왜 이렇게 — 1~2문장


# ============================================================
# 테마별 SYSTEM 메시지.
# ============================================================
_THEME_LABEL: dict[Theme, str] = {
    Theme.SMALL_TALK: "💬 일상 잡담·인사 (어디서든 만나는 사람들과)",
    Theme.SOFT_SKILLS: "🤝 부탁·거절·사과 (정중하게 의사 전달)",
    Theme.REACTIONS: "😊 감정·반응 (맞장구·공감·솔직한 감정)",
    Theme.IDIOMS: "🎯 미국식 관용구 (자연스러운 표현·슬랭)",
}

_THEME_SCOPE: dict[Theme, str] = {
    Theme.SMALL_TALK: (
        "미국인이 어디서나 매일 쓰는 짧은 인사·잡담 표현. "
        "예: \"How's it going?\", \"Can't complain\", \"Same here\", \"Long time no see\", "
        "\"How's your day so far?\", \"What's new with you?\", \"Take care\". "
        "상황은 가게 계산대·이웃·바리스타·동료·낯선 사람 등 어디서든 가능. "
        "직업·장소 의존 표현 회피 — 누구나 들으면 \"아 이거 자주 듣는다\" 싶은 것."
    ),
    Theme.SOFT_SKILLS: (
        "부탁·거절·사과를 부드럽게 전달하는 미국식 표현. "
        "예: \"Would you mind ~ing?\", \"I'd appreciate it if ...\", \"I'm afraid I can't\", "
        "\"My bad\", \"No worries\", \"Could I ask a favor?\", \"I'll have to pass\". "
        "미국 사회에서 직설보다 한 단계 부드러움이 중요한 상황을 매끄럽게 풀어가는 어휘."
    ),
    Theme.REACTIONS: (
        "대화 중 자연스러운 반응·맞장구·감정 표현. "
        "예: \"No kidding!\", \"That makes sense\", \"Tell me about it\", \"You bet\", "
        "\"I'm beat\", \"I'm thrilled\", \"Honestly...\", \"To be fair...\". "
        "한국어 직역만 익히면 어색한 — 미국 회화 특유의 짧고 임팩트 있는 응답 표현 중심."
    ),
    Theme.IDIOMS: (
        "미국 일상 대화에서 매일 듣지만 직역으론 이해 안 되는 관용구. "
        "예: \"Piece of cake\", \"Hit the road\", \"On the same page\", \"Break the ice\", "
        "\"Out of the blue\", \"Lucked out\", \"Up in the air\", \"Down to earth\". "
        "어원·문화 배경을 함께 알려주면 한 번에 각인되는 영어 회화의 핵심 표현."
    ),
}


_DEFAULT_LEARNER_PROFILE = (
    "- 영어회화 중급(B1) 수준 학습자\n"
    "- 가게·이웃·병원·학교·관공서 등 어디서든 자연스럽게 대화하고 싶음\n"
    "- 너무 초급은 지루, 너무 고급은 좌절 — 딱 \"오늘 바로 써먹을 수 있는\" 표현\n"
    "- 학습 동기: 어원·미국 문화 스토리로 \"한 번 들으면 각인되는\" 표현 선호"
)


ENGLISH_PHRASE_SYSTEM = """당신은 미국 일상 영어를 가르치는 친근한 영어 코치입니다.

【학습자 프로필】
{learner_profile}

【오늘의 테마】
{theme_label}

【범위 가이드】
{theme_scope}

【출력 형식 — 반드시 유효한 JSON 만 출력 (markdown fence·주석·설명 없이)】
{{
  "phrase_en": "What have you been up to?",
  "phrase_ko": "그동안 뭘 하고 지내셨어요?",
  "context": "이 표현은 오랜만에 만난 사람에게 안부를 묻거나 근황을 물을 때 자주 쓰입니다. 미국 일상에서 자주 들리는 자연스러운 인사말이며, 형식적인 \\"How are you?\\" 보다 친근하고 진심 있는 관심을 표현합니다.",
  "scenarios": [
    {{
      "setup": "동네 커피숍에서 우연히 만난 이웃에게",
      "english": "Hello, Sarah! Long time no see. What have you been up to?",
      "korean": "안녕하세요, 사라! 오랜만이에요. 그동안 뭘 하고 지내셨어요?"
    }},
    {{
      "setup": "마트 계산대에서 친해진 캐셔에게 (오랜만에 만났을 때)",
      "english": "Hey, Mike! Good to see you again. What have you been up to?",
      "korean": "안녕, 마이크! 오랜만에 만나서 반가워. 그동안 뭐 하고 지냈어?"
    }}
  ],
  "responses": [
    {{
      "english": "Not much, just working a lot. How about you?",
      "korean": "별일 없어, 일이 많아. 너는 어때?"
    }},
    {{
      "english": "Oh, just enjoying the summer weather.",
      "korean": "그냥 여름 날씨를 즐기고 있어."
    }}
  ],
  "pronunciation": "와[강]트 해브 유 빈 업 투?",
  "etymology": "Up to 는 원래 '무엇을 하고 있는지' 또는 '어떤 활동에 참여하고 있는지'를 나타내는 구동사입니다. 과거부터 지금까지 어떤 일을 해오고 있었는지를 묻는 표현."
}}

【규칙】
- phrase_en: 학습 대상 영어 1문장 (또는 짧은 구문)
- phrase_ko: phrase_en 의 자연스러운 한국어 직역 (한 줄)
- context: 3~4문장 한국어 설명. 어떤 상황에서 자연스러운지, 왜 효과적인지, 미국 문화 한 조각 포함
- scenarios: 정확히 2개. setup (한국어 장면) + english (영어 예시) + korean (영어의 자연스러운 한국어 직역)
- responses: 정확히 2개. english (영어 응답) + korean (한국어 직역)
- pronunciation: 한국어 발음 표기 (음절·강세 표시 [강])
- etymology: 1~2문장. 어원·구문·문법의 재미있는 배경
- 표현은 미국에서 실제 자주 쓰는 자연스러운 회화 (교과서 X, 과한 슬랭 X)
- 직업·장소 의존 표현 회피 — 어디서든 쓸 수 있는 universal
- 정치·종교·민감 사회 이슈 회피
- ⚠️ 출력은 반드시 유효 JSON 만.  코드 펜스 (```) ·설명·markdown 사용 금지
- 한국어 문자열 내 큰따옴표는 \\" 로 escape

【최근 28일 발송 이력 — 절대 중복 금지】
{recent_phrases}

위 표현들과 다른 새로운 표현으로 JSON 작성하세요.
"""


def _format_recent(recent: Sequence[str]) -> str:
    if not recent:
        return "(첫 발송 — 중복 회피 필요 없음)"
    return "\n".join(f"  - {p}" for p in recent[:30])


# ============================================================
# JSON 파서 — LLM 출력의 다양한 quirks 흡수.
# ============================================================
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> dict | None:
    """LLM 출력에서 첫 JSON object 추출.

    우선순위:
      1) 통째로 json.loads 시도 (이상적 케이스)
      2) ```json ... ``` 코드 펜스 안 추출
      3) 첫 { 부터 마지막 } 까지 추출 (greedy)
    """
    text = (text or "").strip()
    if not text:
        return None

    # 1) 통째 시도
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 2) 코드 펜스
    m = _JSON_FENCE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 3) greedy 매칭
    m = _JSON_BLOCK.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return None


def parse_phrase_data(raw_text: str, theme: Theme) -> EnglishPhraseData | None:
    """LLM 출력 → EnglishPhraseData.  필수 필드 누락·타입 mismatch 시 None.

    호출자 (`LlmEnglishPhraseGenerator.generate`) 는 None 받으면 결정론 폴백 사용.
    """
    obj = _extract_json_object(raw_text)
    if obj is None:
        return None
    try:
        scenarios = tuple(
            Scenario(
                setup=str(s["setup"]),
                english=str(s["english"]),
                korean=str(s.get("korean", "")),  # 신규 필드 — 누락 시 빈 문자열 (renderer 가 생략)
            )
            for s in obj.get("scenarios", [])[:4]  # 안전 상한
        )
        responses = tuple(
            ResponseExample(english=str(r["english"]), korean=str(r["korean"]))
            for r in obj.get("responses", [])[:4]
        )
        return EnglishPhraseData(
            theme=theme,
            phrase_en=str(obj["phrase_en"]).strip(),
            phrase_ko=str(obj["phrase_ko"]).strip(),
            context=str(obj.get("context", "")).strip(),
            scenarios=scenarios,
            responses=responses,
            pronunciation=str(obj.get("pronunciation", "")).strip(),
            etymology=str(obj.get("etymology", "")).strip(),
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("phrase_data 파싱 실패 (필드 누락·타입 mismatch): %s", exc)
        return None


# ============================================================
# 결정론 폴백 — LLM 3단 모두 실패 시.
# ============================================================
_FALLBACK_BY_THEME: dict[Theme, EnglishPhraseData] = {
    Theme.SMALL_TALK: EnglishPhraseData(
        theme=Theme.SMALL_TALK,
        phrase_en="How's it going?",
        phrase_ko="어떻게 지내요? 요즘 좀 어때요?",
        context=(
            "How are you 의 캐주얼 버전입니다. 친구·동료·가게 점원·이웃 누구에게나 통합니다. "
            "사실 진짜 안부를 묻는 게 아니라 인사말 그 자체에 가깝습니다. "
            "미국인은 \"Hey, how's it going?\" 한 마디로 어색함을 풀고 대화를 시작합니다."
        ),
        scenarios=(
            Scenario(setup="카페에서 바리스타에게",
                     english="Hey, how's it going? Just a small coffee, please.",
                     korean="안녕하세요, 좀 어떠세요? 작은 커피 한 잔 부탁드려요."),
            Scenario(setup="이웃에게 아침 인사",
                     english="Morning! How's it going?",
                     korean="좋은 아침! 잘 지내요?"),
        ),
        responses=(
            ResponseExample(english="Pretty good, thanks. How about you?",
                            korean="잘 지내요, 고마워요. 당신은요?"),
            ResponseExample(english="Not bad, can't complain.",
                            korean="나쁘지 않아요. 불평할 게 없네요."),
        ),
        pronunciation="[하]우즈잇 [고]잉",
        etymology=(
            "How is it going 의 축약입니다. it 은 \"요즘 사정 전반\" 을 가리킵니다. "
            "응답은 Pretty good / Not bad / Can't complain 셋 중 하나면 거의 다 통합니다."
        ),
    ),
    Theme.SOFT_SKILLS: EnglishPhraseData(
        theme=Theme.SOFT_SKILLS,
        phrase_en="Would you mind helping me with this?",
        phrase_ko="혹시 이거 도와주시는 거 괜찮으세요?",
        context=(
            "Can you help me 보다 한 단계 정중한 표현입니다. 직역하면 \"꺼리시나요?\" 라 부담을 줄이는 뉘앙스입니다. "
            "사무실 동료·이웃·낯선 사람 누구에게나 자연스럽고, 거절당해도 어색하지 않습니다. "
            "미국에서 부탁은 직설보다 한 단계 부드럽게 묻는 게 사회적 매너입니다."
        ),
        scenarios=(
            Scenario(setup="이웃에게 택배 부탁",
                     english="Would you mind keeping an eye on my package?",
                     korean="혹시 제 택배 좀 봐주시는 거 괜찮으세요?"),
            Scenario(setup="동료에게 파일 재전송 부탁",
                     english="Would you mind sending that file again?",
                     korean="그 파일 한 번 더 보내주실 수 있을까요?"),
        ),
        responses=(
            ResponseExample(english="No, not at all. Happy to help.",
                            korean="전혀요. 도와드릴게요."),
            ResponseExample(english="Sure thing, I can do that.",
                            korean="그럼요, 할 수 있어요."),
        ),
        pronunciation="[우쥬] 마인드 [헬]핑 미",
        etymology=(
            "mind 는 \"신경 쓰다·꺼리다\" 라는 동사입니다. 직역 \"꺼리시나요?\" → 의역 \"괜찮으시면 ~해 주실래요?\". "
            "응답 시 No (\"안 꺼려요\" = 해줄게요) 가 OK 인 게 헷갈리니 주의."
        ),
    ),
    Theme.REACTIONS: EnglishPhraseData(
        theme=Theme.REACTIONS,
        phrase_en="That makes sense.",
        phrase_ko="그게 말이 되네요. 이해가 가요.",
        context=(
            "상대 설명을 듣고 \"아, 그래서 그렇구나\" 라고 자연스럽게 동의할 때 미국인이 가장 자주 쓰는 표현입니다. "
            "I understand 보다 부드럽고, I see 보다 적극적인 공감을 전달합니다. "
            "회의·전화 상담·일상 대화 어디서든 톤을 부드럽게 이어가는 윤활제 같은 한 마디."
        ),
        scenarios=(
            Scenario(setup="동료 설명을 듣고",
                     english="Oh, that makes sense. Thanks for explaining.",
                     korean="아, 그러네요. 설명해주셔서 감사해요."),
            Scenario(setup="안내를 들었을 때",
                     english="That makes sense. I'll follow that approach.",
                     korean="이해됐어요. 그 방법대로 할게요."),
        ),
        responses=(
            ResponseExample(english="Glad it makes sense! Let me know if you have questions.",
                            korean="이해되었다니 다행이에요. 질문 있으면 알려주세요."),
            ResponseExample(english="Right, exactly. That's the idea.",
                            korean="맞아요, 바로 그거예요."),
        ),
        pronunciation="[댓] [메]잌스 쎈스",
        etymology=(
            "make sense 는 \"의미를 만들다\" → \"이치에 맞다\" 라는 1500년대부터의 관용구입니다. "
            "반대는 That doesn't make sense (이해 안 돼) — 같은 패턴으로 응용."
        ),
    ),
    Theme.IDIOMS: EnglishPhraseData(
        theme=Theme.IDIOMS,
        phrase_en="It's a piece of cake.",
        phrase_ko="이건 케이크 한 조각이에요. (식은 죽 먹기예요)",
        context=(
            "쉬운 일을 강조할 때 미국인이 매일 쓰는 표현입니다. easy 보다 입에 잘 붙고 친근합니다. "
            "동료가 \"이거 어렵지 않아?\" 물으면 \"No, it's a piece of cake\" 한 마디로 가볍게 안심시킵니다. "
            "케이크 한 조각 먹는 게 누구에게나 쉽다는 비유에서 출발했습니다."
        ),
        scenarios=(
            Scenario(setup="부탁받았을 때",
                     english="Sure, that's a piece of cake. I'll do it now.",
                     korean="그럼요, 식은 죽 먹기예요. 지금 바로 할게요."),
            Scenario(setup="일 끝났을 때",
                     english="Done already? Yeah, piece of cake.",
                     korean="벌써 끝났냐고요? 네, 식은 죽 먹기였어요."),
        ),
        responses=(
            ResponseExample(english="Oh nice, you make it sound easy.",
                            korean="와, 쉬워 보이게 말하시네요."),
            ResponseExample(english="Cool, thanks for handling that.",
                            korean="좋아요, 처리해주셔서 감사해요."),
        ),
        pronunciation="잇츠 어 [피]스 어브 [케]잌",
        etymology=(
            "1930년대 미국 공군 파일럿이 \"쉬운 임무\" 를 가리키며 처음 썼다는 설이 유력합니다. "
            "비슷한 표현 → easy as pie, walk in the park 같이 묶어 외우면 효율적."
        ),
    ),
}


# ============================================================
# LLM Generator (Protocol 충족).
# ============================================================
class LlmEnglishPhraseGenerator:
    """LLM 1회 호출 + JSON 파싱 + 렌더링 (TTS 옵션).

    Generator 가 최종 Telegram HTML body 까지 만들어 EnglishPhrase.body_md 에 담는다
    (rendering 분리하지 않은 이유 — caller 단순화: store/telegram 만 주입하면 됨).

    TTS 인프라 (synthesizer + uploader) 미주입 시 음성 링크 없이 메시지만 발송.
    """

    def __init__(
        self,
        *,
        synthesizer: "TtsSynthesizer | None" = None,
        uploader: "TtsUploader | None" = None,
    ) -> None:
        self._synthesizer = synthesizer
        self._uploader = uploader

    def generate(
        self,
        inp: EnglishPhraseInputs,
        providers: Sequence[Provider],
        *,
        config: ChainConfig | None = None,
    ) -> EnglishPhrase:
        # 지연 import — renderer 가 english_phrase_llm 의 dataclass 를 사용하므로
        # 모듈 로딩 순환 회피 (renderer 는 EnglishPhraseData 의존, 본 모듈은 renderer 의존).
        from execution.github_actions.english_phrase_renderer import render_message

        # 학습자 프로필은 env (USER_PERSONA_PROMPT) 우선, 없으면 generic 기본값.
        from src.config.user_profile import load_user_persona_system_prompt
        learner_profile = load_user_persona_system_prompt() or _DEFAULT_LEARNER_PROFILE
        prompt = ENGLISH_PHRASE_SYSTEM.format(
            learner_profile=learner_profile,
            theme_label=_THEME_LABEL[inp.theme],
            theme_scope=_THEME_SCOPE[inp.theme],
            recent_phrases=_format_recent(inp.recent_phrases),
        )
        payload = {"english_phrase": {"theme": inp.theme.value}}
        cfg = config or ChainConfig(max_retries_per_provider=1, base_backoff_seconds=1.0)

        data: EnglishPhraseData | None = None
        try:
            result = call_with_fallback(
                payload=payload, prompt=prompt,
                providers=list(providers), drug_map={}, config=cfg,
            )
            data = parse_phrase_data(result.text or "", inp.theme)
        except AllProvidersFailed:
            logger.warning(
                "영어회화 LLM 3단 폴백 모두 실패 — 결정론 폴백 사용 (theme=%s)",
                inp.theme.value,
            )

        if data is None:
            logger.warning(
                "영어회화 JSON 파싱 실패 — 결정론 폴백 사용 (theme=%s)",
                inp.theme.value,
            )
            data = _FALLBACK_BY_THEME[inp.theme]

        # JSON → HTML 렌더링 (TTS 옵션 포함).  실패 시 음성 링크 없이 발송.
        body_md = render_message(
            data,
            synthesizer=self._synthesizer,
            uploader=self._uploader,
        )
        return EnglishPhrase(
            theme=data.theme,
            phrase_en=data.phrase_en,
            body_md=body_md,
        )


# Forward declarations — renderer 의 Protocol 재사용 (실 모듈 import 는 generate 내부에서).
class TtsSynthesizer(Protocol):
    def __call__(self, text: str) -> bytes: ...


class TtsUploader(Protocol):
    def exists(self, text: str) -> bool: ...
    def upload_or_get_url(self, text: str, audio_bytes: bytes) -> str: ...
    def get_url(self, text: str) -> str: ...


def _data_to_dict(data: EnglishPhraseData) -> dict:
    """EnglishPhraseData → dict (JSON 직렬화 가능).  renderer 의 역직렬화 진입점."""
    return {
        "theme": data.theme.value,
        "phrase_en": data.phrase_en,
        "phrase_ko": data.phrase_ko,
        "context": data.context,
        "scenarios": [
            {"setup": s.setup, "english": s.english, "korean": s.korean}
            for s in data.scenarios
        ],
        "responses": [{"english": r.english, "korean": r.korean} for r in data.responses],
        "pronunciation": data.pronunciation,
        "etymology": data.etymology,
    }


def dict_to_data(d: dict, theme: Theme | None = None) -> EnglishPhraseData:
    """renderer 가 body_md (JSON 문자열) 을 역직렬화할 때 호출.

    theme 인자가 있으면 우선, 없으면 d['theme'] 에서.
    """
    t = theme or Theme(d.get("theme", "small_talk"))
    return EnglishPhraseData(
        theme=t,
        phrase_en=str(d["phrase_en"]),
        phrase_ko=str(d.get("phrase_ko", "")),
        context=str(d.get("context", "")),
        scenarios=tuple(
            Scenario(
                setup=str(s["setup"]),
                english=str(s["english"]),
                korean=str(s.get("korean", "")),
            )
            for s in d.get("scenarios", [])
        ),
        responses=tuple(
            ResponseExample(english=str(r["english"]), korean=str(r["korean"]))
            for r in d.get("responses", [])
        ),
        pronunciation=str(d.get("pronunciation", "")),
        etymology=str(d.get("etymology", "")),
    )


__all__ = [
    "ENGLISH_PHRASE_SYSTEM",
    "EnglishPhraseData",
    "LlmEnglishPhraseGenerator",
    "ResponseExample",
    "Scenario",
    "dict_to_data",
    "parse_phrase_data",
]
