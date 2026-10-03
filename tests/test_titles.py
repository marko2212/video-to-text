"""Tests for the AI title: cleaning, the request, and how titles are shown."""

import httpx
import openai
import pytest

import config
import openai_api
import titles
from exceptions import OpenAIAccountError, TitleError


class FakeChatClient:
    """Answers with ``answer`` (or raises it) and keeps the requests it got.

    A list gives one answer per request, in order.
    """

    def __init__(self, answer):
        self.requests = []
        self.closed = False
        self._answers = list(answer) if isinstance(answer, list) else None
        self._answer = answer
        completions = type("Completions", (), {"create": self.create})()
        self.chat = type("Chat", (), {"completions": completions})()

    def create(self, **kwargs):
        # A copy: the caller extends its message list for a second request.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        answer = self._answers.pop(0) if self._answers is not None else self._answer
        if isinstance(answer, Exception):
            raise answer
        tokens = {"prompt_tokens": 3_000, "completion_tokens": 12}
        if answer is None:
            return type("Response", (), {"choices": [], "usage": tokens})()
        message = type("Message", (), {"content": answer})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice], "usage": tokens})()

    def close(self):
        self.closed = True


_NANO = "gpt-5.4-nano"


def _use(monkeypatch, client):
    monkeypatch.setattr(openai_api, "make_client", lambda key: client)
    return client


# --- clean_title ------------------------------------------------------------


def test_a_plain_answer_is_kept_as_it_is():
    assert titles.clean_title("Dogovor o migraciji baze") == "Dogovor o migraciji baze"


def test_quotes_a_label_and_a_trailing_period_are_removed():
    answer = 'Title: "Budget review for Q3".'
    assert titles.clean_title(answer) == "Budget review for Q3"
    assert titles.clean_title("Naslov: „Plan puštanja u rad“") == "Plan puštanja u rad"
    assert titles.clean_title("**Наслов:** Преглед буџета") == "Преглед буџета"


def test_only_the_first_line_of_the_answer_is_used():
    answer = "\n\nRelease plan for the mobile app\nThis meeting covered…"
    assert titles.clean_title(answer) == "Release plan for the mobile app"


def test_characters_windows_forbids_in_file_names_are_removed():
    # The title becomes the name of a downloaded file.
    title = titles.clean_title("Q3 plan: sales/marketing <draft> | v2?*")
    assert title == "Q3 plan - sales marketing draft v2"
    for char in '<>:"/|?*':
        assert char not in title
    assert chr(92) not in titles.clean_title(f"C{chr(92)}D migration")


def test_a_long_answer_is_cut_at_a_word():
    title = titles.clean_title("word " * 40)
    assert len(title) <= config.TITLE_MAX_CHARS
    assert title.endswith("word")


def test_a_reserved_windows_name_is_not_used_alone():
    assert titles.clean_title("CON") == "CON recording"


def test_an_empty_answer_is_no_title():
    assert titles.clean_title("") is None
    assert titles.clean_title('  "" . ') is None


# --- make_title --------------------------------------------------------------


def test_a_title_is_made_from_the_transcript_and_priced(monkeypatch):
    client = _use(monkeypatch, FakeChatClient("Budget review"))

    title, records = titles.make_title("We went over the budget.", "key", _NANO)

    assert title == "Budget review"
    [request] = client.requests
    assert request["model"] == "gpt-5.4-nano"
    assert request["messages"][-1]["content"].startswith("We went over the budget.")
    [record] = records
    assert record["kind"] == "title"
    # $0.20 per 1M input tokens, $1.25 per 1M output tokens.
    assert record["cost_usd"] == pytest.approx((3_000 * 0.20 + 12 * 1.25) / 1e6)
    assert client.closed


def test_a_long_transcript_is_sent_as_its_beginning_and_end(monkeypatch):
    client = _use(monkeypatch, FakeChatClient("Long meeting"))
    transcript = "A" * config.TITLE_MAX_TRANSCRIPT_CHARS + "B" * 1000

    titles.make_title(transcript, "key", "gpt-5.4-nano")

    sent = client.requests[0]["messages"][-1]["content"]
    excerpt = sent.rsplit("\n\n---\n", 1)[0]
    assert excerpt.startswith("A") and excerpt.endswith("B")
    assert len(excerpt) < config.TITLE_MAX_TRANSCRIPT_CHARS + 20


def test_an_empty_transcript_is_not_sent(monkeypatch):
    client = _use(monkeypatch, FakeChatClient("Nothing"))

    with pytest.raises(TitleError):
        titles.make_title("  \n", "key", "gpt-5.4-nano")
    assert client.requests == []


def test_an_answer_without_a_title_is_an_error_but_still_counted(monkeypatch):
    _use(monkeypatch, FakeChatClient(None))

    with pytest.raises(TitleError) as caught:
        titles.make_title("Some talk.", "key", "gpt-5.4-nano")
    [record] = caught.value.usage_records
    assert record["input_tokens"] == 3_000


def _no_credit() -> openai.RateLimitError:
    body = {"message": "no", "code": "insufficient_quota", "type": "insufficient_quota"}
    response = httpx.Response(
        429, request=httpx.Request("POST", "https://api.openai.com/v1/x"), json=body
    )
    return openai.RateLimitError("Error code: 429", response=response, body=body)


def test_no_credit_is_an_account_error(monkeypatch):
    client = _use(monkeypatch, FakeChatClient(_no_credit()))

    with pytest.raises(OpenAIAccountError, match="credit exhausted"):
        titles.make_title("Some talk.", "key", "gpt-5.4-nano")
    assert client.closed


# --- the transcript's script ---------------------------------------------------


def test_letters_are_counted_by_script():
    scripts = titles.letter_scripts("Čačak i Шабац, 部署 2026!")
    assert scripts == {"LATIN": 6, "CYRILLIC": 5, "CJK": 2}


def test_a_title_in_a_script_the_transcript_never_uses_is_foreign():
    # The owner's English meeting got this title from gpt-5.4-nano (2026-09-29).
    english = titles.letter_scripts("We need the TIS subscriptions for Dev and UAT.")
    assert titles.foreign_scripts("部署自动化与TIS订阅环境选型讨论", english) == ["CJK"]
    assert titles.foreign_scripts("TIS subscriptions for Dev and UAT", english) == []


def test_a_latin_name_in_a_cyrillic_title_is_not_foreign():
    serbian = titles.letter_scripts("Прешли смо на Azure претплате.")
    assert titles.foreign_scripts("Прелазак на Azure", serbian) == []


def test_a_transcript_without_letters_accepts_any_title():
    assert titles.foreign_scripts("Anything", titles.letter_scripts("1, 2, 3…")) == []


def test_one_stray_chinese_line_does_not_let_a_chinese_title_through():
    # Whisper writes Chinese subtitle credits into silence; one such line must
    # not make Chinese a script of an English meeting.
    english = titles.letter_scripts("We went over the release plan. " * 300)
    english += titles.letter_scripts("请不吝点赞 订阅 转发 打赏支持明镜与点点栏目")

    assert titles.foreign_scripts("部署自动化与TIS订阅环境选型讨论", english) == ["CJK"]
    # A Latin title with a Chinese name in it is still fine.
    assert titles.foreign_scripts("Release plan for 明镜", english) == []


def test_either_script_of_a_mixed_serbian_meeting_is_accepted():
    # The API picks the script per chunk, so a Serbian meeting can be part
    # Latin, part Cyrillic.
    mixed = titles.letter_scripts("Dogovor o migraciji baze. " * 30)
    mixed += titles.letter_scripts("Договор о миграцији базе. " * 10)

    assert titles.foreign_scripts("Migracija baze", mixed) == []
    assert titles.foreign_scripts("Миграција базе", mixed) == []


def test_letter_like_symbols_count_as_the_letters_they_stand_for():
    # Found by review: each of these read as a script of its own ("MASCULINE",
    # "FULLWIDTH", "MODIFIER") and would have sent a good title back.
    latin = titles.letter_scripts("First quarter review of the AI roadmap.")
    for title in ["1º trimestre", "2ª reunião", "ＡＩ roadmap", "Q3 ʼ26 review"]:
        assert titles.foreign_scripts(title, latin) == [], title
    assert titles.letter_scripts("ª º ＡＩ ʼ") == {"LATIN": 4}


def test_japanese_counts_as_one_script():
    scripts = titles.letter_scripts("会議のスケジュールについて話しました")
    assert set(scripts) == {"CJK"}


def test_a_title_request_has_its_own_short_timeout(monkeypatch):
    client = _use(monkeypatch, FakeChatClient("Budget review"))

    titles.make_title("We went over the budget.", "key", _NANO)

    assert client.requests[0]["timeout"] == config.TITLE_TIMEOUT_SECONDS


def test_the_request_ends_by_naming_the_transcript_s_script(monkeypatch):
    # A small model follows what it read last; the reminder comes after the
    # transcript, not only in the instructions before it.
    client = _use(monkeypatch, FakeChatClient("Преглед буџета"))

    titles.make_title("Прегледали смо буџет за Q3.", "key", _NANO)

    sent = client.requests[0]["messages"][-1]["content"]
    assert sent.startswith("Прегледали смо буџет за Q3.")
    assert sent.endswith("written in Cyrillic script like the transcript.")


def test_a_title_in_a_foreign_script_is_sent_back_once(monkeypatch):
    client = _use(
        monkeypatch, FakeChatClient(["部署自动化讨论", "Deployment automation"])
    )

    title, records = titles.make_title("We talked about deployments.", "key", _NANO)

    assert title == "Deployment automation"
    assert len(records) == 2  # both answers were paid for
    first, second = client.requests
    assert second["messages"][:2] == first["messages"]
    assert second["messages"][2] == {"role": "assistant", "content": "部署自动化讨论"}
    correction = second["messages"][3]
    assert correction["role"] == "user"
    assert "CJK script" in correction["content"]
    assert "in Latin script" in correction["content"]
    assert client.closed


def test_a_foreign_script_twice_is_no_title_but_both_answers_count(monkeypatch):
    client = _use(monkeypatch, FakeChatClient(["部署", "自动化"]))

    with pytest.raises(TitleError, match=r"CJK script twice.*in Latin") as caught:
        titles.make_title("We talked about deployments.", "key", _NANO)

    assert len(client.requests) == 2  # no third try
    assert len(caught.value.usage_records) == 2
    assert client.closed


def test_a_failure_on_the_second_try_still_counts_the_first(monkeypatch):
    _use(monkeypatch, FakeChatClient(["部署", _no_credit()]))

    with pytest.raises(TitleError, match="credit exhausted") as caught:
        titles.make_title("We talked about deployments.", "key", _NANO)

    [record] = caught.value.usage_records
    assert record["kind"] == "title"


# --- display_name and the cost hint ------------------------------------------


def test_the_mode_decides_how_a_title_is_shown():
    stem, title = "Meeting Recording", "Budget review"
    assert titles.display_name(stem, title, config.TITLE_MODE_REPLACE) == title
    assert (
        titles.display_name(stem, title, config.TITLE_MODE_APPEND)
        == "Meeting Recording - Budget review"
    )
    assert titles.display_name(stem, title, config.TITLE_MODE_OFF) is None


def test_a_row_without_a_title_keeps_its_usual_name():
    for mode in config.TITLE_MODES:
        assert titles.display_name("Meeting", None, mode) is None


def test_every_offered_title_model_has_a_price():
    # The setting shows a cost hint per model; a missing price hides it and
    # records the title's cost as unknown.
    for model in config.TITLE_MODELS:
        assert model in config.CHAT_PRICE_PER_MTOK
        assert titles.estimate_cost_per_hour(model) > 0


def test_an_hour_of_title_costs_under_a_cent_on_the_default_model():
    assert titles.estimate_cost_per_hour(config.DEFAULT_TITLE_MODEL) < 0.01
    assert titles.estimate_cost_per_hour("some-future-model") is None
