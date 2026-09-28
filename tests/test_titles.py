"""Tests for the AI title: cleaning, the request, and how titles are shown."""

import httpx
import openai
import pytest

import config
import openai_api
import titles
from exceptions import OpenAIAccountError, TitleError


class FakeChatClient:
    """Answers with ``answer`` (or raises it) and keeps the request it got."""

    def __init__(self, answer):
        self.requests = []
        self.closed = False
        self._answer = answer
        completions = type("Completions", (), {"create": self.create})()
        self.chat = type("Chat", (), {"completions": completions})()

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if isinstance(self._answer, Exception):
            raise self._answer
        tokens = {"prompt_tokens": 3_000, "completion_tokens": 12}
        if self._answer is None:
            return type("Response", (), {"choices": [], "usage": tokens})()
        message = type("Message", (), {"content": self._answer})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice], "usage": tokens})()

    def close(self):
        self.closed = True


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

    title, record = titles.make_title("We went over the budget.", "key", "gpt-5.4-nano")

    assert title == "Budget review"
    [request] = client.requests
    assert request["model"] == "gpt-5.4-nano"
    assert request["messages"][-1]["content"] == "We went over the budget."
    assert record["kind"] == "title"
    # $0.20 per 1M input tokens, $1.25 per 1M output tokens.
    assert record["cost_usd"] == pytest.approx((3_000 * 0.20 + 12 * 1.25) / 1e6)
    assert client.closed


def test_a_long_transcript_is_sent_as_its_beginning_and_end(monkeypatch):
    client = _use(monkeypatch, FakeChatClient("Long meeting"))
    transcript = "A" * config.TITLE_MAX_TRANSCRIPT_CHARS + "B" * 1000

    titles.make_title(transcript, "key", "gpt-5.4-nano")

    sent = client.requests[0]["messages"][-1]["content"]
    assert sent.startswith("A") and sent.endswith("B")
    assert len(sent) < config.TITLE_MAX_TRANSCRIPT_CHARS + 20


def test_an_empty_transcript_is_not_sent(monkeypatch):
    client = _use(monkeypatch, FakeChatClient("Nothing"))

    with pytest.raises(TitleError):
        titles.make_title("  \n", "key", "gpt-5.4-nano")
    assert client.requests == []


def test_an_answer_without_a_title_is_an_error_but_still_counted(monkeypatch):
    _use(monkeypatch, FakeChatClient(None))

    with pytest.raises(TitleError) as caught:
        titles.make_title("Some talk.", "key", "gpt-5.4-nano")
    assert caught.value.usage_record["input_tokens"] == 3_000


def test_no_credit_is_an_account_error(monkeypatch):
    body = {"message": "no", "code": "insufficient_quota", "type": "insufficient_quota"}
    response = httpx.Response(
        429, request=httpx.Request("POST", "https://api.openai.com/v1/x"), json=body
    )
    refused = openai.RateLimitError("Error code: 429", response=response, body=body)
    client = _use(monkeypatch, FakeChatClient(refused))

    with pytest.raises(OpenAIAccountError, match="credit exhausted"):
        titles.make_title("Some talk.", "key", "gpt-5.4-nano")
    assert client.closed


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
