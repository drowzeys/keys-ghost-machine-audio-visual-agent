"""The brain's prompt keeps the same start from turn to turn, so a server's prompt cache can resume each turn: the
system prompt carries the date only, the clock time goes with a newest user message that asks about the time, and the
warm request is built from the same messages as the turn."""

import datetime

from voice_agent import server


def conversation(*history):
    talk = object.__new__(server.Conversation)
    talk.history = [{"role": "system", "content": "Be brief."}, *history]
    talk.name = "Boo"
    return talk


def test_system_prompt_has_the_date_but_no_clock():
    system = conversation({"role": "user", "content": "Make a fox clip."}).messages()[0]["content"]
    today = datetime.datetime.now().astimezone()
    assert f"Today is {today:%A %-d %B %Y}." in system
    assert "It is now" not in system and ":" not in system.split("Today is")[1].split(".")[0]


def test_system_prompt_is_the_same_in_a_later_minute(monkeypatch):
    talk = conversation({"role": "user", "content": "Make a fox clip."})
    first = talk.messages()[0]["content"]

    real = datetime.datetime

    class Later(real):
        @classmethod
        def now(cls, tz=None):
            return real.now(tz) + datetime.timedelta(minutes=3)
    monkeypatch.setattr(server.datetime, "datetime", Later)
    assert talk.messages()[0]["content"] == first


def test_clock_time_goes_with_a_question_about_the_time():
    asked = conversation({"role": "user", "content": "What time is it?"}).messages()
    assert "(It is now " in asked[-1]["content"] and asked[-1]["content"].startswith("What time is it?")
    other = conversation({"role": "user", "content": "Make a fox clip."}).messages()
    assert other[-1]["content"] == "Make a fox clip."


def test_last_replaces_the_newest_user_message_and_can_carry_the_time():
    talk = conversation({"role": "user", "content": "what is this"})
    assert talk.messages(last="what is this [camera: a red mug]")[-1]["content"] == "what is this [camera: a red mug]"
    timed = talk.messages(last="is it late now [camera: a dark window]")[-1]["content"]
    assert timed.startswith("is it late now [camera: a dark window] (It is now ")


def test_history_is_not_changed():
    talk = conversation({"role": "user", "content": "What time is it?"})
    talk.messages()
    assert talk.history[-1]["content"] == "What time is it?"
