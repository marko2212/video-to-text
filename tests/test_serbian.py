"""Tests for the Serbian Cyrillic to Latin transliteration."""

import serbian


def test_the_whole_alphabet_maps_one_to_one():
    cyrillic = "абвгдђежзијклљмнњопрстћуфхцчџш"
    latin = "abvgdđežzijklljmnnjoprstćufhcčdžš"
    assert serbian.cyrillic_to_latin(cyrillic) == latin


def test_digraphs_follow_the_case_of_the_word():
    assert serbian.cyrillic_to_latin("Љубав") == "Ljubav"
    assert serbian.cyrillic_to_latin("ЉУБАВ") == "LJUBAV"
    assert serbian.cyrillic_to_latin("Његош") == "Njegoš"
    assert serbian.cyrillic_to_latin("ЏЕП") == "DŽEP"
    assert serbian.cyrillic_to_latin("Џеп") == "Džep"


def test_a_single_capital_digraph_letter_takes_title_case():
    assert serbian.cyrillic_to_latin("Љ.") == "Lj."


def test_latin_text_and_punctuation_are_untouched():
    text = "Sastanak u 10:30, (0:05) — Q3 rezultati."
    assert serbian.cyrillic_to_latin(text) == text


def test_mixed_script_becomes_latin_throughout():
    text = "Dobar dan. Данас причамо о буџету."
    assert serbian.to_latin_if_serbian(text) == "Dobar dan. Danas pričamo o budžetu."


def test_macedonian_and_russian_glitch_letters_seen_in_output():
    # ќ, ѓ and й appeared in real Serbian transcripts.
    assert serbian.cyrillic_to_latin("ќе ѓак мој") == "će đak moj"


def test_serbian_is_recognised_by_its_own_letters():
    assert serbian.is_serbian_cyrillic("Ћао, како си, љубави?")


def test_russian_is_left_alone():
    russian = "Привет, как дела? Всё хорошо, спасибо."
    assert not serbian.is_serbian_cyrillic(russian)
    assert serbian.to_latin_if_serbian(russian) == russian


def test_cyrillic_without_serbian_only_letters_is_left_alone():
    # Could be Serbian or Russian; rewriting it would be a guess.
    text = "English meeting notes: сада"
    assert serbian.to_latin_if_serbian(text) == text


def test_english_is_left_alone():
    text = "We shipped the release on Tuesday."
    assert serbian.to_latin_if_serbian(text) == text
