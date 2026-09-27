"""Serbian Cyrillic to Latin transliteration.

Serbian is written in both scripts, and the transcription API picks one per
request: without a language hint, a long meeting sent in chunks came back as
alternating blocks of Latin and Cyrillic, and a few phone calls entirely in
Cyrillic. A deterministic 1:1 transliteration can make a transcript Latin
regardless of what the model chose (opt-in via ``SERBIAN_LATIN``, off by default,
so Macedonian stays Cyrillic). It runs only on text that is recognisably
Serbian, so Russian or Bulgarian Cyrillic is never rewritten. Pure functions,
no I/O.
"""

# Letters that exist in Serbian Cyrillic but not in Russian: their presence is
# what marks a text as Serbian.
_SERBIAN_ONLY = set("ЂђЋћЏџЉљЊњЈј")
# Letters Serbian never uses. A few do appear in Serbian model output as
# glitches (й, ќ, ѓ were seen), so they are transliterated too, but a text where
# they outnumber the Serbian-only letters is treated as another language.
_NON_SERBIAN = set("ЫыЭэЪъЁёЩщЯяЮюЙйЬьІіЇїЄєҐґ")

_LOWER = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "ђ": "đ",
    "е": "e",
    "ж": "ž",
    "з": "z",
    "и": "i",
    "ј": "j",
    "к": "k",
    "л": "l",
    "љ": "lj",
    "м": "m",
    "н": "n",
    "њ": "nj",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "ћ": "ć",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "c",
    "ч": "č",
    "џ": "dž",
    "ш": "š",
    # Macedonian letters the model sometimes emits for Serbian sounds.
    "ќ": "ć",
    "ѓ": "đ",
    "ѕ": "dz",
    # Russian and Ukrainian letters seen as glitches in Serbian output.
    "й": "j",
    "щ": "šč",
    "я": "ja",
    "ю": "ju",
    "ё": "jo",
    "є": "je",
    "ї": "ji",
    "ы": "i",
    "э": "e",
    "і": "i",
    "ґ": "g",
    "ъ": "",
    "ь": "",
}
_UPPER = {cyr.upper(): lat[:1].upper() + lat[1:] for cyr, lat in _LOWER.items() if lat}
_UPPER.update({cyr.upper(): "" for cyr, lat in _LOWER.items() if not lat})


def is_serbian_cyrillic(text: str) -> bool:
    """Return True when the text contains Cyrillic that is recognisably Serbian.

    Args:
        text: Any text.

    Returns:
        True if Serbian-only letters (ђ ћ џ љ њ ј) are present and outnumber
        letters Serbian never uses.
    """
    serbian = sum(1 for char in text if char in _SERBIAN_ONLY)
    foreign = sum(1 for char in text if char in _NON_SERBIAN)
    return serbian > 0 and serbian > foreign


def _upper_digraph(text: str, index: int, latin: str) -> str:
    """Choose ``LJ`` or ``Lj`` for an uppercase digraph letter by its context.

    An all-caps word (``ЉУБАВ``) needs ``LJUBAV``; a capitalised one
    (``Љубав``) needs ``Ljubav``.

    Args:
        text: The whole text.
        index: Position of the uppercase letter.
        latin: Its title-case transliteration, e.g. ``"Lj"``.

    Returns:
        The transliteration in the right case.
    """
    following = text[index + 1] if index + 1 < len(text) else ""
    preceding = text[index - 1] if index > 0 else ""
    if following.isalpha():
        return latin.upper() if following.isupper() else latin
    return latin.upper() if preceding.isalpha() and preceding.isupper() else latin


def cyrillic_to_latin(text: str) -> str:
    """Transliterate Cyrillic letters to Serbian Latin, leaving the rest as is.

    Args:
        text: Text in Cyrillic, Latin or a mix of both.

    Returns:
        The text with every Cyrillic letter in Latin script.
    """
    out: list[str] = []
    for index, char in enumerate(text):
        if char in _LOWER:
            out.append(_LOWER[char])
        elif char in _UPPER:
            latin = _UPPER[char]
            out.append(_upper_digraph(text, index, latin) if len(latin) > 1 else latin)
        else:
            out.append(char)
    return "".join(out)


def to_latin_if_serbian(text: str) -> str:
    """Transliterate the text only when it is recognisably Serbian Cyrillic.

    Args:
        text: A transcript or subtitle document.

    Returns:
        The Latin-script text, or the input unchanged.
    """
    return cyrillic_to_latin(text) if is_serbian_cyrillic(text) else text
