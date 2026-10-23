# basic text cleaners for the ACE step model
# I didn't copy the ones from the reference code because I didn't want to deal with the dependencies
# TODO: more languages than english?

import re

def japanese_to_romaji(japanese_text):
    """
    Convert Japanese hiragana and katakana to romaji (Latin alphabet representation).

    Args:
        japanese_text (str): Text containing hiragana and/or katakana characters

    Returns:
        str: The romaji (Latin alphabet) equivalent
    """
    # Dictionary mapping kana characters to their romaji equivalents
    kana_map = {
        # Katakana characters
        '\u30a2': 'a', '\u30a4': 'i', '\u30a6': 'u', '\u30a8': 'e', '\u30aa': 'o',
        '\u30ab': 'ka', '\u30ad': 'ki', '\u30af': 'ku', '\u30b1': 'ke', '\u30b3': 'ko',
        '\u30b5': 'sa', '\u30b7': 'shi', '\u30b9': 'su', '\u30bb': 'se', '\u30bd': 'so',
        '\u30bf': 'ta', '\u30c1': 'chi', '\u30c4': 'tsu', '\u30c6': 'te', '\u30c8': 'to',
        '\u30ca': 'na', '\u30cb': 'ni', '\u30cc': 'nu', '\u30cd': 'ne', '\u30ce': 'no',
        '\u30cf': 'ha', '\u30d2': 'hi', '\u30d5': 'fu', '\u30d8': 'he', '\u30db': 'ho',
        '\u30de': 'ma', '\u30df': 'mi', '\u30e0': 'mu', '\u30e1': 'me', '\u30e2': 'mo',
        '\u30e4': 'ya', '\u30e6': 'yu', '\u30e8': 'yo',
        '\u30e9': 'ra', '\u30ea': 'ri', '\u30eb': 'ru', '\u30ec': 're', '\u30ed': 'ro',
        '\u30ef': 'wa', '\u30f2': 'wo', '\u30f3': 'n',

        # Katakana voiced consonants
        '\u30ac': 'ga', '\u30ae': 'gi', '\u30b0': 'gu', '\u30b2': 'ge', '\u30b4': 'go',
        '\u30b6': 'za', '\u30b8': 'ji', '\u30ba': 'zu', '\u30bc': 'ze', '\u30be': 'zo',
        '\u30c0': 'da', '\u30c2': 'ji', '\u30c5': 'zu', '\u30c7': 'de', '\u30c9': 'do',
        '\u30d0': 'ba', '\u30d3': 'bi', '\u30d6': 'bu', '\u30d9': 'be', '\u30dc': 'bo',
        '\u30d1': 'pa', '\u30d4': 'pi', '\u30d7': 'pu', '\u30da': 'pe', '\u30dd': 'po',

        # Katakana combinations
        '\u30ad\u30e3': 'kya', '\u30ad\u30e5': 'kyu', '\u30ad\u30e7': 'kyo',
        '\u30b7\u30e3': 'sha', '\u30b7\u30e5': 'shu', '\u30b7\u30e7': 'sho',
        '\u30c1\u30e3': 'cha', '\u30c1\u30e5': 'chu', '\u30c1\u30e7': 'cho',
        '\u30cb\u30e3': 'nya', '\u30cb\u30e5': 'nyu', '\u30cb\u30e7': 'nyo',
        '\u30d2\u30e3': 'hya', '\u30d2\u30e5': 'hyu', '\u30d2\u30e7': 'hyo',
        '\u30df\u30e3': 'mya', '\u30df\u30e5': 'myu', '\u30df\u30e7': 'myo',
        '\u30ea\u30e3': 'rya', '\u30ea\u30e5': 'ryu', '\u30ea\u30e7': 'ryo',
        '\u30ae\u30e3': 'gya', '\u30ae\u30e5': 'gyu', '\u30ae\u30e7': 'gyo',
        '\u30b8\u30e3': 'ja', '\u30b8\u30e5': 'ju', '\u30b8\u30e7': 'jo',
        '\u30d3\u30e3': 'bya', '\u30d3\u30e5': 'byu', '\u30d3\u30e7': 'byo',
        '\u30d4\u30e3': 'pya', '\u30d4\u30e5': 'pyu', '\u30d4\u30e7': 'pyo',

        # Katakana small characters and special cases
        '\u30c3': '', # Small tsu (doubles the following consonant)
        '\u30e3': 'ya', '\u30e5': 'yu', '\u30e7': 'yo',

        # Katakana extras
        '\u30f4': 'vu', '\u30d5\u30a1': 'fa', '\u30d5\u30a3': 'fi', '\u30d5\u30a7': 'fe', '\u30d5\u30a9': 'fo',
        '\u30a6\u30a3': 'wi', '\u30a6\u30a7': 'we', '\u30a6\u30a9': 'wo',

        # Hiragana characters
        '\u3042': 'a', '\u3044': 'i', '\u3046': 'u', '\u3048': 'e', '\u304a': 'o',
        '\u304b': 'ka', '\u304d': 'ki', '\u304f': 'ku', '\u3051': 'ke', '\u3053': 'ko',
        '\u3055': 'sa', '\u3057': 'shi', '\u3059': 'su', '\u305b': 'se', '\u305d': 'so',
        '\u305f': 'ta', '\u3061': 'chi', '\u3064': 'tsu', '\u3066': 'te', '\u3068': 'to',
        '\u306a': 'na', '\u306b': 'ni', '\u306c': 'nu', '\u306d': 'ne', '\u306e': 'no',
        '\u306f': 'ha', '\u3072': 'hi', '\u3075': 'fu', '\u3078': 'he', '\u307b': 'ho',
        '\u307e': 'ma', '\u307f': 'mi', '\u3080': 'mu', '\u3081': 'me', '\u3082': 'mo',
        '\u3084': 'ya', '\u3086': 'yu', '\u3088': 'yo',
        '\u3089': 'ra', '\u308a': 'ri', '\u308b': 'ru', '\u308c': 're', '\u308d': 'ro',
        '\u308f': 'wa', '\u3092': 'wo', '\u3093': 'n',

        # Hiragana voiced consonants
        '\u304c': 'ga', '\u304e': 'gi', '\u3050': 'gu', '\u3052': 'ge', '\u3054': 'go',
        '\u3056': 'za', '\u3058': 'ji', '\u305a': 'zu', '\u305c': 'ze', '\u305e': 'zo',
        '\u3060': 'da', '\u3062': 'ji', '\u3065': 'zu', '\u3067': 'de', '\u3069': 'do',
        '\u3070': 'ba', '\u3073': 'bi', '\u3076': 'bu', '\u3079': 'be', '\u307c': 'bo',
        '\u3071': 'pa', '\u3074': 'pi', '\u3077': 'pu', '\u307a': 'pe', '\u307d': 'po',

        # Hiragana combinations
        '\u304d\u3083': 'kya', '\u304d\u3085': 'kyu', '\u304d\u3087': 'kyo',
        '\u3057\u3083': 'sha', '\u3057\u3085': 'shu', '\u3057\u3087': 'sho',
        '\u3061\u3083': 'cha', '\u3061\u3085': 'chu', '\u3061\u3087': 'cho',
        '\u306b\u3083': 'nya', '\u306b\u3085': 'nyu', '\u306b\u3087': 'nyo',
        '\u3072\u3083': 'hya', '\u3072\u3085': 'hyu', '\u3072\u3087': 'hyo',
        '\u307f\u3083': 'mya', '\u307f\u3085': 'myu', '\u307f\u3087': 'myo',
        '\u308a\u3083': 'rya', '\u308a\u3085': 'ryu', '\u308a\u3087': 'ryo',
        '\u304e\u3083': 'gya', '\u304e\u3085': 'gyu', '\u304e\u3087': 'gyo',
        '\u3058\u3083': 'ja', '\u3058\u3085': 'ju', '\u3058\u3087': 'jo',
        '\u3073\u3083': 'bya', '\u3073\u3085': 'byu', '\u3073\u3087': 'byo',
        '\u3074\u3083': 'pya', '\u3074\u3085': 'pyu', '\u3074\u3087': 'pyo',

        # Hiragana small characters and special cases
        '\u3063': '', # Small tsu (doubles the following consonant)
        '\u3083': 'ya', '\u3085': 'yu', '\u3087': 'yo',

        # Common punctuation and spaces
        '\u3000': ' ', # Japanese space
        '\u3001': ', ', '\u3002': '. ',
    }

    result = []
    i = 0

    while i < len(japanese_text):
        # Check for small tsu (doubling the following consonant)
        if i < len(japanese_text) - 1 and (japanese_text[i] == '\u3063' or japanese_text[i] == '\u30c3'):
            if i < len(japanese_text) - 1 and japanese_text[i+1] in kana_map:
                next_romaji = kana_map[japanese_text[i+1]]
                if next_romaji and next_romaji[0] not in 'aiueon':
                    result.append(next_romaji[0])  # Double the consonant
            i += 1
            continue

        # Check for combinations with small ya, yu, yo
        if i < len(japanese_text) - 1 and japanese_text[i+1] in ('\u3083', '\u3085', '\u3087', '\u30e3', '\u30e5', '\u30e7'):
            combo = japanese_text[i:i+2]
            if combo in kana_map:
                result.append(kana_map[combo])
                i += 2
                continue

        # Regular character
        if japanese_text[i] in kana_map:
            result.append(kana_map[japanese_text[i]])
        else:
            # If it's not in our map, keep it as is (might be kanji, romaji, etc.)
            result.append(japanese_text[i])

        i += 1

    return ''.join(result)

def number_to_text(num, ordinal=False):
    """
    Convert a number (int or float) to its text representation.

    Args:
        num: The number to convert

    Returns:
        str: Text representation of the number
    """

    if not isinstance(num, (int, float)):
        return "Input must be a number"

    # Handle special case of zero
    if num == 0:
        return "zero"

    # Handle negative numbers
    negative = num < 0
    num = abs(num)

    # Handle floats
    if isinstance(num, float):
        # Split into integer and decimal parts
        int_part = int(num)

        # Convert both parts
        int_text = _int_to_text(int_part)

        # Handle decimal part (convert to string and remove '0.')
        decimal_str = str(num).split('.')[1]
        decimal_text = " point " + " ".join(_digit_to_text(int(digit)) for digit in decimal_str)

        result = int_text + decimal_text
    else:
        # Handle integers
        result = _int_to_text(num)

    # Add 'negative' prefix for negative numbers
    if negative:
        result = "negative " + result

    return result


def _int_to_text(num):
    """Helper function to convert an integer to text"""

    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen"]

    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    if num < 20:
        return ones[num]

    if num < 100:
        return tens[num // 10] + (" " + ones[num % 10] if num % 10 != 0 else "")

    if num < 1000:
        return ones[num // 100] + " hundred" + (" " + _int_to_text(num % 100) if num % 100 != 0 else "")

    if num < 1000000:
        return _int_to_text(num // 1000) + " thousand" + (" " + _int_to_text(num % 1000) if num % 1000 != 0 else "")

    if num < 1000000000:
        return _int_to_text(num // 1000000) + " million" + (" " + _int_to_text(num % 1000000) if num % 1000000 != 0 else "")

    return _int_to_text(num // 1000000000) + " billion" + (" " + _int_to_text(num % 1000000000) if num % 1000000000 != 0 else "")


def _digit_to_text(digit):
    """Convert a single digit to text"""
    digits = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    return digits[digit]


_whitespace_re = re.compile(r"\s+")


# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = {
    "en": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("mrs", "misess"),
            ("mr", "mister"),
            ("dr", "doctor"),
            ("st", "saint"),
            ("co", "company"),
            ("jr", "junior"),
            ("maj", "major"),
            ("gen", "general"),
            ("drs", "doctors"),
            ("rev", "reverend"),
            ("lt", "lieutenant"),
            ("hon", "honorable"),
            ("sgt", "sergeant"),
            ("capt", "captain"),
            ("esq", "esquire"),
            ("ltd", "limited"),
            ("col", "colonel"),
            ("ft", "fort"),
        ]
    ],
}


def expand_abbreviations_multilingual(text, lang="en"):
    for regex, replacement in _abbreviations[lang]:
        text = re.sub(regex, replacement, text)
    return text


_symbols_multilingual = {
    "en": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " and "),
            ("@", " at "),
            ("%", " percent "),
            ("#", " hash "),
            ("$", " dollar "),
            ('\xa3', " pound "),
            ('\xb0', " degree "),
        ]
    ],
}


def expand_symbols_multilingual(text, lang="en"):
    for regex, replacement in _symbols_multilingual[lang]:
        text = re.sub(regex, replacement, text)
        text = text.replace("  ", " ")  # Ensure there are no double spaces
    return text.strip()


_ordinal_re = {
    "en": re.compile(r"([0-9]+)(st|nd|rd|th)"),
}
_number_re = re.compile(r"[0-9]+")
_currency_re = {
    "USD": re.compile(r"((\$[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+\$))"),
    "GBP": re.compile('((\xa3[0-9\\.\\,]*[0-9]+)|([0-9\\.\\,]*[0-9]+\xa3))'),
    "EUR": re.compile('(([0-9\\.\\,]*[0-9]+\u20ac)|((\u20ac[0-9\\.\\,]*[0-9]+)))'),
}

_comma_number_re = re.compile(r"\b\d{1,3}(,\d{3})*(\.\d+)?\b")
_dot_number_re = re.compile(r"\b\d{1,3}(.\d{3})*(\,\d+)?\b")
_decimal_number_re = re.compile(r"([0-9]+[.,][0-9]+)")


def _remove_commas(m):
    text = m.group(0)
    if "," in text:
        text = text.replace(",", "")
    return text


def _remove_dots(m):
    text = m.group(0)
    if "." in text:
        text = text.replace(".", "")
    return text


def _expand_decimal_point(m, lang="en"):
    amount = m.group(1).replace(",", ".")
    return number_to_text(float(amount))


def _expand_currency(m, lang="en", currency="USD"):
    amount = float((re.sub(r"[^\d.]", "", m.group(0).replace(",", "."))))
    full_amount = number_to_text(amount)

    and_equivalents = {
        "en": ", ",
        "es": " con ",
        "fr": " et ",
        "de": " und ",
        "pt": " e ",
        "it": " e ",
        "pl": ", ",
        "cs": ", ",
        "ru": ", ",
        "nl": ", ",
        "ar": ", ",
        "tr": ", ",
        "hu": ", ",
        "ko": ", ",
    }

    if amount.is_integer():
        last_and = full_amount.rfind(and_equivalents[lang])
        if last_and != -1:
            full_amount = full_amount[:last_and]

    return full_amount


def _expand_ordinal(m, lang="en"):
    return number_to_text(int(m.group(1)), ordinal=True)


def _expand_number(m, lang="en"):
    return number_to_text(int(m.group(0)))


def expand_numbers_multilingual(text, lang="en"):
    if lang in ["en", "ru"]:
        text = re.sub(_comma_number_re, _remove_commas, text)
    else:
        text = re.sub(_dot_number_re, _remove_dots, text)
    try:
        text = re.sub(_currency_re["GBP"], lambda m: _expand_currency(m, lang, "GBP"), text)
        text = re.sub(_currency_re["USD"], lambda m: _expand_currency(m, lang, "USD"), text)
        text = re.sub(_currency_re["EUR"], lambda m: _expand_currency(m, lang, "EUR"), text)
    except:
        pass

    text = re.sub(_decimal_number_re, lambda m: _expand_decimal_point(m, lang), text)
    text = re.sub(_ordinal_re[lang], lambda m: _expand_ordinal(m, lang), text)
    text = re.sub(_number_re, lambda m: _expand_number(m, lang), text)
    return text


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def multilingual_cleaners(text, lang):
    text = text.replace('"', "")
    if lang == "tr":
        text = text.replace('\u0130', "i")
        text = text.replace('\xd6', '\xf6')
        text = text.replace('\xdc', '\xfc')
    text = lowercase(text)
    try:
        text = expand_numbers_multilingual(text, lang)
    except:
        pass
    try:
        text = expand_abbreviations_multilingual(text, lang)
    except:
        pass
    try:
        text = expand_symbols_multilingual(text, lang=lang)
    except:
        pass
    text = collapse_whitespace(text)
    return text


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text
