import re

CHATTER = re.compile(r"(?i)^.*\b(?:plate|matr[ií]cula)\b.*?(?:\bis\b|\bes\b|:)")
CURRENT_PLATE = re.compile(r"^[0-9]{4}[BCDFGHJKLMNPRSTVWXYZ]{3}$")
OLD_PLATE = re.compile(r"^[A-Z]{1,2}[0-9]{4}[A-Z]{1,2}$")


def normalize(text):
    text = CHATTER.sub("", " ".join(text.split()), count=1)
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_valid_plate(text):
    return bool(CURRENT_PLATE.match(text) or OLD_PLATE.match(text))


def levenshtein(a, b):
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def cer(pred, ann):
    return levenshtein(pred, ann) / len(ann)
