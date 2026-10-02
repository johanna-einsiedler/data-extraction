import re
import string


def normalize(text: str) -> str:
    """Strip all non-word characters and lowercase. Used for choice/label matching."""
    return re.sub(r"\W+", "", text.strip().lower())


def normalize_text(s: str) -> str:
    """Lowercase, remove punctuation, and normalize spaces. Used for loose string comparison."""
    if not isinstance(s, str):
        return ""
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s)
    return s.strip()
