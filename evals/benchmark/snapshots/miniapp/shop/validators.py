"""Input validation helpers."""

import re


def is_slug(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9-]+", value))


def is_positive_id(value: object) -> bool:
    return isinstance(value, int) and value > 0
