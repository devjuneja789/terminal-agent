"""User helpers."""


def display_name(user: dict[str, str]) -> str:
    return user.get("display_name") or user.get("username", "")


def is_active(user: dict[str, object]) -> bool:
    return user.get("active") is True
