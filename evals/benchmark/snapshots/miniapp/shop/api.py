"""Small route table for the example service."""

from shop.validators import is_positive_id


ROUTES = {
    "GET /health": "health",
    "GET /items": "list_items",
    "GET /items/{item_id}": "get_item",
}


def resolve_route(method: str, path: str) -> str | None:
    key = f"{method.upper()} {path}"
    if key in ROUTES:
        return ROUTES[key]
    if path.startswith("/items/") and method.upper() == "GET":
        suffix = path.removeprefix("/items/")
        if suffix.isdigit() and is_positive_id(int(suffix)):
            return "get_item"
    return None
