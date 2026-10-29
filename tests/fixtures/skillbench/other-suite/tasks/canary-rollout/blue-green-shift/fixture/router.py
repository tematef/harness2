"""Weighted traffic router with sticky session cookie."""


def select_backend(cookie_version: str | None, weight_v2: int) -> str:
    if cookie_version in ("v1", "v2"):
        return cookie_version
    return "v2" if weight_v2 >= 50 else "v1"
