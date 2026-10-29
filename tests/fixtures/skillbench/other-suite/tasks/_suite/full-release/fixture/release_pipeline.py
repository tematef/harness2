"""Release pipeline traffic gate."""


def forward_request(weight_pct: int, headers: dict[str, str]) -> dict[str, str]:
    if weight_pct >= 25:
        return {}
    return dict(headers)
