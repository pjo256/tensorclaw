import shlex
from typing import Any, Mapping


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def render_template(template: str, values: Mapping[str, Any]) -> str:
    if not template:
        return ""
    return template.format_map(_SafeDict(values))


def shell_escape(value: Any) -> str:
    return shlex.quote(str(value))
