from __future__ import annotations

from collections.abc import MutableMapping, MutableSequence
from typing import Any


class JSONPointerError(ValueError):
    """Base class for JSON Pointer validation and traversal errors."""


class JSONPointerSyntaxError(JSONPointerError):
    """Raised when a JSON Pointer string is not RFC 6901 syntax."""


class JSONPointerMissingError(JSONPointerError):
    """Raised when a JSON Pointer segment is missing."""


class JSONPointerTraversalError(JSONPointerError):
    """Raised when traversal reaches a scalar value before the pointer ends."""


def parse_json_pointer(pointer: str) -> tuple[str, ...]:
    """Parse an RFC 6901 JSON Pointer into unescaped reference tokens."""
    if not isinstance(pointer, str):
        raise JSONPointerSyntaxError("JSON Pointer must be a string")
    if pointer == "":
        return ()
    if not pointer.startswith("/"):
        raise JSONPointerSyntaxError(
            f"invalid JSON Pointer {pointer!r}: pointer must be empty or start with '/'"
        )
    return tuple(_unescape_token(token) for token in pointer.split("/")[1:])


def read_json_pointer(document: Any, pointer: str) -> Any:
    """Read a value from a JSON-compatible document using an RFC 6901 pointer."""
    current = document
    for token in parse_json_pointer(pointer):
        current = _read_child(current, token, pointer=pointer)
    return current


def set_json_pointer(
    document: Any,
    pointer: str,
    value: Any,
    *,
    create_missing: bool = False,
) -> Any:
    """Set a value in a JSON-compatible document and return the root object."""
    tokens = parse_json_pointer(pointer)
    if not tokens:
        return value

    current = document
    for token, next_token in zip(tokens[:-1], tokens[1:], strict=False):
        if isinstance(current, MutableMapping):
            if token not in current:
                if not create_missing:
                    raise JSONPointerMissingError(
                        f"missing object key {token!r} while resolving {pointer!r}"
                    )
                current[token] = _new_container_for_next_token(next_token)
            current = current[token]
            continue

        if isinstance(current, MutableSequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            index = _parse_list_index(token, pointer=pointer)
            try:
                current = current[index]
            except IndexError as exc:
                raise JSONPointerMissingError(
                    f"missing array index {token!r} while resolving {pointer!r}"
                ) from exc
            continue

        raise JSONPointerTraversalError(
            f"cannot traverse through {type(current).__name__} while resolving {pointer!r}"
        )

    parent = current
    final_token = tokens[-1]
    if isinstance(parent, MutableMapping):
        if not create_missing and final_token not in parent:
            raise JSONPointerMissingError(
                f"missing object key {final_token!r} while resolving {pointer!r}"
            )
        parent[final_token] = value
        return document

    if isinstance(parent, MutableSequence) and not isinstance(
        parent, (str, bytes, bytearray)
    ):
        if final_token == "-":
            parent.append(value)
            return document
        index = _parse_list_index(final_token, pointer=pointer)
        try:
            parent[index] = value
        except IndexError as exc:
            raise JSONPointerMissingError(
                f"missing array index {final_token!r} while resolving {pointer!r}"
            ) from exc
        return document

    raise JSONPointerTraversalError(
        f"cannot set child on {type(parent).__name__} while resolving {pointer!r}"
    )


def _unescape_token(token: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(token):
        char = token[index]
        if char != "~":
            result.append(char)
            index += 1
            continue
        if index + 1 >= len(token):
            raise JSONPointerSyntaxError(
                "invalid JSON Pointer escape: '~' must be followed by '0' or '1'"
            )
        escape = token[index + 1]
        if escape == "0":
            result.append("~")
        elif escape == "1":
            result.append("/")
        else:
            raise JSONPointerSyntaxError(
                f"invalid JSON Pointer escape '~{escape}': expected '~0' or '~1'"
            )
        index += 2
    return "".join(result)


def _read_child(value: Any, token: str, *, pointer: str) -> Any:
    if isinstance(value, dict):
        try:
            return value[token]
        except KeyError as exc:
            raise JSONPointerMissingError(
                f"missing object key {token!r} while resolving {pointer!r}"
            ) from exc

    if isinstance(value, list):
        index = _parse_list_index(token, pointer=pointer)
        try:
            return value[index]
        except IndexError as exc:
            raise JSONPointerMissingError(
                f"missing array index {token!r} while resolving {pointer!r}"
            ) from exc

    raise JSONPointerTraversalError(
        f"cannot traverse through {type(value).__name__} while resolving {pointer!r}"
    )


def _parse_list_index(token: str, *, pointer: str) -> int:
    if not token:
        raise JSONPointerSyntaxError(
            f"invalid array index in JSON Pointer {pointer!r}"
        )
    if token != "0" and token.startswith("0"):
        raise JSONPointerSyntaxError(
            f"invalid array index {token!r} in JSON Pointer {pointer!r}"
        )
    try:
        index = int(token)
    except ValueError as exc:
        raise JSONPointerSyntaxError(
            f"invalid array index {token!r} in JSON Pointer {pointer!r}"
        ) from exc
    if index < 0:
        raise JSONPointerSyntaxError(
            f"invalid negative array index {token!r} in JSON Pointer {pointer!r}"
        )
    return index


def _new_container_for_next_token(
    next_token: str,
) -> dict[str, Any]:
    del next_token
    return {}
