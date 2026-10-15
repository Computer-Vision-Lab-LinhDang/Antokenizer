"""Stream the objects of a top-level JSON array without loading the file (stdlib only).

nuScenes v1.0-trainval tables are single JSON arrays of 0.6-1.4 GB; ``json.load`` needs several
times that in RAM. ``iter_json_array`` decodes one object at a time with ``JSONDecoder.raw_decode``
over a text buffer refilled in chunks, so memory is O(chunk + largest object).
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator

_WS = re.compile(r"[ \t\n\r]*")
MAX_OBJECT_CHARS = 1 << 27  # ponytail: one element > 128 M chars means malformed input, not a big row


def iter_json_array(path: str | os.PathLike, chunk_size: int = 1 << 20) -> Iterator[dict]:
    """Yield each element of the top-level array in ``path``; every element must be a JSON object.

    Raises ValueError on anything else (not an array, non-object element, missing comma/bracket,
    truncated file).
    """
    decode = json.JSONDecoder().raw_decode
    with open(path, encoding="utf-8") as f:  # text mode: a chunk never splits a UTF-8 character
        buf, pos = "", 0

        def refill() -> bool:
            nonlocal buf, pos
            chunk = f.read(chunk_size)
            if not chunk:
                return False
            buf, pos = buf[pos:] + chunk, 0  # copies only the unread tail: O(chunk) per refill
            return True

        def peek() -> str:
            """Next non-whitespace character, '' at end of file."""
            nonlocal pos
            while True:
                pos = _WS.match(buf, pos).end()
                if pos < len(buf):
                    return buf[pos]
                if not refill():
                    return ""

        if peek() != "[":
            raise ValueError(f"{path}: expected a top-level JSON array")
        pos += 1
        if peek() == "]":
            return
        while True:
            if peek() != "{":
                raise ValueError(f"{path}: expected an object at char {pos} of the current buffer")
            while True:  # an object is self-delimiting: a successful decode is complete
                try:
                    obj, end = decode(buf, pos)
                    break
                except json.JSONDecodeError as e:
                    if len(buf) - pos > MAX_OBJECT_CHARS or not refill():
                        raise ValueError(f"{path}: truncated or malformed object ({e.msg})") from e
            pos = end
            yield obj
            c = peek()
            if c == "]":
                return
            if c != ",":
                raise ValueError(f"{path}: expected ',' or ']' after an object, got {c!r}")
            pos += 1
