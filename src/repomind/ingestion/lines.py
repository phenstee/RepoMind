"""Source line splitting that agrees with ``ast`` line numbers and editors.

``str.splitlines`` also breaks on vertical tab, form feed, the file/group/
record separators, NEL, and the Unicode line/paragraph separators. Python's
tokenizer (and therefore ``ast`` line numbers), editors, and ``git`` treat those
as ordinary characters, so using it would silently shift every citation after
the first such character. Every ingestion path that numbers lines must go
through :func:`split_source_lines` instead.
"""

from io import StringIO
from math import ceil


def split_source_lines(content: str) -> list[str]:
    """Split on ``\\r\\n``, ``\\r``, and ``\\n`` only, keeping each line ending.

    Joining the result reproduces ``content`` exactly. An empty string has no
    lines, and a trailing line without a terminator is still a line.
    """

    # newline="" enables universal-newline splitting on exactly \r\n, \r, and
    # \n while returning each line ending untranslated.
    return list(StringIO(content, newline=""))


def count_source_lines(content: str) -> int:
    """Return the number of lines :func:`split_source_lines` would produce."""

    return len(split_source_lines(content))


def split_oversized_line(text: str, max_chars: int) -> list[str]:
    """Split one line into the fewest near-equal pieces of at most ``max_chars``.

    The pieces concatenate back to ``text`` exactly. An empty string yields no
    pieces.
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not text:
        return []
    piece_count = ceil(len(text) / max_chars)
    piece_size, larger_pieces = divmod(len(text), piece_count)
    pieces: list[str] = []
    offset = 0
    for piece_index in range(piece_count):
        end = offset + piece_size + (piece_index < larger_pieces)
        pieces.append(text[offset:end])
        offset = end
    return pieces
