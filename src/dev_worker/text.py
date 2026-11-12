"""Text made safe to hand on: to a JSON answer, a process argument or a tracker."""

NUL = "\x00"
NUL_ESCAPE = "\\x00"


def writable_text(text: str) -> str:
    """Return `text` with every NUL and every character UTF-8 cannot encode written as its backslash escape.

    A NUL becomes the four characters `\\x00`, a lone surrogate such as U+D800 `\\ud800`.
    """
    encodable = text.encode("utf-8", "backslashreplace").decode("utf-8")
    return encodable.replace(NUL, NUL_ESCAPE)
