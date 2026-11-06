"""Result objects shared by the memory store and tools.

Every tool result is a dict with ``ok``. Failures also include ``error`` and
``message`` so the agent loop can record them on the trace and hand them back
to the model without raising.
"""


def fail(error: str, message: str, **extra: object) -> dict:
    body: dict = {"ok": False, "error": error, "message": message}
    body.update(extra)
    return body


def succeed(**payload: object) -> dict:
    body: dict = {"ok": True}
    body.update(payload)
    return body
