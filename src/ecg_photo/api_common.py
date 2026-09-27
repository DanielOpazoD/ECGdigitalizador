"""Helpers shared by the API modules: error shape and identifier check."""

import re

from fastapi import HTTPException

ID_RE = re.compile(r"^[a-z0-9-]{1,64}$")


def api_error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def check_id(value: str) -> str:
    """Study / run ids are generated lowercase slugs; anything else is not
    found (also keeps path traversal out of the store)."""
    if not ID_RE.match(value):
        raise api_error(404, "NOT_FOUND", "invalid identifier")
    return value
