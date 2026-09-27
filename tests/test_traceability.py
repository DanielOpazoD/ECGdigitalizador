"""The regulatory documents cite tests by name; keep them honest."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "regulatorio"
CITED = re.compile(r"`(test_\w+\.py)::(test_\w+)`")


def _tests_in(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}


def test_every_cited_test_exists() -> None:
    missing = []
    for doc in ("analisis_riesgos.md", "trazabilidad.md"):
        for file, name in CITED.findall((DOCS / doc).read_text(encoding="utf-8")):
            path = ROOT / "tests" / file
            if not path.exists() or name not in _tests_in(path):
                missing.append(f"{doc}: {file}::{name}")
    assert not missing, missing


def test_every_cited_module_exists() -> None:
    text = (DOCS / "trazabilidad.md").read_text(encoding="utf-8")
    for row in text.splitlines():
        if row.startswith("| REQ-"):
            for mod in re.findall(r"`([\w/]+(?:\.py)?)`", row.split("|")[3]):
                assert (ROOT / "src" / "ecg_photo" / mod).exists(), (row[:8], mod)


def test_every_risk_is_verified_and_traced() -> None:
    risks = (DOCS / "analisis_riesgos.md").read_text(encoding="utf-8")
    trace = (DOCS / "trazabilidad.md").read_text(encoding="utf-8")
    ids = re.findall(r"^\| (R\d\d) \|", risks, flags=re.MULTILINE)
    assert ids
    for row in risks.splitlines():
        m = re.match(r"^\| (R\d\d) \|", row)
        if m and m.group(1) != "R09":  # R09 is eliminated: ST is not reported
            assert CITED.search(row), f"{m.group(1)} has no automated test"
    traced = set(re.findall(r"\bR\d\d\b", trace))
    assert set(ids) - traced == set(), set(ids) - traced
