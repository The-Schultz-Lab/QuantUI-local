"""Static analysis guards for patterns that fail silently at runtime."""

import re
from pathlib import Path

SRC = Path(__file__).parent.parent / "quantui"


def _grep(pattern: str) -> list[str]:
    hits = []
    for path in SRC.rglob("*.py"):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(pattern, line):
                hits.append(f"{path.relative_to(SRC.parent)}:{i}: {line.strip()}")
    return hits


def test_no_cdn_plotlyjs():
    hits = _grep(r'include_plotlyjs\s*=\s*["\']cdn["\']')
    assert not hits, "CDN plotlyjs detected (fails silently offline):\n" + "\n".join(
        hits
    )


def test_no_bare_except_pass():
    hits = _grep(r"^\s*except\s*(\(\s*\))?\s*:\s*(pass\s*)?$")
    assert not hits, "Bare except/pass detected (swallows all errors):\n" + "\n".join(
        hits
    )
