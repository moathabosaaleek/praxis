import re

from core.version import get_version

SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


def test_returns_a_semantic_version():
    assert SEMVER_PATTERN.match(get_version())
