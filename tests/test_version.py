from core.version import get_version


def test_returns_the_version_from_pyproject():
    assert get_version() == "0.1.0"
