import pytest


def test_tests_1():
    assert True


def test_tests_2():
    with pytest.raises(AssertionError):
        assert False
