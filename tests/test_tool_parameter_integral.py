"""An integer parameter written as an integral number (17.0) decodes to that integer, not the raw text."""

import pytest

from tensorfold.tool_parameters import decode_parameter


@pytest.mark.parametrize("value,expected", [
    ("17.0", 17),     # integral number: the integer it spells
    ("17", 17),       # plain integer unchanged
    ("17.5", "17.5"), # a fraction is no integer: the text stays
])
def test_integral_number_is_an_integer(value, expected):
    for python in (True, False):
        decoded = decode_parameter(value, {"type": "integer"}, python=python)
        assert decoded == expected and type(decoded) is type(expected)


def test_number_parameters_keep_their_float():
    assert type(decode_parameter("17.0", {"type": "number"})) is float
