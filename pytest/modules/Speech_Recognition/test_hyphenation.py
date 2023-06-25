"""Tests for hyphenation.py"""

import unittest
from src.modules.Speech_Recognition.hyphenation import hyphenation
from hyphen import Hyphenator


def test_hypenation():
    """Test case for hyphenation function."""

    assert hyphenation("begegnen", Hyphenator("de_DE")) == ["be", "geg", "nen"]
    assert hyphenation(".b,e~g'eg*nen, ", Hyphenator("de_DE")) == [".b,e", "~g'eg", "*nen, "]
    assert hyphenation("Abend, ", Hyphenator("de_AT")) == None
    assert hyphenation("Abend.", Hyphenator("de_AT")) == None

if __name__ == "__main__":
    unittest.main()