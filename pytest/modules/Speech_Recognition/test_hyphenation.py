"""Tests for hyphenation.py"""

import unittest
from unittest.mock import patch
from src.modules.Speech_Recognition.hyphenation import hyphenation, language_check
from hyphen import Hyphenator, dictools


class TestHypenation(unittest.TestCase):

    def test_hypenation(self):
        """Test case for hyphenation function."""

        # prepare test
        installed = dictools.list_installed()
        for lang in installed:
            dictools.uninstall(lang)

        assert hyphenation("darkness", Hyphenator("en")) == ["dark", "ness"]
        assert hyphenation("Hombre", Hyphenator("es")) == ["Hom", "bre"]
        assert hyphenation("begegnen", Hyphenator("de_DE")) == ["be", "geg", "nen"]
        assert hyphenation(".b,e~g'eg*nen, ", Hyphenator("de_DE")) == [".b,e", "~g'eg", "*nen, "]
        assert hyphenation("Abend, ", Hyphenator("de_AT")) == None
        assert hyphenation("Abend.", Hyphenator("de_AT")) == None

    @patch('hyphen.dictools.list_installed')
    def test_language_check_has_installed_language(self, mock_list_installed):
        mock_list_installed.return_value = ['de', 'de_DE', 'en', 'en_US', 'en_GB']
        assert language_check("en") == "en_US"
        assert language_check("de") == "de_DE"

    @patch('hyphen.dictools.list_installed')
    def test_language_check_not_installed_language(self, mock_list_installed):
        mock_list_installed.return_value = []
        assert language_check("fr") == "fr_FR"
        assert language_check("de") == "de"
        assert language_check("none") == None


if __name__ == "__main__":
    unittest.main()
