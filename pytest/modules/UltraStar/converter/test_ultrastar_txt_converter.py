"""Tests for ultrastar_txt_converter.py"""

import unittest
from modules.Ultrastar.coverter.ultrastar_txt_converter import extract_year


class TestUltrastarTxtConverter(unittest.TestCase):

    def test_extract_year(self):
        years = {extract_year("2023-12-31"), extract_year("2023-12-31 23:59:59"), extract_year("2023/12/31"),
                 extract_year("2023\\12\\31"), extract_year("2023.12.31"), extract_year("2023 12 31"),
                 extract_year("12-31-2023"), extract_year("12/31/2023"), extract_year("12\\31\\2023"),
                 extract_year("12.31.2023"), extract_year("12 31 2023"), extract_year("12-2023"),
                 extract_year("12/2023"), extract_year("2023")}

        for year in years:
            self.assertEqual(year, "2023")
