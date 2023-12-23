"""Tests for UltraSinger.py"""

import unittest
from src.UltraSinger import format_separated_string
from src.UltraSinger import extract_year

class TestUltraSinger(unittest.TestCase):
    def test_format_separated_string(self):

        self.assertEqual(format_separated_string('rock,pop,rock-pop,'), 'Rock, Pop, Rock-Pop')
        self.assertEqual(format_separated_string('rock;pop;rock-pop,'), 'Rock, Pop, Rock-Pop')
        self.assertEqual(format_separated_string('rock/pop/rock-pop,'), 'Rock, Pop, Rock-Pop')
        self.assertEqual(format_separated_string('rock,pop/rock-pop;80s,'), 'Rock, Pop, Rock-Pop, 80s')
        self.assertEqual(format_separated_string('rock, pop, rock-pop, '), 'Rock, Pop, Rock-Pop')

    def test_extract_year(self):
        years = {extract_year("2023-12-31"), extract_year("2023-12-31 23:59:59"), extract_year("2023/12/31"),
                 extract_year("2023\\12\\31"), extract_year("2023.12.31"), extract_year("2023 12 31"),
                 extract_year("12-31-2023"), extract_year("12/31/2023"), extract_year("12\\31\\2023"),
                 extract_year("12.31.2023"), extract_year("12 31 2023"), extract_year("12-2023"),
                 extract_year("12/2023"), extract_year("2023")}

        for year in years:
            self.assertEqual(year, "2023")
