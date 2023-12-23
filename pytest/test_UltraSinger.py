"""Tests for UltraSinger.py"""

import unittest
from src.UltraSinger import format_separated_string
class TestUltraSinger(unittest.TestCase):
    def test_format_separated_string(self):

        self.assertEqual(format_separated_string('rock,pop,rock-pop,'), 'Rock, Pop, Rock-Pop')
        self.assertEqual(format_separated_string('rock;pop;rock-pop,'), 'Rock, Pop, Rock-Pop')
        self.assertEqual(format_separated_string('rock/pop/rock-pop,'), 'Rock, Pop, Rock-Pop')
        self.assertEqual(format_separated_string('rock,pop/rock-pop;80s,'), 'Rock, Pop, Rock-Pop, 80s')
        self.assertEqual(format_separated_string('rock, pop, rock-pop, '), 'Rock, Pop, Rock-Pop')
