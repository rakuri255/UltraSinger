"""Test the musicbrainz_client module."""

import unittest
from unittest.mock import patch
from src.modules.musicbrainz_client import get_music_infos


class TestGetMusicInfos(unittest.TestCase):

    @patch('musicbrainzngs.search_artists')
    @patch('musicbrainzngs.search_release_groups')
    def test_get_music_infos(self, mock_search_release_groups, mock_search_artists):
        # Arrange
        artist = 'UltraSinger'
        title = 'That\'s Rocking!'
        search = f'{artist} - {title} (UltrStar 2023) FULL HD'

        # Set up mock return values for the MusicBrainz API calls
        mock_search_artists.return_value = {
            'artist-list': [
                {'name': artist}
            ]
        }

        mock_search_release_groups.return_value = {
            'release-group-list': [
                {
                    'title': title,
                    'artist-credit-phrase': artist,
                    'first-release-date': '2023-01-01',
                    'tag-list': [
                        {'name': 'Genre 1'},
                        {'name': 'Genre 2'}
                    ]
                }
            ]
        }

        # Call the function to test
        title, artist, year, genre = get_music_infos(search)

        # Assert the returned values
        self.assertEqual(title, 'That\'s Rocking!')
        self.assertEqual(artist, 'UltraSinger')
        self.assertEqual(year, '2023-01-01')
        self.assertEqual(genre, 'Genre 1,Genre 2,')

    @patch('musicbrainzngs.search_artists')
    @patch('musicbrainzngs.search_release_groups')
    def test_get_music_infos_when_title_and_artist_are_the_same(self, mock_search_release_groups, mock_search_artists):
        # Arrange
        artist = "ArtistIsTitle"
        title = "ArtistIsTitle"
        search_not_same = "ArtistIsTitle - ArtistNotTitle"
        search_is_same = f"{artist} - {title}"

        # Set up mock return values for the MusicBrainz API calls
        mock_search_artists.return_value = {
            'artist-list': [
                {'name': artist}
            ]
        }

        mock_search_release_groups.return_value = {
            'release-group-list': [
                {
                    'title': title,
                    'artist-credit-phrase': artist,
                }
            ]
        }

        # Act search_not_same but musicbrainz returns the same artist and title
        title, artist, year, genre = get_music_infos(search_not_same)

        # Assert
        self.assertEqual(title, None)
        self.assertEqual(artist, None)
        self.assertEqual(year, None)
        self.assertEqual(genre, None)

        # Act search_is_same and musicbrainz returns the same artist and title
        title, artist, year, genre = get_music_infos(search_is_same)

        # Assert
        self.assertEqual(title, 'ArtistIsTitle')
        self.assertEqual(artist, 'ArtistIsTitle')
        self.assertEqual(year, None)
        self.assertEqual(genre, None)


if __name__ == '__main__':
    unittest.main()
