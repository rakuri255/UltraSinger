"""Test the musicbrainz_client module."""

import unittest
from unittest.mock import patch
from src.modules.musicbrainz_client import search_musicbrainz


class TestGetMusicInfos(unittest.TestCase):

    @patch('musicbrainzngs.search_artists')
    @patch('musicbrainzngs.search_recordings')
    def test_get_music_infos(self, mock_search_recordings, mock_search_artists):
        # Arrange
        artist = 'UltraSinger'
        title = 'That\'s Rocking! (UltrStar 2023) FULL HD'

        # Set up mock return values for the MusicBrainz API calls
        mock_search_artists.return_value = {
            'artist-list': [
                {'name': artist}
            ]
        }

        mock_search_recordings.return_value = {
            'recording-list': [
                {'artist-credit': [
                    {'name': 'Half Japanese',
                     'artist': {
                         'name': 'Half Japanese',
                                               'sort-name': 'Half Japanese'
                     }}],
                 'release-list': [{'title': title,
                                   'status': 'Official',
                                   'artist-credit': [{'name': 'Half Japanese'}],
                                   'date': '2014-04-20',
                                   'artist-credit-phrase': artist}],
                 'artist-credit-phrase': artist}]}

        # Call the function to test
        title, artist, year, genre = search_musicbrainz(title, artist)

        # Assert the returned values
        self.assertEqual(title, 'That\'s Rocking!')
        self.assertEqual(artist, 'UltraSinger')
        self.assertEqual(year, '2023-01-01')
        self.assertEqual(genre, 'Genre 1,Genre 2,')

    @patch('musicbrainzngs.search_artists')
    @patch('musicbrainzngs.search_recordings')
    def test_get_music_infos_when_title_and_artist_are_the_same(self, mock_search_release_groups, mock_search_artists):
        # Arrange
        artist = "ArtistIsTitle"
        title = "ArtistIsTitle"

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
        title, artist, year, genre = search_musicbrainz("ArtistIsTitle", "ArtistNotTitle")

        # Assert
        self.assertEqual(title, None)
        self.assertEqual(artist, None)
        self.assertEqual(year, None)
        self.assertEqual(genre, None)

        # Act search_is_same and musicbrainz returns the same artist and title
        title, artist, year, genre = search_musicbrainz(title, artist)

        # Assert
        self.assertEqual(title, 'ArtistIsTitle')
        self.assertEqual(artist, 'ArtistIsTitle')
        self.assertEqual(year, None)
        self.assertEqual(genre, None)

    @patch('musicbrainzngs.search_artists')
    @patch('musicbrainzngs.search_recordings')
    def test_get_empty_artist_music_infos(self, mock_search_release_groups, mock_search_artists):
        # Arrange
        artist = 'UltraSinger'
        title = 'That\'s Rocking! (UltrStar 2023) FULL HD'

        # Set up mock return values for the MusicBrainz API calls
        mock_search_artists.return_value = {
            'artist-list': []
        }

        mock_search_release_groups.return_value = {
            'release-group-list': [
                {
                    'title': f'  {title}  ',  # Also test leading and trailing whitespaces
                    'artist-credit-phrase': f'  {artist}  ',  # Also test leading and trailing whitespaces
                    'first-release-date': '  2023-01-01  ',  # Also test leading and trailing whitespaces
                    'tag-list': [
                        {'name': '  Genre 1  '},  # Also test leading and trailing whitespaces
                        {'name': '  Genre 2  '}  # Also test leading and trailing whitespaces
                    ]
                }
            ]
        }

        # Act
        title, artist, year, genre = search_musicbrainz(title, artist)

        # Assert
        self.assertEqual(title, None)
        self.assertEqual(artist, None)
        self.assertEqual(year, None)
        self.assertEqual(genre, None)

    @patch('musicbrainzngs.search_artists')
    @patch('musicbrainzngs.search_recordings')
    def test_get_empty_release_music_infos(self, mock_search_release_groups, mock_search_artists):
        # Arrange
        artist = 'UltraSinger'
        title = 'That\'s Rocking! (UltrStar 2023) FULL HD'

        # Set up mock return values for the MusicBrainz API calls
        mock_search_artists.return_value = {
            'artist-list': [
                {'name': f'  {artist}  '}  # Also test leading and trailing whitespaces
            ]
        }

        mock_search_release_groups.return_value = {
            'release-group-list': []
        }

        # Act
        title, artist, year, genre = search_musicbrainz(title, artist)

        # Assert
        self.assertEqual(title, None)
        self.assertEqual(artist, None)
        self.assertEqual(year, None)
        self.assertEqual(genre, None)

    #@unittest.skip("Search with real data only test manually")
    def test_search_musicbrainz_with_real_data(self):

        # Arrange
        search_list = [
            #(search_artist, seartch_title, expected_artist, expected_title)

            (None, 't', None, None),  # this should return "Unknown artist"
            ('Căsuța noastră', 'Gică Petrescu', 'Gică Petrescu', 'Căsuța noastră'),  # Gică Petrescu - Gică Petrescu
            ("Shawn James - Through the Valley - Official Music Video", None, "Shawn James", "Through the Valley"),
            # (None, 'Corey Taylor Snuff (Acoustic)', 'Corey Taylor', 'Snuff'), # Fixme: is wrong
            # (None, 'Corey Taylor Snuff', 'Corey Taylor', 'Snuff'), # Fixme: is wrong
            # (None, 'Songs für Liam Kraftklub', 'Kraftklub', 'Songs Für Liam'), # Fixme: is wrong
            # ('Kummer feat. Fred Rabe', 'Der letzte Song (Alles wird gut)', 'Kummer feat. Fred Rabe', 'Der letzte Song (Alles wird gut)'),  # Todo: Wrong image?
            # ('Der letzte Song (Alles wird gut)', 'Kummer feat. Fred Rabe', 'Kummer feat. Fred Rabe', 'Der letzte Song (Alles wird gut)'),  # Todo: Wrong image?
            # (None, 'Der letzte Song (Alles wird gut) Kummer feat. Fred Rabe', 'Kummer feat. Fred Rabe', 'Der letzte Song (Alles wird gut)'), # Todo: Wrong image?
            # (None, 'Thats life Shawn James', 'Shawn James', 'Thats life'),  # Fixme: is wrong
            # (None, 'Gloryhole Explicit Steel Panther', 'Steel Panther', 'Gloryhole Explicit'), # Fixme: is wrong
        ]

        failed = 0
        success = 0
        count = 0
        for i, search_string in enumerate(search_list):
            artist = search_string[0]
            title = search_string[1]
            print(f"({i}) - {artist} - {title}")
            count = i

            song_info = search_musicbrainz(title, artist)
            print(f'\t{search_string}\t -> {song_info.artist}, {song_info.title}, {song_info.year}, {song_info.genres}')
            print('-------------------------------')
        print(f"Faild: {failed} | Success: {success} Count: {count}")


if __name__ == '__main__':
    unittest.main()
