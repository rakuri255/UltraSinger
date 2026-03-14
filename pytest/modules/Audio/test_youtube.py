"""Tests for youtube.py"""

import unittest
from unittest.mock import patch
from src.modules.Audio.youtube import get_youtube_title, strip_unmatched_suffixes
from src.modules.Audio.youtube import download_and_convert_thumbnail


class TestStripUnmatchedSuffixes(unittest.TestCase):
    """Tests for strip_unmatched_suffixes()."""

    def test_strips_live_when_not_in_video_title(self):
        """YT Music says '(live)' but the video is an official studio MV."""
        track = "Pump It (live)"
        video_title = "Electric Callboy - PUMP IT (OFFICIAL VIDEO)"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Pump It")

    def test_strips_live_at_venue_when_not_in_video_title(self):
        """Extended live suffix like '(Live at Wembley)' should also be stripped."""
        track = "Song Title (Live at Wembley)"
        video_title = "Artist - Song Title (Official Video)"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song Title")

    def test_keeps_live_when_present_in_video_title(self):
        """If the video title also says 'live', keep the suffix."""
        track = "Song Title (Live)"
        video_title = "Artist - Song Title (Live at Wembley)"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song Title (Live)")

    def test_strips_acoustic_when_not_in_video_title(self):
        track = "My Song (Acoustic)"
        video_title = "Band - My Song (Official Video)"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "My Song")

    def test_strips_remastered_when_not_in_video_title(self):
        track = "Song (Remastered 2024)"
        video_title = "Artist - Song"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song")

    def test_strips_remaster_when_not_in_video_title(self):
        track = "Song (Remaster)"
        video_title = "Artist - Song"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song")

    def test_strips_radio_edit_when_not_in_video_title(self):
        track = "Song (Radio Edit)"
        video_title = "Artist - Song (Official Video)"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song")

    def test_strips_explicit_when_not_in_video_title(self):
        track = "Song (Explicit)"
        video_title = "Artist - Song"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song")

    def test_keeps_feat_even_when_not_in_video_title(self):
        """Feature credits are not on the whitelist and should never be stripped."""
        track = "Song (feat. Other Artist)"
        video_title = "Main Artist - Song (Official Video)"
        self.assertEqual(
            strip_unmatched_suffixes(track, video_title),
            "Song (feat. Other Artist)",
        )

    def test_keeps_feat_when_in_video_title(self):
        """Feature credits should be kept when the video title mentions them."""
        track = "Song (feat. Other Artist)"
        video_title = "Main Artist - Song (feat. Other Artist) [Official]"
        self.assertEqual(
            strip_unmatched_suffixes(track, video_title),
            "Song (feat. Other Artist)",
        )

    def test_keeps_part_number(self):
        """'(Part 2)' is not on the whitelist and should never be stripped."""
        track = "Song (Part 2)"
        video_title = "Artist - Song"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song (Part 2)")

    def test_strips_multiple_trailing_whitelisted_suffixes(self):
        """Multiple trailing whitelist suffixes that are all absent should be stripped."""
        track = "Track Name (Remastered) (live)"
        video_title = "Artist - Track Name (Official Video)"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Track Name")

    def test_no_parenthetical_returns_unchanged(self):
        track = "Simple Track Name"
        video_title = "Artist - Simple Track Name (Official Video)"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Simple Track Name")

    def test_empty_track_returns_empty(self):
        self.assertEqual(strip_unmatched_suffixes("", "Some Video"), "")

    def test_case_insensitive_matching(self):
        """Suffix matching against video title should be case-insensitive."""
        track = "Song (LIVE)"
        video_title = "Artist - Song (live session)"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song (LIVE)")

    def test_strips_only_whitelisted_keeps_non_whitelisted(self):
        """When a non-whitelisted suffix sits behind a whitelisted one,
        only the whitelisted outer one is stripped."""
        track = "Song (Remix) (live)"
        video_title = "Artist - Song (Remix)"
        # "(live)" is whitelisted and not in title -> strip
        # "(Remix)" is NOT whitelisted -> keep regardless
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song (Remix)")

    def test_strips_lyrics_when_not_in_video_title(self):
        track = "Song (Lyrics)"
        video_title = "Artist - Song"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song")

    def test_strips_lyric_video_when_not_in_video_title(self):
        track = "Song (Lyric Video)"
        video_title = "Artist - Song"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song")

    def test_strips_official_video_suffix(self):
        track = "Song (Official Video)"
        video_title = "Artist - Song"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song")

    def test_no_false_positive_substring_match(self):
        """'live' inside 'delivered' should NOT count as a match."""
        track = "Song (Live)"
        video_title = "Artist - Song delivered to you"
        # "live" appears as substring of "delivered" but NOT as whole word
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song")

    def test_word_boundary_keeps_real_match(self):
        """'live' as a standalone word in the title should keep the suffix."""
        track = "Song (Live)"
        video_title = "Artist - Song live version"
        self.assertEqual(strip_unmatched_suffixes(track, video_title), "Song (Live)")


class TestGetYoutubeTitle(unittest.TestCase):
    @patch("yt_dlp.YoutubeDL")
    def test_get_youtube_title(self, mock_youtube_dl):
        # Arrange
        # Also test leading and trailing whitespaces
        mock_youtube_dl.return_value.__enter__.return_value.extract_info.return_value = {
            "artist": "   Test Artist   ",
            "track": "   Test Track   ",
            "title": "   Test Artist - Test Track   ",
            "channel": "   Test Channel   "
        }
        url = "   https://fakeUrl   "

        # Act
        result = get_youtube_title(url)

        # Assert
        self.assertEqual(result, ("Test Artist", "Test Track"))
        mock_youtube_dl.assert_called_once()

    @patch("yt_dlp.YoutubeDL")
    def test_get_youtube_title_strips_yt_music_suffix(self, mock_youtube_dl):
        """Track metadata says '(live)' but video title does not."""
        mock_youtube_dl.return_value.__enter__.return_value.extract_info.return_value = {
            "artist": "Electric Callboy",
            "track": "Pump It (live)",
            "title": "Electric Callboy - PUMP IT (OFFICIAL VIDEO)",
            "channel": "Electric Callboy",
        }

        artist, title = get_youtube_title("https://fakeUrl")

        self.assertEqual(artist, "Electric Callboy")
        self.assertEqual(title, "Pump It")

    @patch("yt_dlp.YoutubeDL")
    def test_get_youtube_title_keeps_matching_suffix(self, mock_youtube_dl):
        """If track and video title agree on the suffix, keep it."""
        mock_youtube_dl.return_value.__enter__.return_value.extract_info.return_value = {
            "artist": "Artist",
            "track": "Song (Live)",
            "title": "Artist - Song (Live at Festival)",
            "channel": "ArtistChannel",
        }

        artist, title = get_youtube_title("https://fakeUrl")

        self.assertEqual(artist, "Artist")
        self.assertEqual(title, "Song (Live)")

    @patch("yt_dlp.YoutubeDL")
    def test_get_youtube_title_no_artist_with_dash(self, mock_youtube_dl):
        """Fallback: no artist field, title contains dash."""
        mock_youtube_dl.return_value.__enter__.return_value.extract_info.return_value = {
            "title": "  Some Artist - Some Song  ",
            "channel": "SomeChannel",
        }

        artist, title = get_youtube_title("https://fakeUrl")

        self.assertEqual(artist, "Some Artist")
        self.assertEqual(title, "Some Song")

    @patch("yt_dlp.YoutubeDL")
    def test_get_youtube_title_multi_dash_preserves_title(self, mock_youtube_dl):
        """Multiple dashes in title: only split on the first one."""
        mock_youtube_dl.return_value.__enter__.return_value.extract_info.return_value = {
            "title": "Artist Name - Song - Part 2",
            "channel": "ArtistChannel",
        }

        artist, title = get_youtube_title("https://fakeUrl")

        self.assertEqual(artist, "Artist Name")
        self.assertEqual(title, "Song - Part 2")

    @patch("yt_dlp.YoutubeDL")
    def test_get_youtube_title_no_artist_no_dash(self, mock_youtube_dl):
        """Fallback: no artist field, no dash in title -> use channel."""
        mock_youtube_dl.return_value.__enter__.return_value.extract_info.return_value = {
            "title": "Just A Title",
            "channel": "MyChannel",
        }

        artist, title = get_youtube_title("https://fakeUrl")

        self.assertEqual(artist, "MyChannel")
        self.assertEqual(title, "Just A Title")

    @patch("yt_dlp.YoutubeDL")
    def test_get_youtube_title_artist_without_track_falls_through(self, mock_youtube_dl):
        """If yt-dlp provides artist but no track, fall through to title parsing."""
        mock_youtube_dl.return_value.__enter__.return_value.extract_info.return_value = {
            "artist": "Some Artist",
            "title": "Some Artist - Some Song",
            "channel": "SomeChannel",
        }

        artist, title = get_youtube_title("https://fakeUrl")

        # Should fall through to title-split logic
        self.assertEqual(artist, "Some Artist")
        self.assertEqual(title, "Some Song")

    @patch("yt_dlp.YoutubeDL")
    def test_get_youtube_title_track_none_falls_through(self, mock_youtube_dl):
        """If yt-dlp provides artist but track is None, fall through."""
        mock_youtube_dl.return_value.__enter__.return_value.extract_info.return_value = {
            "artist": "Some Artist",
            "track": None,
            "title": "Some Artist - Some Song",
            "channel": "SomeChannel",
        }

        artist, title = get_youtube_title("https://fakeUrl")

        # Should fall through to title-split logic
        self.assertEqual(artist, "Some Artist")
        self.assertEqual(title, "Some Song")

    @patch("src.modules.Audio.youtube.yt_dlp.YoutubeDL")
    @patch("src.modules.Audio.youtube.save_image")
    def test_download_and_convert_thumbnail(self, mock_save_image, mock_youtube_dl):
        # Arrange
        mock_youtube_dl.return_value.__enter__.return_value.extract_info.return_value = {"thumbnail": "test_thumbnail_url"}
        mock_youtube_dl.return_value.__enter__.return_value.urlopen.return_value.read.return_value = b"test_image_data"

        mock_save_image.return_value = None

        ydl_opts = {}
        url = "https://fakeUrl"
        clear_filename = "test"
        output_path = "/path/to/output"

        # Act
        download_and_convert_thumbnail(ydl_opts, url, clear_filename, output_path)

        # Assert
        mock_youtube_dl.assert_called_once_with(ydl_opts)
        mock_youtube_dl.return_value.__enter__.return_value.extract_info.assert_called_once_with(url, download=False)
        mock_youtube_dl.return_value.__enter__.return_value.urlopen.assert_called_once_with("test_thumbnail_url")

if __name__ == "__main__":
    unittest.main()
