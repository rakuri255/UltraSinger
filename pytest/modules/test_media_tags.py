"""Tests for reading song metadata from embedded file tags"""

import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import pytest

from modules.ffmpeg_helper import get_media_tags
from modules.musicbrainz_client import SongInfo

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _create_mp3(path: str, **tags: str) -> None:
    metadata = []
    for key, value in tags.items():
        metadata += ["-metadata", f"{key}={value}"]
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=d=1", "-c:a", "libmp3lame", *metadata, path],
        check=True,
    )


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe not available")
class MediaTagsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.output = os.path.join(self.tmp, "output")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _infos(self, input_path: str, metadata_source: str, musicbrainz_result: SongInfo):
        import UltraSinger

        settings = UltraSinger.settings
        with mock.patch.object(settings, "input_file_path", input_path), \
                mock.patch.object(settings, "output_folder_path", self.output), \
                mock.patch.object(settings, "metadata_source", metadata_source), \
                mock.patch.object(UltraSinger, "search_musicbrainz", return_value=musicbrainz_result) as search:
            basename, _, _, media_info = UltraSinger.infos_from_audio_video_input_file()
        return basename, media_info, search

    def test_reads_media_tags(self):
        mp3 = os.path.join(self.tmp, "song.mp3")
        _create_mp3(mp3, artist="Some Artist", title="Some Title", date="2013", genre="Rock")

        tags = get_media_tags(mp3)

        self.assertEqual("Some Artist", tags.get("artist"))
        self.assertEqual("Some Title", tags.get("title"))
        self.assertEqual("2013", tags.get("date"))
        self.assertEqual("Rock", tags.get("genre"))

    def test_missing_file_returns_empty_tags(self):
        self.assertEqual({}, get_media_tags(os.path.join(self.tmp, "missing.mp3")))

    def test_auto_uses_tags_for_musicbrainz_search(self):
        mp3 = os.path.join(self.tmp, "some_file_name.mp3")
        _create_mp3(mp3, artist="Some Artist", title="Some Title")
        found = SongInfo(title="Found Title", artist="Found Artist", year="2001")

        basename, media_info, search = self._infos(mp3, "auto", found)

        search.assert_called_once_with("Some Title", "Some Artist")
        self.assertEqual("Found Artist - Found Title", basename)
        self.assertEqual("2001", media_info.year)

    def test_auto_falls_back_to_tags_without_musicbrainz_match(self):
        mp3 = os.path.join(self.tmp, "some_file_name.mp3")
        _create_mp3(mp3, artist="Some Artist", title="Some Title: Live?", date="2013-05-01", genre="Rock")
        not_found = SongInfo(title="Some Title: Live?", artist="Unknown Artist")

        basename, media_info, _ = self._infos(mp3, "auto", not_found)

        self.assertEqual("Some Artist", media_info.artist)
        self.assertEqual("Some Title: Live?", media_info.title)
        self.assertEqual("2013", media_info.year)
        self.assertEqual("Rock", media_info.genre)
        # invalid characters are removed from the file name
        self.assertEqual("Some Artist - Some Title Live", basename)

    def test_tags_mode_skips_musicbrainz(self):
        mp3 = os.path.join(self.tmp, "some_file_name.mp3")
        _create_mp3(mp3, artist="Some Artist", title="Some Title")

        basename, media_info, search = self._infos(mp3, "tags", None)

        search.assert_not_called()
        self.assertEqual("Some Artist - Some Title", basename)

    def test_musicbrainz_mode_ignores_tags(self):
        mp3 = os.path.join(self.tmp, "File Artist - File Title.mp3")
        _create_mp3(mp3, artist="Some Artist", title="Some Title")
        not_found = SongInfo(title="File Title", artist="Unknown Artist")

        basename, media_info, search = self._infos(mp3, "musicbrainz", not_found)

        search.assert_called_once_with("File Title", "File Artist")
        self.assertEqual("Unknown Artist", media_info.artist)


if __name__ == "__main__":
    unittest.main()
