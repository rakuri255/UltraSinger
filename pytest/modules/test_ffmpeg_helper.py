"""Tests for ffmpeg_helper.py"""

import os
import shutil
import subprocess
import tempfile
import unittest

import pytest

from modules.ffmpeg_helper import is_video_file

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe not available")
class IsVideoFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_audio_with_cover_art_is_not_video(self):
        cover = os.path.join(self.tmp, "cover.png")
        mp3 = os.path.join(self.tmp, "song.mp3")
        _ffmpeg("-f", "lavfi", "-i", "color=c=red:s=64x64:d=1", "-frames:v", "1", cover)
        _ffmpeg(
            "-f", "lavfi", "-i", "sine=d=1", "-i", cover,
            "-map", "0:a", "-map", "1:v", "-c:a", "libmp3lame", "-c:v", "png",
            "-disposition:v", "attached_pic",
            mp3,
        )
        self.assertFalse(is_video_file(mp3))

    def test_audio_without_video_stream_is_not_video(self):
        mp3 = os.path.join(self.tmp, "song.mp3")
        _ffmpeg("-f", "lavfi", "-i", "sine=d=1", "-c:a", "libmp3lame", mp3)
        self.assertFalse(is_video_file(mp3))

    def test_real_video_is_video(self):
        mp4 = os.path.join(self.tmp, "video.mp4")
        _ffmpeg("-f", "lavfi", "-i", "testsrc=d=1:s=64x64", "-f", "lavfi", "-i", "sine=d=1", "-shortest", mp4)
        self.assertTrue(is_video_file(mp4))


if __name__ == "__main__":
    unittest.main()
