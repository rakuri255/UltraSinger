# programmatically customize settings for evaluation runs

import os

from Settings import Settings


def init_settings() -> Settings:
    settings = Settings()
    settings.language = "en"
    # settings.pitch_loudness_threshold = 10000
    settings.create_midi = False
    settings.create_plot = False
    settings.calculate_score = True
    settings.create_karaoke = False
    settings.keep_cache = True
    settings.ignore_audio = False
    # settings.whisper_batch_size = 12
    # settings.whisper_compute_type = "int8"
    # settings.test_songs_input_folder = "C:/Users/Benedikt/git/songs/Creative Commons"
    # settings.skip_cache_vocal_separation = True
    # settings.skip_cache_denoise_vocal_audio = True
    # settings.skip_cache_transcription = True
    # settings.skip_cache_pitch_detection = True


    dedicated_test_folder = ""
    # dedicated_test_folder = "C:/My/Dedicated/Test/songs"
    dedicated_test_songs_exist = False
    if os.path.isdir(dedicated_test_folder):
        for item in os.listdir(dedicated_test_folder):
            if os.path.isdir(os.path.join(dedicated_test_folder, item)):
                dedicated_test_songs_exist = True

    if dedicated_test_songs_exist:
        settings.test_songs_input_folder = dedicated_test_folder

    return settings


user_settings = init_settings()
