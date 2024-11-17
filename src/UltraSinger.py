"""UltraSinger uses AI to automatically create UltraStar song files"""

import copy
import getopt
import os
import sys
import Levenshtein

from packaging import version

from modules import os_helper, timer
from modules.Audio.denoise import denoise_vocal_audio
from modules.Audio.separation import separate_vocal_from_audio
from modules.Audio.vocal_chunks import (
    create_audio_chunks_from_transcribed_data,
    create_audio_chunks_from_ultrastar_data,
)
from modules.Audio.silence_processing import remove_silence_from_transcription_data, mute_no_singing_parts

from modules.Audio.convert_audio import convert_audio_to_mono_wav, convert_wav_to_mp3
from modules.Audio.youtube import (
    download_from_youtube,
)

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    gold_highlighted,
    red_highlighted,
    green_highlighted,
)
from modules.Midi.midi_creator import (
    create_midi_segments_from_transcribed_data,
    create_repitched_midi_segments_from_ultrastar_txt,
    create_midi_file,
)

from modules.Pitcher.pitcher import (
    get_pitch_with_crepe_file,
)
from modules.Pitcher.pitched_data import PitchedData
from modules.Speech_Recognition.TranscriptionResult import TranscriptionResult
from modules.Speech_Recognition.hyphenation import (
    hyphenate_each_word,
)
from modules.Speech_Recognition.Whisper import transcribe_with_whisper
from modules.Ultrastar import (
    ultrastar_writer,
)
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.Ultrastar.ultrastar_score_calculator import Score, calculate_score_points
from modules.Ultrastar.ultrastar_txt import FILE_ENCODING, FormatVersion
from modules.Ultrastar.converter.ultrastar_txt_converter import from_ultrastar_txt, \
    create_ultrastar_txt_from_midi_segments, create_ultrastar_txt_from_automation
from modules.Ultrastar.ultrastar_parser import parse_ultrastar_txt
from modules.common_print import print_support, print_help, print_version
from modules.os_helper import check_file_exists, get_unused_song_output_dir
from modules.plot import create_plots
from modules.musicbrainz_client import get_music_infos
from modules.sheet import create_sheet
from modules.ProcessData import ProcessData, ProcessDataPaths, MediaInfo
from modules.DeviceDetection.device_detection import check_gpu_support
from modules.Audio.bpm import get_bpm_from_file

from Settings import Settings

settings = Settings()


def add_hyphen_to_data(
        transcribed_data: list[TranscribedData], hyphen_words: list[list[str]]
):
    """Add hyphen to transcribed data return new data list"""
    new_data = []

    for i, data in enumerate(transcribed_data):
        if not hyphen_words[i]:
            new_data.append(data)
        else:
            chunk_duration = data.end - data.start
            chunk_duration = chunk_duration / (len(hyphen_words[i]))

            next_start = data.start
            for j in enumerate(hyphen_words[i]):
                hyphenated_word_index = j[0]
                dup = copy.copy(data)
                dup.start = next_start
                next_start = data.end - chunk_duration * (
                        len(hyphen_words[i]) - 1 - hyphenated_word_index
                )
                dup.end = next_start
                dup.word = hyphen_words[i][hyphenated_word_index]
                dup.is_hyphen = True
                if hyphenated_word_index == len(hyphen_words[i]) - 1:
                    dup.is_word_end = True
                else:
                    dup.is_word_end = False
                new_data.append(dup)

    return new_data


# Todo: Unused
def correct_words(recognized_words, word_list_file):
    """Docstring"""
    with open(word_list_file, "r", encoding="utf-8") as file:
        text = file.read()
    word_list = text.split()

    for i, rec_word in enumerate(recognized_words):
        if rec_word.word in word_list:
            continue

        closest_word = min(
            word_list, key=lambda x: Levenshtein.distance(rec_word.word, x)
        )
        print(recognized_words[i].word + " - " + closest_word)
        recognized_words[i].word = closest_word
    return recognized_words


def remove_unecessary_punctuations(transcribed_data: list[TranscribedData]) -> None:
    """Remove unecessary punctuations from transcribed data"""
    punctuation = ".,"
    for i, data in enumerate(transcribed_data):
        data.word = data.word.translate({ord(i): None for i in punctuation})


def run() -> tuple[str, Score, Score]:
    """The processing function of this program"""

    process_data = InitProcessData()

    process_data.process_data_paths.cache_folder_path = (
        os.path.join(settings.output_folder_path, "cache")
        if settings.cache_override_path is None
        else settings.cache_override_path
    )

    # Create process audio
    process_data.process_data_paths.processing_audio_path = CreateProcessAudio(process_data)

    # Audio transcription
    process_data.media_info.language = settings.language
    if not settings.ignore_audio:
        TranscribeAudio(process_data)

    # Create audio chunks
    if settings.create_audio_chunks:
        create_audio_chunks(process_data)

    # Pitch audio
    process_data.pitched_data = pitch_audio(process_data.process_data_paths)

    # Create Midi_Segments
    if not settings.ignore_audio:
        process_data.midi_segments = create_midi_segments_from_transcribed_data(process_data.transcribed_data,
                                                                                process_data.pitched_data)
    else:
        process_data.midi_segments = create_repitched_midi_segments_from_ultrastar_txt(process_data.pitched_data,
                                                                                       process_data.parsed_file)

    # Create plot
    if settings.create_plot:
        create_plots(process_data, settings.output_folder_path)

    # Create Ultrastar txt
    accurate_score, simple_score, ultrastar_file_output = CreateUltraStarTxt(process_data)

    # Create Midi
    if settings.create_midi:
        create_midi_file(process_data.media_info.bpm, settings.output_folder_path, process_data.midi_segments,
                         process_data.basename)

    # Sheet music
    create_sheet(process_data.midi_segments, settings.output_folder_path,
                 process_data.process_data_paths.cache_folder_path, settings.musescore_path, process_data.basename,
                 process_data.media_info)

    # Cleanup
    if not settings.keep_cache:
        remove_cache_folder(process_data.process_data_paths.cache_folder_path)

    # Print Support
    print_support()
    return ultrastar_file_output, simple_score, accurate_score


def create_audio_chunks(process_data):
    if not settings.ignore_audio:
        create_audio_chunks_from_transcribed_data(
            process_data.process_data_paths,
            process_data.transcribed_data)
    else:
        create_audio_chunks_from_ultrastar_data(
            process_data.process_data_paths,
            process_data.parsed_file
        )


def InitProcessData():
    settings.input_file_is_ultrastar_txt = settings.input_file_path.endswith(".txt")
    if settings.input_file_is_ultrastar_txt:
        # Parse Ultrastar txt
        (
            basename,
            settings.output_folder_path,
            audio_file_path,
            ultrastar_class,
        ) = parse_ultrastar_txt(settings.input_file_path, settings.output_folder_path)
        process_data = from_ultrastar_txt(ultrastar_class)
        process_data.basename = basename
        process_data.process_data_paths.audio_output_file_path = audio_file_path
        # todo: ignore transcribe
        if settings.ignore_audio is None:
            settings.ignore_audio = True

    elif settings.input_file_path.startswith("https:"):
        # Youtube
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('full automatic mode')}")
        process_data = ProcessData()
        (
            process_data.basename,
            settings.output_folder_path,
            process_data.process_data_paths.audio_output_file_path,
            process_data.media_info,
        ) = download_from_youtube(settings.input_file_path, settings.output_folder_path)
    else:
        # Audio File
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('full automatic mode')}")
        process_data = ProcessData()
        (
            process_data.basename,
            settings.output_folder_path,
            process_data.process_data_paths.audio_output_file_path,
            process_data.media_info,
        ) = infos_from_audio_input_file()
    return process_data


def TranscribeAudio(process_data):
    transcription_result = transcribe_audio(process_data.process_data_paths.cache_folder_path,
                                            process_data.process_data_paths.processing_audio_path)

    if process_data.media_info.language is None:
        process_data.media_info.language = transcription_result.detected_language

    process_data.transcribed_data = transcription_result.transcribed_data

    # Hyphen
    # Todo: Is it really unnecessary?
    remove_unecessary_punctuations(process_data.transcribed_data)
    if settings.hyphenation:
        hyphen_words = hyphenate_each_word(process_data.media_info.language, process_data.transcribed_data)

        if hyphen_words is not None:
            process_data.transcribed_data = add_hyphen_to_data(process_data.transcribed_data, hyphen_words)

    process_data.transcribed_data = remove_silence_from_transcription_data(
        process_data.process_data_paths.processing_audio_path, process_data.transcribed_data
    )


def CreateUltraStarTxt(process_data: ProcessData):
    # Move instrumental and vocals
    if settings.create_karaoke and version.parse(settings.format_version.value) < version.parse(FormatVersion.V1_1_0.value):
        karaoke_output_path = os.path.join(settings.output_folder_path, process_data.basename + " [Karaoke].mp3")
        convert_wav_to_mp3(process_data.process_data_paths.instrumental_audio_file_path, karaoke_output_path)

    if version.parse(settings.format_version.value) >= version.parse(FormatVersion.V1_1_0.value):
        instrumental_output_path = os.path.join(settings.output_folder_path,
                                                process_data.basename + " [Instrumental].mp3")
        convert_wav_to_mp3(process_data.process_data_paths.instrumental_audio_file_path, instrumental_output_path)
        vocals_output_path = os.path.join(settings.output_folder_path, process_data.basename + " [Vocals].mp3")
        convert_wav_to_mp3(process_data.process_data_paths.vocals_audio_file_path, vocals_output_path)

    # Create Ultrastar txt
    if not settings.ignore_audio:
        ultrastar_file_output = create_ultrastar_txt_from_automation(
            process_data.basename,
            settings.output_folder_path,
            process_data.midi_segments,
            process_data.media_info,
            settings.format_version,
            settings.create_karaoke,
            settings.APP_VERSION
        )
    else:
        ultrastar_file_output = create_ultrastar_txt_from_midi_segments(
            settings.output_folder_path, settings.input_file_path, process_data.media_info.title, process_data.midi_segments
        )

    # Calc Points
    simple_score = None
    accurate_score = None
    if settings.calculate_score:
        simple_score, accurate_score = calculate_score_points(process_data, ultrastar_file_output)

        # Add calculated score to Ultrastar txt
    #Todo: Missing Karaoke
        ultrastar_writer.add_score_to_ultrastar_txt(ultrastar_file_output, simple_score)
    return accurate_score, simple_score, ultrastar_file_output


def CreateProcessAudio(process_data) -> str:
    # Set processing audio to cache file
    process_data.process_data_paths.processing_audio_path = os.path.join(
        process_data.process_data_paths.cache_folder_path, process_data.basename + ".wav"
    )
    os_helper.create_folder(process_data.process_data_paths.cache_folder_path)

    # Separate vocal from audio
    audio_separation_folder_path = separate_vocal_from_audio(
        process_data.process_data_paths.cache_folder_path,
        process_data.process_data_paths.audio_output_file_path,
        settings.use_separated_vocal,
        settings.create_karaoke,
        settings.pytorch_device,
        settings.demucs_model,
        settings.skip_cache_vocal_separation
    )
    process_data.process_data_paths.vocals_audio_file_path = os.path.join(audio_separation_folder_path, "vocals.wav")
    process_data.process_data_paths.instrumental_audio_file_path = os.path.join(audio_separation_folder_path,
                                                                                "no_vocals.wav")

    if settings.use_separated_vocal:
        input_path = process_data.process_data_paths.vocals_audio_file_path
    else:
        input_path = process_data.process_data_paths.audio_output_file_path

    # Denoise vocal audio
    denoised_output_path = os.path.join(
        process_data.process_data_paths.cache_folder_path, process_data.basename + "_denoised.wav"
    )
    denoise_vocal_audio(input_path, denoised_output_path, settings.skip_cache_denoise_vocal_audio)

    # Convert to mono audio
    mono_output_path = os.path.join(
        process_data.process_data_paths.cache_folder_path, process_data.basename + "_mono.wav"
    )
    convert_audio_to_mono_wav(denoised_output_path, mono_output_path)

    # Mute silence sections
    mute_output_path = os.path.join(
        process_data.process_data_paths.cache_folder_path, process_data.basename + "_mute.wav"
    )
    mute_no_singing_parts(mono_output_path, mute_output_path)

    # Define the audio file to process
    return mute_output_path


def transcribe_audio(cache_folder_path: str, processing_audio_path: str) -> TranscriptionResult:
    """Transcribe audio with AI"""
    transcription_result = None
    if settings.transcriber == "whisper":
        transcription_config = f"{settings.transcriber}_{settings.whisper_model.value}_{settings.pytorch_device}_{settings.whisper_align_model}_{settings.whisper_align_model}_{settings.whisper_batch_size}_{settings.whisper_compute_type}_{settings.language}"
        transcription_path = os.path.join(cache_folder_path, f"{transcription_config}.json")
        cached_transcription_available = check_file_exists(transcription_path)
        if settings.skip_cache_transcription or not cached_transcription_available:
            transcription_result = transcribe_with_whisper(
                processing_audio_path,
                settings.whisper_model,
                settings.pytorch_device,
                settings.whisper_align_model,
                settings.whisper_batch_size,
                settings.whisper_compute_type,
                settings.language,
            )
            with open(transcription_path, "w", encoding=FILE_ENCODING) as file:
                file.write(transcription_result.to_json())
        else:
            print(f"{ULTRASINGER_HEAD} {green_highlighted('cache')} reusing cached transcribed data")
            with open(transcription_path) as file:
                json = file.read()
                transcription_result = TranscriptionResult.from_json(json)
    else:
        raise NotImplementedError
    return transcription_result


def infos_from_audio_input_file() -> tuple[str, str, str, MediaInfo]:
    """Infos from audio input file"""
    basename = os.path.basename(settings.input_file_path)
    basename_without_ext = os.path.splitext(basename)[0]

    artist, title = None, None
    if " - " in basename_without_ext:
        artist, title = basename_without_ext.split(" - ", 1)
        search_string = f"{artist} - {title}"
    else:
        search_string = basename_without_ext

    # Get additional data for song
    (title_info, artist_info, year_info, genre_info) = get_music_infos(search_string)

    if title_info is not None:
        title = title_info
        artist = artist_info
    else:
        title = basename_without_ext
        artist = "Unknown Artist"

    if artist is not None and title is not None:
        basename_without_ext = f"{artist} - {title}"
        extension = os.path.splitext(basename)[1]
        basename = f"{basename_without_ext}{extension}"

    song_folder_output_path = os.path.join(settings.output_folder_path, basename_without_ext)
    song_folder_output_path = get_unused_song_output_dir(song_folder_output_path)
    os_helper.create_folder(song_folder_output_path)
    os_helper.copy(settings.input_file_path, song_folder_output_path)
    os_helper.rename(
        os.path.join(song_folder_output_path, os.path.basename(settings.input_file_path)),
        os.path.join(song_folder_output_path, basename),
    )
    # Todo: Read ID3 tags
    ultrastar_audio_input_path = os.path.join(song_folder_output_path, basename)
    real_bpm = get_bpm_from_file(settings.input_file_path)
    return (
        basename_without_ext,
        song_folder_output_path,
        ultrastar_audio_input_path,
        MediaInfo(artist=artist, title=title, year=year_info, genre=genre_info, bpm=real_bpm),
    )


def pitch_audio(
        process_data_paths: ProcessDataPaths) -> PitchedData:
    """Pitch audio"""

    pitching_config = f"crepe_{settings.ignore_audio}_{settings.crepe_model_capacity}_{settings.crepe_step_size}_{settings.tensorflow_device}"
    pitched_data_path = os.path.join(process_data_paths.cache_folder_path, f"{pitching_config}.json")
    cache_available = check_file_exists(pitched_data_path)

    if settings.skip_cache_transcription or not cache_available:
        pitched_data = get_pitch_with_crepe_file(
            process_data_paths.processing_audio_path,
            settings.crepe_model_capacity,
            settings.crepe_step_size,
            settings.tensorflow_device,
        )

        pitched_data_json = pitched_data.to_json()
        with open(pitched_data_path, "w", encoding=FILE_ENCODING) as file:
            file.write(pitched_data_json)
    else:
        print(f"{ULTRASINGER_HEAD} {green_highlighted('cache')} reusing cached pitch data")
        with open(pitched_data_path) as file:
            json = file.read()
            pitched_data = PitchedData.from_json(json)

    return pitched_data


def main(argv: list[str]) -> None:
    """Main function"""
    print_version(settings.APP_VERSION)
    init_settings(argv)
    run()
    sys.exit()


def remove_cache_folder(cache_folder_path: str) -> None:
    """Remove cache folder"""
    os_helper.remove_folder(cache_folder_path)


def init_settings(argv: list[str]) -> Settings:
    """Init settings"""
    long, short = arg_options()
    opts, args = getopt.getopt(argv, short, long)
    if len(opts) == 0:
        print_help()
        sys.exit()
    for opt, arg in opts:
        if opt == "-h":
            print_help()
            sys.exit()
        elif opt in ("-i", "--ifile"):
            settings.input_file_path = arg
        elif opt in ("-o", "--ofile"):
            settings.output_folder_path = arg
        elif opt in ("--whisper"):
            settings.transcriber = "whisper"
            settings.whisper_model = arg
        elif opt in ("--whisper_align_model"):
            settings.whisper_align_model = arg
        elif opt in ("--whisper_batch_size"):
            settings.whisper_batch_size = int(arg)
        elif opt in ("--whisper_compute_type"):
            settings.whisper_compute_type = arg
        elif opt in ("--language"):
            settings.language = arg
        elif opt in ("--crepe"):
            settings.crepe_model_capacity = arg
        elif opt in ("--crepe_step_size"):
            settings.crepe_step_size = int(arg)
        elif opt in ("--plot"):
            settings.create_plot = arg in ["True", "true"]
        elif opt in ("--midi"):
            settings.create_midi = arg in ["True", "true"]
        elif opt in ("--hyphenation"):
            settings.hyphenation = eval(arg.title())
        elif opt in ("--disable_separation"):
            settings.use_separated_vocal = not arg
        elif opt in ("--disable_karaoke"):
            settings.create_karaoke = not arg
        elif opt in ("--create_audio_chunks"):
            settings.create_audio_chunks = arg
        elif opt in ("--ignore_audio"):
            settings.ignore_audio = arg in ["True", "true"]
        elif opt in ("--force_cpu"):
            settings.force_cpu = arg
            if settings.force_cpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        elif opt in ("--force_whisper_cpu"):
            settings.force_whisper_cpu = eval(arg.title())
        elif opt in ("--force_crepe_cpu"):
            settings.force_crepe_cpu = eval(arg.title())
        elif opt in ("--format_version"):
            if arg == FormatVersion.V0_3_0.value:
                settings.format_version = FormatVersion.V0_3_0
            elif arg == FormatVersion.V1_0_0.value:
                settings.format_version = FormatVersion.V1_0_0
            elif arg == FormatVersion.V1_1_0.value:
                settings.format_version = FormatVersion.V1_1_0
            else:
                print(
                    f"{ULTRASINGER_HEAD} {red_highlighted('Error: Format version')} {blue_highlighted(arg)} {red_highlighted('is not supported.')}"
                )
                sys.exit(1)
        elif opt in ("--keep_cache"):
            settings.keep_cache = arg
        elif opt in ("--musescore_path"):
            settings.musescore_path = arg
    if settings.output_folder_path == "":
        if settings.input_file_path.startswith("https:"):
            dirname = os.getcwd()
        else:
            dirname = os.path.dirname(settings.input_file_path)
        settings.output_folder_path = os.path.join(dirname, "output")

    if not settings.force_cpu:
        settings.tensorflow_device, settings.pytorch_device = check_gpu_support()

    return settings


def arg_options():
    short = "hi:o:amv:"
    long = [
        "ifile=",
        "ofile=",
        "crepe=",
        "crepe_step_size=",
        "whisper=",
        "whisper_align_model=",
        "whisper_batch_size=",
        "whisper_compute_type=",
        "language=",
        "plot=",
        "midi=",
        "hyphenation=",
        "disable_separation=",
        "disable_karaoke=",
        "create_audio_chunks=",
        "ignore_audio=",
        "force_cpu=",
        "force_whisper_cpu=",
        "force_crepe_cpu=",
        "format_version=",
        "keep_cache",
        "musescore_path="
    ]
    return long, short


if __name__ == "__main__":
    main(sys.argv[1:])
