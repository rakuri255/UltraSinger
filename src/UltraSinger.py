"""UltraSinger uses AI to automatically create UltraStar song files"""

import copy
import getopt
import os
import sys
import re

import Levenshtein
import librosa
from tqdm import tqdm

from modules import os_helper
from modules.Audio.denoise import ffmpeg_reduce_noise
from modules.Audio.separation import separate_audio
from modules.Audio.vocal_chunks import (
    export_chunks_from_transcribed_data,
    export_chunks_from_ultrastar_data,
)
from modules.Audio.silence_processing import remove_silence_from_transcription_data
from modules.csv_handler import export_transcribed_data_to_csv
from modules.Audio.convert_audio import convert_audio_to_mono_wav, convert_wav_to_mp3
from modules.Audio.youtube import (
    download_youtube_audio,
    download_youtube_thumbnail,
    download_youtube_video,
    get_youtube_title,
)
from modules.DeviceDetection.device_detection import check_gpu_support
from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    gold_highlighted,
    light_blue_highlighted,
    red_highlighted,
)
from modules.Midi import midi_creator
from modules.Midi.midi_creator import (
    convert_frequencies_to_notes,
    create_midi_notes_from_pitched_data,
    most_frequent,
)
from modules.Pitcher.pitcher import (
    get_frequencies_with_high_confidence,
    get_pitch_with_crepe_file,
)
from modules.Pitcher.pitched_data import PitchedData
from modules.Speech_Recognition.hyphenation import hyphenation, language_check, create_hyphenator
from modules.Speech_Recognition.Whisper import transcribe_with_whisper
from modules.Ultrastar import ultrastar_score_calculator, ultrastar_writer, ultrastar_converter, ultrastar_parser
from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue
from Settings import Settings
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.plot import plot, plot_spectrogram
from modules.musicbrainz_client import get_music_infos

settings = Settings()


def convert_midi_notes_to_ultrastar_notes(midi_notes: list[str]) -> list[int]:
    """Convert midi notes to ultrastar notes"""
    print(f"{ULTRASINGER_HEAD} Creating Ultrastar notes from midi data")

    ultrastar_note_numbers = []
    for i in enumerate(midi_notes):
        pos = i[0]
        note_number_librosa = librosa.note_to_midi(midi_notes[pos])
        pitch = ultrastar_converter.midi_note_to_ultrastar_note(
            note_number_librosa
        )
        ultrastar_note_numbers.append(pitch)
        # todo: Progress?
        # print(
        #    f"Note: {midi_notes[i]} midi_note: {str(note_number_librosa)} pitch: {str(pitch)}"
        # )
    return ultrastar_note_numbers


def pitch_each_chunk_with_crepe(directory: str) -> list[str]:
    """Pitch each chunk with crepe and return midi notes"""
    print(
        f"{ULTRASINGER_HEAD} Pitching each chunk with {blue_highlighted('crepe')}"
    )

    midi_notes = []
    for filename in sorted(
        [f for f in os.listdir(directory) if f.endswith(".wav")],
        key=lambda x: int(x.split("_")[1]),
    ):
        filepath = os.path.join(directory, filename)
        # todo: stepsize = duration? then when shorter than "it" it should take the duration. Otherwise there a more notes
        pitched_data = get_pitch_with_crepe_file(
            filepath,
            settings.crepe_model_capacity,
            settings.crepe_step_size,
            settings.tensorflow_device,
        )
        conf_f = get_frequencies_with_high_confidence(
            pitched_data.frequencies, pitched_data.confidence
        )

        notes = convert_frequencies_to_notes(conf_f)
        note = most_frequent(notes)[0][0]

        midi_notes.append(note)
        # todo: Progress?
        # print(filename + " f: " + str(mean))

    return midi_notes


def add_hyphen_to_data(transcribed_data: list[TranscribedData], hyphen_words: list[list[str]]):
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
                new_data.append(dup)

    return new_data


def get_bpm_from_data(data, sampling_rate):
    """Get real bpm from audio data"""
    onset_env = librosa.onset.onset_strength(y=data, sr=sampling_rate)
    wav_tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sampling_rate)

    print(
        f"{ULTRASINGER_HEAD} BPM is {blue_highlighted(str(round(wav_tempo[0], 2)))}"
    )
    return wav_tempo[0]


def get_bpm_from_file(wav_file: str) -> float:
    """Get real bpm from audio file"""
    data, sampling_rate = librosa.load(wav_file, sr=None)
    return get_bpm_from_data(data, sampling_rate)


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


def print_help() -> None:
    """Print help text"""
    help_string = """
    UltraSinger.py [opt] [mode] [transcription] [pitcher] [extra]
    
    [opt]
    -h      This help text.
    -i      Ultrastar.txt
            audio like .mp3, .wav, youtube link
    -o      Output folder
    
    [mode]
    ## INPUT is audio ##
    default  Creates all
    
    # Single file creation selection is in progress, you currently getting all!
    (-u      Create ultrastar txt file) # In Progress
    (-m      Create midi file) # In Progress
    (-s      Create sheet file) # In Progress
    
    ## INPUT is ultrastar.txt ##
    default  Creates all

    # Single selection is in progress, you currently getting all!
    (-r      repitch Ultrastar.txt (input has to be audio)) # In Progress
    (-p      Check pitch of Ultrastar.txt input) # In Progress
    (-m      Create midi file) # In Progress

    [transcription]
    # Default is whisper
    --whisper               Multilingual model > tiny|base|small|medium|large-v1|large-v2  >> ((default) is large-v2
                            English-only model > tiny.en|base.en|small.en|medium.en
    --whisper_align_model   Use other languages model for Whisper provided from huggingface.co
    --language              Override the language detected by whisper, does not affect transcription but steps after transcription
    --whisper_batch_size    Reduce if low on GPU mem >> ((default) is 16)
    --whisper_compute_type  Change to "int8" if low on GPU mem (may reduce accuracy) >> ((default) is "float16" for cuda devices, "int8" for cpu)
    
    [pitcher]
    # Default is crepe
    --crepe            tiny|full >> ((default) is full)
    --crepe_step_size  unit is miliseconds >> ((default) is 10)
    
    [extra]
    --hyphenation           True|False >> ((default) is True)
    --disable_separation    True|False >> ((default) is False)
    --disable_karaoke       True|False >> ((default) is False)
    --create_audio_chunks   True|False >> ((default) is False)
    --plot                  True|False >> ((default) is False)
    
    [device]
    --force_cpu             True|False >> ((default) is False)  All steps will be forced to cpu
    --force_whisper_cpu     True|False >> ((default) is False)  Only whisper will be forced to cpu
    --force_crepe_cpu       True|False >> ((default) is False)  Only crepe will be forced to cpu
    """
    print(help_string)


def remove_unecessary_punctuations(transcribed_data: list[TranscribedData]) -> None:
    """Remove unecessary punctuations from transcribed data"""
    punctuation = ".,"
    for i, data in enumerate(transcribed_data):
        data.word = data.word.translate(
            {ord(i): None for i in punctuation}
        )


def hyphenate_each_word(language: str, transcribed_data: list[TranscribedData]) -> list[list[str]] | None:
    """Hyphenate each word in the transcribed data."""
    lang_region = language_check(language)
    if lang_region is None:
        print(
            f"{ULTRASINGER_HEAD} {red_highlighted('Error in hyphenation for language ')} {blue_highlighted(language)}{red_highlighted(', maybe you want to disable it?')}"
        )
        return None

    hyphenated_word = []
    try:
        hyphenator = create_hyphenator(lang_region)
        for i in tqdm(enumerate(transcribed_data)):
            pos = i[0]
            hyphenated_word.append(
                hyphenation(transcribed_data[pos].word, hyphenator)
            )
    except:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Error in hyphenation for language ')} {blue_highlighted(language)}{red_highlighted(', maybe you want to disable it?')}")
        return None

    return hyphenated_word


def print_support() -> None:
    """Print support text"""
    print()
    print(
        f"{ULTRASINGER_HEAD} {gold_highlighted('Do you like UltraSinger? Want it to be even better? Then help with your')} {light_blue_highlighted('support')}{gold_highlighted('!')}"
    )
    print(
        f"{ULTRASINGER_HEAD} See project page -> https://github.com/rakuri255/UltraSinger"
    )
    print(
        f"{ULTRASINGER_HEAD} {gold_highlighted('This will help a lot to keep this project alive and improved.')}"
    )

def print_version() -> None:
    """Print version text"""
    print()
    print(
        f"{ULTRASINGER_HEAD} {gold_highlighted('*****************************')}"
    )
    print(
        f"{ULTRASINGER_HEAD} {gold_highlighted('UltraSinger Version:')} {light_blue_highlighted(settings.APP_VERSION)}"
    )
    print(
        f"{ULTRASINGER_HEAD} {gold_highlighted('*****************************')}"
    )

def run() -> None:
    """The processing function of this program"""
    is_audio = ".txt" not in settings.input_file_path
    ultrastar_class = None
    real_bpm = None
    (title, artist, year, genre) = (None, None, None, None)

    if not is_audio:  # Parse Ultrastar txt
        print(
            f"{ULTRASINGER_HEAD} {gold_highlighted('re-pitch mode')}"
        )
        (
            basename_without_ext,
            real_bpm,
            song_output,
            ultrastar_audio_input_path,
            ultrastar_class,
        ) = parse_ultrastar_txt()
    elif settings.input_file_path.startswith("https:"):  # Youtube
        print(
            f"{ULTRASINGER_HEAD} {gold_highlighted('full automatic mode')}"
        )
        (
            basename_without_ext,
            song_output,
            ultrastar_audio_input_path,
            (title, artist, year, genre)
        ) = download_from_youtube()
    else:  # Audio File
        print(
            f"{ULTRASINGER_HEAD} {gold_highlighted('full automatic mode')}"
        )
        (
            basename_without_ext,
            song_output,
            ultrastar_audio_input_path,
            (title, artist, year, genre)
        ) = infos_from_audio_input_file()

    cache_path = os.path.join(song_output, "cache")
    settings.mono_audio_path = os.path.join(
        cache_path, basename_without_ext + ".wav"
    )
    os_helper.create_folder(cache_path)

    # Separate vocal from audio
    audio_separation_path = separate_vocal_from_audio(
        basename_without_ext, cache_path, ultrastar_audio_input_path
    )

    if settings.use_separated_vocal:
        input_path = os.path.join(audio_separation_path, "vocals.wav")
    else:
        input_path = ultrastar_audio_input_path

    # Denoise vocal audio
    denoised_output_path = os.path.join(
        cache_path, basename_without_ext + "_denoised.wav"
    )
    denoise_vocal_audio(input_path, denoised_output_path)

    # Convert to mono audio
    mono_output_path = os.path.join(
        cache_path, basename_without_ext + "_mono.wav"
    )
    convert_audio_to_mono_wav(denoised_output_path, mono_output_path)
    settings.mono_audio_path = mono_output_path

    # Audio transcription
    transcribed_data = None
    language = settings.language
    if is_audio:
        detected_language, transcribed_data = transcribe_audio()
        if language is None:
            language = detected_language

        remove_unecessary_punctuations(transcribed_data)
        transcribed_data = remove_silence_from_transcription_data(
            settings.mono_audio_path, transcribed_data
        )

        if settings.hyphenation:
            hyphen_words = hyphenate_each_word(language, transcribed_data)
            if hyphen_words is not None:
                transcribed_data = add_hyphen_to_data(transcribed_data, hyphen_words)

        # todo: do we need to correct words?
        # lyric = 'input/faber_lyric.txt'
        # --corrected_words = correct_words(vosk_speech, lyric)

    # Create audio chunks
    if settings.create_audio_chunks:
        create_audio_chunks(
            cache_path,
            is_audio,
            transcribed_data,
            ultrastar_audio_input_path,
            ultrastar_class,
        )

    # Pitch the audio
    midi_notes, pitched_data, ultrastar_note_numbers = pitch_audio(
        is_audio, transcribed_data, ultrastar_class
    )

    # Create plot
    if settings.create_plot:
        vocals_path = os.path.join(audio_separation_path, "vocals.wav")
        plot_spectrogram(vocals_path, song_output)
        plot(pitched_data, song_output, transcribed_data, midi_notes)

    # Write Ultrastar txt
    if is_audio:
        real_bpm, ultrastar_file_output = create_ultrastar_txt_from_automation(
            audio_separation_path,
            basename_without_ext,
            song_output,
            transcribed_data,
            ultrastar_audio_input_path,
            ultrastar_note_numbers,
            language,
            title,
            artist,
            year,
            genre
        )
    else:
        ultrastar_file_output = create_ultrastar_txt_from_ultrastar_data(
            song_output, ultrastar_class, ultrastar_note_numbers
        )

    # Calc Points
    ultrastar_class, simple_score, accurate_score = calculate_score_points(
        is_audio, pitched_data, ultrastar_class, ultrastar_file_output
    )

    # Add calculated score to Ultrastar txt #Todo: Missing Karaoke
    ultrastar_writer.add_score_to_ultrastar_txt(
        ultrastar_file_output, simple_score
    )

    # Midi
    if settings.create_midi:
        create_midi_file(real_bpm, song_output, ultrastar_class, basename_without_ext)

    # Print Support
    print_support()


def get_unused_song_output_dir(path: str) -> str:
    """Get an unused song output dir"""
    # check if dir exists and add (i) if it does
    i = 1
    if os_helper.check_if_folder_exists(path):
        path = f"{path} ({i})"
    else:
        return path

    while os_helper.check_if_folder_exists(path):
        path = path.replace(f"({i - 1})", f"({i})")
        i += 1
        if i > 999:
            print(
                f"{ULTRASINGER_HEAD} {red_highlighted('Error: Could not create output folder! (999) is the maximum number of tries.')}"
            )
            sys.exit(1)
    return path


def transcribe_audio() -> (str, list[TranscribedData]):
    """Transcribe audio with AI"""
    if settings.transcriber == "whisper":
        device = "cpu" if settings.force_whisper_cpu else settings.pytorch_device
        transcribed_data, detected_language = transcribe_with_whisper(
            settings.mono_audio_path,
            settings.whisper_model,
            device,
            settings.whisper_align_model,
            settings.whisper_batch_size,
            settings.whisper_compute_type,
            settings.language,
        )
    else:
        raise NotImplementedError
    return detected_language, transcribed_data


def separate_vocal_from_audio(
        basename_without_ext: str, cache_path: str, ultrastar_audio_input_path: str
) -> str:
    """Separate vocal from audio"""
    audio_separation_path = os.path.join(
        cache_path, "separated", "htdemucs", basename_without_ext
    )

    if settings.use_separated_vocal or settings.create_karaoke:
        separate_audio(ultrastar_audio_input_path, cache_path, settings.pytorch_device)

    return audio_separation_path

def calculate_score_points(
    is_audio: bool, pitched_data: PitchedData, ultrastar_class: UltrastarTxtValue, ultrastar_file_output: str
):
    """Calculate score points"""
    if is_audio:
        ultrastar_class = ultrastar_parser.parse_ultrastar_txt(
            ultrastar_file_output
        )
        (
            simple_score,
            accurate_score,
        ) = ultrastar_score_calculator.calculate_score(
            pitched_data, ultrastar_class
        )
        ultrastar_score_calculator.print_score_calculation(
            simple_score, accurate_score
        )
    else:
        print(
            f"{ULTRASINGER_HEAD} {blue_highlighted('Score of original Ultrastar txt')}"
        )
        (
            simple_score,
            accurate_score,
        ) = ultrastar_score_calculator.calculate_score(
            pitched_data, ultrastar_class
        )
        ultrastar_score_calculator.print_score_calculation(
            simple_score, accurate_score
        )
        print(
            f"{ULTRASINGER_HEAD} {blue_highlighted('Score of re-pitched Ultrastar txt')}"
        )
        ultrastar_class = ultrastar_parser.parse_ultrastar_txt(
            ultrastar_file_output
        )
        (
            simple_score,
            accurate_score,
        ) = ultrastar_score_calculator.calculate_score(
            pitched_data, ultrastar_class
        )
        ultrastar_score_calculator.print_score_calculation(
            simple_score, accurate_score
        )
    return ultrastar_class, simple_score, accurate_score


def create_ultrastar_txt_from_ultrastar_data(
    song_output: str, ultrastar_class: UltrastarTxtValue, ultrastar_note_numbers: list[int]
) -> str:
    """Create Ultrastar txt from Ultrastar data"""
    output_repitched_ultrastar = os.path.join(
        song_output, ultrastar_class.title + ".txt"
    )
    ultrastar_writer.create_repitched_txt_from_ultrastar_data(
        settings.input_file_path,
        ultrastar_note_numbers,
        output_repitched_ultrastar,
    )
    return output_repitched_ultrastar


def create_ultrastar_txt_from_automation(
    audio_separation_path: str,
    basename_without_ext: str,
    song_output: str,
    transcribed_data: list[TranscribedData],
    ultrastar_audio_input_path: str,
    ultrastar_note_numbers: list[int],
    language: str,
    title: str,
    artist: str,
    year: str,
    genre: str
):
    """Create Ultrastar txt from automation"""
    ultrastar_header = UltrastarTxtValue()
    ultrastar_header.title = basename_without_ext
    ultrastar_header.artist = basename_without_ext
    ultrastar_header.mp3 = basename_without_ext + ".mp3"
    ultrastar_header.video = basename_without_ext + ".mp4"
    ultrastar_header.language = language
    cover = basename_without_ext + " [CO].jpg"
    ultrastar_header.cover = (
        cover
        if os_helper.check_file_exists(os.path.join(song_output, cover))
        else None
    )
    ultrastar_header.creator = f"{ultrastar_header.creator} {Settings.APP_VERSION}"
    ultrastar_header.comment = f"{ultrastar_header.comment} {Settings.APP_VERSION}"

    # Additional data
    if title is not None:
        ultrastar_header.title = title
    if artist is not None:
        ultrastar_header.artist = artist
    if year is not None:
        ultrastar_header.year = extract_year(year)
    if genre is not None:
        ultrastar_header.genre = genre

    real_bpm = get_bpm_from_file(ultrastar_audio_input_path)
    ultrastar_file_output = os.path.join(
        song_output, basename_without_ext + ".txt"
    )
    ultrastar_writer.create_ultrastar_txt_from_automation(
        transcribed_data,
        ultrastar_note_numbers,
        ultrastar_file_output,
        ultrastar_header,
        real_bpm,
    )
    if settings.create_karaoke:
        no_vocals_path = os.path.join(audio_separation_path, "no_vocals.wav")
        title = basename_without_ext + " [Karaoke]"
        ultrastar_header.title = title
        ultrastar_header.mp3 = title + ".mp3"
        karaoke_output_path = os.path.join(song_output, title)
        karaoke_audio_output_path = karaoke_output_path + ".mp3"
        convert_wav_to_mp3(no_vocals_path, karaoke_audio_output_path)
        karaoke_txt_output_path = karaoke_output_path + ".txt"
        ultrastar_writer.create_ultrastar_txt_from_automation(
            transcribed_data,
            ultrastar_note_numbers,
            karaoke_txt_output_path,
            ultrastar_header,
            real_bpm,
        )
    return real_bpm, ultrastar_file_output

def extract_year(date: str) -> str:
    match = re.search(r'\b\d{4}\b', date)
    if match:
        return match.group(0)
    else:
        return date

def infos_from_audio_input_file() -> tuple[str, str, str, tuple[str, str, str, str]]:
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

    if artist is not None and title is not None:
        basename_without_ext = f"{artist} - {title}"
        extension = os.path.splitext(basename)[1]
        basename = f"{basename_without_ext}{extension}"

    song_output = os.path.join(settings.output_file_path, basename_without_ext)
    song_output = get_unused_song_output_dir(song_output)
    os_helper.create_folder(song_output)
    os_helper.copy(settings.input_file_path, song_output)
    os_helper.rename(os.path.join(song_output, os.path.basename(settings.input_file_path)), os.path.join(song_output, basename))
    ultrastar_audio_input_path = os.path.join(song_output, basename)
    return basename_without_ext, song_output, ultrastar_audio_input_path, (title, artist, year_info, genre_info)


FILENAME_REPLACEMENTS = (('?:"', ""), ("<", "("), (">", ")"), ("/\\|*", "-"))


def sanitize_filename(fname: str) -> str:
    """Sanitize filename"""
    for old, new in FILENAME_REPLACEMENTS:
        for char in old:
            fname = fname.replace(char, new)
    if fname.endswith("."):
        fname = fname.rstrip(" .")  # Windows does not like trailing periods
    return fname


def download_from_youtube() -> tuple[str, str, str, tuple[str, str, str, str]]:
    """Download from YouTube"""
    (artist, title) = get_youtube_title(settings.input_file_path)

    # Get additional data for song
    (title_info, artist_info, year_info, genre_info) = get_music_infos(f"{artist} - {title}")

    if title_info is not None:
        title = title_info
        artist = artist_info

    basename_without_ext = sanitize_filename(f"{artist} - {title}")
    basename = basename_without_ext + ".mp3"
    song_output = os.path.join(settings.output_file_path, basename_without_ext)
    song_output = get_unused_song_output_dir(song_output)
    os_helper.create_folder(song_output)
    download_youtube_audio(
        settings.input_file_path, basename_without_ext, song_output
    )
    download_youtube_video(
        settings.input_file_path, basename_without_ext, song_output
    )
    download_youtube_thumbnail(
        settings.input_file_path, basename_without_ext, song_output
    )
    ultrastar_audio_input_path = os.path.join(song_output, basename)
    return basename_without_ext, song_output, ultrastar_audio_input_path, (title, artist, year_info, genre_info)


def parse_ultrastar_txt() -> tuple[str, float, str, str, UltrastarTxtValue]:
    """Parse Ultrastar txt"""
    ultrastar_class = ultrastar_parser.parse_ultrastar_txt(
        settings.input_file_path
    )
    real_bpm = ultrastar_converter.ultrastar_bpm_to_real_bpm(
        float(ultrastar_class.bpm.replace(",", "."))
    )
    ultrastar_mp3_name = ultrastar_class.mp3
    basename_without_ext = os.path.splitext(ultrastar_mp3_name)[0]
    dirname = os.path.dirname(settings.input_file_path)
    ultrastar_audio_input_path = os.path.join(dirname, ultrastar_mp3_name)
    song_output = os.path.join(
        settings.output_file_path,
        ultrastar_class.artist + " - " + ultrastar_class.title,
    )
    song_output = get_unused_song_output_dir(song_output)
    os_helper.create_folder(song_output)
    return (
        basename_without_ext,
        real_bpm,
        song_output,
        ultrastar_audio_input_path,
        ultrastar_class,
    )


def create_midi_file(real_bpm: float,
                     song_output: str,
                     ultrastar_class: UltrastarTxtValue,
                     basename_without_ext: str) -> None:
    """Create midi file"""
    print(
        f"{ULTRASINGER_HEAD} Creating Midi with {blue_highlighted('pretty_midi')}"
    )

    voice_instrument = [
        midi_creator.convert_ultrastar_to_midi_instrument(ultrastar_class)
    ]
    midi_output = os.path.join(song_output, f"{basename_without_ext}.mid")
    midi_creator.instruments_to_midi(
        voice_instrument, real_bpm, midi_output
    )


def pitch_audio(is_audio: bool, transcribed_data: list[TranscribedData], ultrastar_class: UltrastarTxtValue) -> tuple[
    list[str], PitchedData, list[int]]:
    """Pitch audio"""
    # todo: chunk pitching as option?
    # midi_notes = pitch_each_chunk_with_crepe(chunk_folder_name)
    device = "cpu" if settings.force_crepe_cpu else settings.tensorflow_device
    pitched_data = get_pitch_with_crepe_file(
        settings.mono_audio_path,
        settings.crepe_model_capacity,
        settings.crepe_step_size,
        device,
    )
    if is_audio:
        start_times = []
        end_times = []
        for i, data in enumerate(transcribed_data):
            start_times.append(data.start)
            end_times.append(data.end)
        midi_notes = create_midi_notes_from_pitched_data(
            start_times, end_times, pitched_data
        )

    else:
        midi_notes = create_midi_notes_from_pitched_data(
            ultrastar_class.startTimes, ultrastar_class.endTimes, pitched_data
        )
    ultrastar_note_numbers = convert_midi_notes_to_ultrastar_notes(midi_notes)
    return midi_notes, pitched_data, ultrastar_note_numbers


def create_audio_chunks(
    cache_path: str,
    is_audio: bool,
    transcribed_data: list[TranscribedData],
    ultrastar_audio_input_path: str,
    ultrastar_class: UltrastarTxtValue
) -> None:
    """Create audio chunks"""
    audio_chunks_path = os.path.join(
        cache_path, settings.audio_chunk_folder_name
    )
    os_helper.create_folder(audio_chunks_path)
    if is_audio:  # and csv
        csv_filename = os.path.join(audio_chunks_path, "_chunks.csv")
        export_chunks_from_transcribed_data(
            settings.mono_audio_path, transcribed_data, audio_chunks_path
        )
        export_transcribed_data_to_csv(transcribed_data, csv_filename)
    else:
        export_chunks_from_ultrastar_data(
            ultrastar_audio_input_path, ultrastar_class, audio_chunks_path
        )


def denoise_vocal_audio(input_path: str, output_path: str) -> None:
    """Denoise vocal audio"""
    ffmpeg_reduce_noise(input_path, output_path)


def main(argv: list[str]) -> None:
    """Main function"""
    print_version()
    init_settings(argv)
    run()
    # todo: cleanup
    sys.exit()


def init_settings(argv: list[str]) -> None:
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
            settings.output_file_path = arg
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
        elif opt in ("--force_cpu"):
            settings.force_cpu = arg
            if settings.force_cpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        elif opt in ("--force_whisper_cpu"):
            settings.force_whisper_cpu = eval(arg.title())
        elif opt in ("--force_crepe_cpu"):
            settings.force_crepe_cpu = eval(arg.title())

    if settings.output_file_path == "":
        if settings.input_file_path.startswith("https:"):
            dirname = os.getcwd()
        else:
            dirname = os.path.dirname(settings.input_file_path)
        settings.output_file_path = os.path.join(dirname, "output")

    if not settings.force_cpu:
        settings.tensorflow_device, settings.pytorch_device = check_gpu_support()


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
        "force_cpu=",
        "force_whisper_cpu=",
        "force_crepe_cpu=",
    ]
    return long, short


if __name__ == "__main__":
    main(sys.argv[1:])
