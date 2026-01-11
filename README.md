[![Discord](https://img.shields.io/discord/1048892118732656731?logo=discord)](https://discord.gg/rYz9wsxYYK)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rakuri255/UltraSinger/blob/master/colab/UltraSinger.ipynb)
![Status](https://img.shields.io/badge/status-development-yellow)

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/rakuri255/UltraSinger/main.yml)
[![GitHub](https://img.shields.io/github/license/rakuri255/UltraSinger)](https://github.com/rakuri255/UltraSinger/blob/main/LICENSE)
[![CodeFactor](https://www.codefactor.io/repository/github/rakuri255/ultrasinger/badge)](https://www.codefactor.io/repository/github/rakuri255/ultrasinger)
[![Check Requirements](https://github.com/rakuri255/UltraSinger/actions/workflows/main.yml/badge.svg)](https://github.com/rakuri255/UltraSinger/actions/workflows/main.yml)
[![Pytest](https://github.com/rakuri255/UltraSinger/actions/workflows/pytest.yml/badge.svg)](https://github.com/rakuri255/UltraSinger/actions/workflows/pytest.yml)
[![docker](https://github.com/rakuri255/UltraSinger/actions/workflows/docker.yml/badge.svg)](https://hub.docker.com/r/rakuri255/ultrasinger)

<p align="center" dir="auto">
<img src="https://repository-images.githubusercontent.com/594208922/4befe3da-a448-4cbc-b6ef-93899119071b" style="height: 300px;width: auto;" alt="UltraSinger Logo">
</p>

# UltraSinger

> ⚠️ _This project is still under development!_

UltraSinger is a tool to automatically create UltraStar.txt, midi and notes from music.
It automatically pitches UltraStar files, adding text and tapping to UltraStar files and creates separate UltraStar karaoke files.
It also can re-pitch current UltraStar files and calculates the possible in-game score.

Multiple AI models are used to extract text from the voice and to determine the pitch.

Please mention UltraSinger in your UltraStar.txt file if you use it. It helps others find this tool, and it helps this tool get improved and maintained.
You should only use it on Creative Commons licensed songs.

## ❤️ Support
There are many ways to support this project. Starring ⭐️ the repo is just one 🙏

You can also support this work on <a href="https://github.com/sponsors/rakuri255">GitHub sponsors</a> or <a href="https://patreon.com/Rakuri">Patreon</a> or <a href="https://www.buymeacoffee.com/rakuri255" target="_blank">Buy Me a Coffee</a>.

This will help me a lot to keep this project alive and improve it.

<a href="https://www.buymeacoffee.com/rakuri255" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee" style="height: 60px !important;width: 217px !important;" ></a>
<a href="https://patreon.com/Rakuri"><img src="https://raw.githubusercontent.com/rakuri255/UltraSinger/main/assets/patreon.png" alt="Become a Patron" style="height: 60px !important;width: 217px !important;"/> </a>
<a href="https://github.com/sponsors/rakuri255"><img src="https://raw.githubusercontent.com/rakuri255/UltraSinger/main/assets/mona-heart-featured.webp" alt="GitHub Sponsor" style="height: 60px !important;width: auto;"/> </a>

## Table of Contents

- [UltraSinger](#ultrasinger)
  - [❤️ Support](#️-support)
  - [Table of Contents](#table-of-contents)
  - [💻 How to use this source code](#-how-to-use-this-source-code)
    - [Installation](#installation)
    - [Run](#run)
  - [📖 How to use the App](#-how-to-use-the-app)
    - [🎶 Input](#-input)
      - [Audio (full automatic)](#audio-full-automatic)
        - [Local file](#local-file)
        - [Youtube](#youtube)
      - [UltraStar (re-pitch)](#ultrastar-re-pitch)
    - [🗣 Transcriber](#-transcriber)
      - [Whisper](#whisper)
        - [Whisper languages](#whisper-languages)
      - [✍️ Hyphenation](#️-hyphenation)
    - [👂 Pitcher](#-pitcher)
    - [👄 Separation](#-separation)
    - [Sheet Music](#sheet-music)
    - [Format Version](#format-version)
    - [🏆 Ultrastar Score Calculation](#-ultrastar-score-calculation)
    - [📟 Use GPU](#-use-gpu)
      - [Considerations for Windows users](#considerations-for-windows-users)
      - [Crashes due to low VRAM](#crashes-due-to-low-vram)
    - [📦 Containerized](#containerized-docker-or-podman)

## 💻 How to use this source code

### Installation

* Install Python 3.10 **(older and newer versions has some breaking changes)**. [Download](https://www.python.org/downloads/)
* Also download or install ffmpeg with PATH. [Download](https://www.ffmpeg.org/download.html)
* Go to folder `install` and run install script for your OS.
  * Choose `GPU` if you have an nvidia CUDA GPU.
  * Choose `CPU` if you don't have an nvidia CUDA GPU.

### Run

* In root folder just run `run_on_windows.bat`, `run_on_linux.sh` or `run_on_macos.command` to start the app.
* Now you can use the UltraSinger source code with `py UltraSinger.py [opt] [mode] [transcription] [pitcher] [extra]`. See [How to use](#how-to-use) for more information.

## 📖 How to use the App

_Not all options working now!_
```commandline
    UltraSinger.py [opt] [mode] [transcription] [pitcher] [extra]

    [opt]
    -h      This help text.
    -i      Ultrastar.txt
            audio like .mp3, .wav, youtube link
    -o      Output folder

    [mode]
    ## if INPUT is audio ##
    default (Full Automatic Mode) - Creates all, depending on command line options
    --interactive - Interactive Mode. All options are asked at runtime for easier configuration

    # Single file creation selection is in progress, you currently getting all!
    (-u      Create ultrastar txt file) # In Progress
    (-m      Create midi file) # In Progress
    (-s      Create sheet file) # In Progress

    ## if INPUT is ultrastar.txt ##
    default  Creates all

    [separation]
    # Default is htdemucs
    --demucs              Model name htdemucs|htdemucs_ft|htdemucs_6s|hdemucs_mmi|mdx|mdx_extra|mdx_q|mdx_extra_q >> ((default) is htdemucs)

    [transcription]
    # Default is whisper
    --whisper               Multilingual model > tiny|base|small|medium|large-v1|large-v2|large-v3  >> ((default) is large-v2)
                            English-only model > tiny.en|base.en|small.en|medium.en
    --whisper_align_model   Use other languages model for Whisper provided from huggingface.co
    --language              Override the language detected by whisper, does not affect transcription but steps after transcription
    --whisper_batch_size    Reduce if low on GPU mem >> ((default) is 16)
    --whisper_compute_type  Change to "int8" if low on GPU mem (may reduce accuracy) >> ((default) is "float16" for cuda devices, "int8" for cpu)
    --keep_numbers          Numbers will be transcribed as numerics instead of as words

    [pitcher]
    # Default is crepe
    --crepe            tiny|full >> ((default) is full)
    --crepe_step_size  unit is miliseconds >> ((default) is 10)

    [extra]
    --disable_hyphenation   Disable word hyphenation. Hyphenation is enabled by default.
    --disable_separation    Disable track separation. Track separation is enabled by default.
    --disable_karaoke       Disable creation of karaoke style txt file. Karaoke is enabled by default.
    --create_audio_chunks   Enable creation of audio chunks. Audio chunks are disabled by default.
    --keep_cache            Keep cache folder after creation. Cache folder is removed by default.
    --plot                  Enable creation of plots. Plots are disabled by default.
    --format_version        0.3.0|1.0.0|1.1.0|1.2.0 >> ((default) is 1.2.0)
    --musescore_path        path to MuseScore executable
    --keep_numbers          Transcribe numbers as digits and not words
    --ffmpeg                Path to ffmpeg and ffprobe executable

    [yt-dlp]
    --cookiefile            File name where cookies should be read from

    [device]
    --force_cpu             Force all steps to be processed on CPU.
    --force_whisper_cpu     Only whisper will be forced to cpu
    --force_crepe_cpu       Only crepe will be forced to cpu
```

For standard use, you only need to use [opt]. All other options are optional.

### 🎶 Input

### Mode
default (Full Automatic Mode) - Creates all, depending on command line options
--interactive - Interactive Mode. All options are asked at runtime for easier configuration
```commandline
--interactive
```
#### Audio (full automatic)

##### Local file

```commandline
-i "input/music.mp3"
```

##### Youtube

```commandline
-i https://www.youtube.com/watch?v=YwNs1Z0qRY0
```

Note that if you run into a yt-dlp error such as `Sign in to confirm you’re not a bot. This helps protect our community` ([yt-dlp issue](https://github.com/yt-dlp/yt-dlp/issues/10128)) you can follow these steps:

* generate a cookies.txt file with [yt-dlp](https://github.com/yt-dlp/yt-dlp/wiki/Installation) `yt-dlp --cookies cookies.txt --cookies-from-browser firefox`
* then pass the cookies.txt to UltraSinger `--cookiefile cookies.txt`

#### UltraStar (re-pitch)

This re-pitch the audio and creates a new txt file.

```commandline
-i "input/ultrastar.txt"
```

### 🗣 Transcriber

Keep in mind that while a larger model is more accurate, it also takes longer to transcribe.

#### Whisper

For the first test run, use the `tiny`, to be accurate use the `large-v2` model.

```commandline
-i XYZ --whisper large-v2
```

##### Whisper languages

Currently provided default language models are `en, fr, de, es, it, ja, zh, nl, uk, pt`.
If the language is not in this list, you need to find a phoneme-based ASR model from
[🤗 huggingface model hub](https://huggingface.co). It will download automatically.

Example for romanian:
```commandline
-i XYZ --whisper_align_model "gigant/romanian-wav2vec2"
```

#### ✍️ Hyphenation

Is on by default. Can also be deactivated if hyphenation does not produce
anything useful. Note that the word is simply split,
without paying attention to whether the separated word really
starts at the place or is heard. To disable:

```commandline
-i XYZ --disable_hyphenation
```

### 👂 Pitcher

Pitching is done with the `crepe` model.
Also consider that a bigger model is more accurate, but also takes longer to pitch.
For just testing you should use `tiny`.
If you want solid accurate, then use the `full` model.

```commandline
-i XYZ --crepe full
```

### 👄 Separation

The vocals are separated from the audio before they are passed to the models. If problems occur with this,
you have the option to disable this function; in which case the original audio file is used instead.

```commandline
-i XYZ --disable_separation
```

### Sheet Music

For Sheet Music generation you need to have `MuseScore` installed on your system.
Or provide the path to the `MuseScore` executable.

```commandline
-i XYZ --musescore_path "C:/Program Files/MuseScore 4/bin/MuseScore4.exe"
```

### Format Version

This defines the format version of the UltraStar.txt file. For more info see [Official UltraStar format specification](https://usdx.eu/format/).

You can choose between different format versions. The default is `1.2.0`.
* `0.3.0` is the first format version. Use this if you have an old UltraStar program and problems with the newer format.
* `1.0.0` should be supported by the most UltraStar programs. Use this if you have problems with the newest format version
* `1.1.0` is the current format version.
* `1.2.0` is the upcoming format version. It is not finished yet.
* `2.0.0` is the next format version. It is not finished yet.

```commandline
-i XYZ --format_version 1.2.0
```

### 🏆 Ultrastar Score Calculation

The score that the singer in the audio would receive will be measured.
You get 2 scores, simple and accurate. You wonder where the difference is?
Ultrastar is not interested in pitch hights. As long as it is in the pitch range A-G you get one point.
This makes sense for the game, because otherwise men don't get points for high female voices and women don't get points
for low male voices. Accurate is the real tone specified in the txt. I had txt files where the pitch was in a range not
singable by humans, but you could still reach the 10k points in the game. The accuracy is important here, because from
this MIDI and sheet are created. And you also want to have accurate files


### 📟 Use GPU

With a GPU you can speed up the process. Also the quality of the transcription and pitching is better.

You need a cuda device for this to work. Sorry, there is no cuda device for macOS.

It is optional (but recommended) to install the cuda driver for your gpu: see [driver](https://developer.nvidia.com/cuda-downloads).
Install torch with cuda separately in your `venv`. See [tourch+cuda](https://pytorch.org/get-started/locally/).
Also check you GPU cuda support. See [cuda support](https://gist.github.com/standaloneSA/99788f30466516dbcc00338b36ad5acf)

Command for `pip`:
```
pip3 install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --index-url https://download.pytorch.org/whl/cu117
```

When you want to use `conda` instead you need a [different installation command](https://pytorch.org/get-started/locally/).

#### Considerations for Windows users

The pitch tracker used by UltraSinger (crepe) uses TensorFlow as its backend.
TensorFlow dropped GPU support for Windows for versions >2.10 as you can see in this [release note](https://github.com/tensorflow/tensorflow/releases/tag/v2.11.1) and their [installation instructions](https://www.tensorflow.org/install/pip#windows-native).

For now UltraSinger runs the latest version available that still supports GPUs on windows.

For running later versions of TensorFlow on windows while still taking advantage of GPU support the suggested solution is to [run UltraSinger in a container](container/README.md).
#### Crashes due to low VRAM

If something crashes because of low VRAM then use a smaller model.
Whisper needs more than 8GB VRAM in the `large` model!

You can also force cpu usage with the extra option `--force_cpu`.

### 📦 Containerized (Docker or Podman)

See [container/README.md](container/README.md)
