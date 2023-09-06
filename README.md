[![Discord](https://img.shields.io/discord/1048892118732656731?logo=discord)](https://discord.gg/7EqhhjFd)
![Status](https://img.shields.io/badge/status-development-yellow)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/rakuri255/UltraSinger/main.yml)
[![GitHub](https://img.shields.io/github/license/rakuri255/UltraSinger)](https://github.com/rakuri255/UltraSinger/blob/main/LICENSE)
[![CodeFactor](https://www.codefactor.io/repository/github/rakuri255/ultrasinger/badge)](https://www.codefactor.io/repository/github/rakuri255/ultrasinger)

[![Check Requirements](https://github.com/rakuri255/UltraSinger/actions/workflows/main.yml/badge.svg)](https://github.com/rakuri255/UltraSinger/actions/workflows/main.yml)
[![Pytest](https://github.com/rakuri255/UltraSinger/actions/workflows/pytest.yml/badge.svg)](https://github.com/rakuri255/UltraSinger/actions/workflows/pytest.yml)

# UltraSinger

> ⚠️ _This project is still under development!_

UltraSinger is a tool to automatically create UltraStar.txt, midi and notes from music. 
Meaning it automaticly pitch UltraStar files, adding text and tapping to UltraStar files and creates separate UltraStar karaoke files.
It also can re-pitch current UltraStar files and calculates the possible in-game score.

Multiple AI models are used to extract text from the voice and to determine the pitch.

Please mention UltraSinger in your UltraStar.txt file if you use it. It helps other to find this tool.
And it helps you that this tool gets improved and maintained.
You should only use it on Creative Commons licensed songs.

## Support
There are many ways to support this project. Starring ⭐️ the repo is just one 🙏

You can also support this work on Patreon or Buy Me A Coffee.

This will help me alot to keep this project alive and improve it.

<a href="https://www.buymeacoffee.com/rakuri255" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>
<a href="https://patreon.com/Rakuri"><img src="https://raw.githubusercontent.com/rakuri255/UltraSinger/main/assets/patreon.png" alt="Become a Patron" style="height: 60px !important;width: 217px !important;"/> </a>

## How to use this source code

### With Console (Windows)

* Install Python 3.10 **(older and newer versions has some breaking changes)**. [Download](https://www.python.org/downloads/)
* Also install ffmpeg separately with PATH. [Download](https://www.ffmpeg.org/download.html)
* Open a console (CMD) and navigate to the project folder.
* Type `py -3.10 -m venv .venv` and press enter. If this does not work, try instead of `py` `python` or `python3`.
  * If you have multiple versions installed, you can use `py -0p` to see all installed versions.
  * Build with the newest version use `py -m venv .venv`. But currently it only works with 3.10.
* Wait until the console is done with creating the environment. This can take a while.
* Type `.venv\Scripts\activate` and press enter.
* You should see now a `(.venv)` in front of your console line.
* Install the requirements with `pip install -r requirements.txt`.
* Install gpu requirements `pip3 install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --index-url https://download.pytorch.org/whl/cu117`
* Now you can use the UltraSinger source code with `py UltraSinger.py [opt] [mode] [transcription] [pitcher] [extra]`. See [How to use](#how-to-use) for more information.

For more information about Python environments look [here](https://code.visualstudio.com/docs/python/environments#_global-virtual-or-conda-environments).

Installation As copy:
    
```commandline
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip3 install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --index-url https://download.pytorch.org/whl/cu117

```

Run UltraSinger:

* Activate the environment with `.venv\Scripts\activate`. (You dont need this if you already activated it, or just installed with the step above)
* Navigate to src folder `cd src`
* Start UltraSinger `py UltraSinger.py`

Start environment only once:

```commandline
.venv\Scripts\activate
cd src

```

Start UltraSinger:

```commandline
py UltraSinger.py

```

## How to use the App

_Not all options working now!_
```commandline
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
    --vosk                  Needs model
    
    [pitcher]
    # Default is crepe
    --crepe            tiny|full >> ((default) is full)
    --crepe_step_size  unit is miliseconds >> ((default) is 10)
    --crepe_batch_size Reduce if low on GPU mem >> ((default) is 2048)
    
    [extra]
    --hyphenation           True|False >> ((default) is True)
    --disable_separation    True|False >> ((default) is False)
    --disable_karaoke       True|False >> ((default) is False)
    --create_audio_chunks   True|False >> ((default) is False)
    --plot                  True|False >> ((default) is False)
    --force_whisper_cpu     True|False >> ((default) is False)
    --force_separation_cpu  True|False >> ((default) is False)
    --force_crepe_cpu       True|False >> ((default) is False)
```

For standard use, you only need to use [opt]. All other options are optional.

### Input

#### Audio (full automatic)

##### Local file

```commandline
-i "input/music.mp3"
```

##### Youtube

```commandline
-i https://www.youtube.com/watch?v=BaW_jenozKc
```

#### UltraStar (re-pitch)

This re-pitch the audio and creates a new txt file.

```commandline
-i "input/ultrastar.txt"
```

### Transcriber

For transcription, `whisper` is used by default. It is more accurate than the other even with the `large` model.
And it finds the language automatically.
But anyway, it depends! Try the other one if `Whisper` does not suit you.
Also keep in mind that while a larger model is more accurate, it also takes longer to transcribe.

#### Whisper

For the first test run, use the `tiny`, to be accurate use the `large-v2` model.

```commandline
-i XYZ --whisper large-v2
```

##### Whisper languages

Currently provided default language models are `en, fr, de, es, it, ja, zh, nl, uk, pt`. 
If the language is not in this list, you need to find a phoneme-based ASR model from 
[huggingface model hub](https://huggingface.co). It will download automatically.

Example for romanian:
```commandline
-i XYZ --align_model "gigant/romanian-wav2vec2"
```

#### Vosk

If you want to use `Vosk` than you need the model. It is not included. You can download it here [Link](https://alphacephei.com/vosk/models).
Make sure you take the right language. 
For the first test run, use the `small model`, to be accurate use the `gigaspeech` model

```commandline
-i XYZ -v "models\vosk-model-en-us-0.42-gigaspeech"
```

#### Hyphenation

Is on by default. Can also be deactivated if hyphenation does not produce 
anything useful. Note that the word is simply split, 
without paying attention to whether the separated word really 
starts at the place or is heard.  

```commandline
-i XYZ --hyphenation True
```

### Pitcher

Pitching is done with the `crepe` model.
Also consider that a bigger model is more accurate, but also takes longer to pitch.
For just testing you should use `tiny`.
If you want solid accurate, then use the `full` model.

```commandline
-i XYZ --crepe full
```

### Separation

The vocals are separated from the audio before they are passed to the models. If problems occur with this, 
you have the option to disable this function and the original audio file is used instead.

```commandline
-i XYZ --disable_separation True
```

### Ultrastar Score Calculation

The score what the singer in the audio would receive will be measured. 
You get 2 scores, simple and accurate. You wonder where the difference is? 
Ultrastar is not interested in pitch hights. As long as it is in the pitch range A-G you get one point. 
This makes sense for the game, because otherwise men don't get points for high female voices and women don't get points 
for low male voices. Accurate is the real tone specified in the txt. I had txt files where the pitch was in a range not 
singable by humans, but you could still reach the 10k points in the game. The accuracy is important here, because from
this MIDI and sheet are created. And you also want to have accurate files


### Use GPU

With an GPU you can speed up the process and also the quality of the transcription and pitching is better.

You need a cuda device for this to work.
If you use an MAC-System than sorry, there is no cuda device for MAC machines.

It is recommended, but optional, to install the cuda driver for your gpu see [driver](https://developer.nvidia.com/cuda-downloads).
Install torch with cuda separately in your `venv`. See [tourch+cuda](https://pytorch.org/get-started/locally/).
Also check you GPU cuda support. See [cuda support](https://gist.github.com/standaloneSA/99788f30466516dbcc00338b36ad5acf)

Command for `pip`:
```
pip3 install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --index-url https://download.pytorch.org/whl/cu117
```

When you want to use `conda` instead you need a different installation command. See this [link](https://pytorch.org/get-started/locally/).

#### Info

If something crashes because of low VRAM than use a smaller model.
Whisper needs more than 8GB VRAM in the `large` model!

But you can force cpu usage with the extra options `--force_whisper_cpu`, `--force_separation_cpu` and `--force_crepe_cpu`.
