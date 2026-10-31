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
  - [🏗️ System Architecture & SOTA Engine Catalog](#️-system-architecture--sota-engine-catalog)
  - [📁 Standard Directory Structure & Smart Ingestion](#-standard-directory-structure--smart-ingestion)
  - [🎛️ Multi-Engine Strategy & CLI Options](#️-multi-engine-strategy--cli-options)
  - [💎 Production Profiles (Lightweight, SOTA 9.8, Duets)](#-production-profiles-lightweight-sota-98-duets)
  - [🎵 Acoustic Melisma Detection & Duet Architecture](#-acoustic-melisma-detection--duet-architecture)
  - [🔬 MIR & DSP Algorithmic Deep-Dive](#-mir--dsp-algorithmic-deep-dive)
  - [💻 How to use this source code](#-how-to-use-this-source-code)
    - [Installation](#installation)
    - [Run](#run)
  - [📖 How to use the App](#-how-to-use-the-app)
    - [🎶 Input](#-input)
      - [Audio (full automatic)](#audio-full-automatic)
      - [Youtube](#youtube)
      - [UltraStar (re-pitch)](#ultrastar-re-pitch)
    - [🗣 Transcriber](#-transcriber)
    - [👂 Pitcher](#-pitcher)
    - [👄 Separation](#-separation)
    - [🎵 Key Quantization](#-key-quantization)
    - [Sheet Music](#sheet-music)
    - [Format Version](#format-version)
    - [🏆 Ultrastar Score Calculation](#-ultrastar-score-calculation)
    - [📟 Use GPU](#-use-gpu)
    - [📦 Containerized](#containerized-docker-or-podman)

## 🏗️ System Architecture & SOTA Engine Catalog

UltraSinger integrates digital signal processing (DSP), deep learning networks, and information theory to convert raw audio into synchronized, pitched UltraStar karaoke files.

```mermaid
graph TD
    subgraph IN["1. INGESTION & RUNTIME"]
        HostIn["Audio (.mp3/.wav) + lyrics.txt + GENRE.txt"] --> InputFolder["Input Folder"]
        GenreData["GENRE.txt / --genre"] --> GenreEngine["12 Genre Acoustic Profiles"]
    end

    subgraph SEP["2. SOTA VOCAL ISOLATION"]
        InputFolder --> SeparatorEngine{"--separator-engine"}
        SeparatorEngine -->|Default| BSRoformer["BS-Roformer (SDR 12.98 dB)"]
        SeparatorEngine -->|Mel scale| MelRoformer["Mel-Band Roformer"]
        SeparatorEngine -->|Fast| SCNet["SCNet Large"]
        SeparatorEngine -->|Multi-stem| HTDemucs["HTDemucs"]
        BSRoformer & MelRoformer & SCNet & HTDemucs --> DeReverb{"--dereverb (UVR ONNX)"}
    end

    subgraph ASR["3. ASR & PHONETIC ALIGNMENT"]
        DeReverb --> VAD["Pyannote.audio VAD"]
        VAD --> WhisperModel{"--asr-engine (WhisperX / Turbo)"}
        GenreEngine -->|bilingual initial_prompt| WhisperModel
        RefLyrics["lyrics.txt (Official Text)"] --> FuzzyEngine["Fuzzy Matcher & Intro Ad-lib Purge"]
        WhisperModel --> FuzzyEngine
        FuzzyEngine --> Aligner{"--aligner-engine (Wav2Vec2 / MMS-FA)"}
        Aligner --> SuperFlux["SuperFlux Onset Snapping & Hilbert Trimmer"]
    end

    subgraph PITCH["4. F0 EXTRACTION & QUANTIZATION"]
        DeReverb --> PitchEngine{"--pitch-engine"}
        PitchEngine -->|Default| RMVPE["RMVPE (ISMIR Median Filter)"]
        PitchEngine -->|Deep 22M| CREPE["CREPE Full"]
        PitchEngine -->|Audio-to-MIDI| BasicPitch["Spotify Basic Pitch ONNX"]
        RMVPE & CREPE & BasicPitch --> KDEMode["Gaussian Weighted KDE Mode (h=35 cents)"]
        KDEMode --> KeyDetect["Musical Key Quantization"]
    end

    subgraph OUT["5. ASSEMBLY, DUET & QA"]
        KeyDetect & SuperFlux --> DuetCheck{"--duet"}
        DuetCheck -->|Duet Mode| DuetEngine["Duet Separator (ECAPA-TDNN / MFCC)"]
        DuetCheck -->|Solo| Composer["USDX Engine (.txt, .mid, sync_test.mp3)"]
        DuetEngine -->|Continuous P1 & P2 Blocks| Composer
        Composer --> AutoCorrect["--autocorrect (7-Equation USDX Scoring Engine)"]
    end
```

### SOTA Library & Technology Catalog

| Library / Technology | Base Model / Version | Role in UltraSinger | Technical Advantage |
|---|---|---|---|
| **BS-Roformer** | `model_bs_roformer_ep_317` | Clean lead vocal separation | **12.98 dB SDR**, eliminates guitar & cymbal bleed into vocals. |
| **Mel-Band Roformer** | `mel_band_roformer_karaoke_aufr33` | Psychoacoustic vocal isolation | Isolates lead vocals from complex backing vocals and harmonies. |
| **SCNet** | `SCNet-Large` | Fast, lightweight separation | 3x faster processing, low VRAM footprint (< 2.5 GB). |
| **HTDemucs** | `htdemucs_ft` / `htdemucs_6s` | Multi-stem track separation | 4 to 6 stem separation (drums, bass, vocals, other). |
| **UVR De-Echo / De-Reverb** | `UVR-DeEcho-DeReverb.onnx` | Surgical de-reverberation | Removes room reverb tails and live echoes before F0 tracking. |
| **WhisperX** | `faster-whisper` + `large-v2` | High-fidelity transcription | Millisecond-accurate phonetic alignment with Wav2Vec2 CTC. |
| **Whisper Turbo** | `large-v3-turbo` | Low-latency transcription | Half the VRAM requirement of large-v2 with high fidelity. |
| **MMS-FA** | `torchaudio.pipelines.MMS_FA` | Meta multilingual phonetic aligner | Preserves accurate alignment during sustained vocals (> 3s). |
| **RMVPE** | `rmvpe.pt` | Neural F0 pitch extraction | Trained on complex singing voices; suppresses breath noise. |
| **CREPE** | `torchcrepe` (`full`) | Deep CNN F0 estimation | 22M parameters with microscopic 10ms frame resolution. |
| **Spotify Basic Pitch** | `basic-pitch` (ICASSP 2022) | Polyphonic audio-to-MIDI | Robust onset tracking and continuous pitch contour projection. |
| **ECAPA-TDNN** | `speechbrain/spkrec-ecapa` | Duet speaker diarization | 192-dimensional d-vectors for cosine clustering of 2 singers. |
| **Fuzzy Lyric Matcher** | Internal (`difflib` + Levenshtein) | Ad-lib purging & lyrics anchoring | Removes intro scatting (*"Ahhh"*, *"Mmmm"*) and anchors `#GAP`. |

## 📁 Standard Directory Structure & Smart Ingestion

UltraSinger natively organizes input and output files for batch processing and automated container runs:

```text
UltraSinger/
├── input/                                    <-- Mapped to /input in Docker
│   └── Artist - Song Name/
│       ├── Artist - Song Name.mp3            (Original audio file)
│       ├── lyrics.txt                        (⭐ Official reference lyrics - plain text)
│       └── GENRE.txt                         (🎵 Musical genre / #GENRE: Style)
│
├── output/                                   <-- Mapped to /output in Docker
│   └── Artist - Song Name/
│       ├── Artist - Song Name.txt            (Final UltraStar 1.2.0 karaoke file)
│       ├── Artist - Song Name [Vocals].mp3   (Isolated lead vocal track)
│       ├── Artist - Song Name [Instrumental].mp3
│       ├── Artist - Song Name [Sync_Test].mp3(Audio with audible beat check clicks)
│       ├── Artist - Song Name.mid            (Generated MIDI melody)
│       └── whisper.txt                       (Natural paragraph transcription)
│
└── cache/                                    <-- Cached models and stem separations
```

### Reference Files: `lyrics.txt` & `GENRE.txt`

1. **Official Lyrics (`lyrics.txt`, `letra.txt`, `lyrics.lrc`):**
   * Enables **Forced Phonetic Alignment**, purges intro vocalizations (*"Ahhh"*, *"Mmmm"*), and ensures 100% spelling accuracy.
   * ⚠️ **Golden Rule for `lyrics.txt`:** Provide **clean plain text only**. **NEVER add manual melisma symbols or tildes (`~`)**. UltraSinger detects and splits vocal melismas automatically from physical neural F0 curves.
2. **Musical Genre (`GENRE.txt` or `--genre "<name>"`):**
   * Configures VAD sensitivity, rhythmic grids (e.g., Dembow 3-3-2, Swing), and vibrato tolerances across 12 acoustic families.
   * **Hybrid / Mixed Genres:** Supports multi-genre fusions (`Merengue/House/Hip Hop`, `K-Pop, Lo-Fi`) with acoustic dominance hierarchy (the primary genre defines physical DSP parameters).
   * **Canonically Formatted USDX Header:** Injected as `#GENRE: Rock` or `#GENRE: Merengue, House, Hip Hop`.

| Scenario | `lyrics.txt` | `GENRE.txt` | Word Accuracy | Acoustic Tuning | USDX Score |
|---|:---:|:---:|---|---|:---:|
| **Optimal SOTA** | ✅ Yes | ✅ Yes | Forced Alignment (100% official text) | Style-tailored VAD & rhythmic grid | **10.0 / 10** |
| **Lyrics Only** | ✅ Yes | ❌ No | Forced Alignment (100% official text) | Standard Pop profile | **9.5 / 10** |
| **Genre Only** | ❌ No | ✅ Yes | Pure Whisper transcription with genre prompt | Style-tailored acoustic profile | **9.0 / 10** |
| **Zero-Touch** | ❌ No | ❌ No | Pure Whisper transcription from audio | Universal default profile | **8.5 / 10** |

## 🎛️ Multi-Engine Strategy & CLI Options

UltraSinger features a modular **Strategy & Factory** pattern allowing surgical selection across every stage of the audio pipeline:

| Pipeline Stage | CLI Argument | Supported Engines / Options | Default | Description & Strengths |
|---|---|---|---|---|
| **1. Vocal Separation** | `--separator-engine`<br/>`--separator` | `bs_roformer`<br/>`mel_roformer`<br/>`scnet`<br/>`htdemucs`<br/>`demucs` | `bs_roformer` | • `bs_roformer`: SOTA fidelity (12.98 dB SDR).<br/>• `mel_roformer`: Mel-scale backing vocal rejection.<br/>• `scnet`: 3x speed, low VRAM (< 2.5 GB).<br/>• `htdemucs`: 4-to-6 stem separation. |
| **Post-Separation De-Reverb** | `--dereverb` | `none`<br/>`uvr_dereverb`<br/>`kim_vocal_2` | `none` | Removes room acoustics and reverb tails using ONNX de-reverberation models prior to pitch extraction. |
| **2. Speech Recognition (ASR)** | `--asr-engine`<br/>`--whisper` | `whisper_v2`<br/>`whisper_turbo` | `whisper_v2` | • `whisper_v2`: Standard production `large-v2`.<br/>• `whisper_turbo`: `large-v3-turbo` with 50% VRAM usage. |
| **Phonetic Aligner** | `--aligner-engine` | `wav2vec2`<br/>`mms_fa` | `wav2vec2` | • `wav2vec2`: Millisecond WhisperX CTC aligner.<br/>• `mms_fa`: Meta multilingual aligner for sustained vocals ($> 3\text{ s}$). |
| **3. Pitch Estimation (F0)** | `--pitch-engine`<br/>`--pitcher` | `rmvpe`<br/>`crepe`<br/>`swiftf0`<br/>`basic-pitch` | `rmvpe` | • `rmvpe`: RVC neural model robust against breath artifacts.<br/>• `crepe`: 22M parameter CNN with continuous resolution. |
| **MIDI Extraction** | `--midi-engine` | `basic_pitch`<br/>`some`<br/>`omnizart` | `basic_pitch` | • `basic_pitch`: Spotify ICASSP 2022 ONNX engine.<br/>• `some`: Syllable-oriented singing MIDI extractor.<br/>• `omnizart`: Polyphonic multi-timbre acoustic transcription. |
| **Expressive Humanization** | `--magenta-humanize` | Flag (`True`/`False`) | `False` | Modulates micro-timing ($\pm 5\text{--}15\text{ ms}$) and dynamics to avoid mechanical rigidity. |
| **4. Duet Diarization** | `--duet-engine` | `mfcc_kmeans`<br/>`ecapa_embeddings` | `mfcc_kmeans` | • `mfcc_kmeans`: 13 MFCC formants + IACC/CSR spatial metrics.<br/>• `ecapa_embeddings`: SpeechBrain 192-dim d-vector clustering. |
| **Post-Processing QA** | `--autocorrect` | Flag | Off | Runs the 7-equation acoustic calibration and official USDX scoring simulation on existing `.txt`. |
| **Quick Lyrics Preview** | `--to_whisper_out` | Flag | Off | Fast mode: separates audio and exports `whisper.txt` transcription without waiting for pitch tracking. |

## 💎 Production Profiles (Lightweight, SOTA 9.8, Duets)

### Performance Comparison

| Metric / Feature | ⚡ Fast Mode | 💎 SOTA Master Profile (9.8/10) |
|---|---|---|
| **Vocal Separator** | `scnet` / `demucs` | `bs_roformer` or `mel_roformer` (SDR 12.98 dB) |
| **Pitch Extractor** | `rmvpe` / `swiftf0` | `rmvpe` + ISMIR temporal median filter |
| **Whisper Model** | `whisper_turbo` or `medium` | `whisper_v2` (`large-v2`) with anti-hallucination |
| **Melisma Detection (`~`)** | ✅ Enabled by default (F0) | ✅ Enabled by default (neural F0) |
| **Reference Lyrics** | Optional (`lyrics.txt`) | Highly Recommended (`lyrics.txt`) |
| **Processing Time** | ~20–35s per song | ~1–2 minutes per song |
| **GPU VRAM Usage** | < 2.5 GB | ~6.5–7.5 GB |
| **USDX Score Target** | 8,800–9,200 pts [A / S] | **9,795–9,850 / 10,000 pts [S+ (PERFECT)]** |

---

### Profile 1: Ultra-Fast & Lightweight (< 2.5 GB VRAM)
Optimized for laptops and budget GPUs (4 GB to 6 GB VRAM) or batch jobs running at 3x speed:
```bash
# Step 1: Fast Generation
python3 UltraSinger.py -i "/input/Artist - Song" -o "/output" \
  --separator-engine scnet --asr-engine whisper_turbo --pitch-engine rmvpe --midi-engine basic_pitch --keep_cache

# Step 2: Fast Autocorrect & Validation
python3 UltraSinger.py -i "/output/Artist - Song/Artist - Song.txt" --autocorrect --pitch-engine rmvpe
```

---

### Profile 2: Surgical Vocal Studio with De-Reverb (⭐ 9.8/10 SOTA Master)
Master studio quality with psychoacoustic vocal isolation, room de-reverberation, Meta MMS-FA alignment for sustained vocals, RMVPE neural pitch tracking, and Magenta dynamic humanization:
```bash
# Step 1: Master Studio Generation
python3 UltraSinger.py -i "/input/Artist - Song" -o "/output" \
  --separator-engine mel_roformer --dereverb uvr_dereverb --asr-engine whisper_v2 \
  --aligner-engine mms_fa --pitch-engine rmvpe --midi-engine basic_pitch --magenta-humanize --keep_cache

# Step 2: Certification & Autocorrection
python3 UltraSinger.py -i "/output/Artist - Song/Artist - Song.txt" --autocorrect --pitch-engine rmvpe
```

---

### Profile 3: Advanced Duets with ECAPA-TDNN Speaker Diarization
Employs 192-dimensional d-vector embeddings to diarize and output official USDX continuous `P 1` and `P 2` blocks:
```bash
# Step 1: Duet Generation
python3 UltraSinger.py -i "/input/Artist - Duet" -o "/output" \
  --separator-engine bs_roformer --asr-engine whisper_v2 --pitch-engine rmvpe \
  --duet --duet-engine ecapa_embeddings --p1_name "Singer 1" --p2_name "Singer 2" --keep_cache

# Step 2: Duet Autocorrect & Block Verification
python3 UltraSinger.py -i "/output/Artist - Duet/Artist - Duet.txt" --autocorrect --pitch-engine rmvpe
```

## 🎵 Acoustic Melisma Detection & Duet Architecture

### Acoustic Melisma Detection (`~`)

In UltraStar Deluxe, a **melisma** occurs when a singer articulates multiple distinct pitches across a single prolonged vowel or syllable (e.g. *"faaaa-lliiiing"*). The primary note holds the syllable text, and consecutive pitch shifts are denoted by `~`.

* **100% Acoustic Detection:** UltraSinger **never** expects melisma marks in `lyrics.txt`. It dynamically monitors the fundamental frequency ($F_0$) trajectory generated by the neural pitcher (`rmvpe`, `crepe`, `basic-pitch`). When a sustained syllable ($\ge 0.28\text{ s}$) transitions across $\ge 1.8\text{ semitones}$ for $\ge 160\text{ ms}$, UltraSinger splits the note and inserts `~`.
* **Enabled by Default:** Active in all standard runs without extra flags.
* **Disable Melismas (`--disable_melismas`):** Produces strictly one solid note per syllable (ideal for casual or novice gameplay).
* **Sensitivity Tuning (`--melisma_threshold <semitones>`):**
  * Default: `1.8` st (suppresses 5–7 Hz natural human vibrato while capturing deliberate melodic turns).
  * High (e.g., `2.5` st): Captures only wide melodic intervals.
  * Low (e.g., `1.3` st): Captures subtle micro-vocal ornaments and runs.

---

### Official Duet Two-Block USDX Architecture

Previous community tools often alternated `P 1` and `P 2` line-by-line or word-by-word, breaking words in half and collapsing the USDX rendering engine into a single illegible 1-pixel line.

UltraSinger implements the **official UltraStar Deluxe multi-player specification**:

```mermaid
graph TD
    DuetFile["Duet .txt File"] --> Headers["Headers: #P1:Singer 1 | #P2:Singer 2"]
    Headers --> P1Block["PLAYER 1 BLOCK: P 1\n(All notes, line-breaks '-' and lyrics for Singer 1)"]
    P1Block --> P2Block["PLAYER 2 BLOCK: P 2\n(All notes, line-breaks '-' and lyrics for Singer 2)"]
    P2Block --> FileEnd["End Tag: E"]
```

1. **Phrase-Level Timbral Clustering:** Songs are divided into phrases across acoustic silence pauses ($\ge 0.85\text{ s}$). Phrases are never split between singers.
2. **Sequential Block Generation:** All `P 1` verses are written contiguously with independent line breaks (`- <beat>`), followed by all `P 2` verses, and closed by `E`.
3. **Autocorrect & Visual QA Isolation:** `--autocorrect` preserves both blocks independently, and the headless visual QA plotter renders Singer 1 in Blue/Gold and Singer 2 in Purple/Coral with `[P1]` and `[P2]` badges.

## 🔬 MIR & DSP Algorithmic Deep-Dive

### The 4 Mathematical Audio Processing Pillars

1. **Vibrato-Immune Onsets via SuperFlux & HPSS:** Harmonics are separated from percussive transients via Harmonic-Percussive Separation (HPSS). SuperFlux applies a spectral max-filter ($\mu = 3, \text{lag} = 2$) across frequency bins:
   $$\text{Novelty}_{\text{SuperFlux}}(t) = \sum_{f} \max\left(0, S[f, t] - \max_{k \in [-\mu, \mu]} S[f+k, t-\text{lag}]\right)$$
   canceling false note triggers caused by vocal pitch vibrato while preserving true consonant onsets.
2. **Optimal Pitch Mode via Gaussian Weighted KDE:** Rather than raw mean or median (which get skewed by glottal attacks and release drops), pitch is determined by continuous Parzen-Rosenblatt kernel density estimation ($h = 35\text{ cents}$):
   $$\hat{p}(m) = \frac{1}{\sum w_i} \sum_{i=1}^{N} w_i \cdot \frac{1}{\sqrt{2\pi} h} \exp\left( - \frac{(m - m_i)^2}{2 h^2} \right), \quad w_i = c_i^2 \cdot \text{RMS}(t_i)$$
3. **Instantaneous Envelope Trimming with Hilbert Transform:** Computes the analytic signal $z(t) = s(t) + j\mathcal{H}\{s(t)\}$ to extract amplitude $|z(t)|$ at $\approx 2.9\text{ ms}$ resolution, trimming dead tails before reverberation begins.
4. **IRCAM Vibrato Detrending & Scat Discrimination:** 250 ms moving median filter identifies regular $4.5\text{--}7.5\text{ Hz}$ pitch oscillations, preventing unwanted note splitting. Wiener spectral flatness classifies unpitched screams and growls into Freestyle (`F`) notes.

---

### Objective Sync Benchmark Validation

Evaluated against official studio Ground Truth files using the multi-metric evaluation harness:

| Benchmark Evaluation | Metric | Baseline | UltraSinger (Current Engine) | Impact |
|---|---|:---:|:---:|---|
| **Test 1**<br/>*(Studio Ground Truth EN)* | **Onset Median \|dt\|**<br/>**Line Start Median \|dt\|**<br/>**Lexical Coverage**<br/>**Syllabification Agreement**<br/>**USDX Score Simulation** | `24,034.2 ms`<br/>`20,482.3 ms`<br/>`62.58 %`<br/>`52.58 %`<br/>`7,640 pts` | **`113.5 ms`**<br/>**`74.3 ms`**<br/>**`100.0 %`**<br/>**`99.35 %`**<br/>**`9,853 / 10,000 [S+]`** | 🚀 **200x error reduction**<br/>🚀 **Sub-beat studio timing**<br/>🚀 **100% lyrics intact**<br/>🚀 **SingStar-spec syllabification**<br/>🏆 **Record S+ score** |
| **Test 2**<br/>*(Studio Ground Truth ES)* | **Onset Median \|dt\|**<br/>**Lexical Coverage**<br/>**Syllabification Agreement**<br/>**USDX Score Simulation** | `103.7 ms`<br/>`99.26 %`<br/>`90.17 %`<br/>`9,618 pts` | **`67.3 ms`**<br/>**`99.51 %`**<br/>**`90.66 %`**<br/>**`9,958 / 10,000 [S+]`** | 🚀 **Sub-beat acoustic precision**<br/>✅ **Full text preservation**<br/>✅ **Strict RAE hyphenation**<br/>🏆 **Near-perfect S+ score** |

## 💻 How to use this source code

### Installation

* Install Python 3.12 **(older/newer versions may have compatibility issues)**. [Download](https://www.python.org/downloads/)
* Also download or install ffmpeg with PATH. [Download](https://www.ffmpeg.org/download.html)
* Go to folder `install` and run install script for your OS:
  * Choose `GPU` if you have an NVIDIA CUDA GPU.
  * Choose `CPU` if you don't have an NVIDIA GPU or want CPU-only processing.

### Run

* In root folder just run `run_on_windows.bat`, `run_on_linux.sh` or `run_on_mac.command` to start the app.
* Now you can use the UltraSinger source code with `py UltraSinger.py [opt] [mode] [transcription] [pitcher] [extra]`. See [How to use](#-how-to-use-the-app) for more information.

## 📖 How to use the App

_Not all options working now!_
```commandline
    UltraSinger.py [opt] [mode] [transcription] [pitcher] [extra]

    [opt]
    -h      This help text.
    -i      Ultrastar.txt
            audio/video like .mp3, .mp4, .wav, youtube link
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
    --separator-engine      Separation engine: bs_roformer|mel_roformer|scnet|htdemucs|demucs >> ((default) is bs_roformer)
    --dereverb              Post-separation de-reverberation: none|uvr_dereverb|kim_vocal_2 >> ((default) is none)
    --disable_separation    Disable track separation.

    [transcription & ASR]
    --asr-engine            ASR engine: whisper_v2|whisper_turbo >> ((default) is whisper_v2)
    --whisper               Multilingual model > tiny|base|small|medium|large-v1|large-v2|large-v3 >> ((default) is large-v2)
                            English-only model > tiny.en|base.en|small.en|medium.en
    --aligner-engine        Phonetic aligner: wav2vec2|mms_fa >> ((default) is wav2vec2)
    --language              Language code (e.g., "es", "en") for Whisper & phonetic alignment
    --genre                 Music genre for acoustic VAD & rhythmic grids (e.g., "Rock", "Hip Hop/R&B")
    --keep_numbers          Transcribe numbers as digits instead of words

    [pitch & midi]
    --pitch-engine          Pitch tracker: rmvpe|crepe|swiftf0|basic-pitch >> ((default) is rmvpe)
    --crepe_model           CREPE model capacity: tiny|small|medium|large|full >> ((default) is full)
    --midi-engine           MIDI extractor: basic_pitch|some|omnizart >> ((default) is basic_pitch)
    --magenta-humanize      Enable dynamic velocity and micro-timing humanization
    --disable_melismas      Disable melisma detection (force 1 solid note per syllable)
    --melisma_threshold     Minimum semitone deviation to split melisma note >> ((default) is 1.8)
    --quantize_to_key       Quantize notes to detected musical key >> ((default) is enabled)

    [duet]
    --duet                  Enable duet mode (generates official USDX P 1 and P 2 blocks)
    --duet-engine           Duet diarization engine: mfcc_kmeans|ecapa_embeddings >> ((default) is mfcc_kmeans)
    --p1_name               Name of Singer 1
    --p2_name               Name of Singer 2

    [qa & extra]
    --autocorrect           Run 7-equation acoustic calibration and official USDX scoring simulation
    --to_whisper_out        Quick mode: export whisper.txt transcription and skip pitch extraction
    --keep_cache            Keep cache folder after creation. Cache folder is removed by default.
    --plot                  Enable creation of visual QA plots. Plots are disabled by default.
    --format_version        0.3.0|1.0.0|1.1.0|1.2.0 >> ((default) is 1.2.0)
    --musescore_path        Path to MuseScore executable
    --ffmpeg                Path to ffmpeg and ffprobe executable

    [yt-dlp]
    --cookiefile            File name where cookies should be read from

    [device]
    --force_cpu             Force all steps to be processed on CPU.
    --force_whisper_cpu     Only whisper will be forced to cpu
```

For standard use, you only need to use [opt]. All other options are optional.

### 🎶 Input

### Mode
default (Full Automatic Mode) - Creates all, depending on command line options
--interactive - Interactive Mode. All options are asked at runtime for easier configuration
```commandline
--interactive
```
#### Audio / Video (full automatic)

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

Pitching is done with the `SwiftF0` model, which is faster and more accurate than CREPE.
SwiftF0 automatically detects pitch frequencies between 46.875 Hz (G1) and 2093.75 Hz (C7).
UltraSinger uses 60hz and 400hz

### 👄 Separation

The vocals are separated from the audio before they are passed to the models. If problems occur with this,
you have the option to disable this function; in which case the original audio file is used instead.

```commandline
-i XYZ --disable_separation
```

### 🎵 Key Quantization

By default, UltraSinger quantizes all detected notes to the detected musical key of the song. This helps to:
* Remove pitch slides and vocal transitions between notes
* Correct out-of-key notes from pitch detection errors

If you want to keep the raw pitch detection without quantization, set `quantize_to_key = False`.
For most songs  quantization should produces better results.

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

You need an NVIDIA CUDA device for this to work. Sorry, there is no CUDA device for macOS.

For GPU support on Windows and Linux, the installation script automatically installs PyTorch with CUDA support.

It is optional (but recommended) to install the CUDA driver for your GPU: see [CUDA driver](https://developer.nvidia.com/cuda-downloads).
Also check your GPU CUDA support. See [CUDA support](https://gist.github.com/standaloneSA/99788f30466516dbcc00338b36ad5acf)

For manual installation, you can use:
```bash
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

#### Crashes due to low VRAM

If something crashes because of low VRAM then use a smaller Whisper model.
Whisper needs more than 8GB VRAM in the `large` model!

You can also force CPU usage with the extra option `--force_cpu`.

### 📦 Containerized (Docker or Podman)

See [container/README.md](container/README.md)
