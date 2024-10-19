# UltraSinger evaluation

This tool exists to measure the accuracy of UltraSinger.

It takes a directory of known-good UltraStar format files, runs them through UltraSinger, and compares the output to the
original files.

The idea is, that as you make changes to UltraSinger, you can run this tool to see how the changes affect the accuracy
of UltraSinger. The tool will reuse any cached files from previous runs, as long as the configuration used to generate the cache is the same.

## Measurements taken

### Pitch

#### Base measurements

| measurement                          | description                                                                                                                       |
|--------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| input_match_ratio                    | ratio of how many of the pitch datapoints in the **input** can be found as an exact match in the _output_                         |
| output_match_ratio                   | ratio of how many of the pitch datapoints in the _output_ can be found as an exact match in the **input**                         |
| no_pitch_where_should_be_pitch_ratio | ratio of how many of the datapoints in the **input** have a pitch, where the corresponding datapoint in the _output_ has no pitch |
| pitch_where_should_be_no_pitch_ratio | ratio of how many of the datapoints in the _output_ have a pitch, where the corresponding datapoint in the **input** has no pitch |

#### Measurements after transposing the output

For these measurements the output is transposed by up to 12 half-steps, and the octave is being ignored when comparing
to the input. Whichever half-step value scores highest is used. This accounts for octave mismatches and wrongly
transposed inputs

| measurement                                        | description                                                                                                                        |
|----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| best_input_pitch_shift_match_ratio                 | same as input_match_ratio but after transposing the _output_ to achieve the highest possible input_match_ratio                     |
| matching_input_best_output_pitch_shift_match_ratio | the corresponding output_match_ratio when transposing the same amount of half-steps as used for best_input_pitch_shift_match_ratio |
| best_output_pitch_shift_match_ratio                | same as output_match_ratio but after transposing the _output_ to achieve the highest possible output_match_ratio                   |
| matching_output_best_input_pitch_shift_match_ratio | the corresponding input_match_ratio when transposing the same amount of half-steps as used for best_output_pitch_shift_match_ratio |



## Running the evaluation

- Copy the `example.local.py` file in the `evaluation/input/config` directory and name it `local.py`. This file is used to configure the evaluation tool.
- Add songs to the `evaluation/input/songs` directory. You can use the songs from https://github.com/UltraStar-Deluxe/songs.
- Simply run `py UltraSingerEvaluation.py` after following the "How to use this source code/Run" instructions in the root README.md.
- The evaluation tool will create a directory in the `evaluation/output` directory with the current date and time as the name. The output of the evaluation will be stored in this directory.

### Comparing runs

- To compare the results of all runs in the `evaluation/output` folder, run `py UltraSingerMetaEvaluation.py`. This will output each run's measurements to the console.

## Directory structure

```
evaluation
├───input
│   ├───config # programmatic configuration of UltraSingerEvaluation
│   │   │   example.local.py # example configuration file, copy this and name it local.py
│   │   │   local.py # your configuration file, UltraSingerEvaluation will look for this file
│   │   │
│   └───songs # this is the directory that contains the known-good songs to run through UltraSinger and then compare against
│       ├───Jonathan Coulton - A Talk with George
│       │   │   audio.mp3
│       │   │   background.jpg
│       │   │   cover.jpg
│       │   │   license.txt
│       │   │   song.txt # known good input UltraStar txt file. UltraSingerEvaluation compares this to the output of UltraSinger
│       │   │
│       │   └───cache # this cache will be reused for subsequent evaluation runs
│       │       │   crepe_False_full_10_cuda.json # the cached file's name contains the configuration used to generate it
│       │       │   Jonathan Coulton - A Talk with George.wav
│       │       │   Jonathan Coulton - A Talk with George_denoised.wav
│       │       │   Jonathan Coulton - A Talk with George_mono.wav
│       │       │   Jonathan Coulton - A Talk with George_mute.wav
│       │       │   whisper_large-v2_cuda_None_None_16_None_en.json # the cached file's name contains the configuration used to generate it
│       │       │
│       │       └───separated
│       │           └───htdemucs
│       │               └───audio
│       │                       no_vocals.wav
│       │                       vocals.wav
│       │
│       ├───...
│       │   │   ...
│       │
│       └───Many - Songs
│           │   ...
│
└───output
    └───2024-07-27_16-58-27
        │   run.json
        │
        └───songs
            ├───Jonathan Coulton - A Talk with George
            │       Jonathan Coulton - A Talk with George.txt # UltraStar txt file generated by UltraSinger
            │
            ├───...
            │       ....txt # UltraStar txt file generated by UltraSinger
            │
            └───Many - Songs
                    Many - Songs.txt # UltraStar txt file generated by UltraSinger
```

## TODO

- automate comparison in [UltraSingerMetaEvaluation.py](..%2Fsrc%2FUltraSingerMetaEvaluation.py) instead of just printing each run's measurements
- currently only pitch accuracy is being measured, text accuracy should be measured as well
- the cached file's configuration is part of their filename, this will quickly become unmanageable, a better way to store this information should be found
- the tool could verify that there are no changes according to git and store the latest commit hash for a test run ([TestRun.py](..%2Fsrc%2Fmodules%2FEvaluation%2FTestRun.py))