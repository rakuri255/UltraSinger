# UltraSinger 

<a href="https://www.buymeacoffee.com/rakuri255" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>

_This project is still under development!_

_The output files of the full automatic are currently not really usable!
But the re-pitched files can be used now._

UltraSinger is a tool to automatically create UltraStar.txt, midi and notes from music. 
It also can re-pitch current UltraStar files.

Multiple AI models are used to extract text from the voice and to determine the pitch. 

## Requirement

You need FFmpeg installed.

## How to use

_Not all options working now!_
```commandline
    this.py [opt] [mode] [rec model] [pitcher] [extra]
    
    [opt]
    -h      This help text.
    
    -i      Ultrastar.txt
            audio like .mp3, .wav, youtube link
    
    -o      Output folder
    
    [mode]
    ## INPUT is audio ##
    -a      Is default
            Creates all
    (-u      Create ultrastar txt file) # In Progress
    (-m      Create midi file) # In Progress
    (-s      Create sheet file) # In Progress
    
    ## INPUT is ultrastar.txt ##
    -a    Is default
            Creates all
    (-r      repitch Ultrastar.txt (input has to be audio)) # In Progress
    (-p      Check pitch of Ultrastar.txt input) # In Progress
    (-m      Create midi file # In Progress)

    [rec model]
    -v     vosk model path
      
    [extra]
    (-k      Keep audio chunks) # In Progress
    
    [pitcher]
    -crepe              default
    --crepe_model       tiny|small|medium|large|full
```

### Input

#### Audio

For audio it uses Vosk transcription model. This model is not included. You can download it here [Link](https://alphacephei.com/vosk/models).
Make sure you take the right language. Also consider that a bigger model is more accurate, but also takes longer to transcribe.
For the first test run, use the `small model`, to be accurate use the `gigaspeech model`

##### Local file

```commandline
-i "input/music.mp3" -v "models\vosk-model-en-us-0.42-gigaspeech"
```

##### Youtube

```commandline
-i https://www.youtube.com/watch?v=BaW_jenozKc -v "models\vosk-model-en-us-0.42-gigaspeech"
```

#### UltraStar

This re-pitch the audio and creates a new txt file.

```commandline
-i "input/ultrastar.txt"
```

### Pitcher

Pitching is done with the crepe model. 
Also consider that a bigger model is more accurate, but also takes longer to pitch.
For just testing you should use `tiny`, which is currently default.
If you want solid accurate, then use the `full` model.

```commandline
-i XYZ --crepe_model full
```